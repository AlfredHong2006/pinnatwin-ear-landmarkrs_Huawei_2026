"""Role D inference entry point.

Raw Huawei PLY -> left/right 85x3 landmarks in the original Huawei frame.

The estimator deliberately consumes Role A/B/C interfaces rather than
reimplementing crop/geometry/model logic.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from src.data import SIDES, load_mesh
from src.geometry import N_LANDMARKS, N_POINTS, canonicalize_ear, inverse_transform_points, load_crop_config
from src.train import load_checkpoint


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = "configs/infer.yaml"


def _resolve(path: str | Path) -> Path:
    """Resolve a relative path against the CWD, falling back to the repo root.

    Lets the evaluator instantiate LandmarkExtractor from any working directory.
    """
    p = Path(path)
    if p.is_absolute() or p.exists():
        return p
    return REPO_ROOT / p


class LandmarkExtractor:
    """End-to-end inference wrapper for the frozen A/B/C interfaces.

    Parameters
    ----------
    config:
        Path to the inference YAML. ``mode`` must be ``model`` or ``fallback``.
    device:
        Torch device override. Defaults to the config value or CPU.
    """

    def __init__(self, config: str | Path = DEFAULT_CONFIG, device: str | None = None) -> None:
        self.config_path = _resolve(config)
        if not self.config_path.is_file():
            raise FileNotFoundError(f"inference config not found: {self.config_path}")

        with self.config_path.open("r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
        if not isinstance(cfg, dict):
            raise ValueError(f"{self.config_path}: expected a YAML mapping")
        self.config: dict[str, Any] = cfg

        self.mode = str(cfg.get("mode", "fallback")).lower()
        if self.mode not in {"fallback", "model"}:
            raise ValueError(f"{self.config_path}: mode must be 'fallback' or 'model', got {self.mode!r}")

        self.n_points = int(cfg.get("n_points", N_POINTS))
        if self.n_points != N_POINTS:
            raise ValueError(f"n_points={self.n_points}; Role A/B contract requires {N_POINTS}")

        self.n_landmarks = int(cfg.get("n_landmarks", N_LANDMARKS))
        if self.n_landmarks != N_LANDMARKS:
            raise ValueError(f"n_landmarks={self.n_landmarks}; expected {N_LANDMARKS}")

        self.mirror_side = str(cfg.get("mirror_side", "right"))
        self.mirror_axis = int(cfg.get("mirror_axis", 1))
        if self.mirror_side not in SIDES:
            raise ValueError(f"mirror_side must be one of {SIDES}, got {self.mirror_side!r}")
        if self.mirror_axis not in (0, 1, 2):
            raise ValueError(f"mirror_axis must be 0, 1 or 2, got {self.mirror_axis}")

        self.crop_config_path = _resolve(cfg.get("crop_config", "configs/crop.yaml"))
        self.template_path = _resolve(cfg.get("template_file", "outputs/role_c/canonical_template_shared.npz"))
        self.template_version = str(cfg.get("template_version", "B1-shared-canonical"))
        self.seed = int(cfg.get("seed", 0))

        self.template = self._load_template(self.template_path)
        self.crop_configs = {
            side: load_crop_config(side, self.crop_config_path) for side in SIDES
        }

        self.device = torch.device(device or cfg.get("device", "cpu"))
        self.model = None
        self.checkpoint: dict[str, Any] | None = None

        if self.mode == "model":
            checkpoint_path = cfg.get("checkpoint")
            if not checkpoint_path:
                raise ValueError(f"{self.config_path}: model mode requires 'checkpoint'")
            self.model, self.checkpoint = load_checkpoint(_resolve(checkpoint_path), map_location=self.device)
            self.model.to(self.device)
            self.model.eval()

            ckpt_points = int(self.checkpoint.get("n_points", self.n_points))
            if ckpt_points != self.n_points:
                raise ValueError(
                    f"checkpoint expects n_points={ckpt_points}, config expects {self.n_points}"
                )
            ckpt_landmarks = int(self.checkpoint.get("n_landmarks", self.n_landmarks))
            if ckpt_landmarks != self.n_landmarks:
                raise ValueError(
                    f"checkpoint expects n_landmarks={ckpt_landmarks}, config expects {self.n_landmarks}"
                )

            ckpt_template_version = self.checkpoint.get("template_version")
            if ckpt_template_version and str(ckpt_template_version) != self.template_version:
                raise ValueError(
                    "template version mismatch: "
                    f"checkpoint={ckpt_template_version!r}, config={self.template_version!r}"
                )

    @staticmethod
    def _load_template(path: Path) -> np.ndarray:
        if not path.is_file():
            raise FileNotFoundError(f"canonical template not found: {path}")
        with np.load(path) as data:
            if "template" not in data:
                raise ValueError(f"{path}: expected npz key 'template'")
            template = np.asarray(data["template"], dtype=np.float64)
        if template.shape != (N_LANDMARKS, 3):
            raise ValueError(f"{path}: template must have shape ({N_LANDMARKS}, 3), got {template.shape}")
        if not np.isfinite(template).all():
            raise ValueError(f"{path}: template contains non-finite values")
        return template

    def _ear_seed(self, subject_id: str, side: str) -> int:
        # Same deterministic policy as scripts/preprocess.py / DATA_SPEC.
        digest = hashlib.sha256(f"{self.seed}|{subject_id}|{side}".encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)

    def predict_subject(self, mesh_path: str | Path) -> dict[str, np.ndarray]:
        """Predict left/right landmarks for one raw Huawei PLY."""
        raw = load_mesh(mesh_path)
        result: dict[str, np.ndarray] = {}

        for side in SIDES:
            ear = canonicalize_ear(
                raw,
                side,
                self.crop_configs[side],
                mirror_side=self.mirror_side,
                n_points=self.n_points,
                seed=self._ear_seed(raw.subject_id, side),
                mirror_axis=self.mirror_axis,
            )

            pred_canonical = self.template.copy()
            if self.mode == "model":
                assert self.model is not None
                points = torch.from_numpy(np.asarray(ear.points, dtype=np.float32))
                with torch.no_grad():
                    residual = self.model(points.unsqueeze(0).to(self.device))[0].cpu().numpy()
                if residual.shape != (self.n_landmarks, 3):
                    raise ValueError(
                        f"model returned {residual.shape}; expected ({self.n_landmarks}, 3)"
                    )
                pred_canonical = pred_canonical + np.asarray(residual, dtype=np.float64)

            pred_original = inverse_transform_points(pred_canonical, ear.transform)
            if pred_original.shape != (self.n_landmarks, 3):
                raise ValueError(
                    f"{side} prediction has shape {pred_original.shape}; expected ({self.n_landmarks}, 3)"
                )
            if not np.isfinite(pred_original).all():
                raise ValueError(f"{side} prediction contains non-finite values")
            result[side] = pred_original

        return result

    # Compatibility aliases for callers that use the entry point as an extractor.
    def predict(self, mesh_path: str | Path) -> dict[str, np.ndarray]:
        return self.predict_subject(mesh_path)

    def extract(self, mesh_path: str | Path) -> dict[str, np.ndarray]:
        return self.predict_subject(mesh_path)

    def __call__(self, mesh_path: str | Path) -> dict[str, np.ndarray]:
        return self.predict_subject(mesh_path)
