"""Role D inference orchestration.

This module owns the runtime boundary between Roles A, B and C. It intentionally
contains no crop, canonicalisation, model architecture, metric, template-building,
or post-processing implementation. Those remain in their owners' modules.

Runtime contract::

    raw PLY -> {"left": [85,3], "right": [85,3]}

Predictions are returned in Huawei's original coordinate frame.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from .data import RawSubject, load_mesh, subject_id_from_path
from .geometry import (
    DEFAULT_CROP_CONFIG,
    N_LANDMARKS,
    N_POINTS,
    CanonicalEar,
    canonicalize_ear,
    inverse_transform_points,
    load_crop_config,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANONICAL_TEMPLATE = "outputs/role_c/canonical_template_shared.npz"
DEFAULT_GLOBAL_TEMPLATE = "outputs/role_c/global_template.npz"


@dataclass(frozen=True)
class InferenceConfig:
    """Small, explicit configuration surface for inference."""

    mode: str = "fallback"  # "fallback" | "model"
    crop_config: str = DEFAULT_CROP_CONFIG
    canonical_template: str = DEFAULT_CANONICAL_TEMPLATE
    global_template: str = DEFAULT_GLOBAL_TEMPLATE
    checkpoint: str | None = None
    ensemble_checkpoints: tuple[str, ...] = field(default_factory=tuple)
    n_points: int = N_POINTS
    mirror_side: str = "right"
    mirror_axis: int = 1
    seed: int = 0
    device: str = "auto"  # "auto" | "cpu" | "cuda"
    tta_samples: int = 1
    postprocess_enabled: bool = False
    template_version: str = "B1-shared-canonical"

    def __post_init__(self) -> None:
        if self.mode not in {"fallback", "model"}:
            raise ValueError("mode must be 'fallback' or 'model'")
        if self.n_points <= 0:
            raise ValueError("n_points must be > 0")
        if self.mirror_side not in {"left", "right"}:
            raise ValueError("mirror_side must be 'left' or 'right'")
        if self.mirror_axis not in {0, 1, 2}:
            raise ValueError("mirror_axis must be 0, 1 or 2")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be 'auto', 'cpu' or 'cuda'")
        if self.tta_samples <= 0:
            raise ValueError("tta_samples must be >= 1")
        if self.mode == "model" and not (self.checkpoint or self.ensemble_checkpoints):
            raise ValueError("model mode requires checkpoint or ensemble_checkpoints")
        if self.ensemble_checkpoints and self.checkpoint:
            raise ValueError("set checkpoint or ensemble_checkpoints, not both")
        if self.tta_samples > 1 and self.postprocess_enabled:
            # Not mathematically unsafe, but keeps the ordering explicit: D only
            # wires C's post-processing after its validated prediction averaging.
            raise ValueError(
                "tta_samples > 1 with postprocess_enabled is unsupported until C's "
                "validated post-processing ordering is supplied"
            )


@dataclass(frozen=True)
class RuntimeComponents:
    """Loaded artifacts. Tests can inject these without touching A/B/C code."""

    canonical_template: np.ndarray | None = None
    global_template: dict[str, np.ndarray] | None = None
    models: tuple[Any, ...] = field(default_factory=tuple)
    device: Any = None


@dataclass(frozen=True)
class SubjectPrediction:
    """Validated prediction object with an explicit subject id."""

    subject_id: str
    left: np.ndarray
    right: np.ndarray

    def as_dict(self) -> dict[str, np.ndarray]:
        return {"left": self.left, "right": self.right}



def _resolve_artifact_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_file():
        return p
    if not p.is_absolute():
        alt = REPO_ROOT / p
        if alt.is_file():
            return alt
    raise FileNotFoundError(f"artifact not found: {p}")



def _load_canonical_template(path: str | Path) -> np.ndarray:
    p = _resolve_artifact_path(path)
    with np.load(p) as npz:
        if "template" not in npz.files:
            raise ValueError(f"{p}: expected Role C template key 'template'")
        template = np.asarray(npz["template"], dtype=np.float64)
    _validate_landmark_array(template, "canonical template")
    return template



def _load_global_template(path: str | Path) -> dict[str, np.ndarray]:
    p = _resolve_artifact_path(path)
    with np.load(p) as npz:
        missing = {side for side in ("left", "right") if side not in npz.files}
        if missing:
            raise ValueError(f"{p}: global template missing keys {sorted(missing)}")
        out = {side: np.asarray(npz[side], dtype=np.float64) for side in ("left", "right")}
    for side, arr in out.items():
        _validate_landmark_array(arr, f"global template {side}")
    return out



def _validate_landmark_array(array: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(array)
    if arr.shape != (N_LANDMARKS, 3):
        raise ValueError(f"{name} must have shape ({N_LANDMARKS}, 3), got {arr.shape}")
    if not np.issubdtype(arr.dtype, np.number):
        raise TypeError(f"{name} must be numeric, got {arr.dtype}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return arr



def validate_prediction_result(result: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Enforce the challenge-facing left/right prediction contract."""
    if set(result.keys()) != {"left", "right"}:
        raise ValueError("prediction result must contain exactly 'left' and 'right'")
    checked: dict[str, np.ndarray] = {}
    for side in ("left", "right"):
        checked[side] = _validate_landmark_array(np.asarray(result[side]), f"prediction {side}")
    return checked



def load_inference_config(path: str | Path) -> InferenceConfig:
    """Load the intentionally small YAML runtime config."""
    p = Path(path)
    if not p.is_file() and not p.is_absolute():
        p = REPO_ROOT / p
    if not p.is_file():
        raise FileNotFoundError(f"inference config not found: {path}")

    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{p}: config must be a YAML mapping")

    allowed = set(InferenceConfig.__dataclass_fields__)
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"{p}: unknown inference config keys: {sorted(unknown)}")

    data = dict(raw)
    if "ensemble_checkpoints" in data:
        data["ensemble_checkpoints"] = tuple(str(x) for x in data["ensemble_checkpoints"])
    return InferenceConfig(**data)



def _derive_ear_seed(seed: int, subject_id: str, side: str, sample_idx: int) -> int:
    """Stable per-ear sample seed; never depends on Python's salted hash()."""
    raw = f"{seed}|{subject_id}|{side}|{sample_idx}".encode("utf-8")
    value = int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")
    return value & ((1 << 63) - 1)



def _select_device(requested: str) -> Any:
    import torch

    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("config requested CUDA, but torch.cuda.is_available() is false")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")



def _load_models(config: InferenceConfig, device: Any) -> tuple[Any, ...]:
    from .train import load_checkpoint

    paths = [config.checkpoint] if config.checkpoint else list(config.ensemble_checkpoints)
    models: list[Any] = []
    model_signature: dict[str, Any] | None = None

    for raw_path in paths:
        assert raw_path is not None
        path = _resolve_artifact_path(raw_path)
        try:
            model, ckpt = load_checkpoint(path, map_location=device)
        except Exception as exc:
            raise RuntimeError(f"failed to load checkpoint {path}: {exc}") from exc

        required_keys = ("model_config", "n_points", "template_version", "usage")
        missing = [key for key in required_keys if key not in ckpt]
        if missing:
            raise ValueError(f"{path}: checkpoint missing required Role B keys {missing}")

        cfg = dict(ckpt["model_config"])
        if cfg.get("in_dim") != 3:
            raise ValueError(f"{path}: expected model in_dim=3, got {cfg.get('in_dim')}")
        if cfg.get("n_landmarks") != N_LANDMARKS:
            raise ValueError(
                f"{path}: expected n_landmarks={N_LANDMARKS}, got {cfg.get('n_landmarks')}"
            )
        if int(ckpt["n_points"]) != config.n_points:
            raise ValueError(
                f"{path}: checkpoint n_points={ckpt['n_points']} but inference config "
                f"n_points={config.n_points}"
            )
        if str(ckpt["template_version"]) != config.template_version:
            raise ValueError(
                f"{path}: checkpoint template_version={ckpt['template_version']!r} "
                f"does not match inference template_version={config.template_version!r}"
            )

        signature = {
            key: cfg.get(key)
            for key in (
                "in_dim",
                "n_landmarks",
                "point_dim1",
                "point_dim2",
                "point_dim3",
                "head_dim1",
                "head_dim2",
                "dropout",
            )
        }
        if model_signature is None:
            model_signature = signature
        elif signature != model_signature:
            raise ValueError(f"{path}: ensemble checkpoint architecture/config is incompatible")

        model.to(device)
        model.eval()
        models.append(model)

    if not models:
        raise ValueError("no model checkpoints supplied")
    return tuple(models)



def load_runtime(config: InferenceConfig) -> RuntimeComponents:
    """Load exactly the artifacts requested by the runtime config."""
    if config.mode == "fallback":
        return RuntimeComponents(global_template=_load_global_template(config.global_template))

    import torch  # lazy: fallback mode does not need to import Torch here

    device = _select_device(config.device)
    canonical = _load_canonical_template(config.canonical_template)
    models = _load_models(config, device)
    return RuntimeComponents(canonical_template=canonical, models=models, device=device)



def _predict_residuals(models: Sequence[Any], points: np.ndarray, device: Any) -> np.ndarray:
    import torch

    x = torch.from_numpy(np.asarray(points, dtype=np.float32))[None, ...].to(device)
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for model in models:
            pred = model(x)
            arr = pred[0].detach().cpu().numpy().astype(np.float64)
            _validate_landmark_array(arr, "model residual")
            predictions.append(arr)
    residual = np.mean(np.stack(predictions, axis=0), axis=0)
    return residual



def _apply_postprocessing(
    prediction: np.ndarray,
    ear: CanonicalEar,
    config: InferenceConfig,
) -> np.ndarray:
    if not config.postprocess_enabled:
        return prediction

    # C owns implementation and validation. D only calls the agreed hook.
    try:
        from .postprocess import apply_enabled_postprocessing  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "postprocess_enabled=true but Role C's src.postprocess.py is not present"
        ) from exc
    processed = apply_enabled_postprocessing(prediction, ear, config)  # type: ignore[arg-type]
    return _validate_landmark_array(np.asarray(processed), "postprocessed prediction")


class InferencePipeline:
    """Loaded end-to-end predictor; safe to reuse for many subjects."""

    def __init__(self, config: InferenceConfig, runtime: RuntimeComponents) -> None:
        self.config = config
        self.runtime = runtime

    @classmethod
    def from_config(cls, config: InferenceConfig) -> "InferencePipeline":
        return cls(config, load_runtime(config))

    def _predict_fallback(self) -> dict[str, np.ndarray]:
        assert self.runtime.global_template is not None
        return {
            "left": self.runtime.global_template["left"].copy(),
            "right": self.runtime.global_template["right"].copy(),
        }

    def _predict_model(self, raw: RawSubject) -> dict[str, np.ndarray]:
        assert self.runtime.canonical_template is not None
        assert self.runtime.models
        template = self.runtime.canonical_template
        out: dict[str, np.ndarray] = {}

        for side in ("left", "right"):
            crop_cfg = load_crop_config(side, self.config.crop_config)
            tta_preds: list[np.ndarray] = []
            for sample_idx in range(self.config.tta_samples):
                ear_seed = _derive_ear_seed(self.config.seed, raw.subject_id, side, sample_idx)
                ear = canonicalize_ear(
                    raw,
                    side,
                    crop_cfg,
                    mirror_side=self.config.mirror_side,
                    n_points=self.config.n_points,
                    seed=ear_seed,
                    mirror_axis=self.config.mirror_axis,
                )
                residual = _predict_residuals(self.runtime.models, ear.points, self.runtime.device)
                pred_canonical = template + residual
                pred_canonical = _validate_landmark_array(
                    pred_canonical, f"canonical prediction {side}"
                )
                pred_canonical = _apply_postprocessing(pred_canonical, ear, self.config)
                pred_original = inverse_transform_points(pred_canonical, ear.transform)
                tta_preds.append(_validate_landmark_array(pred_original, f"global prediction {side}"))

            out[side] = np.mean(np.stack(tta_preds, axis=0), axis=0)

        return validate_prediction_result(out)

    def predict(self, mesh_path: str | Path) -> SubjectPrediction:
        raw = load_mesh(mesh_path)
        if self.config.mode == "fallback":
            result = self._predict_fallback()
        else:
            result = self._predict_model(raw)
        checked = validate_prediction_result(result)
        return SubjectPrediction(raw.subject_id, checked["left"], checked["right"])



def predict_subject(
    mesh_path: str | Path,
    config: InferenceConfig | Mapping[str, Any] | str | Path,
    checkpoint: str | None = None,
) -> dict[str, np.ndarray]:
    """Public Role D API: raw PLY -> left/right 85x3 global-coordinate predictions."""
    if isinstance(config, InferenceConfig):
        cfg = config
    elif isinstance(config, (str, Path)):
        cfg = load_inference_config(config)
    else:
        cfg = InferenceConfig(**dict(config))

    if checkpoint is not None:
        cfg = replace(cfg, checkpoint=checkpoint, ensemble_checkpoints=())

    pipeline = InferencePipeline.from_config(cfg)
    return pipeline.predict(mesh_path).as_dict()


def _format_csv_rows(points: np.ndarray) -> str:
    lines = []
    for idx, xyz in enumerate(points):
        lines.append(f"{idx},[{xyz[0]:.10g} {xyz[1]:.10g} {xyz[2]:.10g}]")
    return "\n".join(lines) + "\n"



def write_prediction(
    result: Mapping[str, Any],
    output_dir: str | Path,
    subject_id: str,
    *,
    fmt: str = "npz",
) -> tuple[Path, ...]:
    """Write a validated, self-contained prediction artifact.

    ``npz`` is the canonical repo output because the public challenge page specifies
    two 85x3 landmark matrices but does not publish a public filename/serialization
    schema. ``csv`` mirrors the verified annotation row layout for inspection only.
    """
    checked = validate_prediction_result(result)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if fmt not in {"npz", "csv", "both"}:
        raise ValueError("fmt must be 'npz', 'csv' or 'both'")

    paths: list[Path] = []
    if fmt in {"npz", "both"}:
        final = out_dir / f"{subject_id}.npz"
        tmp = out_dir / f".{subject_id}.npz.tmp"
        with open(tmp, "wb") as fh:
            np.savez(
                fh,
                subject_id=np.asarray(subject_id),
                left=checked["left"].astype(np.float64),
                right=checked["right"].astype(np.float64),
            )
        tmp.replace(final)
        paths.append(final)

    if fmt in {"csv", "both"}:
        for side in ("left", "right"):
            path = out_dir / f"{subject_id}_{side}.csv"
            path.write_text(_format_csv_rows(checked[side]), encoding="utf-8")
            paths.append(path)

    return tuple(paths)


__all__ = [
    "InferenceConfig",
    "InferencePipeline",
    "RuntimeComponents",
    "SubjectPrediction",
    "load_inference_config",
    "load_runtime",
    "predict_subject",
    "validate_prediction_result",
    "write_prediction",
]
