from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from src.estimator import LandmarkExtractor


def test_model_mode_is_explicit_in_inference_config():
    cfg_path = Path("configs/infer.yaml")
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    assert cfg["mode"] == "model"
    assert cfg["checkpoint"] == "outputs/role_b/augmented_seed0_best.pt"


def test_fallback_mode_preserves_canonical_template_contract(tmp_path, monkeypatch):
    template_path = tmp_path / "template.npz"
    template = np.zeros((85, 3), dtype=np.float32)
    np.savez(template_path, template=template)

    crop_path = tmp_path / "crop.yaml"
    crop_path.write_text(
        yaml.safe_dump(
            {
                "sides": {
                    "left": {"lo": [-2, -2, -2], "hi": [2, 2, 2], "min_vertices": 1},
                    "right": {"lo": [-2, -2, -2], "hi": [2, 2, 2], "min_vertices": 1},
                }
            }
        ),
        encoding="utf-8",
    )

    cfg = tmp_path / "infer.yaml"
    cfg.write_text(
        yaml.safe_dump(
            {
                "mode": "fallback",
                "n_points": 2048,
                "n_landmarks": 85,
                "crop_config": str(crop_path),
                "template_file": str(template_path),
            }
        ),
        encoding="utf-8",
    )

    extractor = LandmarkExtractor(cfg)
    assert extractor.mode == "fallback"
    assert extractor.template.shape == (85, 3)
