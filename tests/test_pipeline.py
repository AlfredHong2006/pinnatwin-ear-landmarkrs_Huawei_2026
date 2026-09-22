from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.data import RawSubject
from src.geometry import EarTransform, CanonicalEar
from src.pipeline import (
    InferenceConfig,
    InferencePipeline,
    RuntimeComponents,
    load_inference_config,
    predict_subject,
    validate_prediction_result,
    write_prediction,
)


class _FakeModel:
    def __init__(self, residual_value: float = 0.0) -> None:
        self.residual_value = residual_value

    def __call__(self, x):
        import torch

        return torch.full((x.shape[0], 85, 3), self.residual_value, dtype=torch.float32)

    def eval(self):
        return self

    def to(self, device):
        return self


def _fake_raw() -> RawSubject:
    return RawSubject(
        subject_id="P0001",
        vertices=np.zeros((8, 3), dtype=np.float64),
        faces=None,
        normals=None,
        mesh_path=Path("P0001.ply"),
    )


def _fake_ear(raw, side, cfg, mirror_side, n_points, seed, mirror_axis):
    centre = np.zeros(3, dtype=np.float64)
    transform = EarTransform(centre=centre, scale=1.0, mirror_axis=None)
    points = np.zeros((n_points, 3), dtype=np.float64)
    return CanonicalEar(raw.subject_id, side, points, None, transform, {"seed": seed})


def _model_pipeline(monkeypatch, template=None, residual_value=0.0, tta_samples=1):
    monkeypatch.setattr("src.pipeline.load_mesh", lambda path: _fake_raw())
    monkeypatch.setattr("src.pipeline.load_crop_config", lambda side, path: object())
    monkeypatch.setattr("src.pipeline.canonicalize_ear", _fake_ear)

    template = np.zeros((85, 3), dtype=np.float64) if template is None else template
    cfg = InferenceConfig(
        mode="model",
        checkpoint="dummy.pt",
        n_points=32,
        tta_samples=tta_samples,
    )
    runtime = RuntimeComponents(canonical_template=template, models=(_FakeModel(residual_value),), device="cpu")
    return InferencePipeline(cfg, runtime)


def test_model_end_to_end_smoke(monkeypatch):
    pipeline = _model_pipeline(monkeypatch, residual_value=0.25)
    result = pipeline.predict("P0001.ply")
    assert result.subject_id == "P0001"
    assert result.left.shape == (85, 3)
    assert result.right.shape == (85, 3)
    assert np.allclose(result.left, 0.25)
    assert np.allclose(result.right, 0.25)


def test_tta_is_deterministic_and_side_specific(monkeypatch):
    seen = []

    def fake_ear(raw, side, cfg, mirror_side, n_points, seed, mirror_axis):
        seen.append((side, seed))
        return _fake_ear(raw, side, cfg, mirror_side, n_points, seed, mirror_axis)

    monkeypatch.setattr("src.pipeline.load_mesh", lambda path: _fake_raw())
    monkeypatch.setattr("src.pipeline.load_crop_config", lambda side, path: object())
    monkeypatch.setattr("src.pipeline.canonicalize_ear", fake_ear)

    cfg = InferenceConfig(mode="model", checkpoint="dummy.pt", n_points=32, tta_samples=3, seed=7)
    runtime = RuntimeComponents(canonical_template=np.zeros((85, 3)), models=(_FakeModel(),), device="cpu")
    pipeline = InferencePipeline(cfg, runtime)
    pipeline.predict("P0001.ply")
    assert len(seen) == 6
    assert len({seed for _, seed in seen}) == 6
    assert [side for side, _ in seen].count("left") == 3
    assert [side for side, _ in seen].count("right") == 3


def test_fallback_does_not_need_a_or_b(monkeypatch, tmp_path):
    monkeypatch.setattr("src.pipeline.load_mesh", lambda path: _fake_raw())
    left = np.ones((85, 3), dtype=np.float64)
    right = np.full((85, 3), 2.0, dtype=np.float64)
    runtime = RuntimeComponents(global_template={"left": left, "right": right})
    cfg = InferenceConfig(mode="fallback", global_template=str(tmp_path / "not-used.npz"))
    prediction = InferencePipeline(cfg, runtime).predict("P0001.ply")
    assert np.array_equal(prediction.left, left)
    assert np.array_equal(prediction.right, right)


def test_validate_prediction_result_rejects_missing_side():
    with pytest.raises(ValueError, match="exactly 'left' and 'right'"):
        validate_prediction_result({"left": np.zeros((85, 3))})


def test_validate_prediction_result_rejects_wrong_shape():
    bad = {"left": np.zeros((84, 3)), "right": np.zeros((85, 3))}
    with pytest.raises(ValueError, match="shape"):
        validate_prediction_result(bad)


def test_validate_prediction_result_rejects_nan():
    left = np.zeros((85, 3))
    left[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN or Inf"):
        validate_prediction_result({"left": left, "right": np.zeros((85, 3))})


def test_writer_npz_and_csv(tmp_path):
    result = {"left": np.zeros((85, 3)), "right": np.ones((85, 3))}
    paths = write_prediction(result, tmp_path, "P0001", fmt="both")
    assert (tmp_path / "P0001.npz") in paths
    assert (tmp_path / "P0001_left.csv") in paths
    assert (tmp_path / "P0001_right.csv") in paths
    loaded = np.load(tmp_path / "P0001.npz")
    assert loaded["left"].shape == (85, 3)
    assert loaded["right"].shape == (85, 3)
    assert (tmp_path / "P0001_left.csv").read_text().splitlines()[0].startswith("0,[")


def test_config_loader_is_strict(tmp_path):
    path = tmp_path / "infer.yaml"
    path.write_text("mode: fallback\nunknown: true\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown inference config keys"):
        load_inference_config(path)


def test_predict_subject_accepts_mapping(monkeypatch):
    template = np.zeros((85, 3), dtype=np.float64)
    monkeypatch.setattr("src.pipeline.load_mesh", lambda path: _fake_raw())
    monkeypatch.setattr("src.pipeline._load_global_template", lambda path: {"left": template.copy(), "right": template.copy()})
    result = predict_subject("P0001.ply", {"mode": "fallback", "global_template": "unused.npz"})
    assert result["left"].shape == (85, 3)
    assert result["right"].shape == (85, 3)


def test_model_runtime_rejects_incompatible_checkpoint_points(tmp_path):
    import torch
    from src.model import EarLandmarkNet
    from src.train import TrainConfig, save_checkpoint
    from src.pipeline import load_runtime

    model = EarLandmarkNet().eval()
    ckpt = tmp_path / "run_002_best.pt"
    save_checkpoint(ckpt, model, TrainConfig(n_points=2048), epoch=300)
    template = tmp_path / "template.npz"
    np.savez(template, template=np.zeros((85, 3), dtype=np.float32))

    cfg = InferenceConfig(
        mode="model",
        canonical_template=str(template),
        checkpoint=str(ckpt),
        n_points=1024,
        device="cpu",
    )
    with pytest.raises(ValueError, match="n_points"):
        load_runtime(cfg)


def test_inverse_transform_failure_is_not_silenced(monkeypatch):
    pipeline = _model_pipeline(monkeypatch)
    monkeypatch.setattr(
        "src.pipeline.inverse_transform_points",
        lambda points, transform: (_ for _ in ()).throw(RuntimeError("inverse failed")),
    )
    with pytest.raises(RuntimeError, match="inverse failed"):
        pipeline.predict("P0001.ply")


def test_model_inference_does_not_require_annotations(monkeypatch):
    pipeline = _model_pipeline(monkeypatch)
    monkeypatch.setattr("src.data.load_landmarks", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("GT loaded")))
    result = pipeline.predict("P0001.ply")
    assert result.left.shape == (85, 3)
    assert result.right.shape == (85, 3)
