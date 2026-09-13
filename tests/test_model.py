"""
Role B — tests for the global landmark model.

    pytest tests/test_model.py -q

Fast by design (seconds), so they can be run constantly while developing. They
catch the failure mode that matters most in deep learning: shape bugs that do
NOT crash — the code runs, produces numbers, and the numbers are meaningless.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from src.augment import AugmentConfig, augment_ear
from src.cache import write_cached_ear
from src.geometry import N_LANDMARKS, CanonicalEar, EarTransform, inverse_transform_points
from src.losses import (
    canonical_to_mm,
    get_loss_fn,
    mean_euclidean_error,
    per_ear_error,
    per_landmark_error,
    smooth_l1_loss,
)
from src.model import EarLandmarkNet
from src.train import (
    CanonicalEarCacheDataset,
    SyntheticEarDataset,
    TrainConfig,
    evaluate_global_mm,
    load_checkpoint,
    load_split,
    load_template,
    save_checkpoint,
    set_seed,
)

N_POINTS_TEST = 256


# ---------------------------------------------------------------------
# Shape contract
# ---------------------------------------------------------------------


def test_output_shape_matches_contract():
    """[B, 2048, 3] in -> [B, 85, 3] out. The interface the team agreed."""
    model = EarLandmarkNet()
    assert model(torch.randn(4, 2048, 3)).shape == (4, 85, 3)


@pytest.mark.parametrize("batch_size", [2, 4, 16])
@pytest.mark.parametrize("n_points", [1024, 2048])
def test_shape_across_batch_and_point_counts(batch_size, n_points):
    """Point count must not be baked in — the 1024-vs-2048 experiment needs this."""
    model = EarLandmarkNet()
    assert model(torch.randn(batch_size, n_points, 3)).shape == (batch_size, 85, 3)


def test_configurable_widths():
    """Width is a controlled experiment in the brief, so it must be a parameter."""
    model = EarLandmarkNet(point_dim1=32, point_dim2=64, point_dim3=128, head_dim1=256)
    assert model(torch.randn(2, 512, 3)).shape == (2, 85, 3)


def test_rejects_wrong_input_dim():
    """A [B, F, N] tensor passed by mistake must fail loudly, not silently."""
    model = EarLandmarkNet(in_dim=3)
    with pytest.raises(ValueError):
        model(torch.randn(4, 3, 2048))


def test_n_landmarks_matches_role_a_constant():
    """85 is Role A's N_LANDMARKS, not a number we invented."""
    assert EarLandmarkNet().config["n_landmarks"] == N_LANDMARKS


# ---------------------------------------------------------------------
# Permutation invariance
# ---------------------------------------------------------------------


def test_permutation_invariance():
    """
    Shuffling an ear's points must not change the prediction. A scan has no
    natural point ordering, so order-dependence would be learned noise.

    eval() mode so dropout is off and the comparison is deterministic.
    """
    model = EarLandmarkNet()
    model.eval()
    x = torch.randn(2, 1024, 3)
    perm = torch.randperm(1024)
    with torch.no_grad():
        assert torch.allclose(model(x), model(x[:, perm, :]), atol=1e-5)


# ---------------------------------------------------------------------
# Training mechanics
# ---------------------------------------------------------------------


def test_all_parameters_receive_gradients():
    model = EarLandmarkNet(dropout=0.0)
    model.train()
    smooth_l1_loss(model(torch.randn(4, 512, 3)), torch.randn(4, 85, 3)).backward()
    missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, f"no gradient reached: {missing}"


def test_model_can_overfit_a_tiny_batch():
    """
    Miniature of the tiny-overfit gate. If this fails the bug is in the
    plumbing, not the architecture — do not make the network bigger.
    """
    set_seed(0)
    model = EarLandmarkNet(dropout=0.0)
    model.train()
    x = torch.randn(4, N_POINTS_TEST, 3)
    target = torch.randn(4, 85, 3) * 0.05
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    first = None
    for _ in range(300):
        opt.zero_grad()
        loss = smooth_l1_loss(model(x), target)
        loss.backward()
        opt.step()
        if first is None:
            first = loss.item()
    assert loss.item() < first * 0.1, f"loss only fell {first:.6f} -> {loss.item():.6f}"


def test_determinism_under_seed():
    set_seed(123)
    a = EarLandmarkNet().eval()
    set_seed(123)
    b = EarLandmarkNet().eval()
    x = torch.randn(2, 512, 3)
    with torch.no_grad():
        assert torch.allclose(a(x), b(x), atol=1e-6)


# ---------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------


def test_losses_are_zero_for_perfect_prediction():
    pred = torch.randn(4, 85, 3)
    assert smooth_l1_loss(pred, pred.clone()).item() == pytest.approx(0.0, abs=1e-7)
    assert mean_euclidean_error(pred, pred.clone()).item() == pytest.approx(0.0, abs=1e-7)


def test_mean_euclidean_error_is_a_real_distance():
    """A uniform (3,4,0) offset must give exactly 5 — the 3-4-5 triangle."""
    pred = torch.zeros(2, 85, 3)
    target = torch.zeros(2, 85, 3)
    target[..., 0] = 3.0
    target[..., 1] = 4.0
    assert mean_euclidean_error(pred, target).item() == pytest.approx(5.0)


def test_error_matches_role_c_metric():
    """
    Our canonical monitoring number must agree with Role C's official
    landmark_errors on the same arrays. If these ever diverge, one of us has a
    bug and it is much cheaper to find it here.
    """
    from src.evaluate import landmark_errors

    pred = torch.randn(6, 85, 3)
    target = torch.randn(6, 85, 3)
    ours = mean_euclidean_error(pred, target).item()
    theirs = float(landmark_errors(pred.numpy(), target.numpy()).mean())
    assert ours == pytest.approx(theirs, rel=1e-5)


def test_beta_default_is_appropriate_for_canonical_scale():
    """
    Guards the beta trap: with PyTorch's default beta=1.0, Smooth-L1 on
    canonical-scale errors is indistinguishable from MSE. Our default must not
    be.
    """
    pred = torch.zeros(2, 85, 3)
    target = torch.full((2, 85, 3), 0.3)
    with_default = smooth_l1_loss(pred, target).item()
    with_pytorch_default = smooth_l1_loss(pred, target, beta=1.0).item()
    assert with_default != pytest.approx(with_pytorch_default)


def test_canonical_to_mm_is_exact():
    """
    mm_error == canonical_error * transform.scale, exactly.

    Distance is invariant under translation and axis negation, and scales
    linearly with `scale`, so this is an identity rather than an approximation.
    Verified here against Role A's real inverse_transform_points.
    """
    rng = np.random.default_rng(0)
    transform = EarTransform(
        centre=np.array([12.0, -30.0, 4.5]), scale=41.7, mirror_axis=1
    )
    gt_can = rng.normal(size=(85, 3))
    pred_can = gt_can + rng.normal(scale=0.02, size=(85, 3))

    canonical = float(np.linalg.norm(pred_can - gt_can, axis=-1).mean())
    via_identity = canonical * transform.scale

    gt_mm = inverse_transform_points(gt_can, transform)
    pred_mm = inverse_transform_points(pred_can, transform)
    actual_mm = float(np.linalg.norm(pred_mm - gt_mm, axis=-1).mean())

    assert via_identity == pytest.approx(actual_mm, rel=1e-9)

    tensor_version = canonical_to_mm(
        torch.tensor([canonical]), torch.tensor([transform.scale])
    ).item()
    assert tensor_version == pytest.approx(actual_mm, rel=1e-6)


def test_per_landmark_and_per_ear_shapes():
    pred, target = torch.randn(4, 85, 3), torch.randn(4, 85, 3)
    assert per_landmark_error(pred, target).shape == (85,)
    assert per_ear_error(pred, target).shape == (4,)


def test_loss_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        smooth_l1_loss(torch.randn(4, 85, 3), torch.randn(4, 84, 3))


@pytest.mark.parametrize("name", ["smooth_l1", "l1", "mse"])
def test_get_loss_fn_known_names(name):
    fn = get_loss_fn(name)
    assert fn(torch.zeros(2, 85, 3), torch.zeros(2, 85, 3)).item() == pytest.approx(0.0)


def test_get_loss_fn_rejects_unknown():
    with pytest.raises(ValueError):
        get_loss_fn("chamfer")


# ---------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------


def test_checkpoint_round_trip_reproduces_outputs(tmp_path):
    """Rebuild from the file alone and get identical output — Role D depends on this."""
    set_seed(7)
    model = EarLandmarkNet(point_dim1=32, point_dim2=64, point_dim3=128, dropout=0.1).eval()
    x = torch.randn(2, 512, 3)
    with torch.no_grad():
        before = model(x)

    cfg = TrainConfig(seed=7, template_version="B1-shared-canonical")
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model, cfg, epoch=3, val_mm=5.1, val_canonical=0.0123)

    restored, ckpt = load_checkpoint(path)
    with torch.no_grad():
        after = restored(x)

    assert torch.allclose(before, after, atol=1e-6)
    assert ckpt["epoch"] == 3
    assert ckpt["seed"] == 7
    assert ckpt["template_version"] == "B1-shared-canonical"
    assert ckpt["model_config"]["point_dim1"] == 32
    assert ckpt["val_mean_mm"] == pytest.approx(5.1)


def test_checkpoint_carries_everything_role_d_needs(tmp_path):
    model = EarLandmarkNet()
    cfg = TrainConfig()
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model, cfg, epoch=1)
    _, ckpt = load_checkpoint(path)
    for key in (
        "model_class", "model_state_dict", "model_config", "in_dim", "n_landmarks",
        "n_points", "template_version", "template_file", "split_file", "epoch",
        "seed", "train_config", "torch_version", "created_utc", "usage",
    ):
        assert key in ckpt, f"checkpoint is missing {key!r}"


# ---------------------------------------------------------------------
# Synthetic dataset
# ---------------------------------------------------------------------


def test_synthetic_dataset_shapes():
    ds = SyntheticEarDataset(n_samples=8, n_points=512)
    points, residual, idx = ds[0]
    assert points.shape == (512, 3)
    assert residual.shape == (85, 3)
    assert idx == 0
    assert len(ds) == 8


def test_synthetic_targets_are_deterministic_not_noise():
    """A flat loss on noise targets would be uninformative — the mapping must be real."""
    ds = SyntheticEarDataset(n_samples=4, n_points=256, seed=3)
    assert torch.allclose(ds[1][1], ds[1][1])
    assert not torch.allclose(ds[0][1], ds[1][1])


# ---------------------------------------------------------------------
# Role C artefact loading
# ---------------------------------------------------------------------


def test_load_template_reads_role_c_format(tmp_path):
    """C writes an npz with key 'template'. Anything else must fail clearly."""
    path = tmp_path / "canonical_template_shared.npz"
    np.savez(path, template=np.zeros((85, 3), dtype=np.float32), n_train_ears=np.int64(320))
    assert load_template(path).shape == (85, 3)


def test_load_template_rejects_wrong_key(tmp_path):
    path = tmp_path / "bad.npz"
    np.savez(path, mean_ear=np.zeros((85, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="template"):
        load_template(path)


def test_load_split_reads_role_c_keys(tmp_path):
    """C's keys are train_subjects / val_subjects, not train / val."""
    path = tmp_path / "split.json"
    path.write_text(json.dumps({
        "seed": 42, "train_subjects": ["P0001", "P0002"], "val_subjects": ["P0003"]
    }))
    train, val = load_split(path)
    assert train == ["P0001", "P0002"]
    assert val == ["P0003"]


def test_load_split_rejects_overlap(tmp_path):
    path = tmp_path / "split.json"
    path.write_text(json.dumps({
        "train_subjects": ["P0001"], "val_subjects": ["P0001"]
    }))
    with pytest.raises(ValueError, match="overlap"):
        load_split(path)


# ---------------------------------------------------------------------
# Cache-backed dataset, using Role A's own writer
# ---------------------------------------------------------------------


def _write_fake_ear(cache_dir, subject_id, side, n_points, rng):
    """Build one cache file through Role A's writer so the schema stays honest."""
    transform = EarTransform(
        centre=rng.normal(scale=20.0, size=3),
        scale=float(rng.uniform(35.0, 50.0)),
        mirror_axis=1 if side == "right" else None,
    )
    ear = CanonicalEar(
        subject_id=subject_id,
        side=side,
        points=rng.normal(scale=0.4, size=(n_points, 3)),
        normals=None,
        transform=transform,
        qa={"n_vertices": 5000, "suspicious": False},
    )
    targets = rng.normal(scale=0.3, size=(N_LANDMARKS, 3))
    path = cache_dir / f"{subject_id}_{side}.npz"
    write_cached_ear(
        path, ear, seed=0, ear_seed=1, crop_config_sha256="deadbeef",
        split_file="fake", split_sha256="cafe", targets=targets,
    )
    return targets


@pytest.fixture
def fake_cache(tmp_path):
    """A four-ear cache plus a matching template, written via Role A's API."""
    rng = np.random.default_rng(0)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    all_targets = []
    for sid in ("P0001", "P0002"):
        for side in ("left", "right"):
            all_targets.append(_write_fake_ear(cache_dir, sid, side, N_POINTS_TEST, rng))
    template = np.stack(all_targets).mean(axis=0)
    template_path = tmp_path / "template.npz"
    np.savez(template_path, template=template.astype(np.float32))
    return cache_dir, template_path


def test_cache_dataset_reads_through_role_a_api(fake_cache):
    cache_dir, template_path = fake_cache
    template = load_template(template_path)
    ds = CanonicalEarCacheDataset(
        cache_dir, ["P0001", "P0002"], template, n_points=N_POINTS_TEST
    )
    assert len(ds) == 4
    points, residual, idx = ds[0]
    assert points.shape == (N_POINTS_TEST, 3)
    assert residual.shape == (85, 3)
    assert points.dtype == torch.float32
    assert idx == 0


def test_cache_dataset_residual_is_target_minus_template(fake_cache):
    """The one arithmetic fact the whole role rests on."""
    cache_dir, template_path = fake_cache
    template = load_template(template_path)
    ds = CanonicalEarCacheDataset(cache_dir, ["P0001"], template, n_points=N_POINTS_TEST)
    for i in range(len(ds)):
        recovered = ds.residuals[i].numpy() + template.numpy()
        assert np.allclose(recovered, ds.targets_canonical[i], atol=1e-5)


def test_cache_dataset_honours_subject_filter(fake_cache):
    """Passing only train subjects must never pull in a val subject."""
    cache_dir, template_path = fake_cache
    template = load_template(template_path)
    ds = CanonicalEarCacheDataset(cache_dir, ["P0001"], template, n_points=N_POINTS_TEST)
    assert len(ds) == 2
    assert {sid for sid, _ in ds.meta} == {"P0001"}


def test_cache_dataset_limit_for_tiny_overfit(fake_cache):
    cache_dir, template_path = fake_cache
    template = load_template(template_path)
    ds = CanonicalEarCacheDataset(
        cache_dir, ["P0001", "P0002"], template, n_points=N_POINTS_TEST, limit=2
    )
    assert len(ds) == 2


def test_cache_dataset_rejects_wrong_point_count(fake_cache):
    cache_dir, template_path = fake_cache
    template = load_template(template_path)
    with pytest.raises(ValueError, match="n_points"):
        CanonicalEarCacheDataset(cache_dir, ["P0001"], template, n_points=2048)


def test_zero_residual_reproduces_the_template_baseline(fake_cache):
    """
    A model that outputs exactly zero must score exactly Role C's canonical
    template baseline. This is the sanity check that the residual framing is
    wired up correctly: your job really is 'beat zero'.
    """
    cache_dir, template_path = fake_cache
    template = load_template(template_path)
    ds = CanonicalEarCacheDataset(
        cache_dir, ["P0001", "P0002"], template, n_points=N_POINTS_TEST
    )

    class ZeroModel(torch.nn.Module):
        def forward(self, x):
            return torch.zeros(x.shape[0], 85, 3)

    metrics = evaluate_global_mm(ZeroModel(), ds, torch.device("cpu"), batch_size=2)

    # Reproduce the baseline independently, the way run_canonical_baseline.py does.
    expected = []
    for i in range(len(ds)):
        t = ds.transforms[i]
        pred_g = inverse_transform_points(template.numpy().astype(np.float64), t)
        gt_g = inverse_transform_points(ds.targets_canonical[i], t)
        expected.append(np.linalg.norm(pred_g - gt_g, axis=-1))
    assert metrics["mean"] == pytest.approx(float(np.stack(expected).mean()), rel=1e-6)
    assert metrics["n_val_ears"] == 4


def test_evaluate_global_mm_reports_both_sides(fake_cache):
    cache_dir, template_path = fake_cache
    template = load_template(template_path)
    ds = CanonicalEarCacheDataset(
        cache_dir, ["P0001", "P0002"], template, n_points=N_POINTS_TEST
    )
    model = EarLandmarkNet().eval()
    metrics = evaluate_global_mm(model, ds, torch.device("cpu"), batch_size=2)
    assert metrics["left_mean"] is not None
    assert metrics["right_mean"] is not None
    assert len(metrics["per_landmark_mean"]) == 85


# ---------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------


def test_augment_disabled_is_identity():
    points = np.random.default_rng(0).normal(size=(64, 3))
    target = np.random.default_rng(1).normal(size=(85, 3))
    p_aug, t_aug = augment_ear(points, target, AugmentConfig(enabled=False))
    assert p_aug is points
    assert t_aug is target


def test_augment_rotation_preserves_pairwise_distances():
    """A rotation must not distort the ear -- only reorient it."""
    rng = np.random.default_rng(2)
    points = rng.normal(size=(32, 3))
    target = rng.normal(size=(85, 3))
    cfg = AugmentConfig(enabled=True, rotation_deg=30.0, scale_min=1.0, scale_max=1.0)
    np.random.seed(0)
    p_aug, _ = augment_ear(points, target, cfg)
    d_before = np.linalg.norm(points[0] - points[1])
    d_after = np.linalg.norm(p_aug[0] - p_aug[1])
    assert d_after == pytest.approx(d_before, rel=1e-6)


def test_augment_scale_and_rotation_move_target_consistently():
    """
    Fixing rotation_deg=0 and scale=2.0 makes the whole transform a pure,
    deterministic doubling -- the one case simple enough to assert exactly.
    """
    rng = np.random.default_rng(3)
    points = rng.normal(size=(16, 3))
    target = rng.normal(size=(85, 3))
    cfg = AugmentConfig(enabled=True, rotation_deg=0.0, scale_min=2.0, scale_max=2.0)
    p_aug, t_aug = augment_ear(points, target, cfg)
    assert np.allclose(p_aug, points * 2.0)
    assert np.allclose(t_aug, target * 2.0)


def test_augment_jitter_changes_points_not_target():
    rng = np.random.default_rng(4)
    points = rng.normal(size=(32, 3))
    target = rng.normal(size=(85, 3))
    cfg = AugmentConfig(
        enabled=True, rotation_deg=0.0, scale_min=1.0, scale_max=1.0, jitter_std=0.05
    )
    p_aug, t_aug = augment_ear(points, target, cfg)
    assert not np.allclose(p_aug, points)
    assert np.allclose(t_aug, target)


def test_augment_dropout_resamples_from_existing_points_only():
    rng = np.random.default_rng(5)
    points = rng.normal(size=(40, 3))
    target = rng.normal(size=(85, 3))
    cfg = AugmentConfig(
        enabled=True, rotation_deg=0.0, scale_min=1.0, scale_max=1.0, dropout_frac=0.3
    )
    p_aug, t_aug = augment_ear(points, target, cfg)
    original_rows = {tuple(row) for row in points.round(8)}
    aug_rows = {tuple(row) for row in p_aug.round(8)}
    assert aug_rows <= original_rows
    assert np.allclose(t_aug, target)


def test_dataset_augmentation_disabled_by_default_matches_precomputed(fake_cache):
    """Passing augment=None must reproduce the exact old (pre-augmentation) path."""
    cache_dir, template_path = fake_cache
    template = load_template(template_path)
    ds = CanonicalEarCacheDataset(cache_dir, ["P0001"], template, n_points=N_POINTS_TEST)
    assert ds.augment.enabled is False
    points, residual, idx = ds[0]
    assert points is ds.points[0]
    assert residual is ds.residuals[0]


def test_dataset_augmentation_recomputes_residual_against_augmented_target(fake_cache):
    """
    rotation_deg=0, scale fixed at 2.0: deterministic enough to assert the
    exact residual, proving the dataset recomputes against target_aug and
    never reuses the precomputed (un-augmented) residual.
    """
    cache_dir, template_path = fake_cache
    template = load_template(template_path)
    aug = AugmentConfig(enabled=True, rotation_deg=0.0, scale_min=2.0, scale_max=2.0)
    ds = CanonicalEarCacheDataset(
        cache_dir, ["P0001"], template, n_points=N_POINTS_TEST, augment=aug
    )
    points_aug, residual_aug, idx = ds[0]
    expected_target = ds.targets_canonical[0] * 2.0
    recovered = residual_aug.numpy() + template.numpy()
    assert np.allclose(recovered, expected_target, atol=1e-4)
    assert np.allclose(points_aug.numpy(), ds.points[0].numpy() * 2.0, atol=1e-4)


def test_dataset_augmentation_never_applies_to_validation(fake_cache):
    """build_datasets wires augment only into the training dataset -- this
    pins the dataset-level contract that a caller must opt in explicitly."""
    cache_dir, template_path = fake_cache
    template = load_template(template_path)
    val_ds = CanonicalEarCacheDataset(cache_dir, ["P0001"], template, n_points=N_POINTS_TEST)
    assert val_ds.augment.enabled is False