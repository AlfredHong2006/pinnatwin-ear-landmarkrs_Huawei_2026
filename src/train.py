"""
Role B — training, checkpointing, and datasets.

Synthetic smoke test (needs nothing from anyone):
    python -m src.train --config configs/train_synthetic.yaml

Tiny-overfit gate (needs A's cache + C's template):
    python -m src.train --config configs/overfit_tiny.yaml

First full run:
    python -m src.train --config configs/train.yaml

INTERFACES THIS FILE CONSUMES (all owned by other roles — never reimplemented)
-----------------------------------------------------------------------------
Role A  src.cache.list_cached / load_cached_ear   the ONLY way to read the cache
        src.geometry.inverse_transform_points     canonical -> global mm
Role C  configs/split_seed42.json                 frozen split, keys
                                                  train_subjects / val_subjects
        outputs/role_c/canonical_template_shared.npz   key "template", [85,3]
        src.evaluate.landmark_errors / summarize_errors   the OFFICIAL metric

The bar to beat (EXPERIMENTS.md, Role C):
    B0 global mean template      val mean 6.393 mm
    B1 canonical mean template   val mean 5.537 mm   <-- beat this
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Dataset

# --- other roles' modules -------------------------------------------------
from src.cache import list_cached, load_cached_ear
from src.evaluate import landmark_errors, summarize_errors
from src.geometry import N_LANDMARKS, N_POINTS, inverse_transform_points

# --- our own --------------------------------------------------------------
from src.augment import AugmentConfig, augment_ear
from src.losses import get_loss_fn, mean_euclidean_error
from src.model import EarLandmarkNet

DEFAULT_SPLIT = "configs/split_seed42.json"
DEFAULT_TEMPLATE = "outputs/role_c/canonical_template_shared.npz"


# =====================================================================
# Reproducibility
# =====================================================================


def set_seed(seed: int) -> None:
    """Seed every generator that can affect a run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =====================================================================
# Config
# =====================================================================


@dataclass
class TrainConfig:
    # data
    mode: str = "synthetic"                  # "synthetic" | "cache"
    cache_dir: str = "cache"
    split_file: str = DEFAULT_SPLIT
    template_file: str = DEFAULT_TEMPLATE
    template_version: str = "B1-shared-canonical"
    limit_train_ears: int = 0                # >0 = tiny-overfit gate
    n_points: int = N_POINTS
    n_synthetic_train: int = 256
    n_synthetic_val: int = 64

    # model
    in_dim: int = 3                          # real scans carry no normals
    n_landmarks: int = N_LANDMARKS
    point_dim1: int = 64
    point_dim2: int = 128
    point_dim3: int = 256
    head_dim1: int = 512
    head_dim2: int = 256
    dropout: float = 0.3

    # optimisation
    epochs: int = 300
    batch_size: int = 16
    lr: float = 1e-3
    weight_decay: float = 1e-4
    loss: str = "smooth_l1"                  # "smooth_l1" | "l1" | "mse"
    beta: float = 0.05

    # augmentation (train split only; off by default — see src/augment.py)
    augment: bool = False
    aug_rotation_deg: float = 5.0
    aug_scale_min: float = 0.95
    aug_scale_max: float = 1.05
    aug_jitter_std: float = 0.0
    aug_dropout_frac: float = 0.0

    # bookkeeping
    seed: int = 0
    out_dir: str = "outputs/role_b"
    run_name: str = "run"
    num_workers: int = 0
    eval_every: int = 1

    def model_kwargs(self) -> dict[str, Any]:
        return {
            "in_dim": self.in_dim,
            "n_landmarks": self.n_landmarks,
            "point_dim1": self.point_dim1,
            "point_dim2": self.point_dim2,
            "point_dim3": self.point_dim3,
            "head_dim1": self.head_dim1,
            "head_dim2": self.head_dim2,
            "dropout": self.dropout,
        }

    def aug_config(self) -> AugmentConfig:
        return AugmentConfig(
            enabled=self.augment,
            rotation_deg=self.aug_rotation_deg,
            scale_min=self.aug_scale_min,
            scale_max=self.aug_scale_max,
            jitter_std=self.aug_jitter_std,
            dropout_frac=self.aug_dropout_frac,
        )


def load_config(path: str | Path) -> TrainConfig:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    known = set(TrainConfig.__dataclass_fields__)
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    return TrainConfig(**raw)


# =====================================================================
# Role C's artefacts — loaded, never computed
# =====================================================================


def load_template(path: str | Path, n_landmarks: int = N_LANDMARKS) -> torch.Tensor:
    """
    Load Role C's shared canonical template.

    Written by scripts/build_canonical_template.py as an npz with key
    ``template``, float32 [85, 3], averaged over TRAIN ears only. One shared
    template covers both sides, because Role A mirrors right ears into the left
    ear's frame.

    We consume this. We never compute template statistics ourselves — that is
    what keeps validation ground truth out of the training signal.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"template not found at {path}. Role C builds it with:\n"
            f"  python -m scripts.build_canonical_template --cache_dir cache"
        )
    with np.load(path) as npz:
        if "template" not in npz.files:
            raise ValueError(f"{path}: expected a 'template' key, found {list(npz.files)}")
        template = np.asarray(npz["template"], dtype=np.float64)
    if template.shape != (n_landmarks, 3):
        raise ValueError(f"template must be [{n_landmarks}, 3], got {template.shape}")
    return torch.from_numpy(template).float()


def load_split(path: str | Path) -> tuple[list[str], list[str]]:
    """Role C's frozen split. Keys are train_subjects / val_subjects."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"split not found at {path}")
    split = json.loads(path.read_text(encoding="utf-8"))
    train = [str(s) for s in split["train_subjects"]]
    val = [str(s) for s in split["val_subjects"]]
    overlap = set(train) & set(val)
    if overlap:
        raise ValueError(f"split has train/val overlap: {sorted(overlap)[:5]}")
    return train, val


# =====================================================================
# Datasets
# =====================================================================


class SyntheticEarDataset(Dataset):
    """
    Fake ears for the day-one smoke test.

    Targets are a DETERMINISTIC FUNCTION of the inputs, not independent noise.
    If they were noise the model could not learn them, the loss would sit flat,
    and a flat loss would tell you nothing — you could not distinguish "my
    training loop is broken" from "this task is impossible". With a learnable
    mapping, a falling loss is real evidence the machinery works.
    """

    def __init__(
        self,
        n_samples: int,
        n_points: int = N_POINTS,
        in_dim: int = 3,
        n_landmarks: int = N_LANDMARKS,
        seed: int = 0,
        residual_scale: float = 0.05,
    ) -> None:
        self.n_samples = n_samples
        self.n_landmarks = n_landmarks
        self.residual_scale = residual_scale

        g = torch.Generator().manual_seed(seed)
        self.points = torch.randn(n_samples, n_points, in_dim, generator=g) * 0.4
        self.true_map = torch.randn(6, n_landmarks * 3, generator=g)

        # Stand-ins so the same eval code path works in both modes.
        self.scales = torch.ones(n_samples)
        self.meta = [(f"SYN{i:04d}", "left") for i in range(n_samples)]

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        pts = self.points[idx]
        xyz = pts[:, :3]
        summary = torch.cat([xyz.mean(dim=0), xyz.max(dim=0).values - xyz.min(dim=0).values])
        residual = (summary @ self.true_map).view(self.n_landmarks, 3) * self.residual_scale
        return pts, residual, idx


class CanonicalEarCacheDataset(Dataset):
    """
    Role A's canonical ear cache, as residual-regression training data.

    Reads exclusively through ``src.cache.load_cached_ear`` — Role A's module
    docstring is explicit that B and C must never touch npz keys directly, so
    that reader and writer cannot drift apart.

    Each item is ``(points [N,3] float32, residual [85,3] float32, index)``.
    The index lets the global-frame evaluator recover this ear's transform,
    which cannot be collated into a tensor batch.

    Targets::

        residual = ear.qa["targets"] - template

    ``qa["targets"]`` only exists when the cache was built with
    ``--with-targets``; a cache without it raises here with that instruction.

    With ``augment`` enabled, each ``__getitem__`` call draws a fresh random
    view via ``src.augment.augment_ear`` and recomputes the residual against
    that view instead of returning the precomputed one — see src/augment.py
    for why rotation/scale must move the target along with the points. Pass
    ``augment=None`` (the default) for the exact pre-augmentation behaviour.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        subject_ids: list[str],
        template: torch.Tensor,
        n_points: int = N_POINTS,
        limit: int = 0,
        augment: AugmentConfig | None = None,
    ) -> None:
        wanted = set(subject_ids)
        entries = [(sid, side, p) for sid, side, p in list_cached(cache_dir) if sid in wanted]
        if not entries:
            raise FileNotFoundError(
                f"no cache files in {cache_dir} for the {len(wanted)} requested subjects. "
                "Build it with:\n"
                "  python -m scripts.preprocess --subject-list splits/train_ids.txt "
                "--with-targets"
            )
        if limit > 0:
            entries = entries[:limit]

        self.template = template
        self.augment = augment if augment is not None else AugmentConfig(enabled=False)
        self.meta: list[tuple[str, str]] = []
        self.points: list[torch.Tensor] = []
        self.residuals: list[torch.Tensor] = []
        self.targets_canonical: list[np.ndarray] = []
        self.transforms: list[Any] = []
        self.scales: list[float] = []
        self.n_suspicious = 0

        template_np = template.numpy().astype(np.float64)
        self._template_np = template_np

        for sid, side, path in entries:
            ear = load_cached_ear(path)

            if "targets" not in ear.qa:
                raise ValueError(
                    f"{path}: no qa['targets']. Rebuild the cache with --with-targets "
                    "(Role A's scripts/preprocess.py)."
                )
            pts = np.asarray(ear.points, dtype=np.float32)
            if pts.shape != (n_points, 3):
                raise ValueError(
                    f"{path}: expected points ({n_points}, 3), got {pts.shape}. "
                    "Cache n_points and config n_points disagree."
                )
            tgt = np.asarray(ear.qa["targets"], dtype=np.float64)

            self.meta.append((sid, side))
            self.points.append(torch.from_numpy(pts))
            self.residuals.append(torch.from_numpy((tgt - template_np).astype(np.float32)))
            self.targets_canonical.append(tgt)
            self.transforms.append(ear.transform)
            self.scales.append(float(ear.transform.scale))
            if ear.qa.get("suspicious", False):
                self.n_suspicious += 1

    def __len__(self) -> int:
        return len(self.points)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        if not self.augment.enabled:
            return self.points[idx], self.residuals[idx], idx

        points_aug, target_aug = augment_ear(
            self.points[idx].numpy(), self.targets_canonical[idx], self.augment
        )
        residual_aug = (target_aug - self._template_np).astype(np.float32)
        return torch.from_numpy(points_aug.astype(np.float32)), torch.from_numpy(residual_aug), idx


def build_datasets(cfg: TrainConfig) -> tuple[Dataset, Dataset, torch.Tensor | None]:
    if cfg.mode == "synthetic":
        train_ds: Dataset = SyntheticEarDataset(
            cfg.n_synthetic_train, cfg.n_points, cfg.in_dim, cfg.n_landmarks, seed=cfg.seed
        )
        val_ds: Dataset = SyntheticEarDataset(
            cfg.n_synthetic_val, cfg.n_points, cfg.in_dim, cfg.n_landmarks, seed=cfg.seed + 1000
        )
        return train_ds, val_ds, None

    if cfg.mode != "cache":
        raise ValueError(f"unknown mode {cfg.mode!r}, expected 'synthetic' or 'cache'")

    if cfg.in_dim != 3:
        raise ValueError(
            f"in_dim={cfg.in_dim} but the real scans carry no normals "
            "(src/cache.py always returns normals=None). Only in_dim=3 is possible."
        )

    template = load_template(cfg.template_file, cfg.n_landmarks)
    train_ids, val_ids = load_split(cfg.split_file)

    train_ds = CanonicalEarCacheDataset(
        cfg.cache_dir, train_ids, template, cfg.n_points,
        limit=cfg.limit_train_ears, augment=cfg.aug_config(),
    )
    # limit and augmentation never apply to validation: a val set must stay
    # a fixed, honest read of the model, not a moving target.
    val_ds = CanonicalEarCacheDataset(cfg.cache_dir, val_ids, template, cfg.n_points)
    return train_ds, val_ds, template


def build_dataloaders(cfg: TrainConfig, train_ds: Dataset, val_ds: Dataset):
    train_loader = DataLoader(
        train_ds,
        batch_size=min(cfg.batch_size, max(2, len(train_ds))),
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=len(train_ds) > cfg.batch_size,  # BatchNorm dies on a batch of 1
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        drop_last=False,
    )
    return train_loader, val_loader


# =====================================================================
# Checkpoints
# =====================================================================


def save_checkpoint(
    path: str | Path,
    model: EarLandmarkNet,
    cfg: TrainConfig,
    epoch: int,
    val_mm: float | None = None,
    val_canonical: float | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """
    Save everything needed to rebuild and reproduce this model.

    Role D must never have to guess which architecture a checkpoint uses, so the
    full model config travels inside the file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "model_class": "EarLandmarkNet",
        "model_state_dict": model.state_dict(),
        "model_config": model.config,
        "in_dim": model.config["in_dim"],
        "n_landmarks": model.config["n_landmarks"],
        "n_points": cfg.n_points,
        "template_version": cfg.template_version,
        "template_file": cfg.template_file,
        "split_file": cfg.split_file,
        "epoch": epoch,
        "seed": cfg.seed,
        "val_mean_mm": val_mm,
        "val_mean_canonical": val_canonical,
        "train_config": asdict(cfg),
        "torch_version": torch.__version__,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "role": "B",
        "format_version": 2,
        "usage": "residual = model(points[B,N,3]); pred_canonical = template + residual",
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if extra:
        payload.update(extra)

    torch.save(payload, path)


def load_checkpoint(
    path: str | Path,
    map_location: str | torch.device = "cpu",
) -> tuple[EarLandmarkNet, dict[str, Any]]:
    """Rebuild a model from a checkpoint using only the file's own contents."""
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model = EarLandmarkNet.from_config(ckpt["model_config"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


# =====================================================================
# Train / eval
# =====================================================================


def train_one_epoch(model, loader, optimizer, loss_fn, device) -> tuple[float, float]:
    model.train()
    total_loss = total_mee = 0.0
    n = 0
    for points, target, _ in loader:
        points, target = points.to(device), target.to(device)
        optimizer.zero_grad()
        pred = model(points)
        loss = loss_fn(pred, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_mee += mean_euclidean_error(pred, target).item()
        n += 1
    return total_loss / n, total_mee / n


@torch.no_grad()
def evaluate_canonical(model, loader, loss_fn, device) -> tuple[float, float]:
    """Cheap per-epoch readout in canonical units. Not the official score."""
    model.eval()
    total_loss = total_mee = 0.0
    n = 0
    for points, target, _ in loader:
        points, target = points.to(device), target.to(device)
        pred = model(points)
        total_loss += loss_fn(pred, target).item()
        total_mee += mean_euclidean_error(pred, target).item()
        n += 1
    return total_loss / n, total_mee / n


@torch.no_grad()
def evaluate_global_mm(
    model: nn.Module,
    dataset: CanonicalEarCacheDataset,
    device: torch.device,
    batch_size: int = 16,
) -> dict[str, Any]:
    """
    The number that is actually comparable to Role C's B1 baseline (5.537 mm).

    Reconstructs ``pred_canonical = template + residual``, maps both prediction
    and ground truth back to global millimetres with Role A's
    ``inverse_transform_points``, then calls Role C's ``landmark_errors`` /
    ``summarize_errors``. Neither the transform nor the metric is reimplemented
    here — this is the same code path that produced the baseline number.
    """
    model.eval()
    template = dataset.template.numpy().astype(np.float64)

    preds_global: list[np.ndarray] = []
    gts_global: list[np.ndarray] = []
    sides: list[str] = []

    for start in range(0, len(dataset), batch_size):
        idxs = list(range(start, min(start + batch_size, len(dataset))))
        points = torch.stack([dataset.points[i] for i in idxs]).to(device)
        residual = model(points).cpu().numpy().astype(np.float64)

        for k, i in enumerate(idxs):
            transform = dataset.transforms[i]
            pred_can = template + residual[k]
            preds_global.append(inverse_transform_points(pred_can, transform))
            gts_global.append(inverse_transform_points(dataset.targets_canonical[i], transform))
            sides.append(dataset.meta[i][1])

    pred_arr = np.stack(preds_global)
    gt_arr = np.stack(gts_global)
    err = landmark_errors(pred_arr, gt_arr)          # [B, 85], millimetres
    summary = summarize_errors(err)

    sides_arr = np.array(sides)
    out: dict[str, Any] = {
        "n_val_ears": int(err.shape[0]),
        "mean": summary["mean"],
        "median": summary["median"],
        "p95": summary["p95"],
        "per_landmark_mean": summary["per_landmark_mean"].tolist(),
    }
    for side in ("left", "right"):
        mask = sides_arr == side
        out[f"{side}_mean"] = float(err[mask].mean()) if mask.any() else None
    return out


def train(cfg: TrainConfig) -> dict[str, Any]:
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds, val_ds, _ = build_datasets(cfg)
    train_loader, val_loader = build_dataloaders(cfg, train_ds, val_ds)

    model = EarLandmarkNet(**cfg.model_kwargs()).to(device)
    loss_fn = get_loss_fn(cfg.loss, cfg.beta)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    real = cfg.mode == "cache"
    print(f"device={device}  params={model.n_parameters():,}  mode={cfg.mode}")
    print(f"train ears={len(train_ds)}  val ears={len(val_ds)}")
    if real and getattr(train_ds, "n_suspicious", 0):
        print(f"note: {train_ds.n_suspicious} training ears flagged qa_suspicious by Role A")

    out_dir = Path(cfg.out_dir)
    best = float("inf")
    best_mm: float | None = None
    history: list[dict[str, float]] = []

    for epoch in range(1, cfg.epochs + 1):
        tr_loss, tr_mee = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        va_loss, va_mee = evaluate_canonical(model, val_loader, loss_fn, device)
        history.append(
            {"epoch": epoch, "train_loss": tr_loss, "train_mee": tr_mee,
             "val_loss": va_loss, "val_mee": va_mee}
        )

        if epoch == 1 or epoch % 10 == 0 or epoch == cfg.epochs:
            print(
                f"epoch {epoch:4d} | train {tr_loss:.6f} / {tr_mee:.5f} "
                f"| val {va_loss:.6f} / {va_mee:.5f}"
            )

        if va_mee < best:
            best = va_mee
            if real:
                best_mm = evaluate_global_mm(model, val_ds, device, cfg.batch_size)["mean"]
            save_checkpoint(
                out_dir / f"{cfg.run_name}_best.pt", model, cfg, epoch, best_mm, best, optimizer
            )

    save_checkpoint(
        out_dir / f"{cfg.run_name}_last.pt", model, cfg, cfg.epochs, best_mm, best, optimizer
    )

    result: dict[str, Any] = {"best_val_mee_canonical": best, "history": history}
    if real:
        restored, _ = load_checkpoint(out_dir / f"{cfg.run_name}_best.pt", map_location=device)
        metrics = evaluate_global_mm(restored.to(device), val_ds, device, cfg.batch_size)
        result["official"] = metrics
        print("\n--- official metric (Role C's evaluate.py, global mm) ---")
        print(f"  mean   {metrics['mean']:.4f} mm    (B1 baseline 5.5373)")
        print(f"  median {metrics['median']:.4f} mm  (B1 baseline 5.2597)")
        print(f"  p95    {metrics['p95']:.4f} mm     (B1 baseline 10.1252)")
        delta = 5.5373456214629915 - metrics["mean"]
        verdict = "BEATS B1" if delta > 0 else "does NOT beat B1"
        print(f"  -> {verdict} by {abs(delta):.4f} mm")
    else:
        print(f"\nbest val mee (canonical, synthetic): {best:.6f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{cfg.run_name}_history.json", "w") as f:
        json.dump({"config": asdict(cfg), **result}, f, indent=2)
    print(f"checkpoints and history written to {out_dir}/")
    return result


# =====================================================================
# CLI
# =====================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="Role B — train the global landmark model")
    parser.add_argument("--config", required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--cache-dir", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.seed is not None:
        cfg.seed = args.seed
        cfg.run_name = args.run_name or f"{cfg.run_name}_seed{args.seed}"
    if args.run_name is not None:
        cfg.run_name = args.run_name
    if args.cache_dir is not None:
        cfg.cache_dir = args.cache_dir

    train(cfg)


if __name__ == "__main__":
    main()