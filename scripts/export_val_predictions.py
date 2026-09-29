#!/usr/bin/env python3
"""Export Role D model predictions for Role C's final validation scoring.

Reads ONLY raw validation PLY meshes plus the frozen validation ID list.
It never loads validation annotations/ground truth.

Output NPZ schema:
  subject_ids: [N] strings
  left:        [N, 85, 3] original Huawei XYZ
  right:       [N, 85, 3] original Huawei XYZ
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from src.estimator import LandmarkExtractor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data-root",
        default=os.environ.get("HUAWEI_DATA_ROOT"),
        help="Directory containing mesh/ and landmarks/. Defaults to $HUAWEI_DATA_ROOT.",
    )
    p.add_argument(
        "--split",
        default="splits/val_ids.txt",
        help="Frozen validation subject ID list.",
    )
    p.add_argument(
        "--config",
        default="configs/infer.yaml",
        help="Role D inference config.",
    )
    p.add_argument(
        "--output",
        default="outputs/role_d/val_predictions.npz",
        help="Output NPZ path.",
    )
    p.add_argument(
        "--device",
        default=None,
        help="Torch device override, e.g. cpu or cuda.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional smoke-test limit. 0 means all validation subjects.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if not args.data_root:
        raise SystemExit(
            "HUAWEI_DATA_ROOT is not set. Pass --data-root or export HUAWEI_DATA_ROOT."
        )

    data_root = Path(args.data_root).expanduser()
    mesh_dir = data_root / "mesh"
    split_path = Path(args.split)

    if not mesh_dir.is_dir():
        raise SystemExit(f"mesh directory not found: {mesh_dir}")
    if not split_path.is_file():
        raise SystemExit(f"validation split not found: {split_path}")

    subject_ids = [
        line.strip()
        for line in split_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.limit > 0:
        subject_ids = subject_ids[:args.limit]

    extractor = LandmarkExtractor(args.config, device=args.device)

    left_preds: list[np.ndarray] = []
    right_preds: list[np.ndarray] = []
    kept_ids: list[str] = []

    for i, subject_id in enumerate(subject_ids, start=1):
        mesh_path = mesh_dir / f"{subject_id}.ply"
        if not mesh_path.is_file():
            raise SystemExit(f"missing validation mesh: {mesh_path}")

        pred = extractor.predict_subject(mesh_path)

        left = np.asarray(pred["left"], dtype=np.float64)
        right = np.asarray(pred["right"], dtype=np.float64)

        if left.shape != (85, 3):
            raise SystemExit(
                f"{subject_id}: left prediction has shape {left.shape}, expected (85, 3)"
            )
        if right.shape != (85, 3):
            raise SystemExit(
                f"{subject_id}: right prediction has shape {right.shape}, expected (85, 3)"
            )
        if not np.isfinite(left).all() or not np.isfinite(right).all():
            raise SystemExit(f"{subject_id}: prediction contains NaN/Inf")

        kept_ids.append(subject_id)
        left_preds.append(left)
        right_preds.append(right)

        print(f"[{i}/{len(subject_ids)}] {subject_id}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        out,
        subject_ids=np.asarray(kept_ids, dtype=str),
        left=np.stack(left_preds, axis=0),
        right=np.stack(right_preds, axis=0),
    )

    print()
    print(f"wrote: {out}")
    print(f"subject_ids: {len(kept_ids)}")
    print(f"left:  {np.stack(left_preds).shape}")
    print(f"right: {np.stack(right_preds).shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
