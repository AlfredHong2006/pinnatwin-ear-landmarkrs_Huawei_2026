from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
import numpy as np

from src.evaluate import landmark_errors, summarize_errors

FLOAT_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def parse_landmark_file(path: Path) -> np.ndarray:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        nums = [float(x) for x in FLOAT_RE.findall(line)]
        if len(nums) >= 4:
            idx = int(round(nums[0]))
            xyz = nums[1:4]
        elif len(nums) == 3:
            idx = len(rows)
            xyz = nums
        else:
            continue
        rows.append((idx, xyz))

    if len(rows) != 85:
        raise ValueError(f"{path.name}: expected 85 rows, got {len(rows)}")

    idxs = sorted(i for i, _ in rows)
    if idxs != list(range(85)):
        raise ValueError(f"{path.name}: indices are not exactly 0..84")

    rows.sort(key=lambda t: t[0])
    return np.asarray([xyz for _, xyz in rows], dtype=np.float64)


def load_gt_for_subjects(data_root: Path, subject_ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
    lm_dir = data_root / "landmarks"
    gt_left, gt_right = [], []

    for sid in subject_ids:
        lpath = lm_dir / f"{sid}_left_ear_landmarks.csv"
        rpath = lm_dir / f"{sid}_right_ear_landmarks.csv"
        if not lpath.exists() or not rpath.exists():
            raise FileNotFoundError(f"Missing GT landmarks for subject {sid}")
        gt_left.append(parse_landmark_file(lpath))
        gt_right.append(parse_landmark_file(rpath))

    return np.stack(gt_left, axis=0), np.stack(gt_right, axis=0)  # [B,85,3], [B,85,3]


def topk_subjects(subject_ids: list[str], per_subject_mean: np.ndarray, k: int) -> list[dict]:
    order = np.argsort(-per_subject_mean)[: min(k, len(subject_ids))]
    return [{"subject_id": subject_ids[i], "mean_error": float(per_subject_mean[i])} for i in order]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_npz", required=True, type=str,
                    help="NPZ with keys: subject_ids, left, right")
    ap.add_argument("--data_root", required=True, type=str,
                    help="Dataset root containing landmarks/")
    ap.add_argument("--split_json", default="configs/split_seed42.json", type=str,
                    help="If provided, evaluates only val_subjects present in predictions")
    ap.add_argument("--k_worst", default=20, type=int)
    ap.add_argument("--out_json", default="outputs/role_c/model_eval.json", type=str)
    args = ap.parse_args()

    pred = np.load(args.pred_npz, allow_pickle=True)
    subject_ids_all = [str(x) for x in pred["subject_ids"].tolist()]
    pred_left_all = np.asarray(pred["left"], dtype=np.float64)   # [N,85,3]
    pred_right_all = np.asarray(pred["right"], dtype=np.float64) # [N,85,3]

    if pred_left_all.shape != pred_right_all.shape or pred_left_all.shape[1:] != (85, 3):
        raise ValueError(f"Expected left/right [N,85,3], got {pred_left_all.shape} and {pred_right_all.shape}")

    split = json.loads(Path(args.split_json).read_text(encoding="utf-8"))
    val_set = set(split["val_subjects"])

    keep_idx = [i for i, sid in enumerate(subject_ids_all) if sid in val_set]
    if not keep_idx:
        raise RuntimeError("No prediction subject IDs overlap with val split.")

    subject_ids = [subject_ids_all[i] for i in keep_idx]
    pred_left = pred_left_all[keep_idx]
    pred_right = pred_right_all[keep_idx]

    gt_left, gt_right = load_gt_for_subjects(Path(args.data_root), subject_ids)

    err_left = landmark_errors(pred_left, gt_left)   # [B,85]
    err_right = landmark_errors(pred_right, gt_right)
    err_all = np.concatenate([err_left, err_right], axis=0)

    m_left = summarize_errors(err_left)
    m_right = summarize_errors(err_right)
    m_all = summarize_errors(err_all)

    per_subj_left = err_left.mean(axis=1)
    per_subj_right = err_right.mean(axis=1)
    per_subj_both = (per_subj_left + per_subj_right) / 2.0

    out = {
        "n_eval_subjects": len(subject_ids),
        "mean": float(m_all["mean"]),
        "median": float(m_all["median"]),
        "p95": float(m_all["p95"]),
        "left_mean": float(m_left["mean"]),
        "right_mean": float(m_right["mean"]),
        "left_p95": float(m_left["p95"]),
        "right_p95": float(m_right["p95"]),
        "worst_subjects_both": topk_subjects(subject_ids, per_subj_both, args.k_worst),
        "worst_subjects_left": topk_subjects(subject_ids, per_subj_left, args.k_worst),
        "worst_subjects_right": topk_subjects(subject_ids, per_subj_right, args.k_worst),
    }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(json.dumps({
        "eval": {
            "n_eval_subjects": out["n_eval_subjects"],
            "mean": out["mean"],
            "median": out["median"],
            "p95": out["p95"],
        }
    }, indent=2))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()