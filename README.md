# PinnaTwin-Zoom

Huawei Munich Tech Arena 2026 — 3D Pinna Landmark Extraction.

## Team sprint

Start here:

`docs/sprint/TEAM_OVERVIEW.md`

For AI coding assistants:

1. Read `docs/sprint/MASTER_AI_CONTEXT.md`
2. Read your assigned role file
3. Read `DATA_SPEC.md`
4. Inspect the current repository before editing

## Roles

A — 3D Preprocessing & Geometry Engineer      ← Alfred
B — Global Landmark Model Engineer            ← Ojas
C — Validation & Geometric Refinement Engineer - Pravi
D — Inference & Pipeline Engineer - Ayush

## Role A — 3D preprocessing & geometry (COMPLETE)

Owner: Alfred. Handover: `docs/HANDOFF_A.md` (frozen signatures, npz schema, all
numbers). Verified data facts: `DATA_SPEC.md`. Status: `docs/STATUS_A.md`.

**What Role A provides.** Raw Huawei head PLY -> left/right `CanonicalEar`: 2048
canonical-frame points per ear plus an exactly invertible `EarTransform` back to
Huawei coordinates. Concretely: mesh/annotation loaders (`src/data.py`), the frozen
ear crop, canonical transform, inverse and sampler (`src/geometry.py`), the processed
`.npz` cache reader/writer (`src/cache.py`), the frozen crop box (`configs/crop.yaml`)
and the provisional 160/40 subject split (`splits/`). Every transform is derived from
the **mesh and the frozen config only** — ground-truth landmarks are never allowed to
define a crop, centre, scale, rotation or mirror.

**The five verification scripts, and what each proves.**

| script | what it proves |
|---|---|
| `scripts/inspect_dataset.py` | the data conventions themselves — units, file layout, subject IDs, PLY properties (no stored normals), the 85-line `<idx>,[<x> <y> <z>]` annotation layout, and the +Y = subject's-left sign. Aggregates only; never dumps points. |
| `scripts/crop_stats.py` | that a crop box derived from the **training** subjects generalises. Its freeze criterion is **leave-one-out**: every subject is measured against a box rebuilt without it, and the minimum headroom must be > 0 on every axis, both sides. It writes `crop.rejected.yaml` and exits 1 if that fails. |
| `scripts/check_crop_all.py` | that the frozen box still keeps **85/85** GT landmarks inside on subjects it was *not* derived from. Exit 1 on any truncated ear or suspicious crop. This is the real generalisation test; the leave-one-out headroom only estimates it. |
| `scripts/check_roundtrip.py` | that the transform is **exactly invertible**. Builds it from the mesh alone, pushes GT landmarks to canonical and back through `EarTransform.from_dict(to_dict())` — the same path Role D's predictions take — and exits 1 above 1e-9 mm. Also reports the canonical envelope Role C sizes the template with. |
| `scripts/check_mirror.py` | the **mirror convention**: each subject's own two canonical ears compared under no-mirror / mirror-right / mirror-left, rebuilt from the mesh every time (the cache is never read). Decides mirror vs no mirror, and its per-contour spread shows whether left and right index the landmarks the same way. Changes no default — it reports. |

`scripts/preprocess.py` (cache build), `scripts/make_split.py` (split) and
`scripts/plot_ear.py` (QA figures) are build/inspection tools, not checks.

**Reproduce cache and checks from a fresh clone.** Needs the NDA dataset locally
(folders `mesh/` and `landmarks/`); nothing under the data root is ever committed.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install numpy trimesh pyyaml matplotlib pytest torch    # torch is needed for Role B/model inference
$env:HUAWEI_DATA_ROOT = "<path containing mesh/ and landmarks/>"

pytest tests -q                                        # full suite; see CI output for any pre-existing failures

# 1. data conventions (all 200 meshes)
python scripts/inspect_dataset.py --max-meshes 0

# 2. crop box — DO NOT re-run these unless the split actually changes.
#    splits/ and configs/crop.yaml are COMMITTED. crop_stats.py stamps a
#    `generated:` timestamp into crop.yaml, so a re-run produces a file with a
#    DIFFERENT sha256 even when the bounds are identical — and every cached ear
#    stores that sha256, so the next preprocess.py run would declare all 400
#    files stale and fail. To verify the committed box without touching it:
python scripts/crop_stats.py --subject-list splits/train_ids.txt --dry-run

# 3. the three checks
python scripts/check_roundtrip.py --subject-list splits/val_ids.txt
python scripts/check_crop_all.py  --subject-list splits/val_ids.txt
python scripts/check_mirror.py    --subject-list splits/train_ids.txt --limit 40

# 4. build the processed cache (cache/<subject_id>_<side>.npz, git-ignored)
python scripts/preprocess.py --subject-list splits/train_ids.txt --with-targets
python scripts/preprocess.py --subject-list splits/val_ids.txt   --with-targets
```

`check_roundtrip.py`, `check_crop_all.py`, `crop_stats.py` and `preprocess.py`
exit non-zero on failure, so they can be run as a gate. **`check_mirror.py` is the
exception: it reports and exits 0 even when the current default loses** — it is
deliberately not allowed to change a frozen setting, so read its verdict block.

If the split ever changes, the order is fixed: regenerate `configs/crop.yaml`
first, then re-run the checks, then rebuild the cache with `--overwrite`.

B and C read the cache **only** through `src.cache.load_cached_ear` / `list_cached`
— never by parsing npz keys directly.

## Important

Huawei NDA-protected data must not be committed to this repository.

The implementation objective is to establish a complete end-to-end
PLY → left/right 85×3 landmark pipeline before adding optional features.


## Role D — inference and reproduction

Role D owns the runnable path from an unseen PLY to two validated `85 x 3`
arrays in the original Huawei coordinate frame. The runtime is intentionally
small and consumes Role A/B/C interfaces rather than duplicating their logic.

### One-command fallback smoke test

`configs/infer.yaml` is deliberately configured for the train-only global-mean
fallback. The fallback artifact is produced by Role C and is NDA-derived, so it
is not committed to Git.

```bash
python -m src.infer \
  --input /path/to/P0001.ply \
  --config configs/infer.yaml \
  --output predictions/
```

The command writes `predictions/P0001.npz` containing:

```python
{
    "subject_id": "P0001",
    "left":  np.ndarray((85, 3)),
    "right": np.ndarray((85, 3)),
}
```

For inspection, `--format both` also writes `P0001_left.csv` and
`P0001_right.csv` using the repository's verified `idx,[x y z]` row notation.
The NPZ is the canonical reproducibility artifact.

### Learned-model inference

When a Role B checkpoint has been selected by the team, use the same command
with the checkpoint override:

```bash
python -m src.infer \
  --input /path/to/P0001.ply \
  --config configs/infer.yaml \
  --mode model \
  --checkpoint outputs/role_b/augmented_seed0_best.pt \
  --output predictions/
```

The checkpoint is validated before inference for its stored architecture,
`n_points`, `n_landmarks`, `in_dim`, template version and residual-prediction
contract. Multiple compatible checkpoints can be supplied by repeating
`--checkpoint`; their residual predictions are averaged. This is an orchestration
feature only; the choice to retain an ensemble remains a C/B experiment decision.

Test-time multi-sample averaging is available through `tta_samples` in the Role D
config, but the shipped default is `1`. It should only be raised after Role C has
validated the setting. C-owned post-processing is likewise disabled by default;
when enabled later, D calls C's `apply_enabled_postprocessing` hook instead of
reimplementing projection/refinement logic.

### Inference assumptions and boundaries

Inference uses only the supplied mesh and frozen runtime artifacts. It never
loads the training landmark CSVs and therefore does not use ground truth to crop,
centre, scale, mirror or sample points.

The public Huawei Topic Description specifies separate left/right sets of 85
landmarks, but the accessible public page does not publish a filename or binary
serialization schema for the evaluator. The repository therefore keeps its NPZ
writer as a deterministic internal/reproduction format; only the final platform
wrapper should change if the private submission instructions specify another
container format.

Run the D-focused integration suite with:

```bash
pytest tests/test_pipeline.py -q
```
