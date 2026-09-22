# EXPERIMENTS

Use this file for every meaningful training/evaluation run.

---

## Baselines

### B0 — Global-frame mean template
- Date: 07/09/2026 
- Owner: Pravi  
- Split: seed=42, subject-level, train=160, val=40, file=configs/split_seed42.json
- Validation mean: 6.3932424711307645
- Median: 6.159525629613377
- P95: 11.46832211778311
- Notes: Built mean left/right templates from TRAIN subjects only; evaluated on VAL subjects only. Output artifacts: outputs/role_c/b0_metrics.json and outputs/role_c/global_template.npz. Left has heavier high-error tail (p95) than right - more extreme outliers.

### B0 — Global-frame mean template
- Date: 2026-09-08
- Owner: Pravi (Role C)
- Split: seed=42, subject-level, train=160, val=40 (configs/split_seed42.json)
- Validation mean: 6.3932424711307645
- Median: 6.159525629613377
- P95: 11.46832211778311
- Notes:
  - Side metrics:
    - left_mean: 6.348421204522823
    - right_mean: 6.438063737738706
    - left_median: 6.005782685040604
    - right_median: 6.267835125708471
    - left_p95: 13.40592973521226
    - right_p95: 10.81508953959883
  - Worst-case analysis saved: outputs/role_c/worst_b0_subjects.json
  - Top-5 worst left: P0190, P0058, P0041, P0102, P0223
  - Top-5 worst right: P0041, P0298, P0134, P0027, P0223
  - Visual check: apparent “distorted ear” cases are mostly viewpoint-dependent in 3D plots.
- Decision: KEEP (baseline + fallback), INVESTIGATE left-tail outliers.


### B1 — Canonical template
- Date: 2026-09-08
- Owner: Pravi (Role C)
- Transform version: Role A frozen canonical transform (mirror_side=right, mirror_axis=1)
- Validation mean: 5.5373456214629915
- Median: 5.259664938574634
- P95: 10.125245094139707
- Notes:
  - Shared canonical template (single template for both ears).
  - Built from TRAIN cached canonical GT targets only (320 ears total: 160 left + 160 right).
  - Evaluated on VAL ears after inverse transform back to global coordinates.
  - Improves over B0 on mean/median/p95.

---

### Run 001 — PointNet baseline (no augmentation)

- **Date:** <today's date>
- **Owner:** Ojas (Role B)
- **Git commit:** <fill in after committing>
- **Config:** configs/train.yaml
- **Model:** EarLandmarkNet, point_dim 64/128/256, head 512/256, dropout 0.3
- **Input features:** xyz only (in_dim=3)
- **Point count:** 2048
- **Template version:** B1-shared-canonical
- **Loss:** smooth_l1 (beta=0.05)
- **Augmentation:** off
- **Seed:** 0
- **Epochs:** 300
- **Checkpoint path/location:** outputs/role_b/baseline_seed0_best.pt (git-ignored — share with C/D directly)

#### Validation
- Mean: 2.7228 mm
- Median: 2.5075 mm
- P95: 5.2836 mm

#### Notes
- Beats B1 canonical-template baseline (5.5373 mm) by 2.8146 mm.
- Tiny-overfit gate passed first (train_mee -> ~0 on 6 ears), confirming the
  pipeline (target construction, transform, mirror axis, loss) before this run.
- train/val loss stayed close throughout (no divergence) — no sign of overfitting
  despite ~372k params on 320 training ears.

#### Decision
KEEP

### Run 002 — PointNet + conservative augmentation

- **Date:** <today>
- **Owner:** Ojas (Role B)
- **Git commit:** <fill in after committing>
- **Config:** configs/train_augmented.yaml
- **Model:** same as Run 001 (point_dim 64/128/256, head 512/256, dropout 0.3)
- **Input features:** xyz only
- **Point count:** 2048
- **Template version:** B1-shared-canonical
- **Loss:** smooth_l1 (beta=0.05)
- **Augmentation:** on (rotation ±5°, scale 0.95-1.05, jitter_std 0.01, dropout_frac 0.05)
- **Seed:** 0
- **Epochs:** 300
- **Checkpoint path/location:** outputs/role_b/augmented_seed0_best.pt

#### Validation
- Mean: 2.6212 mm (Run 001: 2.7228 mm)
- Median: 2.3744 mm (Run 001: 2.5075 mm)
- P95: 5.1502 mm (Run 001: 5.2836 mm)

#### Notes
- Consistent improvement over Run 001 on mean/median/p95 — augmentation helps.
- Single seed; effect is modest (~4% relative on mean), worth confirming isn't
  seed noise before calling it final.

#### Decision
KEEP

### Run 003 — PointNet + augmentation, seed 1

- **Date:** <today>
- **Owner:** Ojas (Role B)
- **Config:** configs/train_augmented.yaml
- **Seed:** 1
- **Epochs:** 300
- **Checkpoint:** outputs/role_b/augmented_seed1_best.pt

#### Validation
- Mean: 2.6472 mm
- Median: 2.4411 mm
- P95: 5.0692 mm

#### Notes
- Confirms Run 002 (seed 0, 2.6212mm) — augmentation improvement is stable
  across seeds, not a lucky draw.

#### Decision
KEEP — augmentation is confirmed beneficial.

### Run 004 — PointNet + augmentation, seed 2

- **Date:** <today>
- **Owner:** Ojas (Role B)
- **Config:** configs/train_augmented.yaml
- **Seed:** 2
- **Epochs:** 300
- **Checkpoint:** outputs/role_b/augmented_seed2_best.pt

#### Validation
- Mean: 2.6998 mm
- Median: 2.4576 mm
- P95: 5.3021 mm

#### Notes
- Third augmented seed. All 3 augmented seeds (2.6212 / 2.6472 / 2.6998)
  beat the non-augmented run (2.7228) — smaller margin than seeds 0-1, but
  consistent direction, 3/3.

#### Decision
KEEP — set complete (3 seeds), ready to hand to Role C/D

---

## Role D — runtime integration record

- Runtime entry point: `python -m src.infer`
- Runtime config: `configs/infer.yaml`
- Public API: `src.pipeline.predict_subject(mesh_path, config)`
- Output contract: `left` and `right`, each `[85,3]`, finite, original Huawei frame
- Fallback artifact: Role C `outputs/role_c/global_template.npz`
- Learned-model artifact contract: Role B checkpoint format v2 via `src.train.load_checkpoint`
- Canonical template: Role C `outputs/role_c/canonical_template_shared.npz`
- A interface: `load_mesh`, `load_crop_config`, `canonicalize_ear`, `inverse_transform_points`
- No annotations are read during inference.
- Run 002 / Run 003 / Run 004 are compatible Role B candidate checkpoints; the current
  experiment log does **not** record a final single-checkpoint or ensemble decision, so
  Role D does not invent one. The checkpoint is supplied explicitly in the runtime config
  or CLI override once B/C select the exact run.
- C-owned post-processing remains disabled until its implementation and validation are
  present; D only provides the integration hook.
