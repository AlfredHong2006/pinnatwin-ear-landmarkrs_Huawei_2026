# Evaluation (Role C)

## 1) Protocol and split

- Evaluation uses a **fixed subject-level split**:
  - Train: 160 subjects
  - Validation: 40 subjects
  - Seed: 42
  - Canonical split file: `configs/split_seed42.json`
- Both ears from the same subject are always in the same split.
- Templates are built from **train only**.
- Validation is used only for reporting performance.

---

## 2) Metric

Primary metric: mean 3D Euclidean landmark error over both ears and all 85 landmarks per ear.

Also reported:
- Median error
- p95 error (95th percentile)

Lower is better.

---

## 3) Data integrity checks

Validated before running baselines:

- Mesh files: 200
- Left landmark files: 200
- Right landmark files: 200
- Usable subjects (mesh + left + right): 200
- Bad landmark shapes: 0
- Non-finite entries: 0

---

## 4) Baseline results

### B0 — Global-frame mean template

Train-only mean left and right templates in global coordinates.

- Mean: **6.3932424711307645**
- Median: **6.159525629613377**
- p95: **11.46832211778311**

Side breakdown:
- Left mean / median / p95: **6.348421204522823 / 6.005782685040604 / 13.40592973521226**
- Right mean / median / p95: **6.438063737738706 / 6.267835125708471 / 10.81508953959883**

---

### B1 — Shared canonical template

Single shared canonical template built from train cached canonical targets (320 ears total: 160 left + 160 right), then inverse-transformed to global coordinates for scoring.

- Mean: **5.5373456214629915**
- Median: **5.259664938574634**
- p95: **10.125245094139707**

Repro check via generic prediction evaluator on exported validation NPZ:
- Mean: **5.537345627224655**
- Median: **5.259664816854107**
- p95: **10.12524504417699**

---

## 5) B0 vs B1 summary

B1 improves over B0 on all key metrics:

- Mean improvement: ~13.4%
- Median improvement: ~14.6%
- p95 improvement: ~11.7%

Decision:
- **B1: KEEP** (stronger baseline)
- **B0: KEEP as fallback**

---

## 6) Failure analysis notes

- Worst-case subjects are tracked via:
  - `outputs/role_c/worst_b0_subjects.json`
- Initial “distorted ear” visuals were mostly 3D viewpoint effects, confirmed by rotating views and CSV-only plotting.

---

## 7) Leakage controls

- Frozen subject-level split
- Train-only template construction
- No validation GT used to define preprocessing
- Evaluation only on held-out validation subjects

---

## 8) Final model result (to fill after Role D final val run)

- Final model mean: **TBD**
- Final model median: **TBD**
- Final model p95: **TBD**
- Delta vs B1: **TBD**
