# PinnaTwin — 3D Ear Landmark Detection

Predicts **85 anatomical landmarks on each ear** directly from a raw 3D head scan.
Built by a team of four Imperial College London students for the **Huawei Munich Tech Arena 2026**.

**2.66 mm** mean landmark error on held-out subjects — 52% lower than the best geometric baseline.

---

## How it works

```
raw head mesh (.ply)
   │
   ├─ 1. Crop        frozen per-side box, derived from training subjects only
   ├─ 2. Align       centre + scale into a canonical frame; mirror the right ear onto the left
   ├─ 3. Sample      2,048 points per ear
   ├─ 4. Predict     PointNet regresses offsets from a mean ear template
   └─ 5. Invert      exact inverse transform back to scan coordinates
   │
left / right landmarks, 85 × 3 each (mm)
```

The idea is to let geometry do as much of the work as possible, so the network only
has to learn small, subject-specific corrections. With just 200 annotated subjects,
that matters.

- **Exactly invertible alignment.** Everything the model sees and predicts is in a
  canonical frame, and the mapping back to the original scan is closed-form
  (round-trip error < 10⁻¹⁴ mm).
- **One model for both ears.** We measured that mirroring the right ear onto the left
  aligns them 5.9× better than not mirroring, and that the landmark orderings match.
  So a single shared template and model serve both sides, doubling the training data
  to 320 ears.
- **Residual prediction.** The network predicts offsets from a mean canonical ear rather
  than absolute coordinates, which keeps the learning problem small and stable.
- **No leakage.** Crop boxes and templates come from training subjects only, and at
  inference time the transform is computed from the mesh alone — ground-truth
  landmarks never influence cropping or alignment.

## Results

Subject-level split: 160 training / 40 validation subjects (80 ears), seed 42.
Error is the 3D Euclidean distance per landmark, in millimetres.

| Method | Mean | Median | P95 |
|---|---:|---:|---:|
| Mean template, scan frame | 6.39 | 6.16 | 11.47 |
| Mean template, canonical frame | 5.54 | 5.26 | 10.13 |
| PointNet residual, no augmentation | 2.72 | 2.51 | 5.28 |
| **PointNet residual + augmentation (final)** | **2.66** | **2.43** | **5.29** |

Canonical alignment alone cuts error by ~13%; the learned residual halves it again.
Augmentation (±5° rotation, ±5% scale, jitter, point dropout) gave a small but
consistent gain across three random seeds.

## Model

A compact PointNet-style network (~372k parameters): three shared per-point layers
(3 → 64 → 128 → 256), global max-pooling, and an MLP head that outputs 85 × 3
offsets. Trained with Smooth-L1 loss and Adam for 300 epochs on CPU.
A deliberately small model — with 320 training ears, reliability beat capacity.

## Usage

```bash
pip install -r requirements.txt
```

```python
from src.estimator import LandmarkExtractor

extractor = LandmarkExtractor()
landmarks = extractor("path/to/scan.ply")

landmarks["left"].shape    # (85, 3)
landmarks["right"].shape   # (85, 3)
```

> The competition dataset is under NDA, so meshes, annotations and trained weights
> are not included in this repository.

### Training from scratch (requires the dataset)

```bash
export HUAWEI_DATA_ROOT=/path/to/data     # folder containing mesh/ and landmarks/

python scripts/preprocess.py --subject-list splits/train_ids.txt --with-targets
python scripts/preprocess.py --subject-list splits/val_ids.txt   --with-targets
python -m scripts.build_canonical_template
python -m src.train --config configs/train_augmented.yaml
```

### Tests

```bash
pytest tests -q        # synthetic data only, no dataset needed
```

## Repository layout

```
src/
  data.py        mesh and landmark loaders
  geometry.py    cropping, canonical transform, inverse, sampling
  cache.py       preprocessed-ear cache
  model.py       PointNet landmark regressor
  train.py       training loop and checkpointing
  augment.py     point-cloud augmentation
  evaluate.py    official metric
  pipeline.py    inference pipeline
  estimator.py   LandmarkExtractor entry point
scripts/         preprocessing, baselines, verification checks, analysis
configs/         crop boxes, training and inference configs
tests/           unit and integration tests
```

## Team

| | Role |
|---|---|
| **Alfred Hong** | Team lead · 3D preprocessing & geometry · integration |
| **Ojas Joshi** | Landmark model & training |
| **Pravallika Chittapragada** | Evaluation & baselines |
| **Ayush Sahu** | Inference pipeline |

Imperial College London · Huawei Munich Tech Arena 2026

## Acknowledgements

Dataset provided by Huawei for the Munich Tech Arena 2026.
Model architecture inspired by PointNet (Qi et al., CVPR 2017).
