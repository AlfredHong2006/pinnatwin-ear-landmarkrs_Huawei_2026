# Role D — Runtime Handoff

## Contract

```text
raw Huawei PLY
    ↓
Role A mesh-only crop/canonicalisation/sampling
    ↓
Role B residual model + Role C canonical template
    ↓
(optional: Role C validated post-processing)
    ↓
Role A exact inverse transform
    ↓
left (85,3) + right (85,3) in original Huawei coordinates
```

## Files owned by D

- `src/pipeline.py` — runtime config, artifact loading, orchestration, validation, writer
- `src/infer.py` — CLI
- `tests/test_pipeline.py` — integration/smoke coverage
- `configs/infer.yaml` — safe fallback runtime defaults
- `README.md` — reproduction/inference instructions
- `EXPERIMENTS.md` — runtime handoff record
- `docs/ROLE_D_HANDOFF.md` — this handoff

## Interfaces consumed

### Role A

D calls, but does not reimplement:

- `src.data.load_mesh`
- `src.geometry.load_crop_config`
- `src.geometry.canonicalize_ear`
- `src.geometry.inverse_transform_points`

`canonicalize_ear` is given the raw mesh plus the frozen crop/mirror settings and
never receives annotations.

### Role B

D calls `src.train.load_checkpoint`. The checkpoint carries its own model
architecture and records the training/runtime contract. D rejects a checkpoint
when its stored `n_points`, `n_landmarks`, `in_dim`, template version or ensemble
architecture is incompatible with the runtime.

The model output is interpreted exactly as B defines it:

```python
residual = model(points)                 # [1,85,3]
pred_canonical = template + residual
```

### Role C

D reads C's canonical template (`template` key, `[85,3]`) and global fallback
(`left` / `right`, each `[85,3]`). D does not rebuild either statistic.

The post-processing hook is intentionally late-bound to `src.postprocess` so D
can consume C's implementation later without copying it.

## Determinism

Each ear/sample receives a stable SHA-256-derived seed from:

```text
base seed | subject id | side | sample index
```

This avoids Python's salted `hash()` and ensures left/right/sample draws do not
accidentally share the same sample.

## Output writer

The canonical repo artifact is `<subject_id>.npz`:

- `subject_id`: scalar string
- `left`: float64 `[85,3]`
- `right`: float64 `[85,3]`

`--format csv` is intended for inspection. The public Huawei topic description
confirms separate 85-landmark left/right outputs but does not expose a public file
serialization schema, so the NPZ artifact is explicitly an internal reproducibility
format rather than a claim about the private submission container.

## Current selection state

The experiment log contains three compatible augmented Role B runs (seeds 0, 1,
2). All are recorded as retained candidates, but the current log does not contain a
final selection of one checkpoint or an ensemble. Therefore D deliberately requires
the exact selected checkpoint(s) to be supplied rather than silently choosing one.

## QA command

```bash
pytest tests/test_pipeline.py -q
```

Then, on a machine with the NDA dataset and the required C/B artifacts:

```bash
python -m src.infer --input /path/to/P0001.ply --config configs/infer.yaml --output predictions/
```

For learned-model inference, override `--mode model --checkpoint ...` (repeat
`--checkpoint` for compatible ensemble members).
