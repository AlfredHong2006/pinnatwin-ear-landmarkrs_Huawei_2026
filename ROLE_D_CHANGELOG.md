# Role D implementation record

Implemented against the repository snapshot supplied in the chat. Role A/B/C source
modules were not modified.

## Added

- `src/pipeline.py` — runtime configuration, artifact loading, checkpoint validation,
  mesh-only A preprocessing orchestration, B residual inference, optional C hook,
  inverse transform, prediction validation, deterministic TTA/ensemble orchestration,
  and output writer.
- `src/infer.py` — one-command CLI and directory batch support.
- `configs/infer.yaml` — safe global-mean fallback defaults.
- `tests/test_pipeline.py` — 12 focused integration/invariant tests.
- `docs/ROLE_D_HANDOFF.md` — interface, determinism, artifact, and QA handoff.
- `ROLE_D_CHANGELOG.md` — this record.

## Updated

- `README.md` — inference, model/fallback usage, output contract, dependency setup,
  and reproduction instructions.
- `EXPERIMENTS.md` — Role D runtime handoff and current checkpoint-selection state.

## Validation

- `pytest tests/test_pipeline.py -q` → **12 passed**.
- CLI smoke with synthetic-only PLY/template/checkpoint data → successful.
- Produced `left` and `right` arrays were both `(85, 3)`, `float64`, finite.
- Byte-level comparison against the supplied repository snapshot found **no changes
  outside D-owned files plus README/EXPERIMENTS**.

## Existing unrelated issue

The supplied repository snapshot's full test suite has one pre-existing failure in
`tests/test_geometry.py`: the SHA-256 recorded in `configs/crop.yaml` does not match
the checked-in `splits/train_ids.txt`. This is an A-owned frozen-config/provenance issue;
Role D deliberately does not alter it.
