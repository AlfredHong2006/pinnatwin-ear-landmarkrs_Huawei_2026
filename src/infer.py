"""Command-line inference entry point owned by Role D."""

from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import InferenceConfig, InferencePipeline, load_inference_config, write_prediction


def _iter_inputs(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() != ".ply":
            raise ValueError(f"input file must be a .ply mesh, got {path}")
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"input path not found: {path}")
    meshes = sorted(p for p in path.glob("P*.ply") if p.is_file())
    if not meshes:
        raise FileNotFoundError(f"no P*.ply meshes found under {path}")
    return meshes


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Huawei PinnaTwin-Zoom inference: raw PLY -> left/right 85x3 landmarks"
    )
    parser.add_argument("--input", required=True, type=Path, help="one .ply file or a directory of P*.ply meshes")
    parser.add_argument("--config", default="configs/infer.yaml", type=Path, help="Role D runtime YAML")
    parser.add_argument("--output", default="predictions", type=Path, help="output directory")
    parser.add_argument("--format", choices=("npz", "csv", "both"), default="npz")
    parser.add_argument("--checkpoint", action="append", help="override config checkpoint; repeat for a compatible ensemble")
    parser.add_argument("--mode", choices=("fallback", "model"), help="override config mode")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    config = load_inference_config(args.config)

    data = dict(config.__dict__)
    if args.mode is not None:
        data["mode"] = args.mode
    if args.checkpoint:
        if len(args.checkpoint) == 1:
            data["checkpoint"] = args.checkpoint[0]
            data["ensemble_checkpoints"] = ()
        else:
            data["checkpoint"] = None
            data["ensemble_checkpoints"] = tuple(args.checkpoint)
    config = InferenceConfig(**data)

    pipeline = InferencePipeline.from_config(config)
    inputs = _iter_inputs(args.input)

    for mesh in inputs:
        prediction = pipeline.predict(mesh)
        paths = write_prediction(
            prediction.as_dict(),
            args.output,
            prediction.subject_id,
            fmt=args.format,
        )
        print(
            f"{prediction.subject_id}: left={prediction.left.shape} "
            f"right={prediction.right.shape} -> {', '.join(str(p) for p in paths)}"
        )

    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through subprocess/CLI
    raise SystemExit(main())
