"""
CLI wrapper for the synthetic symmetry end-to-end workflow.

Example:
    python tools/symmetry_workflow.py --shape 32 --periodic 1 --output out/symmetry/demo
"""

from __future__ import annotations

import argparse
import json

from worldlab.board.analysis import PolicyConfig, WorkflowConfig, run_synthetic_symmetry_workflow


def _parse_bool_flags(text: str) -> tuple[bool, ...]:
    vals = [v.strip() for v in text.split(",") if v.strip()]
    out = []
    for v in vals:
        if v in {"1", "true", "True", "yes", "y"}:
            out.append(True)
        elif v in {"0", "false", "False", "no", "n"}:
            out.append(False)
        else:
            raise ValueError(f"invalid bool flag: {v}")
    return tuple(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run synthetic symmetry workflow")
    parser.add_argument("--shape", type=str, default="32", help="Comma-separated world shape, e.g. 32 or 8,8")
    parser.add_argument("--periodic", type=str, default=None, help="Comma-separated periodic flags, e.g. 1 or 0,1")
    parser.add_argument("--length", type=int, default=48, help="Sequence length")
    parser.add_argument("--num-sequences", type=int, default=1200, help="Number of generated sequences")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--run-model", action="store_true", help="Also train model and run representation diagnostics")
    parser.add_argument("--output", type=str, default=None, help="Optional output directory")
    args = parser.parse_args(argv)

    shape = tuple(int(x.strip()) for x in args.shape.split(",") if x.strip())
    periodic = _parse_bool_flags(args.periodic) if args.periodic is not None else None

    cfg = WorkflowConfig(
        world_shape=shape,
        periodic=periodic,
        policy=PolicyConfig(length=args.length, num_sequences=args.num_sequences, seed=args.seed),
        run_model=args.run_model,
        output_dir=args.output,
    )
    summary = run_synthetic_symmetry_workflow(cfg)
    print(json.dumps(summary, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
