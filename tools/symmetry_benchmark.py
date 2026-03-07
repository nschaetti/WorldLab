"""
Run a multi-run symmetry benchmark campaign.

This CLI launches multiple workflow runs and aggregates results into:
- CSV table
- Markdown report
- JSON summary
"""

from __future__ import annotations

import argparse
import itertools
import json

from boardGPT.analysis import (
    PolicyConfig,
    WorkflowConfig,
    run_workflow_benchmark,
)


def _parse_int_tuple(text: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in text.split(",") if x.strip())


def _parse_bool_tuple(text: str) -> tuple[bool, ...]:
    vals = [v.strip() for v in text.split(",") if v.strip()]
    out = []
    for v in vals:
        if v in {"1", "true", "True", "yes", "y"}:
            out.append(True)
        elif v in {"0", "false", "False", "no", "n"}:
            out.append(False)
        else:
            raise ValueError(f"invalid bool value: {v}")
    return tuple(out)


def _make_default_campaign(run_model: bool) -> list[tuple[str, WorkflowConfig]]:
    specs = [
        ("ring1d_local", (32,), (True,), 0.18),
        ("line1d_local", (32,), (False,), 0.18),
        ("grid2d_local", (8, 8), (False, False), 0.25),
        ("grid3d_local", (4, 4, 4), (False, False, False), 0.3),
    ]
    runs: list[tuple[str, WorkflowConfig]] = []
    for name, shape, periodic, scale in specs:
        cfg = WorkflowConfig(
            world_shape=shape,
            periodic=periodic,
            world_name=name,
            policy=PolicyConfig(length=48, num_sequences=1200, seed=0, transition_scale=scale),
            run_model=run_model,
            decoding_max_rank=12,
        )
        runs.append((name, cfg))
    return runs


def _make_grid_campaign(
    dims: tuple[int, ...],
    sizes: tuple[int, ...],
    scales: tuple[float, ...],
    run_model: bool,
) -> list[tuple[str, WorkflowConfig]]:
    runs: list[tuple[str, WorkflowConfig]] = []
    for dim, size, scale in itertools.product(dims, sizes, scales):
        shape = tuple([size] * dim)
        periodic = tuple([False] * dim)
        name = f"grid{dim}d_s{size}_sig{scale:g}"
        cfg = WorkflowConfig(
            world_shape=shape,
            periodic=periodic,
            world_name=name,
            policy=PolicyConfig(length=48, num_sequences=1200, seed=0, transition_scale=scale),
            run_model=run_model,
            decoding_max_rank=12,
        )
        runs.append((name, cfg))
    return runs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run symmetry benchmark campaign")
    parser.add_argument("--name", type=str, default="symmetry_benchmark", help="Benchmark campaign name")
    parser.add_argument("--output", type=str, default="out/symmetry_benchmarks", help="Output root directory")
    parser.add_argument("--run-model", action="store_true", help="Include model training/representation analysis")
    parser.add_argument(
        "--preset",
        type=str,
        default="default",
        choices=["default", "grid"],
        help="Campaign preset",
    )
    parser.add_argument("--dims", type=str, default="1,2,3", help="For preset=grid: latent dims")
    parser.add_argument("--sizes", type=str, default="8", help="For preset=grid: lattice size per axis")
    parser.add_argument("--scales", type=str, default="0.18,0.25,0.35", help="For preset=grid: transition scales")
    args = parser.parse_args(argv)

    if args.preset == "default":
        campaign = _make_default_campaign(run_model=args.run_model)
    else:
        dims = _parse_int_tuple(args.dims)
        sizes = _parse_int_tuple(args.sizes)
        scales = tuple(float(x.strip()) for x in args.scales.split(",") if x.strip())
        campaign = _make_grid_campaign(dims=dims, sizes=sizes, scales=scales, run_model=args.run_model)

    summary = run_workflow_benchmark(
        benchmark_name=args.name,
        configs=campaign,
        output_root=args.output,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
