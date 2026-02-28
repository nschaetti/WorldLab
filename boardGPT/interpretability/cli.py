"""
CLI for interpretability workflows (B3).
"""

import argparse
import json
from pathlib import Path
from typing import List

from .io import load_run_artifacts
from .pipeline import run_b3_demo_pipeline
from .real_pipeline import (
    load_exp_config,
    run_b4_batch_from_config_file,
    run_b4_from_config_file,
    run_b4_single_from_config,
)


def _parse_layers(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def _parse_cells(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interpretability CLI (B3)")
    sub = parser.add_subparsers(dest="command")

    smoke = sub.add_parser("smoke", help="Run full B3 demo pipeline")
    smoke.add_argument("--out-root", default="out_interp/jalon_b3", help="Root output directory")
    smoke.add_argument("--run-name", default=None, help="Optional run name")
    smoke.add_argument("--seed", type=int, default=0)
    smoke.add_argument("--layers", default="0,1", help="Comma-separated layers")
    smoke.add_argument("--target-cell", type=int, default=5)
    smoke.add_argument("--target-class", type=int, default=2)
    smoke.add_argument("--target-cells", default="3,5,9", help="Comma-separated cells for attribution")
    smoke.add_argument("--max-examples", type=int, default=8)

    metrics = sub.add_parser("b2-metrics", help="Run B3 demo and print metrics summary")
    metrics.add_argument("--out-root", default="out_interp/jalon_b3")
    metrics.add_argument("--run-name", default=None)
    metrics.add_argument("--seed", type=int, default=0)
    metrics.add_argument("--layers", default="0,1")
    metrics.add_argument("--target-cell", type=int, default=5)
    metrics.add_argument("--target-class", type=int, default=2)
    metrics.add_argument("--max-examples", type=int, default=8)

    attrib = sub.add_parser("b2-attrib", help="Run B3 demo and print attribution top contributors")
    attrib.add_argument("--out-root", default="out_interp/jalon_b3")
    attrib.add_argument("--run-name", default=None)
    attrib.add_argument("--seed", type=int, default=0)
    attrib.add_argument("--layers", default="0,1")
    attrib.add_argument("--target-cells", default="3,5,9")
    attrib.add_argument("--target-class", type=int, default=2)
    attrib.add_argument("--max-examples", type=int, default=8)

    report = sub.add_parser("report", help="Print stored report/summary from a run directory")
    report.add_argument("--run-dir", required=True)

    real_metrics = sub.add_parser("real-metrics", help="Run B4 real metrics from exp config")
    real_metrics.add_argument("--exp-config", required=True)
    real_metrics.add_argument("--run-name", default=None)

    real_attrib = sub.add_parser("real-attrib", help="Run B4 real attribution from exp config")
    real_attrib.add_argument("--exp-config", required=True)
    real_attrib.add_argument("--run-name", default=None)

    real_batch = sub.add_parser("real-batch", help="Run B4 multi-seed/checkpoint batch from exp config")
    real_batch.add_argument("--exp-config", required=True)
    real_batch.add_argument("--run-name", default=None)

    return parser


def _run_with_args(args: argparse.Namespace) -> dict:
    layers = _parse_layers(getattr(args, "layers", "0,1"))
    target_cells = _parse_cells(getattr(args, "target_cells", "3,5,9"))
    target_cell = getattr(args, "target_cell", target_cells[0] if target_cells else 5)
    return run_b3_demo_pipeline(
        out_root=args.out_root,
        run_name=args.run_name,
        seed=args.seed,
        layers=layers,
        target_cell=target_cell,
        target_class=args.target_class,
        target_cells=target_cells,
        max_examples=args.max_examples,
    )


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 1

    if args.command in {"smoke", "b2-metrics", "b2-attrib"}:
        output = _run_with_args(args)
        run_dir = output["run_dir"]
        print(f"Run directory: {run_dir}")
        print(f"Report: {output['report_md']}")

        if args.command == "b2-metrics":
            metrics = output["metrics_report"].get("metrics", {})
            print(json.dumps(metrics, indent=2))
        elif args.command == "b2-attrib":
            attr = output["attribution_report"].get("attribution", {})
            top = attr.get("top_contributors", [])
            print(json.dumps(top, indent=2))
        else:
            print("Completed demo pipeline.")
        return 0

    if args.command == "report":
        loaded = load_run_artifacts(args.run_dir)
        print(json.dumps(loaded.get("summary", {}), indent=2))
        report_md = Path(args.run_dir) / "report.md"
        if report_md.exists():
            print(f"Markdown report: {report_md}")
        return 0

    if args.command in {"real-metrics", "real-attrib"}:
        cfg = load_exp_config(args.exp_config)
        cfg.setdefault("protocol", {})
        if args.command == "real-metrics":
            cfg["protocol"]["type"] = "metrics_only"
        else:
            cfg["protocol"]["type"] = "attribution_only"
        out = run_b4_single_from_config(cfg, run_name=args.run_name)
        print(f"Run directory: {out['run_dir']}")
        if out.get("report_md"):
            print(f"Report: {out['report_md']}")
        metrics = out.get("metrics_report", {})
        if isinstance(metrics, dict):
            print(json.dumps(metrics.get("metrics", {}), indent=2))
        attr = out.get("attribution_report", {})
        if isinstance(attr, dict):
            top = attr.get("attribution", {}).get("top_contributors", [])
            if top:
                print(json.dumps(top[:10], indent=2))
        return 0

    if args.command == "real-batch":
        out = run_b4_batch_from_config_file(args.exp_config, run_name=args.run_name)
        print(f"Batch directory: {out['batch_dir']}")
        print(json.dumps(out.get("summary", {}), indent=2))
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
