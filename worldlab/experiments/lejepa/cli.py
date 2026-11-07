"""Command-line entry points for the LeJEPA reproduction."""

import argparse

from .config import load_config, presets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train", help="pretrain and evaluate an online linear probe")
    training.add_argument("--preset", choices=presets())
    training.add_argument("--config", help="YAML overrides for a preset")
    training.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    training.add_argument("--output", required=True)
    training.add_argument("--device", default="auto")
    training.add_argument("--resume", action="store_true")

    transfer = commands.add_parser("probe", help="train a frozen probe or fine-tune a checkpoint")
    source = transfer.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint")
    source.add_argument("--baseline", help="pretrained timm model identifier")
    transfer.add_argument("--dataset", required=True)
    transfer.add_argument("--data-root", required=True)
    transfer.add_argument("--download", action="store_true")
    transfer.add_argument("--output", required=True)
    transfer.add_argument("--mode", choices=("linear", "finetune"), default="linear")
    transfer.add_argument("--device", default="auto")
    transfer.add_argument("--resolution", type=int)
    transfer.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    transfer.add_argument("--epochs", type=int, default=50)
    transfer.add_argument("--batch-size", type=int, default=256)
    transfer.add_argument("--workers", type=int, default=8)
    transfer.add_argument("--seed", type=int, default=0)
    transfer.add_argument("--shots", type=int)
    transfer.add_argument("--lr", type=float, default=0.001)
    transfer.add_argument("--encoder-lr", type=float, default=0.0001)
    transfer.add_argument("--weight-decay", type=float, default=0.000001)

    from .matrix import SUITES, TRANSFER_SUITES

    matrix = commands.add_parser("matrix", help="list or execute indexed paper runs")
    matrix.add_argument("suite", choices=SUITES)
    matrix.add_argument("--data-root", default="data/lejepa")
    matrix.add_argument("--output", required=True)
    matrix.add_argument("--device", default="auto")
    matrix.add_argument("--index", type=int)
    matrix.add_argument("--resume", action="store_true")

    downstream = commands.add_parser("transfer-matrix", help="list or execute downstream evaluation runs")
    downstream.add_argument("suite", choices=TRANSFER_SUITES)
    downstream.add_argument("--pretrained-runs", required=True)
    downstream.add_argument("--data-root", default="data/lejepa")
    downstream.add_argument("--output", required=True)
    downstream.add_argument("--index", type=int)
    downstream.add_argument("--seed", type=int, nargs="+", default=[0, 1, 2])
    downstream.add_argument("--device", default="auto")
    downstream.add_argument("--epochs", type=int, default=50)
    downstream.add_argument("--batch-size", type=int, default=256)
    downstream.add_argument("--workers", type=int, default=8)
    downstream.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")

    for name in ("summarize", "summarize-probes", "correlate"):
        report = commands.add_parser(name)
        report.add_argument("--runs", required=True)
        report.add_argument("--output", required=True)

    timing = commands.add_parser("timing", help="benchmark the Table 6 workloads")
    timing.add_argument("--device", default="auto")
    timing.add_argument("--output", required=True)
    timing.add_argument("--dimension", type=int, default=512)
    timing.add_argument("--repeats", type=int, default=10)

    synthetic = commands.add_parser("synthetic", help="run the Figures 6–7 projection experiment")
    synthetic.add_argument("--device", default="auto")
    synthetic.add_argument("--output", required=True)
    synthetic.add_argument("--steps", type=int, default=500)
    synthetic.add_argument("--seed", type=int, default=0)

    quadrature = commands.add_parser("quadrature", help="compare Epps–Pulley integration with its exact integral")
    quadrature.add_argument("--output", required=True)
    quadrature.add_argument("--seed", type=int, default=0)

    spatial = commands.add_parser("visualize", help="render ViT patch PCA and CLS attention")
    spatial.add_argument("--checkpoint", required=True)
    spatial.add_argument("--output", required=True)
    spatial.add_argument("--device", default="auto")
    spatial.add_argument("--mass", type=float, default=0.6)
    spatial.add_argument("images", nargs="+")

    args = parser.parse_args()
    if args.command == "train":
        from .train import train

        train(load_config(args.preset, args.config, args.set), args.output, args.device, args.resume)
    elif args.command == "probe":
        from .probe import probe

        probe(args)
    elif args.command == "matrix":
        from .matrix import run_matrix

        run_matrix(args)
    elif args.command == "transfer-matrix":
        from .matrix import run_transfer

        run_transfer(args)
    elif args.command in {"summarize", "summarize-probes", "correlate"}:
        from .analysis import correlations, summarize, summarize_probes

        action = {"summarize": summarize, "summarize-probes": summarize_probes,
                  "correlate": correlations}[args.command]
        action(args.runs, args.output)
    elif args.command == "timing":
        from .analysis import sigreg_timing

        sigreg_timing(args.output, args.device, args.dimension, args.repeats)
    elif args.command == "synthetic":
        from .analysis import synthetic_sigreg

        synthetic_sigreg(args.output, args.steps, args.device, args.seed)
    elif args.command == "quadrature":
        from .analysis import quadrature

        quadrature(args.output, args.seed)
    elif args.command == "visualize":
        from .analysis import visualize

        visualize(args.checkpoint, args.images, args.output, args.device, args.mass)


if __name__ == "__main__":
    main()
