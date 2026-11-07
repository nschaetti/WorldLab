"""Train and compare unimodal JEPAs for the Platonic Representation Hypothesis."""

import argparse

from .config import PRESETS
from .paper import SCRIPTS


def _scene_input(parser):
    parser.add_argument("--data", required=True, help="Scene JSONL or Flickr30k Karpathy CSV/directory")
    parser.add_argument("--root", help="Directory for relative image paths")


def _pairs(parser):
    parser.add_argument("source", help="Source embedding cache")
    parser.add_argument("target", help="Target embedding cache")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="train a single unimodal arm")
    _scene_input(train)
    train.add_argument("--preset", choices=PRESETS, default="good-text")
    train.add_argument("--smoke", action="store_true", help="use a small CPU-friendly model")
    train.add_argument("--config")
    train.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    train.add_argument("--output", required=True)
    train.add_argument("--device", default="auto")
    train.add_argument("--resume", action="store_true")

    embed = commands.add_parser("embed", help="export embeddings keyed by scene ID")
    _scene_input(embed)
    model = embed.add_mutually_exclusive_group(required=True)
    model.add_argument("--checkpoint")
    model.add_argument("--vision-backbone", help="pretrained timm ViT identifier")
    embed.add_argument("--resolution", type=int, default=224)
    embed.add_argument("--split", action="append")
    embed.add_argument("--layers", action="store_true")
    embed.add_argument("--batch-size", type=int, default=64)
    embed.add_argument("--device", default="auto")
    embed.add_argument("--output", required=True)

    align = commands.add_parser("align", help="fit a rotation on validation scenes and score held-out retrieval")
    _pairs(align)
    align.add_argument("--fit-split", default="val")
    align.add_argument("--test-split", default="test")
    align.add_argument("--test-source")
    align.add_argument("--test-target")
    align.add_argument("--features", choices=("embedding", "projection"), default="embedding")
    align.add_argument("--rotation", help="save the fitted rotation")

    benchmark = commands.add_parser("benchmark", help="run the rotation-free max-over-layers PRH protocol")
    _pairs(benchmark)
    benchmark.add_argument("--split", default="test")
    benchmark.add_argument("--metric", choices=("mknn", "cka"), default="mknn")

    nuisance = commands.add_parser("nuisance", help="measure how much a text representation encodes the scene")
    _scene_input(nuisance)
    nuisance.add_argument("--checkpoint", required=True)
    nuisance.add_argument("--split", default="test")
    nuisance.add_argument("--views", type=int, default=5)
    nuisance.add_argument("--dimensions", type=int, nargs="+", default=[32, 128])
    nuisance.add_argument("--batch-size", type=int, default=64)
    nuisance.add_argument("--device", default="auto")
    nuisance.add_argument("--output")

    theory = commands.add_parser("theory", help="check the spectral mismatch bound on random finite worlds")
    theory.add_argument("--cases", type=int, default=300)
    theory.add_argument("--seed", type=int, default=0)
    theory.add_argument("--output")

    scaling = commands.add_parser("scaling", help="fit the paper's alignment law from a phase-grid CSV")
    scaling.add_argument("--data", required=True)
    scaling.add_argument("--fit", choices=("v1", "v2"), default="v2")
    scaling.add_argument("--output")

    registry = commands.add_parser("paper-run", help="show or execute a run in the official registry")
    registry.add_argument("--checkout", required=True)
    registry.add_argument("--run", action="store_true")
    registry.add_argument("tag")
    registry.add_argument("arguments", nargs=argparse.REMAINDER)

    upstream = commands.add_parser("upstream", help="run an official reproduction script")
    upstream.add_argument("--checkout", required=True)
    upstream.add_argument("script", choices=SCRIPTS)
    upstream.add_argument("arguments", nargs=argparse.REMAINDER)

    args = parser.parse_args(argv)
    if args.command == "train":
        from .config import load_config
        from .train import train as run

        config = load_config(args.preset, args.config, args.set, args.smoke)
        run(config, args.data, args.output, root=args.root, device_name=args.device, resume=args.resume)
    elif args.command in {"paper-run", "upstream"}:
        from .paper import registry, upstream

        if args.arguments[:1] == ["--"]:
            args.arguments.pop(0)
        (registry if args.command == "paper-run" else upstream)(args)
    elif args.command in {"scaling", "theory"}:
        from .analysis import check_theory, fit_scaling

        (check_theory if args.command == "theory" else fit_scaling)(args)
    else:
        from .evaluate import align, benchmark, embed, nuisance

        if args.command == "align" and bool(args.test_source) != bool(args.test_target):
            parser.error("Provide both --test-source and --test-target for transfer")
        {"embed": embed, "align": align, "benchmark": benchmark, "nuisance": nuisance}[args.command](args)


if __name__ == "__main__":
    main()
