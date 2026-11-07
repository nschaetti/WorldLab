"""Train, plan and analyze LeWorldModel in WorldLab."""

import argparse
import os

from .config import PRESETS


def _data_output(parser, *, checkpoint=False):
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    if checkpoint:
        parser.add_argument("--checkpoint", required=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train", help="train from offline trajectory HDF5")
    _data_output(training)
    training.add_argument("--preset", choices=PRESETS, default="pusht")
    training.add_argument("--config")
    training.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    training.add_argument("--resume", action="store_true")

    control = commands.add_parser("control", help="evaluate dataset-initialized MPC in a paper environment")
    _data_output(control)
    control.add_argument("--checkpoint")
    control.add_argument("--environment", choices=("pusht", "tworoom", "cube", "reacher"), required=True)
    control.add_argument("--solver", choices=("cem", "adam", "adamw", "sgd", "rmsprop", "random"), default="cem")
    control.add_argument("--episodes", type=int, default=50)
    control.add_argument("--goal-offset", type=int, default=25)
    control.add_argument("--budget", type=int, default=50)
    control.add_argument("--frame-skip", type=int, default=5)
    control.add_argument("--horizon", type=int, default=5)
    control.add_argument("--receding", type=int, default=5)
    control.add_argument("--samples", type=int)
    control.add_argument("--elites", type=int, default=30)
    control.add_argument("--iterations", type=int)
    control.add_argument("--batch-size", type=int, default=1)
    control.add_argument("--solver-lr", type=float, default=0.1)
    control.add_argument("--seed", type=int, default=42)
    control.add_argument("--video")

    matrix = commands.add_parser("matrix", help="list indexed paper ablations and matched evaluation commands")
    matrix.add_argument("suite", choices=("main", "variance", "sigreg", "architecture", "solver", "compute"))
    matrix.add_argument("--data-root", required=True)
    matrix.add_argument("--output", required=True)
    matrix.add_argument("--index", type=int)
    matrix.add_argument("--device", default="auto")
    matrix.add_argument("--run", choices=("train", "control"))

    probe = commands.add_parser("probe", help="fit physical probes to frozen embeddings")
    _data_output(probe, checkpoint=True)
    probe.add_argument("--target", action="append", required=True, metavar="NAME=COLUMN[:I,J,...]")
    probe.add_argument("--frame-skip", type=int, default=5)
    probe.add_argument("--stride", type=int, default=10)
    probe.add_argument("--epochs", type=int, default=50)
    probe.add_argument("--batch-size", type=int, default=128)
    probe.add_argument("--seed", type=int, default=42)
    probe.add_argument("--tsne", type=int, default=0, metavar="POINTS")

    surprise = commands.add_parser("surprise", help="measure prediction errors for matched perturbation episodes")
    _data_output(surprise, checkpoint=True)
    surprise.add_argument("--stats-data", help="training HDF5 for the action normalization statistics")
    surprise.add_argument("--visual", required=True)
    surprise.add_argument("--physical", required=True)
    surprise.add_argument("--perturbation-step", type=int, required=True)
    surprise.add_argument("--frame-skip", type=int, default=5)

    rollout = commands.add_parser("rollout", help="export open-loop errors and latent path straightness")
    _data_output(rollout, checkpoint=True)
    rollout.add_argument("--frame-skip", type=int, default=5)
    rollout.add_argument("--horizon", type=int, default=5)
    rollout.add_argument("--stride", type=int, default=25)
    rollout.add_argument("--batch-size", type=int, default=32)
    rollout.add_argument("--max-windows", type=int, default=512)
    rollout.add_argument("--decoder", help="decoder checkpoint for an imagined/actual PNG strip")

    decoder = commands.add_parser("decoder", help="train a diagnostic decoder on frozen latents")
    _data_output(decoder, checkpoint=True)
    decoder.add_argument("--frame-skip", type=int, default=5)
    decoder.add_argument("--stride", type=int, default=10)
    decoder.add_argument("--epochs", type=int, default=10)
    decoder.add_argument("--batch-size", type=int, default=32)
    decoder.add_argument("--patch-size", type=int, default=16)
    decoder.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    if args.command == "train":
        from .config import load_config
        from .train import train

        train(load_config(args.preset, args.config, args.set), args.data, args.output, args.device, args.resume)
    elif args.command == "control":
        os.environ.setdefault("MUJOCO_GL", "egl")
        from .control import evaluate_control

        evaluate_control(args)
    elif args.command == "matrix":
        from .matrix import list_matrix

        list_matrix(args)
    else:
        from .diagnostics import fit_decoder, physical_probe, rollout_metrics, violation_of_expectation

        {"probe": physical_probe, "surprise": violation_of_expectation,
         "rollout": rollout_metrics, "decoder": fit_decoder}[args.command](args)


if __name__ == "__main__":
    main()
