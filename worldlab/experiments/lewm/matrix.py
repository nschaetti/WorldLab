"""Runnable manifests for control, training variance and LeWM ablations."""

import json
import subprocess
import sys
from itertools import product
from pathlib import Path

DATA_FILES = {
    "pusht": "pusht_expert_train.h5",
    "tworoom": "tworoom.h5",
    "cube": "ogbench/cube_single_expert.h5",
    "reacher": "dmc/reacher_random.h5",
}
SEEDS = (3072, 3073, 3074)


def cases(suite):
    if suite == "main":
        for environment in DATA_FILES:
            yield environment, environment, {}, {}
    elif suite == "variance":
        for seed in SEEDS:
            yield f"seed-{seed}", "pusht", {"seed": seed}, {}
    elif suite == "sigreg":
        axes = {"latent_dim": (8, 24, 96, 192, 384), "slices": (64, 256, 512, 1024),
                "knots": (4, 8, 12, 17), "weight": (0.01, 0.05, 0.09, 0.095, 0.1, 0.2, 0.5)}
        for key, values in axes.items():
            for value, seed in product(values, SEEDS):
                yield f"{key}-{value}-seed-{seed}", "pusht", {key: value, "seed": seed}, {}
    elif suite == "architecture":
        axes = {"encoder": ("vit-tiny", "resnet18"), "predictor_depth": (3, 6, 12),
                "dropout": (0.0, 0.1, 0.2, 0.5), "reconstruction_weight": (0.0, 1.0)}
        for key, values in axes.items():
            for value, seed in product(values, SEEDS):
                yield f"{key}-{value}-seed-{seed}", "pusht", {key: value, "seed": seed}, {}
    elif suite == "solver":
        for solver, seed in product(("cem", "sgd", "rmsprop", "adam"), SEEDS):
            yield f"{solver}-seed-{seed}", "pusht", None, {"solver": solver, "training_seed": seed}
    elif suite == "compute":
        for environment, samples, iterations in product(("pusht", "cube"), (60, 150, 300), (5, 10, 30)):
            yield (f"{environment}-{samples}x{iterations}", environment, None,
                   {"samples": samples, "iterations": iterations})
    else:
        raise ValueError(f"Unknown experiment suite: {suite}")


def manifest(args):
    root = Path(args.output)
    data_root = Path(args.data_root)
    for index, (name, environment, options, control_settings) in enumerate(cases(args.suite)):
        dataset = str(data_root / DATA_FILES[environment])
        output = root / args.suite / name
        checkpoint = output
        training = None
        if options is not None:
            training = ["train", "--preset", environment, "--data", dataset,
                        "--output", str(output), "--device", args.device]
            for key, value in options.items():
                training.extend(("--set", f"{key}={value}"))
        elif args.suite == "solver":
            checkpoint = root / "variance" / f"seed-{control_settings['training_seed']}"
        else:
            checkpoint = root / "main" / environment
        control = ["control", "--environment", environment, "--data", dataset,
                   "--checkpoint", str(checkpoint), "--output", str(output / "control.json"),
                   "--device", args.device]
        for key, value in control_settings.items():
            if key != "training_seed":
                control.extend((f"--{key.replace('_', '-')}", str(value)))
        yield {"index": index, "name": name, "train": training, "control": control}


def list_matrix(args):
    jobs = list(manifest(args))
    if args.index is None:
        if args.run:
            raise ValueError("--run requires an --index")
        for job in jobs:
            print(json.dumps(job))
        return
    if not 0 <= args.index < len(jobs):
        raise ValueError(f"Index must be in [0, {len(jobs) - 1}]")
    job = jobs[args.index]
    if args.run:
        command = job[args.run]
        if command is None:
            raise ValueError("This job evaluates a checkpoint from a different suite")
        subprocess.run([sys.executable, "-m", "worldlab.experiments.lewm", *command], check=True)
    else:
        print(json.dumps(job))
