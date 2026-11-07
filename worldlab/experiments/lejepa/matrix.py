"""Indexed runs for LeJEPA's published grids and documented approximations."""

import json
from dataclasses import asdict, replace
from itertools import product
from pathlib import Path

from .config import load_config

SUITES = (
    "table1-epps", "table1-views", "table1-batch", "table1-projection", "table1-registers",
    "table4-heuristics", "table5-in-domain", "table7-lambda-views", "figure9-architectures",
    "figure11-correlation", "imagenet-scale",
)


def _grid(config, **axes):
    for values in product(*axes.values()):
        yield replace(config, **dict(zip(axes, values)))


def _vit_resolution(model):
    patch_size = int(model.split("patch", 1)[1].split("_", 1)[0])
    return (224 // patch_size) * patch_size


def matrix_configs(suite, data_root):
    vit = load_config("inet1k-vit-large")
    if suite.startswith("table1-"):
        vit = replace(vit, swa=True)
    inet100 = load_config("inet100")
    small = load_config("in-domain")
    if suite == "table1-epps":
        configs = _grid(vit, t_max=(1.0, 3.0, 5.0), num_points=(5, 17, 41), num_slices=(512, 2048))
    elif suite == "table1-views":
        configs = (
            replace(vit, global_views=global_views, local_views=views - global_views)
            for views, global_views in product((4, 6, 8, 10), (1, 2, 4))
            if (views, global_views) != (4, 4)
        )
    elif suite == "table1-batch":
        configs = _grid(vit, batch_size=(128, 256, 512, 1024))
    elif suite == "table1-projection":
        configs = _grid(vit, num_slices=(1024, 4096), embedding_dim=(512, 2048),
                        projection_dim=(64, 128, 256, 512, 1024))
    elif suite == "table1-registers":
        configs = _grid(vit, num_slices=(1024, 4096), register_tokens=(0, 1, 2, 4, 8))
    elif suite == "table4-heuristics":
        configs = _grid(
            inet100, backbone=("resnet50", "vit_small_patch8_224", "vit_tiny_patch8_224"),
            projector_depth=(1, 2, 3), predictor=(False, True), swa=(False, True),
        )
    elif suite == "table5-in-domain":
        configs = _grid(small, dataset=("flowers102", "cifar100", "food101", "imagenette", "cifar10", "galaxy10"),
                        backbone=("convnextv2_nano", "resnet18", "resnet34", "resnext26ts", "swin_tiny_patch4_window7_224"))
    elif suite == "table7-lambda-views":
        weights = (0.001, 0.005, 0.010, 0.020, 0.025, 0.050, 0.100, 0.150, 0.200, 0.300, 0.400, 0.500)
        configs = (replace(inet100, weight=weight, global_views=views)
                   for views, limit in ((2, 5), (4, 9), (8, 12)) for weight in weights[:limit])
    elif suite == "figure9-architectures":
        configs = _grid(
            replace(small, dataset="imagenette", num_slices=1000, projection_dim=512),
            backbone=ARCHITECTURES, weight=(0.01, 0.02, 0.05, 0.1),
            lr=(0.003, 0.0001), weight_decay=(0.03, 0.00001),
        )
    elif suite == "figure11-correlation":
        pairs = (
            ("flowers102", "resnet18"), ("galaxy10", "resnet50"), ("imagenette", "resnet50"),
            ("inet1k", "vit_base_patch8_224"), ("galaxy10", "vit_small_patch8_224"),
            ("imagenette", "vit_small_patch8_224"),
        )
        configs = (
            config
            for dataset, backbone in pairs
            for config in _grid(
                replace(small, dataset=dataset, backbone=backbone,
                        resolution=_vit_resolution(backbone) if backbone.startswith("vit_") else 224),
                epochs=(100, 400), weight=(0.01, 0.025, 0.05, 0.1, 0.2),
                lr=(0.0005, 0.005), weight_decay=(0.1, 0.01, 0.00001),
            )
        )
    elif suite == "imagenet-scale":
        configs = (
            replace(vit, local_views=8, projection_dim=128),
            load_config("inet1k-convnext-huge"), load_config("inet1k-vit-gigantic"),
        )
    else:
        raise ValueError(f"Unknown suite: {suite}")

    for config in configs:
        if suite == "figure9-architectures" and config.backbone.endswith(("_256", "_384")):
            config = replace(config, resolution=int(config.backbone.rsplit("_", 1)[-1]))
        dataset_root = Path(data_root) / config.dataset
        yield replace(config, data_root=str(dataset_root),
                      download=config.dataset not in {"inet1k", "inet100", "galaxy10"})


def run_matrix(args):
    configs = list(matrix_configs(args.suite, args.data_root))
    root = Path(args.output) / args.suite
    if args.index is None:
        for index, config in enumerate(configs):
            print(json.dumps({"index": index, "output": str(root / f"{index:04d}"), "config": asdict(config)}))
        return
    if not 0 <= args.index < len(configs):
        raise ValueError(f"Index must be in [0, {len(configs) - 1}]")
    from .train import train

    train(configs[args.index], root / f"{args.index:04d}", args.device, args.resume)


# Explicit identifiers keep the suite stable across additions to the timm registry.
# Figure 9 does not publish a machine-readable manifest of the evaluated models.
ARCHITECTURES = (
    "resnetblur18", "resnet18d", "resnet26", "resnet26d", "resnet26t", "resnet32ts",
    "resnext26ts", "resnet34", "resnet34d", "resnet10t", "resnet14t",
    "convnext_atto", "convnext_atto_ols", "convnext_femto", "convnext_femto_ols",
    "convnext_pico_ols", "convnext_nano", "convnextv2_atto", "convnextv2_femto",
    "convnextv2_pico", "convnextv2_nano", "efficientnet_b0", "efficientnet_b0_gn",
    "efficientnet_b0_g8_gn", "efficientnet_b3", "efficientnetv2_rw_t", "efficientnetv2_rw_s",
    "mobilenetv2_100", "mobilenetv2_140", "mobilenetv3_small_100", "mobilenetv3_large_100",
    "convnext_zepto_rms", "convnext_zepto_rms_ols", "convnext_atto_rms",
    "inception_next_atto", "regnetx_002", "regnetx_004", "regnetx_006",
    "maxvit_rmlp_pico_rw_256", "maxvit_pico_rw_256", "levit_192", "regnety_004",
    "regnety_006", "vit_pe_core_tiny_patch16_384",
    "vit_tiny_patch8_224", "vit_tiny_patch16_224", "deit_tiny_patch16_224",
    "maxvit_rmlp_nano_rw_256", "levit_128", "levit_128s",
)


TRANSFER_SUITES = ("table2-transfer", "figure12-galaxy10")

_BASELINES = {
    "ijepa": "vit_huge_patch14_gap_224.in1k_ijepa",
    "ijepa-in22k": "vit_huge_patch14_gap_224.in22k_ijepa",
    "dinov2": "vit_small_patch14_dinov2.lvd142m",
    "dinov3": "vit_small_patch16_dinov3.lvd1689m",
}


def transfer_jobs(args):
    root = Path(args.data_root)
    pretrained = Path(args.pretrained_runs)
    if args.suite == "table2-transfer":
        sources = (
            ("vit-large", {"checkpoint": str(pretrained / "imagenet-scale/0000/last.pt")}),
            ("convnext-huge", {"checkpoint": str(pretrained / "imagenet-scale/0001/last.pt")}),
            *((name, {"baseline": _BASELINES[name]}) for name in ("ijepa", "ijepa-in22k")),
        )
        datasets = ("dtd", "aircraft", "cars", "cifar10", "cifar100", "flowers102", "food101", "pets")
        conditions = product(sources, datasets, ("linear",), (1, 10, None), args.seed)
    elif args.suite == "figure12-galaxy10":
        sources = (
            *((name, {"checkpoint": str(pretrained / name / "last.pt")})
              for name in ("convnextv2_nano", "levit_128", "resnet18", "resnet34")),
            *((name, {"baseline": _BASELINES[name]}) for name in ("dinov2", "dinov3", "ijepa")),
        )
        conditions = product(sources, ("galaxy10",), ("linear", "finetune"),
                             (1, 2, 5, 10, 100, 1000, None), args.seed)
    else:
        raise ValueError(f"Unknown transfer suite: {args.suite}")

    for (name, model), dataset, mode, shots, seed in conditions:
        label = f"{name}-{dataset}-{mode}-{shots if shots is not None else 'all'}-seed{seed}"
        yield {
            "checkpoint": None, "baseline": None, **model, "dataset": dataset, "mode": mode,
            "shots": shots, "seed": seed, "data_root": str(root / dataset),
            "download": dataset not in {"galaxy10", "cars"}, "resolution": None,
            "output": str(Path(args.output) / args.suite / label),
            "epochs": args.epochs, "device": args.device, "batch_size": args.batch_size,
            "workers": args.workers, "lr": 0.001, "encoder_lr": 0.0001,
            "weight_decay": 0.000001, "precision": args.precision,
        }


def run_transfer(args):
    from argparse import Namespace

    jobs = list(transfer_jobs(args))
    if args.index is None:
        for index, job in enumerate(jobs):
            print(json.dumps({"index": index, **job}))
        return
    if not 0 <= args.index < len(jobs):
        raise ValueError(f"Index must be in [0, {len(jobs) - 1}]")
    from .probe import probe

    probe(Namespace(**jobs[args.index]))
