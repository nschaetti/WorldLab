"""Frozen linear probes and end-to-end downstream fine-tuning."""

import json
from dataclasses import replace
from pathlib import Path

import timm
import torch
from filelock import FileLock
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from worldlab.training import optimizer_step, warmup_cosine_lr
from worldlab.training.runtime import atomic_write, autocast, get_device, seed_epoch

from .config import RunConfig
from .data import (
    TransformedImages, _augment, check_splits, class_count, evaluation_transform,
    few_shot_subset, image_dataset,
)

from .train import evaluate
from .vision import VisionEncoder


class DownstreamClassifier(nn.Module):
    def __init__(self, encoder, classes, frozen):
        super().__init__()
        self.encoder = encoder
        self.frozen = frozen
        encoder.requires_grad_(not frozen)
        self.head = nn.Sequential(nn.LayerNorm(encoder.output_dim), nn.Linear(encoder.output_dim, classes))

    def classify(self, images):
        return self.head(self.encoder(images))


def load_encoder(checkpoint=None, baseline=None, resolution=None):
    if baseline:
        default_size = timm.get_pretrained_cfg(baseline).input_size[-1]
        config = RunConfig(backbone=baseline, pretrained=True, resolution=resolution or default_size)
        return VisionEncoder(config), config
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = replace(RunConfig(**state["config"]), pretrained=False, patch_mask_ratio=0.0,
                     activation_checkpointing=False)
    encoder = VisionEncoder(config)
    prefix = "jepa.encoder."
    encoder.load_state_dict({
        key.removeprefix(prefix): value for key, value in state["model"].items() if key.startswith(prefix)
    })
    if resolution is not None and resolution != config.resolution:
        raise ValueError("WorldLab checkpoints are evaluated at their pretraining resolution")
    return encoder, config


@torch.inference_mode()
def extract_features(encoder, loader, device, precision):
    features, labels = [], []
    encoder.eval()
    for images, target in loader:
        with autocast(device, precision):
            features.append(encoder(images.to(device, non_blocking=True)).float().cpu())
        labels.append(target)
    return TensorDataset(torch.cat(features), torch.cat(labels))


class FeatureClassifier(nn.Sequential):
    def classify(self, features):
        return self(features)


def probe(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    with FileLock(output / ".run.lock", timeout=0):
        if (output / "probe.json").exists():
            raise ValueError("Probe output already exists. Choose a fresh directory.")
        return _probe(args, output)


def _probe(args, output):
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0:
        raise ValueError("Probe requires positive epochs and batch size and non-negative workers")
    device = get_device(args.device)
    seed_epoch(args.seed, 0, 0)
    encoder, config = load_encoder(args.checkpoint, args.baseline, args.resolution)
    train = image_dataset(args.dataset, args.data_root, True, args.download)
    test = image_dataset(args.dataset, args.data_root, False, args.download)
    check_splits(train, test)
    classes = class_count(train)
    train = few_shot_subset(train, args.shots, args.seed)
    frozen = args.mode == "linear"
    deterministic = evaluation_transform(config.resolution, config.augment)
    transform = deterministic if frozen else _augment(config.resolution, (0.08, 1.0), config.augment)
    options = dict(batch_size=args.batch_size, num_workers=args.workers, pin_memory=device.type == "cuda")
    train_loader = DataLoader(TransformedImages(train, transform), shuffle=not frozen, **options)
    val_loader = DataLoader(TransformedImages(test, deterministic), **options)

    model = DownstreamClassifier(encoder, classes, frozen).to(device)
    precision = args.precision
    if frozen:
        features_train = extract_features(encoder, train_loader, device, precision)
        features_val = extract_features(encoder, val_loader, device, precision)
        train_loader = DataLoader(features_train, shuffle=True, **options)
        val_loader = DataLoader(features_val, **options)
        model = FeatureClassifier(*model.head)
        del encoder

    if frozen:
        groups = [{"params": model.parameters(), "lr": args.lr}]
    else:
        groups = [
            {"params": model.encoder.parameters(), "lr": args.encoder_lr},
            {"params": model.head.parameters(), "lr": args.lr},
        ]
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    peak_lrs = [group["lr"] for group in optimizer.param_groups]
    history = []
    steps = len(train_loader)
    for epoch in range(args.epochs):
        seed_epoch(args.seed, epoch + 1, 0)
        model.train()
        for index, (images, labels) in enumerate(train_loader):
            for group, lr in zip(optimizer.param_groups, peak_lrs):
                group["lr"] = warmup_cosine_lr(
                    epoch * steps + index, lr, lr / 1000, steps, args.epochs * steps, start_factor=0.01,
                )
            with autocast(device, precision):
                logits = model.classify(images.to(device, non_blocking=True))
                loss = F.cross_entropy(logits, labels.to(device, non_blocking=True))
            loss.backward()
            optimizer_step(optimizer)
        record = {"epoch": epoch + 1, **evaluate(model, val_loader, device, precision)}
        history.append(record)
        print(json.dumps(record), flush=True)

    settings = {key: value for key, value in vars(args).items() if key != "command"}
    state = {**settings, "resolution": config.resolution, "num_classes": classes,
             "train_samples": len(train), "history": history}
    atomic_write(output / "probe.pt", lambda path: torch.save(model.state_dict(), path))
    atomic_write(output / "probe.json", lambda path: path.write_text(json.dumps(state, indent=2) + "\n"))
    return state
