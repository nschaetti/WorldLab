"""Pretrain LeJEPA with an online linear probe on CPU, CUDA or torchrun."""

import json
import os
import platform
import random
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import timm
import torch
import torchvision
from filelock import FileLock
from torch import distributed as dist, nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, Subset

from worldlab.training import optimizer_step, warmup_cosine_lr

from .data import (
    TransformedImages, check_splits, class_count, evaluation_transform, image_dataset, view_transform,
)
from .vision import LeJEPAClassifier


def get_device(requested, local_rank=0):
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if requested == "auto" else requested)
    if device.type == "cuda":
        device = torch.device("cuda", local_rank if device.index is None else device.index)
        torch.cuda.set_device(device)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Only CPU and CUDA are supported by the experiment runner")
    return device


def autocast(device, precision):
    return torch.autocast(device.type, enabled=precision != "fp32",
                          dtype=torch.float16 if precision == "fp16" else torch.bfloat16)


def seed_epoch(seed, epoch, rank):
    epoch_seed = seed + epoch * 100003 + rank
    torch.manual_seed(epoch_seed)
    random.seed(epoch_seed)


def atomic_write(path, write):
    temporary = path.with_suffix(path.suffix + ".tmp")
    write(temporary)
    os.replace(temporary, path)


def make_loaders(config, device, rank, world_size):
    if config.batch_size % world_size or config.batch_size // world_size < 2:
        raise ValueError("Global batch size must be divisible by world size with at least 2 samples per rank")
    if world_size > 1 and config.download and rank:
        dist.barrier()
    train = image_dataset(config.dataset, config.data_root, True, config.download)
    test = image_dataset(config.dataset, config.data_root, False, config.download)
    if world_size > 1 and config.download and rank == 0:
        dist.barrier()
    check_splits(train, test)
    classes = class_count(train)
    train = TransformedImages(train, view_transform(config))
    test = TransformedImages(test, evaluation_transform(config.resolution, config.augment))
    sampler = DistributedSampler(train, num_replicas=world_size, rank=rank, seed=config.seed, drop_last=True)
    options = dict(num_workers=config.workers, pin_memory=device.type == "cuda")
    train_loader = DataLoader(train, batch_size=config.batch_size // world_size,
                              sampler=sampler, drop_last=True, **options)
    test_loader = DataLoader(Subset(test, range(rank, len(test), world_size)),
                             batch_size=config.batch_size // world_size, **options)
    if not len(train_loader):
        raise ValueError("No full global batch. Reduce batch_size or use a larger dataset.")
    return train_loader, test_loader, classes


def _reduce(values):
    if dist.is_initialized():
        dist.all_reduce(values)
    return values


@torch.inference_mode()
def evaluate(model, loader, device, precision):
    model.eval()
    totals = torch.zeros(3, dtype=torch.float64, device=device)
    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        with autocast(device, precision):
            logits = model.classify(images)
        totals[0] += (logits.argmax(dim=-1) == labels).sum()
        totals[1] += F.cross_entropy(logits.float(), labels, reduction="sum")
        totals[2] += labels.numel()
    correct, loss, count = _reduce(totals).tolist()
    return {"top1": correct / count, "val_probe_loss": loss / count, "val_samples": int(count)}


def train(config, output, requested_device="auto", resume=False):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    device = get_device(requested_device, int(os.environ.get("LOCAL_RANK", "0")))
    if config.precision == "fp16" and device.type != "cuda":
        raise ValueError("fp16 training requires CUDA")
    if world_size > 1 and requested_device.startswith("cuda:"):
        raise ValueError("Use --device cuda under torchrun so each rank selects its local GPU")
    if world_size > 1:
        dist.init_process_group(backend="nccl" if device.type == "cuda" else "gloo")
    try:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        lock = FileLock(output / ".run.lock", timeout=0) if rank == 0 else nullcontext()
        with lock:
            if world_size > 1:
                dist.barrier()
            return _train(config, output, device, rank, world_size, resume)
    finally:
        if world_size > 1:
            dist.destroy_process_group()


def _train(config, output, device, rank, world_size, resume):
    checkpoint_path = output / "last.pt"
    if checkpoint_path.exists() != resume:
        raise ValueError(f"{output}: use --resume for an existing run, or select a fresh output directory")
    checkpoint = None
    if resume:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if checkpoint["config"] != asdict(config) or checkpoint["world_size"] != world_size:
            raise ValueError("Resume requires the same resolved config and world size")

    train_loader, val_loader, classes = make_loaders(config, device, rank, world_size)
    seed_epoch(config.seed, 0, 0)
    model = LeJEPAClassifier(config, classes)
    if config.sync_batchnorm and world_size > 1:
        if device.type != "cuda":
            raise ValueError("Synchronized BatchNorm requires CUDA")
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.to(device)
    optimizer = torch.optim.AdamW([
        {"params": list(model.ssl_parameters()), "weight_decay": config.weight_decay},
        {"params": model.probe.parameters(), "weight_decay": config.probe_weight_decay},
    ])
    scaler = torch.amp.GradScaler(device.type, enabled=config.precision == "fp16")
    start_epoch, history = 0, []
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch, history = checkpoint["epoch"], checkpoint["history"]
        del checkpoint
    else:
        model.update_teacher()

    wrapped = model
    if world_size > 1:
        wrapped = DistributedDataParallel(model, device_ids=[device.index] if device.type == "cuda" else None)
    steps = len(train_loader)
    warmup_steps = min(config.warmup_epochs * steps, config.epochs * steps)
    metadata = {
        "config": asdict(config), "world_size": world_size, "num_classes": classes,
        "steps_per_epoch": steps, "torch": str(torch.__version__), "timm": timm.__version__,
        "torchvision": torchvision.__version__, "python": platform.python_version(),
    }
    if rank == 0:
        if not resume:
            atomic_write(output / "run.json", lambda path: path.write_text(json.dumps(metadata, indent=2) + "\n"))
        elif history:
            _write_history(output, history)

    for epoch in range(start_epoch, config.epochs):
        seed_epoch(config.seed, epoch + 1, rank)
        train_loader.sampler.set_epoch(epoch)
        wrapped.train()
        totals = torch.zeros(4, dtype=torch.float64, device=device)
        for index, (views, labels) in enumerate(train_loader):
            step = epoch * steps + index
            lrs = config.lr, config.probe_lr
            for group, lr in zip(optimizer.param_groups, lrs):
                floor = config.min_lr if config.min_lr is not None else lr * config.min_lr_ratio
                group["lr"] = warmup_cosine_lr(
                    step, lr, floor, warmup_steps, config.epochs * steps, start_factor=0.01,
                )
            views = tuple(view.to(device, non_blocking=True) for view in views)
            labels = labels.to(device, non_blocking=True)
            with autocast(device, config.precision):
                ssl, logits = wrapped(views, step)
                probe_loss = F.cross_entropy(logits.flatten(0, 1), labels.repeat(logits.size(0)))
                loss = ssl.loss + probe_loss
            scaler.scale(loss).backward()
            optimizer_step(optimizer, scaler=scaler, parameters=model.parameters(), grad_clip=config.grad_clip)
            totals += torch.stack((ssl.loss.detach(), ssl.prediction.detach(), ssl.sigreg.detach(), probe_loss.detach()))
        model.update_teacher()

        keys = "lejepa_loss", "prediction_loss", "sigreg", "probe_loss"
        record = {"epoch": epoch + 1, "steps": (epoch + 1) * steps,
                  **dict(zip(keys, (_reduce(totals) / (steps * world_size)).tolist()))}
        if (epoch + 1) % config.eval_every == 0 or epoch + 1 == config.epochs:
            record.update(evaluate(model, val_loader, device, config.precision))
        history.append(record)
        if rank == 0:
            state = {**metadata, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                     "scaler": scaler.state_dict(), "epoch": epoch + 1, "history": history}
            atomic_write(checkpoint_path, lambda path: torch.save(state, path))
            _write_history(output, history)
            print(json.dumps(record), flush=True)
    return history


def _write_history(output, history):
    records = "".join(json.dumps(record) + "\n" for record in history)
    atomic_write(output / "metrics.jsonl", lambda path: path.write_text(records))
