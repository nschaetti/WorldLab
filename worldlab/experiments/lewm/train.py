"""Train LeWM on offline trajectories with an epoch-boundary checkpoint."""

import json
import os
from dataclasses import asdict
from pathlib import Path

import torch
from filelock import FileLock
from torch.utils.data import DataLoader

from worldlab.training import optimizer_step, warmup_cosine_lr
from worldlab.training.runtime import atomic_write, autocast, get_device, seed_epoch

from .config import build_model
from .data import Trajectories, split_dataset
from .decoder import PatchDecoder


def train(config, dataset_path, output, device_name="auto", resume=False):
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("The LeWM experiment runner uses one device per run")
    device = get_device(device_name)
    if device.type != "cuda" and config.precision == "fp16":
        raise ValueError("fp16 training requires CUDA")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with FileLock(output / ".run.lock", timeout=0):
        return _train(config, dataset_path, output, device, resume)


def _train(config, dataset_path, output, device, resume):
    path = output / "last.pt"
    if path.exists() != resume:
        raise ValueError("Use --resume for an existing run, or choose a fresh output directory")
    dataset_path = str(Path(dataset_path).expanduser().resolve())
    dataset = Trajectories(dataset_path, config.resolution, config.history, config.frame_skip)
    training, validation = split_dataset(dataset, config.val_fraction, config.seed, config.split)
    seed_epoch(config.seed, 0, 0)
    model = build_model(config, dataset.action_dim).to(device)
    decoder = None
    params = list(model.parameters())
    if config.reconstruction_weight:
        decoder = PatchDecoder(
            config.latent_dim, config.resolution, patch_size=16 if config.resolution % 16 == 0 else config.patch_size,
        ).to(device)
        params.extend(decoder.parameters())
    optimizer = torch.optim.AdamW(params, lr=config.lr, weight_decay=config.weight_decay)
    scaler = torch.amp.GradScaler(device.type, enabled=config.precision == "fp16")
    start, history = 0, []
    metadata = {"config": asdict(config), "data": dataset_path, "action_dim": dataset.action_dim,
                "windows": len(dataset), "torch": str(torch.__version__),
                "action_stats": {"mean": dataset.mean.tolist(), "std": dataset.std.tolist()}}
    if resume:
        state = torch.load(path, map_location="cpu", weights_only=True)
        if any(state[key] != metadata[key] for key in ("config", "data", "windows", "action_dim", "action_stats")):
            raise ValueError("Resume requires the same configuration and dataset")
        model.load_state_dict(state["model"])
        if decoder is not None:
            decoder.load_state_dict(state["decoder"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        start, history = state["epoch"], state["history"]
        del state
    else:
        atomic_write(output / "run.json", lambda file: file.write_text(json.dumps(metadata, indent=2) + "\n"))

    shuffle = torch.Generator()
    options = dict(batch_size=config.batch_size, num_workers=config.workers, pin_memory=device.type == "cuda")
    train_loader = DataLoader(training, shuffle=True, drop_last=True, generator=shuffle, **options)
    val_loader = DataLoader(validation, **options)
    if not len(train_loader):
        raise ValueError("No complete training batch. Reduce batch_size or supply a larger dataset.")
    total_steps = config.epochs * len(train_loader)
    warmup = round(config.warmup_fraction * total_steps)
    for epoch in range(start, config.epochs):
        seed_epoch(config.seed, epoch + 1, 0)
        shuffle.manual_seed(config.seed + epoch)
        record = {"epoch": epoch + 1}
        for phase, loader in (("train", train_loader), ("val", val_loader)):
            training_phase = phase == "train"
            model.train(training_phase)
            if decoder is not None:
                decoder.train(training_phase)
            totals = torch.zeros(4, device=device, dtype=torch.float64)
            count = 0
            for index, batch in enumerate(loader):
                if training_phase:
                    step = epoch * len(train_loader) + index
                    lr = warmup_cosine_lr(step, config.lr, 0.0, warmup, total_steps, start_factor=0.0)
                    for group in optimizer.param_groups:
                        group["lr"] = lr
                    record["lr"] = lr
                pixels = batch["pixels"].to(device, non_blocking=True)
                actions = batch["action"].to(device, non_blocking=True)
                with torch.set_grad_enabled(training_phase), autocast(device, config.precision):
                    step = epoch * len(train_loader) + index if training_phase else -1 - index
                    result = model(pixels, actions, step=step)
                    reconstruction = result.loss.new_zeros(())
                    if decoder is not None:
                        reconstruction = (decoder(result.embeddings).float() - pixels).square().mean()
                    loss = result.loss + config.reconstruction_weight * reconstruction
                if training_phase:
                    scaler.scale(loss).backward()
                    optimizer_step(optimizer, scaler=scaler, parameters=params, grad_clip=config.grad_clip)
                totals += torch.stack((loss.detach(), result.prediction.detach(), result.sigreg.detach(),
                                       reconstruction.detach())) * len(pixels)
                count += len(pixels)
            for name, value in zip(("loss", "prediction", "sigreg", "reconstruction"), (totals / count).tolist()):
                record[f"{phase}_{name}"] = value
        history.append(record)
        state = {**metadata, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "scaler": scaler.state_dict(), "epoch": epoch + 1, "history": history}
        if decoder is not None:
            state["decoder"] = decoder.state_dict()
        atomic_write(path, lambda file: torch.save(state, file))
        _write_metrics(output, history)
        print(json.dumps(record), flush=True)

    if start == config.epochs:
        _write_metrics(output, history)
    return history



def _write_metrics(output, history):
    metrics = "".join(json.dumps(record) + "\n" for record in history)
    atomic_write(output / "metrics.jsonl", lambda file: file.write_text(metrics))
