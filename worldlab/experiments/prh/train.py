"""Train independent unimodal arms using WorldLab's optimization helpers."""

import json
import os
from dataclasses import asdict
from pathlib import Path

import torch
from filelock import FileLock
from torch.utils.data import DataLoader

from worldlab.training import optimizer_step, warmup_cosine_lr
from worldlab.training.runtime import atomic_write, autocast, get_device, seed_epoch

from .config import build_model, load_tokenizer
from .data import SceneViews, load_scenes


def train(config, manifest, output, *, root=None, device_name="auto", resume=False):
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("The native PRH runner uses one device. Use the upstream registry for distributed paper runs.")
    device = get_device(device_name)
    if config.precision == "fp16" and device.type != "cuda":
        raise ValueError("fp16 training requires CUDA")
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    with FileLock(destination / ".run.lock", timeout=0):
        return _train(config, manifest, destination, root, device, resume)


def _train(config, manifest, output, root, device, resume):
    checkpoint = output / "last.pt"
    if checkpoint.exists() != resume:
        raise ValueError("Use --resume for an existing checkpoint or select a fresh output directory")
    rows, data_root = load_scenes(manifest, root)
    rows = [row for row in rows if row.get("split") == "train"]
    tokenizer = load_tokenizer(config)
    dataset = SceneViews(rows, data_root, config, tokenizer, training=True)
    if len(dataset) < config.batch_size:
        raise ValueError("No full batch in the training split")
    rng = torch.Generator()
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, drop_last=True,
                        generator=rng, num_workers=config.workers, pin_memory=device.type == "cuda")
    seed_epoch(config.seed, 0, 0)
    model = build_model(config, tokenizer).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scaler = torch.amp.GradScaler(device.type, enabled=config.precision == "fp16")
    run = {"config": asdict(config), "manifest": str(Path(manifest).resolve()), "root": str(data_root.resolve()),
           "scenes": len(rows), "torch": str(torch.__version__)}
    initial_epoch, history = 0, []
    if resume:
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved["run"] != run:
            raise ValueError("Resume requires the same dataset, configuration and Torch version")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        initial_epoch, history = saved["epoch"], saved["history"]
        del saved
    else:
        atomic_write(output / "run.json", lambda path: path.write_text(json.dumps(run, indent=2) + "\n"))

    is_jepa = config.objective == "lejepa"
    total_steps = config.epochs * len(loader)
    warmup = min(config.warmup_epochs * len(loader), total_steps)
    for epoch in range(initial_epoch, config.epochs):
        seed_epoch(config.seed, epoch + 1, 0)
        rng.manual_seed(config.seed + epoch)
        model.train()
        totals = torch.zeros(3, device=device, dtype=torch.float64)
        for index, batch in enumerate(loader):
            step = epoch * len(loader) + index
            lr = warmup_cosine_lr(step, config.lr, config.lr / 1000, warmup, total_steps, start_factor=0.01)
            for group in optimizer.param_groups:
                group["lr"] = lr
            with autocast(device, config.precision):
                if is_jepa:
                    views = [view.to(device, non_blocking=True) for view in batch]
                    result = model(views, config.global_views, step=step)
                    loss = result.loss
                    totals[1] += result.prediction.detach()
                    totals[2] += result.sigreg.detach()
                else:
                    loss = model(batch.to(device, non_blocking=True))
            scaler.scale(loss).backward()
            optimizer_step(optimizer, scaler=scaler, parameters=model.parameters(), grad_clip=config.grad_clip)
            totals[0] += loss.detach()
        loss, prediction, regularizer = (totals / len(loader)).tolist()
        record = {"epoch": epoch + 1, "step": (epoch + 1) * len(loader), "lr": lr, "loss": loss}
        if is_jepa:
            record.update(prediction=prediction, sigreg=regularizer)
        history.append(record)
        state = {"run": run, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "scaler": scaler.state_dict(), "epoch": epoch + 1, "history": history}
        atomic_write(checkpoint, lambda path: torch.save(state, path))
        _write_history(output, history)
        print(json.dumps(record), flush=True)
    if initial_epoch == config.epochs:
        _write_history(output, history)
    return history


def _write_history(output, history):
    atomic_write(output / "metrics.jsonl", lambda path: path.write_text(
        "".join(json.dumps(row) + "\n" for row in history),
    ))
