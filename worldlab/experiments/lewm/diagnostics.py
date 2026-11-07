"""Frozen physical probes, open-loop rollouts and violation-of-expectation."""

import csv
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset

from worldlab.training import optimizer_step
from worldlab.training.runtime import atomic_write, get_device, seed_epoch

from .config import load_checkpoint
from .data import IMAGE_MEAN, IMAGE_STD, Trajectories, split_dataset
from .decoder import PatchDecoder


def _output(args):
    path = Path(args.output)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _json(path, data):
    atomic_write(path, lambda file: file.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n"))


def _csv(path, records):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)


def _targets(specs):
    targets = {}
    for spec in specs:
        name, separator, source = spec.partition("=")
        column, _, indices = source.partition(":")
        if not separator or not name or not column or name in targets:
            raise ValueError("Probe targets must have unique names in NAME=COLUMN[:I,J,...]")
        targets[name] = column, [int(i) for i in indices.split(",")] if indices else None
    return targets


@torch.no_grad()
def _extract(model, subset, targets, batch_size, device):
    features = []
    labels = {name: [] for name in targets}
    for batch in DataLoader(subset, batch_size=batch_size):
        features.append(model.encode(batch["pixels"][:, :1].to(device))[:, 0].cpu())
        for name, (column, indices) in targets.items():
            values = batch[column][:, 0].float().reshape(len(batch["pixels"]), -1)
            labels[name].append(values if indices is None else values[:, indices])
    return torch.cat(features), {name: torch.cat(values) for name, values in labels.items()}


def _probe_scores(predicted, targets):
    squared = (predicted - targets).square().mean(dim=1)
    x = predicted - predicted.mean(0)
    y = targets - targets.mean(0)
    norm = x.norm(dim=0) * y.norm(dim=0)
    defined = norm > 0
    correlations = (x * y).sum(0)[defined] / norm[defined]
    return {"mse": squared.mean().item(), "mse_std": squared.std(unbiased=False).item(),
            "pearson_r": correlations.mean().item() if defined.any() else None,
            "samples": len(targets)}


def physical_probe(args):
    if min(args.epochs, args.batch_size) < 1:
        raise ValueError("Probe epochs and batch size must be positive")
    output = _output(args)
    device = get_device(args.device)
    seed_epoch(args.seed, 0, 0)
    model, size, action_stats = load_checkpoint(args.checkpoint, device)
    targets = _targets(args.target)
    dataset = Trajectories(args.data, size, 1, args.frame_skip,
                           sorted({column for column, _ in targets.values()}), args.stride, action_stats=action_stats)
    train, val = split_dataset(dataset, 0.1, args.seed, "episodes")
    x, y = _extract(model, train, targets, args.batch_size, device)
    vx, vy = _extract(model, val, targets, args.batch_size, device)
    records = []
    for name in targets:
        train_mask = y[name].isfinite().all(1)
        val_mask = vy[name].isfinite().all(1)
        if train_mask.sum() < 2 or not val_mask.any():
            raise ValueError(f"Target {name} has too few finite values")
        train_x, train_y = x[train_mask], y[name][train_mask]
        val_x, val_y = vx[val_mask], vy[name][val_mask]
        mean, std = train_y.mean(0), train_y.std(0, unbiased=False)
        std = torch.where(std > 0, std, torch.ones_like(std))
        train_y, val_y = (train_y - mean) / std, (val_y - mean) / std

        design = F.pad(train_x.double(), (0, 1), value=1)
        coefficients = torch.linalg.lstsq(design, train_y.double(), driver="gelsd").solution
        predicted = (F.pad(val_x.double(), (0, 1), value=1) @ coefficients).float()
        records.append({"target": name, "model": "linear", **_probe_scores(predicted, val_y)})

        mlp = nn.Sequential(nn.Linear(x.size(1), 256), nn.GELU(), nn.Linear(256, train_y.size(1))).to(device)
        optimizer = torch.optim.AdamW(mlp.parameters(), lr=0.001)
        loader = DataLoader(TensorDataset(train_x, train_y), batch_size=args.batch_size, shuffle=True)
        for _ in range(args.epochs):
            for features, target in loader:
                F.mse_loss(mlp(features.to(device)), target.to(device)).backward()
                optimizer_step(optimizer)
        with torch.no_grad():
            predicted = torch.cat([mlp(part.to(device)).cpu() for part in val_x.split(args.batch_size)])
        records.append({"target": name, "model": "mlp", **_probe_scores(predicted, val_y)})
    _csv(output / "probes.csv", records)
    _json(output / "probe.json", {"config": vars(args), "metrics": records})
    if args.tsne:
        from sklearn.manifold import TSNE

        count = min(args.tsne, len(vx))
        if count < 3:
            raise ValueError("t-SNE requires at least three validation examples")
        index = torch.randperm(len(vx), generator=torch.Generator().manual_seed(args.seed))[:count]
        coords = TSNE(n_components=2, perplexity=min(30, count - 1), random_state=args.seed).fit_transform(vx[index])
        np.savez(output / "tsne.npz", embedding=coords, **{name: values[index].numpy() for name, values in vy.items()})
    return records


@torch.no_grad()
def rollout_metrics(args):
    if min(args.horizon, args.batch_size, args.max_windows) < 1:
        raise ValueError("Horizon, batch size and number of windows must be positive")
    output = _output(args)
    device = get_device(args.device)
    model, size, action_stats = load_checkpoint(args.checkpoint, device)
    dataset = Trajectories(args.data, size, model.history + args.horizon - 1, args.frame_skip, stride=args.stride, action_stats=action_stats)
    indices = torch.linspace(0, len(dataset) - 1, min(args.max_windows, len(dataset))).long().tolist()
    errors, straightness, count = torch.zeros(args.horizon), 0.0, 0
    decoder = None
    if args.decoder:
        state = torch.load(args.decoder, map_location="cpu", weights_only=True)
        decoder = PatchDecoder(**state["decoder_config"]).to(device)
        decoder.load_state_dict(state["decoder"])
        decoder.eval()
    for pixels in DataLoader(Subset(dataset, indices), batch_size=args.batch_size):
        frames, actions = pixels["pixels"].to(device), pixels["action"].to(device)
        h = model.history
        latent = model.encode(frames)
        imagined = model.rollout(latent[:, :h], actions[:, None, h-1:-1], actions[:, :h-1])[:, 0]
        errors += (imagined[:, h:] - latent[:, h:]).square().mean(dim=-1).sum(dim=0).cpu()
        velocity = torch.diff(latent, dim=1)
        if velocity.size(1) > 1:
            straightness += F.cosine_similarity(velocity[:, 1:], velocity[:, :-1], dim=-1).mean(1).sum().item()
        if decoder is not None and count == 0:
            _rollout_strip(output / "rollout.png", frames[:1], decoder(imagined[:1]))
        count += len(frames)
    record = {"config": vars(args), "mse_by_horizon": (errors / count).tolist(), "windows": count,
              "straightness": straightness / count if model.history + args.horizon >= 3 else None}
    _json(output / "rollout.json", record)
    return record


def _rollout_strip(path, actual, imagined):
    joined = torch.cat((actual, imagined), dim=0)
    mean, std = (joined.new_tensor(stats)[None, None, :, None, None] for stats in (IMAGE_MEAN, IMAGE_STD))
    rgb = ((joined * std + mean).clamp(0, 1) * 255).byte().cpu()
    array = rgb.permute(0, 3, 1, 4, 2).reshape(2 * rgb.size(3), -1, 3).numpy()
    Image.fromarray(array).save(path)


def fit_decoder(args):
    if min(args.epochs, args.batch_size) < 1:
        raise ValueError("Decoder epochs and batch size must be positive")
    output = _output(args)
    device = get_device(args.device)
    seed_epoch(args.seed, 0, 0)
    model, resolution, action_stats = load_checkpoint(args.checkpoint, device)
    model.requires_grad_(False)
    dataset = Trajectories(args.data, resolution, 1, args.frame_skip, stride=args.stride, action_stats=action_stats)
    train, val = split_dataset(dataset, 0.1, args.seed, "episodes")
    config = {"latent_dim": model.predictor.pos_embedding.size(-1), "resolution": resolution,
              "patch_size": args.patch_size}
    decoder = PatchDecoder(**config).to(device)
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=0.001)
    loaders = [DataLoader(split, batch_size=args.batch_size, shuffle=(index == 0))
               for index, split in enumerate((train, val))]
    history = []
    for epoch in range(args.epochs):
        record = {"epoch": epoch + 1}
        for phase, loader in zip(("train", "val"), loaders):
            training = phase == "train"
            decoder.train(training)
            loss_sum = count = 0
            for batch in loader:
                frames = batch["pixels"][:, :1].to(device)
                with torch.no_grad():
                    features = model.encode(frames)
                with torch.set_grad_enabled(training):
                    loss = F.mse_loss(decoder(features), frames)
                if training:
                    loss.backward()
                    optimizer_step(optimizer)
                loss_sum += loss.item() * len(frames)
                count += len(frames)
            record[phase + "_mse"] = loss_sum / count
        history.append(record)
        print(json.dumps(record), flush=True)
    atomic_write(output / "decoder.pt", lambda path: torch.save(
        {"decoder_config": config, "decoder": decoder.state_dict()}, path,
    ))
    _json(output / "decoder.json", {"config": vars(args), "history": history})


@torch.no_grad()
def violation_of_expectation(args):
    from scipy.stats import ttest_rel

    if args.frame_skip < 1 or args.perturbation_step < args.frame_skip:
        raise ValueError("Perturbation must occur after at least one prediction interval")
    output = _output(args)
    device = get_device(args.device)
    model, size, action_stats = load_checkpoint(args.checkpoint, device)
    normalizer = Trajectories(args.stats_data or args.data, size, 1, args.frame_skip,
                              action_stats=None if args.stats_data else action_stats)
    rows, events, sizes = [], {}, {}
    for condition, path in (("reference", args.data), ("visual", args.visual), ("physical", args.physical)):
        events[condition] = {}
        with h5py.File(path, "r") as source:
            offsets, lengths = source["ep_offset"][:], source["ep_len"][:]
            sizes[condition] = lengths.tolist()
            for ep, (offset, length) in enumerate(zip(offsets, lengths)):
                phase = args.perturbation_step % args.frame_skip
                steps = np.arange(phase, length, args.frame_skip)
                if len(steps) < 2:
                    continue
                positions = offset + steps
                frames = torch.from_numpy(source["pixels"][positions])
                if frames.size(-1) == 3:
                    frames = frames.permute(0, 3, 1, 2)
                latent = torch.cat([model.encode(normalizer.transform(part.contiguous()).unsqueeze(0).to(device))
                                    for part in frames.split(32)], dim=1)
                raw = torch.from_numpy(source["action"][positions[0]:positions[-1]]).float()
                raw = raw.reshape(len(raw), -1)
                actions = torch.nan_to_num((raw - normalizer.mean) / normalizer.std, posinf=0, neginf=0)
                actions = actions.reshape(1, len(steps)-1, -1).to(device)
                for index, step in enumerate(steps[1:]):
                    start = max(0, index + 1 - model.history)
                    predicted = model.predict(latent[:, start:index+1], actions[:, start:index+1])[:, -1]
                    error = F.mse_loss(predicted, latent[:, index+1]).item()
                    rows.append({"condition": condition, "episode": ep, "step": int(step), "mse": error})
                    if step == args.perturbation_step:
                        events[condition][ep] = error
    if sizes["reference"] != sizes["visual"] or sizes["reference"] != sizes["physical"]:
        raise ValueError("Use matched episodes with the same lengths and ordering in all three conditions")
    comparisons = {}
    for condition in ("visual", "physical"):
        common = sorted(events["reference"].keys() & events[condition].keys())
        if not common:
            raise ValueError("No matched episodes contain the perturbation step")
        ref = [events["reference"][ep] for ep in common]
        alt = [events[condition][ep] for ep in common]
        p = ttest_rel(alt, ref).pvalue if len(common) > 1 else np.nan
        comparisons[condition] = {"episodes": len(common), "mean_delta_mse": float(np.mean(alt) - np.mean(ref)),
                                  "paired_p_value": float(p) if np.isfinite(p) else None}
    _csv(output / "surprise.csv", rows)
    _json(output / "surprise.json", {"config": vars(args), "comparisons": comparisons})
    return comparisons
