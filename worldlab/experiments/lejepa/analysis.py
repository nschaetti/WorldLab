"""Analyze loss rankings, SIGReg behavior, and emergent ViT spatial features."""

import csv
import json
import time
from pathlib import Path

import torch
from PIL import Image
from scipy.stats import spearmanr

from worldlab.nn import SIGReg
from worldlab.training import optimizer_step

from .data import evaluation_transform
from .probe import load_encoder
from .train import get_device
from .vision import VisionTransformer, set_unfused_attention


def write_table(path, rows):
    if not rows:
        raise ValueError("No results to write")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(key for row in rows for key in row)))
        writer.writeheader()
        writer.writerows(rows)


def summarize(runs, output):
    rows = []
    for path in sorted(Path(runs).rglob("run.json")):
        metrics = path.with_name("metrics.jsonl")
        if not metrics.exists():
            continue
        run = json.loads(path.read_text())
        records = [json.loads(line) for line in metrics.read_text().splitlines()]
        evaluated = [row for row in records if "top1" in row]
        if not evaluated:
            continue
        row = evaluated[-1]
        rows.append({"run": str(path.parent.relative_to(runs)), **run["config"], **row,
                     "complete": row["epoch"] == run["config"]["epochs"]})
    write_table(output, rows)
    return rows


def correlations(runs, output):
    rows = summarize(runs, Path(output).with_suffix(".runs.csv"))
    groups = {}
    for row in rows:
        if row["complete"] and row["weight"] > 0:
            groups.setdefault((row["dataset"], row["backbone"]), []).append(row)
    results = []
    for (dataset, backbone), runs in groups.items():
        if len(runs) < 3:
            continue
        accuracy = [run["top1"] for run in runs]
        for index in range(-30, 31):
            alpha = index / 10
            scaled_loss = [run["lejepa_loss"] / run["weight"]**alpha for run in runs]
            results.append({"dataset": dataset, "backbone": backbone, "alpha": alpha,
                            "spearman": float(spearmanr([-loss for loss in scaled_loss], accuracy).statistic),
                            "runs": len(runs)})
    write_table(output, results)


def sigreg_timing(output, device_name, dimension, repeats):
    if dimension < 1 or repeats < 1:
        raise ValueError("dimension and repeats must be positive")
    device = get_device(device_name)
    rows = []
    sizes = [(512, 512, 16), (512, 512, 64), (512, 512, 256), (2048, 512, 16),
             (8192, 512, 16), (8192, 8192, 16), (32768, 512, 16), (512, 2048, 16), (512, 8192, 16)]
    for batch, slices, knots in sizes:
        inputs = torch.randn(batch, dimension, device=device, requires_grad=True)
        regularizer = SIGReg(slices, knots, symmetric_grid=True).to(device)
        times = []
        for step in range(repeats + 2):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            regularizer(inputs, step=step).backward()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            if step >= 2:
                times.append(1000 * (time.perf_counter() - started))
            inputs.grad = None
        times = torch.tensor(times)
        rows.append({"batch": batch, "slices": slices, "knots": knots, "dimension": dimension,
                     "device": str(device), "mean_ms": times.mean().item(),
                     "std_ms": times.std(unbiased=False).item()})
    write_table(output, rows)


def _univariate_statistic(samples, test):
    if test in {"variance", "jarque_bera"}:
        mean = samples.mean(dim=0)
        centered = samples - mean
        variance = centered.square().mean(dim=0)
        statistic = mean.square() + 0.5 * (variance - 1).square()
        if test == "jarque_bera":
            standardized = centered * variance.clamp_min(1e-8).rsqrt()
            skewness = standardized.pow(3).mean(dim=0)
            excess = standardized.pow(4).mean(dim=0) - 3
            statistic = statistic + samples.size(0) / 6 * (skewness.square() + excess.square() / 4)
        return statistic.mean()

    order = samples.sort(dim=0).values
    cdf = torch.special.ndtr(order)
    n = samples.size(0)
    rank = torch.arange(1, n + 1, device=samples.device, dtype=samples.dtype).unsqueeze(1)
    if test == "anderson_darling":
        log_cdf = torch.special.log_ndtr(order)
        log_survival = torch.special.log_ndtr(-order.flip(0))
        return (-n - ((2 * rank - 1) * (log_cdf + log_survival)).mean(dim=0)).mean()
    statistic = (cdf - (2 * rank - 1) / (2 * n)).square().sum(dim=0) + 1 / (12 * n)
    if test == "watson":
        statistic = statistic - n * (cdf.mean(dim=0) - 0.5).square()
    return statistic.mean()


def synthetic_sigreg(output, steps, device_name, seed):
    """Optimize X-shaped samples with alternative statistics and slicing policies."""
    if steps < 1:
        raise ValueError("steps must be positive")
    device = get_device(device_name)
    generator = torch.Generator(device=device).manual_seed(seed)
    initial = torch.randn(100, 1024, device=device, generator=generator)
    initial[:, 1] = torch.sign(initial[:, 1]) * initial[:, 0]
    validator = SIGReg(4096).to(device)
    comparisons = (
        (dimension, slices, test, True)
        for dimension in (128, 1024)
        for slices in (10, 100)
        for test in ("variance", "jarque_bera", "cramer_von_mises", "watson", "anderson_darling", "epps")
    )
    resampling = ((1024, slices, "epps", resample)
                  for slices in (16, 64, 256, 1024) for resample in (False, True))
    rows = []
    for dimension, slices, test, resample in (*comparisons, *resampling):
        start = initial[:, :dimension].contiguous()
        samples = torch.nn.Parameter(start.clone())
        optimizer = torch.optim.Adam([samples], lr=0.03)
        criterion = SIGReg(slices, seed=seed).to(device)
        for step in range(steps + 1):
            if step % max(1, steps // 20) == 0 or step == steps:
                with torch.no_grad():
                    value = validator(samples, step=1_000_000)
                rows.append({"dimension": dimension, "slices": slices, "test": test, "resample": resample,
                             "step": step, "heldout_sigreg": value.item()})
            if step == steps:
                break
            directions = criterion.directions(samples, step if resample else 0)
            if test == "epps":
                loss = criterion(samples, directions=directions)
            else:
                loss = _univariate_statistic(samples @ directions, test)
            loss.backward()
            optimizer_step(optimizer)
        label = f"{dimension}.{slices}.{test}.{'random' if resample else 'fixed'}"
        path = Path(output).with_suffix(f".{label}.pt")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"initial": start[:, :4].cpu(), "final": samples.detach()[:, :4].cpu()}, path)
    write_table(output, rows)


@torch.inference_mode()
def visualize(checkpoint, images, output, device_name, mass):
    """Save per-image patch PCA and last-layer CLS attention maps (Figures 13–14)."""
    if not 0 < mass <= 1:
        raise ValueError("Attention mass must be in (0, 1]")
    device = get_device(device_name)
    encoder, config = load_encoder(checkpoint)
    encoder.to(device).eval()
    backbone = encoder.backbone
    if not isinstance(backbone, VisionTransformer) or backbone.cls_token is None:
        raise ValueError("Spatial visualization requires a ViT with a CLS token")
    attention = []
    hook = set_unfused_attention(backbone).register_forward_hook(
        lambda module, args, result: attention.append(result.detach()),
    )
    transform = evaluation_transform(config.resolution, config.augment)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    try:
        for index, path in enumerate(images):
            with Image.open(path) as image:
                inputs = transform(image.convert("RGB")).unsqueeze(0).to(device)
            tokens = backbone.forward_features(inputs)[0, backbone.num_prefix_tokens:].float()
            centered = tokens - tokens.mean(0)
            _, _, basis = torch.linalg.svd(centered, full_matrices=False)
            rgb = centered @ basis[:3].T
            rgb = (rgb - rgb.amin(0)) / (rgb.amax(0) - rgb.amin(0)).clamp_min(1e-8)
            weights = attention.pop()[0, :, 0, backbone.num_prefix_tokens:].mean(0)
            weights = weights / weights.sum()
            values, indices = weights.sort(descending=True)
            mask = torch.zeros_like(weights)
            mask[indices] = (values.cumsum(0) - values < mass).float()
            grid = backbone.patch_embed.dynamic_feat_size(inputs.shape[-2:])
            basename = f"{index:04d}-{Path(path).stem}"
            for name, values in {"pca": rgb, "attention": weights / weights.max(), "mask": mask}.items():
                array = (values.reshape(*grid, *values.shape[1:]) * 255).byte().cpu().numpy()
                rendered = Image.fromarray(array).resize(inputs.shape[-2:][::-1], Image.Resampling.NEAREST)
                rendered.save(output / f"{basename}-{name}.png")
    finally:
        hook.remove()


def summarize_probes(runs, output):
    rows = []
    for path in sorted(Path(runs).rglob("probe.json")):
        settings = json.loads(path.read_text())
        history = settings.pop("history")
        rows.append({"run": str(path.parent.relative_to(runs)), **settings, **history[-1]})
    write_table(output, rows)


def quadrature(output, seed, num_samples=1000):
    """Compare the quadrature to the analytical Gaussian-kernel integral (Figure 20)."""
    import math

    from worldlab.nn import EppsPulley

    generator = torch.Generator().manual_seed(seed)
    normal = torch.randn(num_samples, generator=generator, dtype=torch.float64)
    mixture = normal * 0.5 + 2 * torch.where(torch.rand(num_samples, generator=generator) < 0.5, -1, 1)
    rows = []
    for name, samples in (("normal", normal), ("mixture", mixture)):
        pairwise = samples[:, None] - samples
        exact = num_samples * (
            math.sqrt(2 * math.pi) * torch.exp(-pairwise.square() / 2).mean()
            - 2 * math.sqrt(math.pi) * torch.exp(-samples.square() / 4).mean()
            + math.sqrt(2 * math.pi / 3)
        )
        for points in range(3, 102, 2):
            for half_grid in (False, True):
                estimate = EppsPulley(points, 5.0, symmetric_grid=not half_grid)(samples[:, None]).item()
                rows.append({"distribution": name, "knots": points, "half_grid": half_grid,
                             "exact": exact.item(), "estimate": estimate,
                             "absolute_error": abs(estimate - exact.item())})
    write_table(output, rows)
