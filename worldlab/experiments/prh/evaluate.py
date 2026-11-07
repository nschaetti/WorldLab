"""Cache unimodal embeddings and score alignment on held-out scene IDs."""

import json
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from worldlab.representation import paired_alignment, prh_score, scene_explained_fraction
from worldlab.training.runtime import atomic_write, get_device

from .config import RunConfig, build_model, load_tokenizer
from .data import SceneViews, load_scenes


def _load_model(checkpoint, device):
    path = Path(checkpoint)
    if path.is_dir():
        path = path / "last.pt"
    state = torch.load(path, weights_only=True, map_location="cpu")
    config = RunConfig(**state["run"]["config"])
    tokenizer = load_tokenizer(config)
    model = build_model(config, tokenizer)
    model.load_state_dict(state["model"], strict=True)
    return model.to(device).eval(), config, tokenizer


def _layers(encoder, values, modality):
    if modality == "text":
        hidden = encoder.hidden(values, all_layers=True).hidden_states
        return [encoder.mean_pool(layer, values) for layer in hidden]
    layers = encoder.forward_intermediates(
        values, return_prefix_tokens=True, output_fmt="NLC", intermediates_only=True,
    )
    return [prefix[:, 0] for _, prefix in layers]


@torch.inference_mode()
def embed(args):
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    device = get_device(args.device)
    if args.checkpoint:
        model, cfg, tokenizer = _load_model(args.checkpoint, device)
        encoder = model.encoder
        projector = getattr(model, "projector", None)
    else:
        import timm

        cfg = RunConfig(modality="image", resolution=args.resolution)
        tokenizer, projector = None, None
        encoder = timm.create_model(args.vision_backbone, pretrained=True, num_classes=0,
                                    img_size=args.resolution).to(device).eval()
    rows, root = load_scenes(args.data, args.root)
    allowed = set(args.split or ["val", "test"])
    rows = [row for row in rows if row.get("split", "test") in allowed]
    if not rows:
        raise ValueError("No scenes match the selected splits")
    values = SceneViews(rows, root, cfg, tokenizer)
    cached, projected, layer_values = [], [], []
    ids, splits = [], []
    for inputs, batch_ids, batch_splits in DataLoader(values, batch_size=args.batch_size):
        inputs = inputs.to(device)
        features = encoder(inputs)
        cached.append(features.float().cpu())
        if projector is not None:
            projected.append(projector(features).float().cpu())
        if args.layers:
            outputs = _layers(encoder, inputs, cfg.modality)
            if not layer_values:
                layer_values = [[] for _ in outputs]
            for target, layer in zip(layer_values, outputs):
                target.append(layer.float().cpu())
        ids.extend(batch_ids)
        splits.extend(batch_splits)
    result = {"ids": ids, "splits": splits, "embedding": torch.cat(cached), "config": asdict(cfg)}
    result["projection"] = torch.cat(projected) if projected else result["embedding"]
    if layer_values:
        result["layers"] = [torch.cat(values) for values in layer_values]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(output, lambda path: torch.save(result, path))
    return result


def _cache(path):
    cache = torch.load(path, weights_only=True, map_location="cpu")
    ids, splits = cache["ids"], cache["splits"]
    if len(set(ids)) != len(ids) or len(ids) != len(splits) or len(ids) != len(cache["embedding"]):
        raise ValueError(f"Invalid IDs, splits or embedding rows in {path}")
    return cache


def _join(a, b, split):
    left = {key: i for i, (key, label) in enumerate(zip(a["ids"], a["splits"])) if label == split}
    right = {key: i for i, (key, label) in enumerate(zip(b["ids"], b["splits"])) if label == split}
    if left.keys() != right.keys() or not left:
        raise ValueError(f"The two caches must contain the same nonempty {split!r} scene set")
    keys = sorted(left)
    return [left[key] for key in keys], [right[key] for key in keys], set(keys)


def _json(path, result):
    if path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write(path, lambda temporary: temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n"))
    print(json.dumps(result, allow_nan=False), flush=True)


def align(args):
    a, b = _cache(args.source), _cache(args.target)
    ia, ib, fit_ids = _join(a, b, args.fit_split)
    ta, tb = (_cache(args.test_source), _cache(args.test_target)) if args.test_source else (a, b)
    ja, jb, test_ids = _join(ta, tb, args.test_split)
    same_caches = not args.test_source or (
        Path(args.source).resolve() == Path(args.test_source).resolve()
        and Path(args.target).resolve() == Path(args.test_target).resolve()
    )
    if same_caches and fit_ids & test_ids:
        raise ValueError("Fit and test scenes must be disjoint")
    device = get_device(args.device)
    key = args.features
    tensors = [cache[key][rows].to(device) for cache, rows in ((a, ia), (b, ib), (ta, ja), (tb, jb))]
    scores, rotation = paired_alignment(*tensors, k=args.k, test_means=bool(args.test_source))
    if args.rotation:
        path = Path(args.rotation)
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write(path, lambda temporary: torch.save(
            {"matrix": rotation.matrix.cpu(), "source_mean": rotation.source_mean.cpu(),
             "target_mean": rotation.target_mean.cpu()}, temporary,
        ))
    _json(args.output, scores)
    return scores


def benchmark(args):
    a, b = _cache(args.source), _cache(args.target)
    ia, ib, _ = _join(a, b, args.split)
    if "layers" not in a or "layers" not in b:
        raise ValueError("Both embedding caches need to be generated with --layers")
    device = get_device(args.device)
    score, layer_a, layer_b = prh_score(
        [layer[ia].to(device) for layer in a["layers"]],
        [layer[ib].to(device) for layer in b["layers"]],
        k=args.k, metric=args.metric,
    )
    result = {"metric": args.metric, "score": score, "layer_a": layer_a, "layer_b": layer_b, "pairs": len(ia)}
    _json(args.output, result)
    return result


@torch.inference_mode()
def nuisance(args):
    if args.batch_size < 1 or args.views < 2:
        raise ValueError("Expected positive batch size and at least two captions per scene")
    device = get_device(args.device)
    model, cfg, tokenizer = _load_model(args.checkpoint, device)
    if cfg.modality != "text":
        raise ValueError("The nuisance protocol expects a text checkpoint")
    rows, _ = load_scenes(args.data, args.root)
    captions = [row["captions"][:args.views] for row in rows if row.get("split") == args.split]
    if len(captions) < 2 or any(len(parts) != args.views for parts in captions):
        raise ValueError("Every selected scene needs the requested number of captions")
    texts = [caption for group in captions for caption in group]
    outputs = []
    for start in range(0, len(texts), args.batch_size):
        tokens = tokenizer(texts[start:start + args.batch_size], padding="max_length", truncation=True,
                           max_length=cfg.max_len, return_tensors="pt").input_ids
        outputs.append(model.encoder(tokens.to(device)).float().cpu())
    features = torch.cat(outputs).reshape(len(captions), args.views, -1)
    result = {"scenes": len(captions), "views": args.views,
              "scene_explained": {str(k): scene_explained_fraction(features, k) for k in args.dimensions}}
    _json(args.output, result)
    return result
