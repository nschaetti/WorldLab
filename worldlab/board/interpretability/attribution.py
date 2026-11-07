"""
Attribution helpers derived from intervention outcomes (B2).
"""

from typing import Dict, Iterable, List, Mapping, Sequence

import torch


def compute_intervention_attribution(
    results: Iterable[Dict[str, object]],
    value_key: str = "target_prob_gain",
    reduce: str = "mean",
) -> torch.Tensor:
    """
    Aggregate per-example intervention scores into a per-cell attribution vector [64].
    """
    sums = torch.zeros(64, dtype=torch.float32)
    counts = torch.zeros(64, dtype=torch.float32)

    for row in results:
        cell_obj = row.get("target_cell")
        if not isinstance(cell_obj, (int, float)):
            continue
        cell = int(cell_obj)
        if cell < 0 or cell >= 64:
            continue
        value_obj = row.get(value_key, 0.0)
        if not isinstance(value_obj, (int, float)):
            continue
        value = float(value_obj)
        sums[cell] += value
        counts[cell] += 1

    if reduce == "sum":
        return sums
    if reduce == "mean":
        mask = counts > 0
        out = torch.zeros_like(sums)
        out[mask] = sums[mask] / counts[mask]
        return out
    raise ValueError("reduce must be 'mean' or 'sum'")


def compute_layer_cell_attribution_map(
    layer_results: Mapping[int, Iterable[Dict[str, object]]],
    value_key: str = "target_prob_gain",
    reduce: str = "mean",
) -> torch.Tensor:
    """
    Build attribution map [num_layers, 64] from per-layer intervention results.
    """
    layers_sorted = sorted(layer_results.keys())
    rows: List[torch.Tensor] = []
    for layer in layers_sorted:
        rows.append(compute_intervention_attribution(layer_results[layer], value_key=value_key, reduce=reduce))
    if not rows:
        raise ValueError("No layers provided for attribution map")
    return torch.stack(rows, dim=0)


def rank_top_contributors(
    attribution_map: torch.Tensor,
    layers: Sequence[int],
    top_k: int = 10,
    absolute: bool = True,
) -> List[Dict[str, float]]:
    """
    Return top-k (layer, cell) contributions from attribution map [L,64].
    """
    if attribution_map.dim() != 2 or attribution_map.size(1) != 64:
        raise ValueError(f"Expected attribution map shape [L,64], got {tuple(attribution_map.shape)}")
    if len(layers) != attribution_map.size(0):
        raise ValueError("layers length must match attribution_map first dimension")

    flat = attribution_map.abs() if absolute else attribution_map
    k = min(top_k, flat.numel())
    values, indices = torch.topk(flat.reshape(-1), k=k)

    out: List[Dict[str, float]] = []
    for v, idx in zip(values.tolist(), indices.tolist()):
        layer_i = idx // 64
        cell_i = idx % 64
        raw_score = float(attribution_map[layer_i, cell_i].item())
        out.append(
            {
                "layer": float(layers[layer_i]),
                "cell": float(cell_i),
                "score": raw_score,
                "abs_score": float(v),
            }
        )
    return out


def compute_intervention_attribution_summary(
    layer_results: Mapping[int, Iterable[Dict[str, object]]],
    value_key: str = "target_prob_gain",
    top_k: int = 10,
) -> Dict[str, object]:
    layers = sorted(layer_results.keys())
    amap = compute_layer_cell_attribution_map(layer_results=layer_results, value_key=value_key)
    top = rank_top_contributors(attribution_map=amap, layers=layers, top_k=top_k, absolute=True)
    return {
        "layers": layers,
        "value_key": value_key,
        "attribution_map": amap,
        "top_contributors": top,
    }
