"""
Standardized intervention protocol runners (Sprint B1).
"""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Union

import torch
import torch.nn as nn

from boardGPT.nn import GPT

from .interventions import (
    InterventionConfig,
    InterventionTarget,
    run_probe_intervention,
)
from .metrics_intervention import (
    aggregate_by_layer,
    compute_delta_norm_stats,
    compute_off_target_impact,
    compute_success_rate,
    compute_target_gain_stats,
)
from .attribution import compute_intervention_attribution_summary


@dataclass
class ProtocolConfig:
    """
    Standard protocol parameters for B1 experiments.
    """

    max_examples: int = 32
    intervention_steps: int = 120
    intervention_lr: float = 5e-2
    intervention_reg_strength: float = 1e-3
    clamp_norm: Optional[float] = None
    device: Optional[Union[str, torch.device]] = None


def _extract_tokens(batch: Union[torch.Tensor, Sequence[torch.Tensor]]) -> torch.Tensor:
    if torch.is_tensor(batch):
        return batch
    if isinstance(batch, (list, tuple)) and len(batch) > 0 and torch.is_tensor(batch[0]):
        return batch[0]
    raise ValueError("Protocol loader must yield tensor batches or tuple/list with token tensor first")


def _get_float(row: Dict[str, object], key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    if isinstance(value, (int, float)):
        return float(value)
    return float(default)


def run_single_target_protocol(
    model: GPT,
    probe: nn.Module,
    batches: Iterable[Union[torch.Tensor, Sequence[torch.Tensor]]],
    layer: int,
    target_cell: int,
    target_class: int,
    position: Union[str, int] = "last",
    config: Optional[ProtocolConfig] = None,
) -> Dict[str, object]:
    """
    Run a fixed target intervention protocol over a token stream.

    For each processed sequence, this protocol:
    1) optimizes a residual delta for target cell/class,
    2) applies it in-model at a fixed layer/position,
    3) stores per-example results and aggregates summary metrics.
    """
    config = ProtocolConfig() if config is None else config
    target = InterventionTarget(cell_index=target_cell, target_class=target_class, position=position)

    intervention_cfg = InterventionConfig(
        steps=config.intervention_steps,
        lr=config.intervention_lr,
        reg_strength=config.intervention_reg_strength,
        clamp_norm=config.clamp_norm,
        device=config.device,
    )

    per_example: List[Dict[str, object]] = []
    processed = 0

    for batch in batches:
        tokens = _extract_tokens(batch)
        for row in range(tokens.size(0)):
            seq = tokens[row : row + 1]
            result = run_probe_intervention(
                model=model,
                probe=probe,
                token_batch=seq,
                layer=layer,
                target=target,
                config=intervention_cfg,
                device=config.device,
            )
            per_example.append(result)
            processed += 1
            if processed >= config.max_examples:
                break
        if processed >= config.max_examples:
            break

    if processed == 0:
        raise ValueError("No examples processed in run_single_target_protocol")

    success_rate = sum(1 for x in per_example if bool(x["success"])) / processed
    mean_prob_gain = float(sum(_get_float(x, "target_prob_gain") for x in per_example) / processed)
    mean_delta_norm = float(sum(_get_float(x, "delta_norm") for x in per_example) / processed)

    return {
        "summary": {
            "processed_examples": processed,
            "layer": layer,
            "target_cell": target_cell,
            "target_class": target_class,
            "position": position,
            "success_rate": success_rate,
            "mean_target_prob_gain": mean_prob_gain,
            "mean_delta_norm": mean_delta_norm,
        },
        "results": per_example,
    }


def run_layer_sweep_protocol(
    model: GPT,
    probe: nn.Module,
    batches: Iterable[Union[torch.Tensor, Sequence[torch.Tensor]]],
    layers: Sequence[int],
    target_cell: int,
    target_class: int,
    position: Union[str, int] = "last",
    config: Optional[ProtocolConfig] = None,
) -> Dict[str, object]:
    """
    Run `run_single_target_protocol` across several layers and aggregate summaries.
    """
    config = ProtocolConfig() if config is None else config

    materialized_batches = [
        _extract_tokens(batch).detach().clone() for batch in batches
    ]

    layer_results: Dict[int, Dict[str, object]] = {}
    for layer in layers:
        layer_results[layer] = run_single_target_protocol(
            model=model,
            probe=probe,
            batches=materialized_batches,
            layer=layer,
            target_cell=target_cell,
            target_class=target_class,
            position=position,
            config=config,
        )

    return {
        "layers": list(layers),
        "target_cell": target_cell,
        "target_class": target_class,
        "position": position,
        "layer_results": layer_results,
    }


def run_b2_metrics_protocol(
    model: GPT,
    probe: nn.Module,
    batches: Iterable[Union[torch.Tensor, Sequence[torch.Tensor]]],
    layers: Sequence[int],
    target_cell: int,
    target_class: int,
    position: Union[str, int] = "last",
    config: Optional[ProtocolConfig] = None,
) -> Dict[str, object]:
    """
    Run a layer sweep and compute standardized B2 intervention metrics.
    """
    sweep = run_layer_sweep_protocol(
        model=model,
        probe=probe,
        batches=batches,
        layers=layers,
        target_cell=target_cell,
        target_class=target_class,
        position=position,
        config=config,
    )

    per_layer = aggregate_by_layer(sweep)

    all_results: List[Dict[str, object]] = []
    layer_results_obj = sweep.get("layer_results")
    if not isinstance(layer_results_obj, dict):
        raise RuntimeError("Expected sweep['layer_results'] to be a dict")

    for data in layer_results_obj.values():
        if isinstance(data, dict) and isinstance(data.get("results"), list):
            all_results.extend(data["results"])

    overall = {
        "success_rate": compute_success_rate(all_results),
        "target_gain": compute_target_gain_stats(all_results),
        "delta_norm": compute_delta_norm_stats(all_results),
        "off_target": compute_off_target_impact(all_results),
    }

    return {
        "sweep": sweep,
        "metrics": {
            "overall": overall,
            "by_layer": per_layer,
        },
    }


def run_b2_attribution_protocol(
    model: GPT,
    probe: nn.Module,
    batches: Iterable[Union[torch.Tensor, Sequence[torch.Tensor]]],
    layers: Sequence[int],
    target_cells: Sequence[int],
    target_class: int,
    position: Union[str, int] = "last",
    config: Optional[ProtocolConfig] = None,
    value_key: str = "target_prob_gain",
    top_k: int = 10,
) -> Dict[str, object]:
    """
    Run layer x cell interventions and compute attribution summaries.
    """
    config = ProtocolConfig() if config is None else config
    materialized_batches = [_extract_tokens(batch).detach().clone() for batch in batches]

    layer_to_results: Dict[int, List[Dict[str, object]]] = {int(layer): [] for layer in layers}
    detailed: Dict[int, Dict[int, Dict[str, object]]] = {}

    for layer in layers:
        layer = int(layer)
        detailed[layer] = {}
        for cell in target_cells:
            report = run_single_target_protocol(
                model=model,
                probe=probe,
                batches=materialized_batches,
                layer=layer,
                target_cell=int(cell),
                target_class=target_class,
                position=position,
                config=config,
            )
            detailed[layer][int(cell)] = report
            results = report.get("results")
            if not isinstance(results, list):
                raise RuntimeError("Expected protocol report to contain a list under 'results'")
            layer_to_results[layer].extend(results)

    attribution = compute_intervention_attribution_summary(
        layer_results=layer_to_results,
        value_key=value_key,
        top_k=top_k,
    )

    return {
        "layers": list(layers),
        "target_cells": [int(c) for c in target_cells],
        "target_class": target_class,
        "position": position,
        "value_key": value_key,
        "attribution": attribution,
        "detailed_reports": detailed,
    }
