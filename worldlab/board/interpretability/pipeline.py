"""
High-level B3 pipeline orchestrator for interpretability runs.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Sequence

import torch
from torch.utils.data import DataLoader, TensorDataset

from worldlab.nn.board import GPT, GPTConfig

from .io import save_run_artifacts
from .probes import LinearBoardProbe
from .protocols import ProtocolConfig, run_b2_attribution_protocol, run_b2_metrics_protocol
from .report import export_metrics_csv, export_top_contributors_csv, generate_markdown_report


def build_demo_model_probe(seed: int = 0) -> tuple[GPT, LinearBoardProbe]:
    """
    Build a tiny deterministic model+probe pair for CLI demos.
    """
    torch.manual_seed(seed)
    cfg = GPTConfig(
        block_size=8,
        vocab_size=25,
        n_layer=2,
        n_head=1,
        n_embd=16,
        dropout=0.0,
        bias=True,
    )
    model = GPT(cfg, use_flash=False)
    probe = LinearBoardProbe(input_dim=cfg.n_embd)

    with torch.no_grad():
        probe.proj.weight.zero_()
        probe.proj.bias.zero_()
        base = 5 * 3
        probe.proj.weight[base + 0, 0] = -2.0
        probe.proj.weight[base + 1, 0] = -1.0
        probe.proj.weight[base + 2, 0] = 2.0

    return model, probe


def _build_demo_loader(num_sequences: int, block_size: int, vocab_size: int, batch_size: int, seed: int) -> DataLoader:
    g = torch.Generator().manual_seed(seed)
    tokens = torch.randint(0, vocab_size, (num_sequences, block_size), generator=g)
    return DataLoader(TensorDataset(tokens), batch_size=batch_size, shuffle=False)


def run_b3_demo_pipeline(
    out_root: str = "out_interp/jalon_b3",
    run_name: Optional[str] = None,
    seed: int = 0,
    layers: Sequence[int] = (0, 1),
    target_cell: int = 5,
    target_class: int = 2,
    target_cells: Sequence[int] = (3, 5, 9),
    max_examples: int = 8,
) -> Dict[str, object]:
    """
    Run a complete B3 demo:
    - B2 metrics protocol
    - B2 attribution protocol
    - save JSON/JSONL/PT artifacts
    - export CSV tables
    - generate report.md
    """
    run_name = run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = Path(out_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    model, probe = build_demo_model_probe(seed=seed)
    config = ProtocolConfig(
        max_examples=max_examples,
        intervention_steps=20,
        intervention_lr=0.08,
        intervention_reg_strength=1e-4,
        device="cpu",
    )

    loader_metrics = _build_demo_loader(16, model.config.block_size, model.config.vocab_size, batch_size=4, seed=seed)
    metrics_report = run_b2_metrics_protocol(
        model=model,
        probe=probe,
        batches=loader_metrics,
        layers=list(layers),
        target_cell=target_cell,
        target_class=target_class,
        config=config,
    )

    loader_attr = _build_demo_loader(16, model.config.block_size, model.config.vocab_size, batch_size=4, seed=seed + 1)
    attribution_report = run_b2_attribution_protocol(
        model=model,
        probe=probe,
        batches=loader_attr,
        layers=list(layers),
        target_cells=list(target_cells),
        target_class=target_class,
        config=config,
        top_k=10,
    )

    sweep = metrics_report.get("sweep")
    layer_results = sweep.get("layer_results") if isinstance(sweep, dict) else {}
    first_layer = int(list(layers)[0])
    first_report = layer_results.get(first_layer, {}) if isinstance(layer_results, dict) else {}
    first_results = first_report.get("results", []) if isinstance(first_report, dict) else []

    attr_obj = attribution_report.get("attribution") if isinstance(attribution_report, dict) else {}
    attr_map = attr_obj.get("attribution_map") if isinstance(attr_obj, dict) else None
    tensors = {"attribution_map": attr_map} if torch.is_tensor(attr_map) else None

    artifacts = save_run_artifacts(
        run_dir=str(run_dir),
        config={
            "pipeline": "b3_demo",
            "seed": seed,
            "layers": list(layers),
            "target_cell": target_cell,
            "target_class": target_class,
            "target_cells": list(target_cells),
            "max_examples": max_examples,
        },
        summary=metrics_report.get("metrics", {}),
        results=first_results if isinstance(first_results, list) else None,
        tensors=tensors,
    )

    csv_metrics = export_metrics_csv(str(run_dir), metrics_report)
    contrib_csv = export_top_contributors_csv(str(run_dir), attr_obj if isinstance(attr_obj, dict) else {})

    md_path = generate_markdown_report(
        run_dir=str(run_dir),
        run_name=run_name,
        summary={
            "pipeline": "b3_demo",
            "run_dir": str(run_dir),
            "seed": seed,
            "layers": list(layers),
            "target_cell": target_cell,
            "target_class": target_class,
            "max_examples": max_examples,
        },
        metrics_report=metrics_report,
        attribution_report=attr_obj if isinstance(attr_obj, dict) else {},
    )

    return {
        "run_dir": str(run_dir),
        "artifacts": artifacts,
        "metrics_csv": csv_metrics,
        "contributors_csv": contrib_csv,
        "report_md": md_path,
        "metrics_report": metrics_report,
        "attribution_report": attribution_report,
    }
