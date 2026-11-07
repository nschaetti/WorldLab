from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from worldlab.board.interpretability.io import load_run_artifacts, save_run_artifacts
from worldlab.board.interpretability.probes import LinearBoardProbe
from worldlab.board.interpretability.protocols import (
    ProtocolConfig,
    run_b2_attribution_protocol,
    run_b2_metrics_protocol,
)
from worldlab.nn.board import GPT, GPTConfig


def _tiny_model() -> GPT:
    cfg = GPTConfig(
        block_size=8,
        vocab_size=25,
        n_layer=2,
        n_head=1,
        n_embd=16,
        dropout=0.0,
        bias=True,
    )
    return GPT(cfg, use_flash=False)


def _targetable_probe(input_dim: int = 16, cell: int = 5) -> LinearBoardProbe:
    probe = LinearBoardProbe(input_dim=input_dim)
    with torch.no_grad():
        probe.proj.weight.zero_()
        probe.proj.bias.zero_()
        base = cell * 3
        probe.proj.weight[base + 0, 0] = -2.0
        probe.proj.weight[base + 1, 0] = -1.0
        probe.proj.weight[base + 2, 0] = 2.0
    return probe


def test_b2_protocols_and_io(tmp_path: Path):
    model = _tiny_model()
    probe = _targetable_probe()

    tokens = torch.randint(0, 25, (6, 8))
    loader = DataLoader(TensorDataset(tokens), batch_size=2, shuffle=False)
    cfg = ProtocolConfig(
        max_examples=4,
        intervention_steps=20,
        intervention_lr=0.08,
        intervention_reg_strength=1e-4,
        device="cpu",
    )

    metrics_report = run_b2_metrics_protocol(
        model=model,
        probe=probe,
        batches=loader,
        layers=[0, 1],
        target_cell=5,
        target_class=2,
        config=cfg,
    )
    assert "overall" in metrics_report["metrics"]
    assert "by_layer" in metrics_report["metrics"]

    loader2 = DataLoader(TensorDataset(tokens), batch_size=2, shuffle=False)
    attr_report = run_b2_attribution_protocol(
        model=model,
        probe=probe,
        batches=loader2,
        layers=[0, 1],
        target_cells=[3, 5],
        target_class=2,
        config=cfg,
        top_k=4,
    )
    amap = attr_report["attribution"]["attribution_map"]
    assert tuple(amap.shape) == (2, 64)
    assert len(attr_report["attribution"]["top_contributors"]) == 4

    out = tmp_path / "run_b2"
    paths = save_run_artifacts(
        run_dir=str(out),
        config={"name": "b2-test"},
        summary=metrics_report["metrics"]["overall"],
        results=metrics_report["sweep"]["layer_results"][0]["results"],
        tensors={"attribution_map": amap},
    )
    assert "summary" in paths and "config" in paths

    loaded = load_run_artifacts(str(out))
    assert loaded["config"]["name"] == "b2-test"
    assert "attribution_map" in loaded["tensors"]
