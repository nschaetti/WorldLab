#!/usr/bin/env python3
"""
Smoke test for BoardGPT Jalon A interpretability APIs.

Run:
    python tools/smoke_test_interpretability.py
"""

import random
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from worldlab.board.interpretability import (
    InterventionConfig,
    InterventionTarget,
    LinearBoardProbe,
    MLPBoardProbe,
    ProtocolConfig,
    compute_intervention_attribution_summary,
    compute_success_rate,
    load_run_artifacts,
    run_b2_attribution_protocol,
    run_b2_metrics_protocol,
    run_b3_demo_pipeline,
    run_probe_intervention,
    run_single_target_protocol,
    save_run_artifacts,
    eval_probe,
    extract_dataset_residuals,
    extract_residuals,
    fit_probe,
    get_othello_move_mappings,
    move_ids_to_board_labels,
)
from worldlab.nn.board import GPT, GPTConfig


def set_seed(seed: int = 1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_tiny_gpt() -> GPT:
    cfg = GPTConfig(
        block_size=8,
        vocab_size=20,
        n_layer=2,
        n_head=2,
        n_embd=16,
        dropout=0.0,
        bias=True,
    )
    return GPT(cfg, use_flash=False)


def run_spec_checks():
    print("[1/7] Spec checks")
    move_to_id, _ = get_othello_move_mappings()
    assert move_to_id["BOS"] == 0

    sample_moves = ["d3", "c3", "c4", "e3"]
    sample_ids = [0] + [move_to_id[m] for m in sample_moves]
    board_labels = move_ids_to_board_labels(sample_ids, skip_bos=True)
    assert len(board_labels) == 64
    assert set(board_labels).issubset({0, 1, 2})
    print("    ok - mapping + board labels")


def run_activation_checks(model: GPT):
    print("[2/7] Activation extraction checks")
    tokens = torch.randint(0, model.config.vocab_size, (12, model.config.block_size))

    residuals = extract_residuals(model, tokens, layer=1, device="cpu")
    assert residuals.shape == (12, 8, 16)

    loader = DataLoader(TensorDataset(tokens, torch.zeros(12, 1)), batch_size=4)
    residuals_last = extract_dataset_residuals(
        model=model,
        batches=loader,
        layer=0,
        position_policy="last",
        device="cpu",
    )
    assert residuals_last.shape == (12, 16)
    print("    ok - residual extraction (all + last)")


def build_linear_decodable_dataset(num_samples: int = 256, dim: int = 16):
    x = torch.randn(num_samples, dim)
    w = torch.randn(dim, 64, 3)
    logits = torch.einsum("nd,dck->nck", x, w)
    y = logits.argmax(dim=-1)
    return x, y


def run_probe_checks():
    print("[3/7] Probe training checks")
    x, y = build_linear_decodable_dataset()
    train_x, val_x = x[:192], x[192:]
    train_y, val_y = y[:192], y[192:]

    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=32, shuffle=False)

    linear_probe = LinearBoardProbe(input_dim=16)
    history = fit_probe(
        probe=linear_probe,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=10,
        learning_rate=5e-3,
        device="cpu",
    )
    stats = eval_probe(linear_probe, val_loader, device="cpu")

    assert len(history["train_loss"]) == 10
    assert 0.0 <= stats["acc_global"] <= 1.0
    assert stats["acc_global"] > 0.5
    print(f"    ok - linear probe val_acc={stats['acc_global']:.3f}")

    mlp_probe = MLPBoardProbe(input_dim=16, hidden_dim=32)
    mlp_logits = mlp_probe(val_x[:4])
    assert mlp_logits.shape == (4, 64, 3)
    print("    ok - mlp probe forward")


def run_intervention_checks(model: GPT):
    print("[4/7] Intervention protocol checks")
    probe = LinearBoardProbe(input_dim=model.config.n_embd)
    with torch.no_grad():
        probe.proj.weight.zero_()
        probe.proj.bias.zero_()
        cell = 5
        base = cell * 3
        probe.proj.weight[base + 0, 0] = -2.0
        probe.proj.weight[base + 1, 0] = -1.0
        probe.proj.weight[base + 2, 0] = 2.0

    tokens = torch.randint(0, model.config.vocab_size, (4, model.config.block_size))
    target = InterventionTarget(cell_index=5, target_class=2, position="last")
    intv_cfg = InterventionConfig(steps=30, lr=0.08, reg_strength=1e-4, device="cpu")

    single = run_probe_intervention(
        model=model,
        probe=probe,
        token_batch=tokens[0:1],
        layer=0,
        target=target,
        config=intv_cfg,
        device="cpu",
    )
    before_prob = single["before_target_prob"]
    after_prob = single["after_target_prob"]
    assert isinstance(before_prob, (float, int))
    assert isinstance(after_prob, (float, int))
    assert after_prob >= before_prob

    loader = DataLoader(TensorDataset(tokens, torch.zeros(4, 1)), batch_size=2, shuffle=False)
    prot_cfg = ProtocolConfig(
        max_examples=3,
        intervention_steps=15,
        intervention_lr=0.08,
        intervention_reg_strength=1e-4,
        device="cpu",
    )
    report = run_single_target_protocol(
        model=model,
        probe=probe,
        batches=loader,
        layer=0,
        target_cell=5,
        target_class=2,
        position="last",
        config=prot_cfg,
    )
    summary = report["summary"]
    assert isinstance(summary, dict)
    assert summary["processed_examples"] == 3
    print("    ok - single intervention + protocol")


def run_b2_checks(model: GPT):
    print("[5/7] B2 metrics + attribution checks")
    probe = LinearBoardProbe(input_dim=model.config.n_embd)
    with torch.no_grad():
        probe.proj.weight.zero_()
        probe.proj.bias.zero_()
        base = 5 * 3
        probe.proj.weight[base + 0, 0] = -2.0
        probe.proj.weight[base + 1, 0] = -1.0
        probe.proj.weight[base + 2, 0] = 2.0

    tokens = torch.randint(0, model.config.vocab_size, (8, model.config.block_size))
    loader = DataLoader(TensorDataset(tokens, torch.zeros(8, 1)), batch_size=2, shuffle=False)
    cfg = ProtocolConfig(
        max_examples=4,
        intervention_steps=15,
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
    metrics_obj = metrics_report.get("metrics")
    assert isinstance(metrics_obj, dict)
    overall = metrics_obj.get("overall")
    assert isinstance(overall, dict)
    assert 0.0 <= overall["success_rate"] <= 1.0

    loader2 = DataLoader(TensorDataset(tokens, torch.zeros(8, 1)), batch_size=2, shuffle=False)
    attr_report = run_b2_attribution_protocol(
        model=model,
        probe=probe,
        batches=loader2,
        layers=[0, 1],
        target_cells=[3, 5],
        target_class=2,
        config=cfg,
        top_k=3,
    )
    attr_obj = attr_report.get("attribution")
    assert isinstance(attr_obj, dict)
    attr_map = attr_obj.get("attribution_map")
    assert torch.is_tensor(attr_map)
    assert tuple(attr_map.shape) == (2, 64)

    detailed_obj = attr_report.get("detailed_reports")
    assert isinstance(detailed_obj, dict)
    layer0_obj = detailed_obj.get(0)
    assert isinstance(layer0_obj, dict)
    cell3_obj = layer0_obj.get(3)
    assert isinstance(cell3_obj, dict)
    cell3_results = cell3_obj.get("results")
    assert isinstance(cell3_results, list)
    summary = compute_intervention_attribution_summary(
        layer_results={0: cell3_results},
        top_k=2,
    )
    top_contrib = summary.get("top_contributors")
    assert isinstance(top_contrib, list)
    assert len(top_contrib) == 2

    sweep_obj = metrics_report.get("sweep")
    assert isinstance(sweep_obj, dict)
    layer_results_obj = sweep_obj.get("layer_results")
    assert isinstance(layer_results_obj, dict)
    layer0_report = layer_results_obj.get(0)
    assert isinstance(layer0_report, dict)
    flat_results = layer0_report.get("results")
    assert isinstance(flat_results, list)

    with tempfile.TemporaryDirectory() as tmp_dir:
        paths = save_run_artifacts(
            run_dir=tmp_dir,
            config={"name": "smoke-b2"},
            summary=overall,
            results=flat_results,
            tensors={"attr_map": attr_map},
        )
        assert "summary" in paths
        loaded = load_run_artifacts(tmp_dir)
        assert loaded["config"]["name"] == "smoke-b2"
        assert "attr_map" in loaded["tensors"]

    assert 0.0 <= compute_success_rate(flat_results) <= 1.0
    print("    ok - B2 metrics, attribution, io")


def run_b3_checks():
    print("[6/7] B3 pipeline checks")
    with tempfile.TemporaryDirectory() as tmp_dir:
        out = run_b3_demo_pipeline(
            out_root=tmp_dir,
            run_name="smoke_b3",
            seed=123,
            layers=(0, 1),
            target_cell=5,
            target_class=2,
            target_cells=(3, 5),
            max_examples=4,
        )
        run_dir_obj = out.get("run_dir")
        assert isinstance(run_dir_obj, str)
        run_dir = Path(run_dir_obj)
        assert run_dir.exists()
        assert (run_dir / "report.md").exists()
        assert (run_dir / "metrics_overall.csv").exists()
        assert (run_dir / "metrics_by_layer.csv").exists()
        assert (run_dir / "top_contributors.csv").exists()
    print("    ok - B3 pipeline + report artifacts")


def main():
    set_seed(1234)
    model = build_tiny_gpt()

    run_spec_checks()
    run_activation_checks(model)
    run_probe_checks()
    run_intervention_checks(model)
    run_b2_checks(model)
    run_b3_checks()

    print("[7/7] Smoke test completed successfully")


if __name__ == "__main__":
    main()
