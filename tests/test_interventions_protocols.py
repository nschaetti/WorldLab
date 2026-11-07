import torch
from torch.utils.data import DataLoader, TensorDataset

from worldlab.board.interpretability.interventions import (
    InterventionConfig,
    InterventionTarget,
    optimize_residual_delta,
    run_probe_intervention,
)
from worldlab.board.interpretability.probes import LinearBoardProbe
from worldlab.board.interpretability.protocols import (
    ProtocolConfig,
    run_layer_sweep_protocol,
    run_single_target_protocol,
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


def test_optimize_residual_delta_improves_target_probability():
    probe = _targetable_probe()
    target = InterventionTarget(cell_index=5, target_class=2, position="last")
    cfg = InterventionConfig(steps=80, lr=0.1, reg_strength=1e-4, device="cpu")

    result = optimize_residual_delta(
        probe=probe,
        base_activations=torch.zeros(16),
        target=target,
        config=cfg,
    )

    assert result["after_target_prob"] > result["before_target_prob"]
    assert result["target_prob_gain"] > 0.2


def test_run_probe_intervention_and_protocols():
    model = _tiny_model()
    probe = _targetable_probe()
    target = InterventionTarget(cell_index=5, target_class=2, position="last")
    cfg = InterventionConfig(steps=40, lr=0.08, reg_strength=1e-4, device="cpu")

    tokens = torch.randint(0, 25, (1, 8))
    single = run_probe_intervention(
        model=model,
        probe=probe,
        token_batch=tokens,
        layer=0,
        target=target,
        config=cfg,
        device="cpu",
    )
    assert single["after_target_prob"] >= single["before_target_prob"]
    assert single["target_prob_gain"] > 0.0

    loader = DataLoader(TensorDataset(torch.randint(0, 25, (6, 8))), batch_size=2, shuffle=False)
    protocol_cfg = ProtocolConfig(
        max_examples=4,
        intervention_steps=20,
        intervention_lr=0.08,
        intervention_reg_strength=1e-4,
        device="cpu",
    )
    prot = run_single_target_protocol(
        model=model,
        probe=probe,
        batches=loader,
        layer=0,
        target_cell=5,
        target_class=2,
        position="last",
        config=protocol_cfg,
    )

    assert prot["summary"]["processed_examples"] == 4
    assert len(prot["results"]) == 4

    sweep = run_layer_sweep_protocol(
        model=model,
        probe=probe,
        batches=loader,
        layers=[0, 1],
        target_cell=5,
        target_class=2,
        position="last",
        config=protocol_cfg,
    )
    assert set(sweep["layer_results"].keys()) == {0, 1}
