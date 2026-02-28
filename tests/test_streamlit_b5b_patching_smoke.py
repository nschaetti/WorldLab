import torch

from boardGPT.interpretability.probes import LinearBoardProbe
from boardGPT.interpretability.ui.patching import run_activation_patch_experiment
from boardGPT.nn import GPT, GPTConfig


def _tiny_model() -> GPT:
    cfg = GPTConfig(
        block_size=8,
        vocab_size=61,
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


def test_b5b_activation_patch_experiment_smoke():
    model = _tiny_model()
    probe = _targetable_probe()
    tokens = torch.randint(0, model.config.vocab_size, (1, model.config.block_size))

    out = run_activation_patch_experiment(
        model=model,
        probe=probe,
        tokens=tokens,
        layer=0,
        position="last",
        target_cell=5,
        target_class=2,
        steps=20,
        lr=0.08,
        reg_strength=1e-4,
        clamp_norm=None,
        device="cpu",
    )

    assert "before_inference" in out
    assert "after_inference" in out
    assert "intervention" in out
    assert out["intervention"]["after_target_prob"] >= out["intervention"]["before_target_prob"]
