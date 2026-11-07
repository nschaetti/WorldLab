import torch
from torch.utils.data import DataLoader, TensorDataset

from worldlab.board.interpretability.activation_extractor import (
    extract_dataset_residuals,
    extract_residuals,
)
from worldlab.nn.board import GPT, GPTConfig


def _build_tiny_model() -> GPT:
    config = GPTConfig(
        block_size=8,
        vocab_size=20,
        n_layer=2,
        n_head=2,
        n_embd=16,
        dropout=0.0,
        bias=True,
    )
    return GPT(config=config, use_flash=False)


def test_extract_residuals_returns_expected_shape():
    model = _build_tiny_model()
    tokens = torch.randint(0, 20, (4, 8))

    residuals = extract_residuals(model=model, token_batch=tokens, layer=1, device="cpu")
    assert residuals.shape == (4, 8, 16)


def test_extract_dataset_residuals_position_policies():
    model = _build_tiny_model()
    tokens = torch.randint(0, 20, (10, 8))
    labels = torch.zeros(10, 1)
    loader = DataLoader(TensorDataset(tokens, labels), batch_size=5, shuffle=False)

    all_positions = extract_dataset_residuals(
        model=model,
        batches=loader,
        layer=0,
        position_policy="all",
        device="cpu",
    )
    assert all_positions.shape == (10, 8, 16)

    last_position = extract_dataset_residuals(
        model=model,
        batches=loader,
        layer=0,
        position_policy="last",
        device="cpu",
    )
    assert last_position.shape == (10, 16)

    fixed_position = extract_dataset_residuals(
        model=model,
        batches=loader,
        layer=0,
        position_policy=2,
        device="cpu",
    )
    assert fixed_position.shape == (10, 16)
