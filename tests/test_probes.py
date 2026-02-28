import torch
from torch.utils.data import DataLoader, TensorDataset

from boardGPT.interpretability.probes import (
    LinearBoardProbe,
    MLPBoardProbe,
    eval_probe,
    fit_probe,
    predict_board,
)


def _build_linearly_decodable_dataset(num_samples: int = 256, dim: int = 24):
    torch.manual_seed(7)
    x = torch.randn(num_samples, dim)
    w = torch.randn(dim, 64, 3)
    logits = torch.einsum("nd,dck->nck", x, w)
    y = logits.argmax(dim=-1)
    return x, y


def test_linear_probe_trains_and_predicts():
    x, y = _build_linearly_decodable_dataset()
    train_x, val_x = x[:192], x[192:]
    train_y, val_y = y[:192], y[192:]

    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=32, shuffle=False)

    probe = LinearBoardProbe(input_dim=x.size(1))
    history = fit_probe(
        probe=probe,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=15,
        learning_rate=5e-3,
        device="cpu",
    )

    assert len(history["train_loss"]) == 15
    stats = eval_probe(probe=probe, data_loader=val_loader, device="cpu")
    assert 0.0 <= stats["acc_global"] <= 1.0
    assert stats["acc_global"] > 0.55
    assert stats["acc_per_cell"].shape == (64,)

    preds = predict_board(probe, val_x, device="cpu")
    assert preds.shape == (64, 64)
    assert preds.dtype == torch.long


def test_mlp_probe_forward_shape():
    probe = MLPBoardProbe(input_dim=16, hidden_dim=32)
    x = torch.randn(5, 16)
    logits = probe(x)
    assert logits.shape == (5, 64, 3)
