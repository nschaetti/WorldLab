"""BoardGPT imports and checkpoints continue to resolve to the canonical implementation."""

import importlib
import pickle

import torch

from worldlab.nn import GPTConfig, GameAutoEncoder, GameGPT
from worldlab.training import configure_optimizers


def test_legacy_names_preserve_module_and_class_identity():
    for old, current in (
        ("boardGPT", "worldlab.board"),
        ("boardGPT.nn", "worldlab.nn.board"),
        ("boardGPT.nn.gpt", "worldlab.nn.board.gpt"),
        ("boardGPT.models.gptboard", "worldlab.nn.board.gptboard"),
        ("boardGPT.models.game_autoencoder", "worldlab.nn.board.game_autoencoder"),
        ("boardGPT.games.othello.othello_simulator", "worldlab.board.games.othello.othello_simulator"),
    ):
        assert importlib.import_module(old) is importlib.import_module(current)
    # Protocol 0 GLOBAL opcodes as stored by old object checkpoints.
    assert pickle.loads(b"cboardGPT.nn.gpt\nGPTConfig\n.") is GPTConfig
    assert pickle.loads(b"cboardGPT.models.gptboard\nGameGPT\n.") is GameGPT
    assert pickle.loads(b"cboardGPT.models.game_autoencoder\nGameAutoEncoder\n.") is GameAutoEncoder


def test_shared_adamw_loads_the_original_group_layout():
    config = GPTConfig(vocab_size=16, block_size=4, n_layer=1, n_head=2, n_embd=8, bias=True)
    model = GameGPT(config)
    optimizer, decay, no_decay = model.configure_optimizers(0.1, 0.001, (0.9, 0.95), "cpu")
    tokens = torch.ones(2, 8, dtype=torch.long)
    _, _, loss, _ = model(tokens[:, :4], tokens[:, 4:])
    loss.backward()
    optimizer.step()
    replica = GameGPT(config)
    replica.load_state_dict(model.state_dict(), strict=True)
    shared, shared_decay, shared_no_decay = configure_optimizers(replica, 0.1, 0.001, (0.9, 0.95), "cpu")
    shared.load_state_dict(optimizer.state_dict())
    assert (shared_decay, shared_no_decay) == (decay, no_decay)
    assert [group["weight_decay"] for group in shared.param_groups] == [0.1, 0.0]
    assert len(shared.state) == len(optimizer.state)
