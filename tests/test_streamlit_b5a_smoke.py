import torch

from worldlab.board.interpretability.probes import LinearBoardProbe
from worldlab.board.interpretability.ui.activations import get_residual_vector, summarize_vector
from worldlab.board.interpretability.ui.inference import run_next_token_inference
from worldlab.board.interpretability.ui.parsing import parse_input_to_tensor, parse_position
from worldlab.board.interpretability.ui.probe_view import (
    format_board_confidence,
    format_board_symbols,
    run_probe_on_vector,
)
from worldlab.nn.board import GPT, GPTConfig


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


def test_b5a_parsing_inference_activation_probe_flow():
    model = _tiny_model()
    probe = LinearBoardProbe(input_dim=model.config.n_embd)

    tokens = parse_input_to_tensor("0, 12, 14, 22", mode="token_ids", block_size=8, device="cpu")
    assert tokens.shape == (1, 8)

    infer = run_next_token_inference(model=model, tokens=tokens, top_k=5)
    assert "prediction_token_id" in infer
    assert len(infer["top_k"]) == 5

    pos = parse_position("last", seq_len=tokens.size(1))
    vec = get_residual_vector(model=model, tokens=tokens, layer=1, position=pos, device="cpu")
    assert vec.shape == (16,)

    stats = summarize_vector(vec)
    assert stats["dim"] == 16.0

    pout = run_probe_on_vector(probe=probe, vector=vec, device="cpu")
    assert pout["pred"].shape == (64,)
    assert pout["conf"].shape == (64,)

    board = format_board_symbols(pout["pred"])
    conf = format_board_confidence(pout["conf"])
    assert len(board) == 8 and len(board[0]) == 8
    assert len(conf) == 8 and len(conf[0]) == 8


def test_b5a_move_input_parsing():
    tokens = parse_input_to_tensor("d3,c3,c4,e3", mode="moves", block_size=8, device="cpu")
    assert tokens.shape == (1, 8)
    assert int(tokens[0, 0].item()) == 0
