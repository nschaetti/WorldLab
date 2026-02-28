"""
Input parsing helpers for interpretability UI.
"""

from __future__ import annotations

import re
from typing import List

import torch

from boardGPT.games.othello.othello_simulator import create_move_mapping


def _normalize_tokens(text: str) -> List[str]:
    tokens = re.split(r"[\s,;]+", text.strip())
    return [t for t in tokens if t]


def parse_token_id_text(text: str) -> List[int]:
    chunks = _normalize_tokens(text)
    if not chunks:
        raise ValueError("Empty token id sequence")
    ids = [int(x) for x in chunks]
    if any(x < 0 for x in ids):
        raise ValueError("Token IDs must be non-negative")
    return ids


def parse_move_text(text: str) -> List[int]:
    moves = [m.lower() for m in _normalize_tokens(text)]
    if not moves:
        raise ValueError("Empty move sequence")

    move_to_id = create_move_mapping()
    ids = [0]
    for move in moves:
        if move not in move_to_id:
            raise ValueError(f"Unknown Othello move notation: {move}")
        ids.append(move_to_id[move])
    return ids


def fit_to_block_size(ids: List[int], block_size: int) -> List[int]:
    if len(ids) > block_size:
        return ids[:block_size]
    if len(ids) < block_size:
        return ids + [0] * (block_size - len(ids))
    return ids


def parse_input_to_tensor(text: str, mode: str, block_size: int, device: str = "cpu") -> torch.Tensor:
    mode = mode.strip().lower()
    if mode == "token_ids":
        ids = parse_token_id_text(text)
    elif mode == "moves":
        ids = parse_move_text(text)
    else:
        raise ValueError("mode must be one of: token_ids, moves")
    ids = fit_to_block_size(ids, block_size)
    return torch.tensor([ids], dtype=torch.long, device=device)


def parse_position(position_text: str, seq_len: int) -> int:
    text = position_text.strip().lower()
    if text == "last":
        return seq_len - 1
    idx = int(text)
    if idx < 0 or idx >= seq_len:
        raise ValueError(f"Position must be in [0, {seq_len - 1}] or 'last'")
    return idx
