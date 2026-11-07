"""
Conventions for Othello interpretability in BoardGPT.
"""

from typing import Dict, Iterable, List, Tuple

from worldlab.board.games.othello import create_id_to_move_mapping, create_move_mapping
from worldlab.board.games.othello.othello_utils import game_to_board


OTHELLO_BOARD_SIZE = 8
OTHELLO_NUM_BOARD_CELLS = OTHELLO_BOARD_SIZE * OTHELLO_BOARD_SIZE
OTHELLO_NUM_MOVE_TOKENS = 60
OTHELLO_NUM_VOCAB_TOKENS = OTHELLO_NUM_MOVE_TOKENS + 1  # BOS


def get_othello_move_mappings() -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Return notation <-> token mappings for Othello moves.
    """
    move_to_id = create_move_mapping()
    id_to_move = create_id_to_move_mapping()
    return move_to_id, id_to_move


def move_ids_to_notation(move_ids: Iterable[int], skip_bos: bool = True) -> List[str]:
    """
    Convert token IDs to move notation.

    Args:
        move_ids: Iterable of token IDs.
        skip_bos: Whether to ignore BOS token (ID 0).

    Returns:
        List of notations such as ["d3", "c4"].
    """
    id_to_move = create_id_to_move_mapping()
    notations: List[str] = []
    for token_id in move_ids:
        if token_id == 0 and skip_bos:
            continue
        if token_id not in id_to_move:
            raise ValueError(f"Unknown Othello token ID: {token_id}")
        notations.append(id_to_move[token_id])
    return notations


def notation_to_board_labels(moves: List[str]) -> List[int]:
    """
    Convert move notation sequence to board labels.

    Output labels convention:
    - 0: empty
    - 1: white
    - 2: black
    """
    board = game_to_board(moves)
    if len(board) != OTHELLO_NUM_BOARD_CELLS:
        raise ValueError(f"Expected {OTHELLO_NUM_BOARD_CELLS} board labels, got {len(board)}")
    return board


def move_ids_to_board_labels(move_ids: Iterable[int], skip_bos: bool = True) -> List[int]:
    """
    Convert move token IDs directly to board labels.
    """
    moves = move_ids_to_notation(move_ids, skip_bos=skip_bos)
    return notation_to_board_labels(moves)
