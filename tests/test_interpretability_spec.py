import torch

from boardGPT.games.othello.othello_utils import game_to_board
from boardGPT.interpretability.spec import (
    OTHELLO_NUM_BOARD_CELLS,
    OTHELLO_NUM_VOCAB_TOKENS,
    get_othello_move_mappings,
    move_ids_to_board_labels,
    move_ids_to_notation,
)


def test_othello_mapping_core_properties():
    move_to_id, id_to_move = get_othello_move_mappings()

    assert move_to_id["BOS"] == 0
    assert id_to_move[0] == "BOS"
    assert len(move_to_id) == OTHELLO_NUM_VOCAB_TOKENS
    assert len(id_to_move) == OTHELLO_NUM_VOCAB_TOKENS

    for center in ["d4", "d5", "e4", "e5"]:
        assert center not in move_to_id


def test_move_ids_to_notation_and_board_labels_match_reference():
    move_to_id, _ = get_othello_move_mappings()
    moves = ["d3", "c3", "c4", "e3"]
    move_ids = [0] + [move_to_id[m] for m in moves]

    converted_moves = move_ids_to_notation(move_ids, skip_bos=True)
    assert converted_moves == moves

    board_labels = move_ids_to_board_labels(move_ids, skip_bos=True)
    reference = game_to_board(moves)
    assert board_labels == reference
    assert len(board_labels) == OTHELLO_NUM_BOARD_CELLS
    assert set(torch.tensor(board_labels).tolist()).issubset({0, 1, 2})
