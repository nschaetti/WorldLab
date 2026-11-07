"""
Copyright (C) 2025 boardGPT Contributors

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

from .cli import (
    checkers_generate,
    checkers_visualize,
    checkers_load,
    checkers_play,
    checkers_convert,
    checkers_verify
)

from .checkers_simulator import (
    CheckersGame,
    CheckersBoard,
    create_id_to_move_mapping,
    create_move_mapping,
    convert_ids_to_notation,
    load_games,
    generate_checkers_game
)

from .checkers_utils import verify_game, game_to_board, board_to_string, count_pieces, evaluate_board

__all__ = [
    "checkers_generate",
    "checkers_visualize",
    "checkers_load",
    "checkers_play",
    "checkers_convert",
    "checkers_verify",
    "CheckersGame",
    "CheckersBoard",
    "create_id_to_move_mapping",
    "create_move_mapping",
    "convert_ids_to_notation",
    "load_games",
    "generate_checkers_game",
    "verify_game",
    "game_to_board",
    "board_to_string",
    "count_pieces",
    "evaluate_board"
]

