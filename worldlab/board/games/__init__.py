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

"""
Simulators for board games.
"""

from .game_interface import GameInterface
from .othello import (
    create_id_to_move_mapping as othello_create_id_to_move_mapping,
    convert_ids_to_notation as othello_convert_ids_to_notation,
    OthelloGame
)
try:
    from .checkers import (
        create_id_to_move_mapping as checkers_create_id_to_move_mapping,
        convert_ids_to_notation as checkers_convert_ids_to_notation,
        CheckersGame,
        CheckersBoard
    )
except ModuleNotFoundError:
    checkers_create_id_to_move_mapping = None
    checkers_convert_ids_to_notation = None
    CheckersGame = None
    CheckersBoard = None

__all__ = [
    "GameInterface",
    "OthelloGame",
    "othello_create_id_to_move_mapping",
    "othello_convert_ids_to_notation",
    "CheckersGame",
    "CheckersBoard",
    "checkers_create_id_to_move_mapping",
    "checkers_convert_ids_to_notation"
]
