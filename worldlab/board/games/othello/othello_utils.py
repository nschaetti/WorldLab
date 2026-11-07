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

# Imports
from typing import List, Tuple, Optional
import numpy as np
import sys
import os
from matplotlib import pyplot as plt

from .othello_simulator import OthelloGame


def verify_game(moves: List[str]) -> Tuple[bool, List[str]]:
    """
    Verify the validity of an Othello game by checking if all moves are valid.
    
    Args:
        moves (List[str]): List of moves in standard notation (e.g., ['c4', 'd3', ...])
        
    Returns:
        Tuple[bool, List[str]]: 
            - First element is True if all moves are valid, False otherwise
            - Second element is an empty list if all moves are valid, 
              or a list of invalid moves if any are invalid
    """
    # Create a new board to replay and validate the game
    board = OthelloGame()
    invalid_moves = []
    
    # Track the current player (Black starts in Othello)
    current_player = board.BLACK
    
    for i, move in enumerate(moves):
        # Check move corresponds to a valid square (a1 to h8)
        if not (len(move) == 2 and 'a' <= move[0].lower() <= 'h' and '1' <= move[1] <= '8'):
            invalid_moves.append(move)
            continue
        # end if
        # Convert notation to coordinates
        try:
            row, col = board.notation_to_coords(move)  # end try
        except ValueError:
            # Invalid notation
            invalid_moves.append(move)
            continue
        # end except
        # Set the current player
        board.current_player = current_player
        
        # Check if the move is legal for the current player
        if not board.is_valid_move(row, col):
            # If not valid for the current player, check if it's valid after a pass
            board.current_player = board.WHITE if current_player == board.BLACK else board.BLACK
            
            # Check if the move is valid for the other player
            if not board.is_valid_move(row, col):
                # Not valid for either player
                invalid_moves.append(move)
                continue
            # end if
            # Valid for the other player (implies a pass)
            current_player = board.current_player
        # end if
        # Make the move
        board.make_move(row, col)
        
        # Update current player for next move
        current_player = board.WHITE if current_player == board.BLACK else board.BLACK
    # end for
    # Return True if all moves are valid, False otherwise
    return len(invalid_moves) == 0, invalid_moves
# end def verify_game


def game_to_board(moves: List[str]) -> np.array:
    """
    Transform a game (as a list of str) into a board representation as a list of int.
    
    Args:
        moves (List[str]): List of moves in standard notation (e.g., ['c4', 'd3', ...])
        
    Returns:
        List[int]: Board representation as a 1D list where:
            - 0 = empty
            - 1 = black
            - 2 = white
            - The first entry is "a1", the second "a2", etc.
            
    Raises:
        ValueError: If a move is invalid for both players
    """
    # Create an OthelloGame object with the moves applied
    game = OthelloGame.load_moves(moves)
    
    # Get board
    return game.board.get_flattened_board_state()
# end game_to_board


def visualize_board(board: list):
    """
    Visualize an Othello board from its 1D list representation.

    Args:
        board (list[int]): 1D list of length 64 representing the board.
            0 = empty, 1 = white, 2 = black
    """
    assert len(board) == 64, "Board must have 64 positions"

    # Convert to 8x8 array
    board_array = np.array(board).reshape(8, 8)

    fig, ax = plt.subplots(figsize=(6, 6))

    # Background grid (green board)
    ax.set_facecolor((0.0, 0.6, 0.0))

    # Draw grid lines
    for i in range(9):
        ax.plot([0, 8], [i, i], color="black", linewidth=1)
        ax.plot([i, i], [0, 8], color="black", linewidth=1)
    # end for

    # Place pieces as circles
    for r in range(8):
        for c in range(8):
            if board_array[r, c] == 1:  # white
                circ = plt.Circle((c + 0.5, r + 0.5), 0.4, color="black", ec="black")
                ax.add_patch(circ)
            elif board_array[r, c] == 2:  # black
                circ = plt.Circle((c + 0.5, r + 0.5), 0.4, color="white")
                ax.add_patch(circ)
            # end if
        # end for
    # end for

    # Set ticks for labels
    ax.set_xticks(np.arange(0.5, 8.5, 1))
    ax.set_yticks(np.arange(0.5, 8.5, 1))
    ax.set_xticklabels(['a','b','c','d','e','f','g','h'])
    ax.set_yticklabels(range(1, 9))

    # Adjust axis
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 8)
    ax.invert_yaxis()  # to have row 1 at the top
    ax.set_aspect('equal')

    ax.set_title("Othello Board")

    plt.show()
# end def visualize_board

