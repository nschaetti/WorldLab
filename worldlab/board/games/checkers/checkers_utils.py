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
import sys
import os
from typing import List, Tuple, Optional
from .checkers_simulator import CheckersGame


def verify_game(moves: List[str]) -> Tuple[bool, List[str]]:
    """
    Verify the validity of a Checkers game by checking if all moves are valid.
    
    Args:
        moves (List[str]): List of moves in standard notation (e.g., ['a3-b4', 'c5-d4', ...])
        
    Returns:
        Tuple[bool, List[str]]: 
            - First element is True if all moves are valid, False otherwise
            - Second element is an empty list if all moves are valid, 
              or a list of invalid moves if any are invalid
    """
    # Create a new board to replay and validate the game
    game = CheckersGame()
    invalid_moves = []
    
    # Track the current player (Black starts in Checkers)
    current_player = game.BLACK
    
    for i, move in enumerate(moves):
        # Check move format (e.g., "a3-b4")
        parts = move.split("-")
        if len(parts) != 2:
            invalid_moves.append(move)
            continue
        
        from_notation, to_notation = parts
        
        # Check if notations are valid
        if not (len(from_notation) == 2 and 'a' <= from_notation[0].lower() <= 'h' and '1' <= from_notation[1] <= '8'):
            invalid_moves.append(move)
            continue
        
        if not (len(to_notation) == 2 and 'a' <= to_notation[0].lower() <= 'h' and '1' <= to_notation[1] <= '8'):
            invalid_moves.append(move)
            continue
        
        # Convert notation to coordinates
        try:
            from_row, from_col = game.notation_to_coords(from_notation)
            to_row, to_col = game.notation_to_coords(to_notation)
        except ValueError:
            # Invalid notation
            invalid_moves.append(move)
            continue
        
        # Set the current player
        game.current_player = current_player
        
        # Check if the move is legal for the current player
        if not game.is_valid_move(to_row, to_col):
            invalid_moves.append(move)
            continue
        
        # Make the move
        if not game.make_move(to_row, to_col):
            invalid_moves.append(move)
            continue
        
        # Update current player for next move (if not in a jump sequence)
        if game.current_jump_sequence is None:
            current_player = game.WHITE if current_player == game.BLACK else game.BLACK
    
    # Return True if all moves are valid, False otherwise
    return len(invalid_moves) == 0, invalid_moves


def game_to_board(moves: List[str]) -> List[int]:
    """
    Transform a game (as a list of str) into a board representation as a list of int.
    
    Args:
        moves (List[str]): List of moves in standard notation (e.g., ['a3-b4', 'c5-d4', ...])
        
    Returns:
        List[int]: Board representation as a 1D list where:
            - 0 = empty
            - 1 = black regular piece
            - 2 = white regular piece
            - 3 = black king
            - 4 = white king
            - The first entry is "a1", the second "a2", etc.
            
    Raises:
        ValueError: If a move is invalid
    """
    # Create a new Checkers game
    game = CheckersGame()
    
    # Replay the game by making each move
    for move in moves:
        try:
            # Parse the move notation (e.g., "a3-b4")
            parts = move.split("-")
            if len(parts) != 2:
                raise ValueError(f"Invalid move notation: {move}")
            
            from_notation, to_notation = parts
            from_coords = game.notation_to_coords(from_notation)
            to_coords = game.notation_to_coords(to_notation)
            
            # Check if the move is valid
            if not game.is_valid_move(to_coords[0], to_coords[1]):
                raise ValueError(f"Move {move} is invalid")
            
            # Make the move
            if not game.make_move(to_coords[0], to_coords[1]):
                raise ValueError(f"Failed to make move: {move}")
        except ValueError as e:
            # Re-raise ValueError for invalid moves
            raise ValueError(f"Error processing move {move}: {str(e)}")
    
    # Convert the 2D board to a 1D list in the required order
    board_1d = []
    for col in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']:
        for row in range(1, 9):
            # Position
            pos = f"{col}{row}"
            
            # Get the coordinate
            r, c = game.notation_to_coords(pos)
            
            # Get the piece at this position
            piece = game.board.get_piece(r, c)
            
            # Add the piece to the board representation
            board_1d.append(piece)
    
    assert len(board_1d) == 64, f"Error: board representation must contain 64 elements"
    
    return board_1d


def board_to_string(board: List[int]) -> str:
    """
    Convert a board representation to a string for display.
    
    Args:
        board (List[int]): Board representation as a 1D list
        
    Returns:
        str: String representation of the board
    """
    if len(board) != 64:
        raise ValueError("Board must have 64 elements")
    
    # Piece representations
    pieces = {
        0: "·",  # empty
        1: "●",  # black regular piece
        2: "○",  # white regular piece
        3: "♚",  # black king
        4: "♔"   # white king
    }
    
    result = "  a b c d e f g h\n"
    
    for row in range(8):
        result += f"{8-row} "
        for col in range(8):
            index = col * 8 + row
            piece = board[index]
            result += pieces.get(piece, "?") + " "
        result += f"{8-row}\n"
    
    result += "  a b c d e f g h"
    
    return result


def count_pieces(board: List[int]) -> Tuple[int, int, int, int]:
    """
    Count the number of pieces of each type on the board.
    
    Args:
        board (List[int]): Board representation as a 1D list
        
    Returns:
        Tuple[int, int, int, int]: (black_regular, white_regular, black_king, white_king)
    """
    black_regular = board.count(1)
    white_regular = board.count(2)
    black_king = board.count(3)
    white_king = board.count(4)
    
    return black_regular, white_regular, black_king, white_king


def evaluate_board(board: List[int]) -> int:
    """
    Evaluate the board position from black's perspective.
    
    Args:
        board (List[int]): Board representation as a 1D list
        
    Returns:
        int: Score (positive is good for black, negative is good for white)
    """
    black_regular, white_regular, black_king, white_king = count_pieces(board)
    
    # Basic material count (kings are worth more than regular pieces)
    score = (black_regular - white_regular) + 2 * (black_king - white_king)
    
    return score