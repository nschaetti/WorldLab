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
import math

"""
Checkers game simulator.

This module implements the checkers game logic, including board representation,
move validation, and game state management.
"""

import random
import numpy as np
import pickle
from collections import Counter
from typing import List, Tuple, Set, Dict, Optional
from rich.text import Text
from rich.columns import Columns
from rich.console import Console
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from PIL import Image

from worldlab.board.utils import console, warning, info, error
from worldlab.board.games.game_interface import GameInterface


class CheckersBoard:
    """
    Represents a checkers board.
    
    This class handles the board representation and basic board operations.
    """
    
    def __init__(
            self,
            size: int = 5,
            empty: int = 0,
            black: int = 1,
            white: int = 2,
            black_king: int = 3,
            white_king: int = 4,
            starting_pieces: int = 20
    ):
        """
        Initialize a new checkers board.
        
        Args:
            size (int): Size of the board (default: 5)
            empty (int): Value representing an empty cell (default: 0)
            black (int): Value representing a black piece (default: 1)
            white (int): Value representing a white piece (default: 2)
            black_king (int): Value representing a black king piece (default: 3)
            white_king (int): Value representing a white king piece (default: 4)
            starting_pieces (int): Number of pieces to start at per player (default: 20)
        """
        self.size = size
        self.empty = empty
        self.black = black
        self.white = white
        self.black_king = black_king
        self.white_king = white_king
        self.starting_pieces = starting_pieces
        
        # Initialize the board
        self.board = np.zeros((size * 2, size), dtype=int)
        self.history = []
        
        # Set up the initial board state
        self._setup_board()
    # end __init__
    
    def _setup_board(self):
        """
        Set up the initial board state with pieces in their starting positions.
        """
        # Put the black pieces on the board
        max_manoury = self.size * self.size * 2
        for pi in range(self.starting_pieces):
            # Set black piece
            row, col = self.manoury_to_coords(pi + 1)
            self.set_piece(row, col, self.black)

            # Set white pieces
            row, col = self.manoury_to_coords(max_manoury - pi)
            self.set_piece(row, col, self.white)
        # end for
    # end def _setup_board

    def manoury_to_coords(self, manpos: int) -> Tuple[int, int]:
        """
        Convert Manoury to coordinates.

        Args:
            manpos (int): Manpo coordinate

        Returns:
            Tuple[int, int]: Board coordinate
        """
        assert 1 <= manpos <= (self.size * self.size * 2), \
            f"Manoury notation must be between 1 and {self.size * self.size * 2}, got {manpos}"
        col = (manpos-1) % self.size
        row = int(math.floor((manpos-1) / self.size))
        return row, col
    # end def manoury_to_coords

    def coords_to_manoury(self, row: int, col: int) -> int:
        """
        Convert coordinates to Manoury.

        Args:
            row (int): Row
            col (int): Column

        Returns:
            int: Manoury
        """
        assert 0 <= row < self.size * 2, f"row must be between 0 and {self.size * 2 - 1}, got {row}"
        assert 0 <= col < self.size, f"column must be between 0 and {self.size - 1}, got {col}"
        return row * self.size + col + 1
    # end coords_to_manoury

    def board_coords_to_manoury(self, row: int, col: int) -> Optional[int]:
        """
        Convert board coordinates to Manoury.

        Args:
            row (int): Row
            col (int): Column

        Returns:
            Tuple[int, int]: Manoury
        """
        assert 0 <= row < self.size * 2, f"row must be between 0 and {self.size * 2 - 1}, got {row}"
        assert 0 <= col < self.size * 2, f"column must be between 0 and {self.size * 2 - 1}, got {col}"

        # Empty case 1
        if row % 2 == 0 and col % 2 == 0:
            return None
        # end if

        # Empty case 2
        if row % 2 == 1 and col % 2 == 1:
            return None
        # end if

        if row % 2 == 0:
            return row * self.size + ((col + 1) // 2)
        else:
            return row * self.size + ((col // 2) + 1)
        # end if
    # end def board_coords_to_manoury

    def manoury_to_board_coords(self, manpos: int) -> Tuple[int, int]:
        """
        Convert Manoury to coordinates.

        Args:
            manpos (int): Manpo coordinate

        Returns:
            Tuple[int, int]: Board coordinate
        """
        assert 1 <= manpos <= (self.size * self.size * 2), \
            f"Manoury notation must be between 1 and {self.size * self.size * 2}, got {manpos}"
        row = (manpos-1) // self.size
        col = manpos - (row * self.size)
        if row % 2 == 0:
            col = col * 2 - 1
        else:
            col = col * 2 - 2
        # end if
        return row, col
    # end def manoury_to_board_coords

    def board_coords_to_coords(self, row: int, col: int) -> Optional[Tuple[int, int]]:
        """
        Convert Board coordinates to coordinates.

        Args:
            row (int): Row
            col (int): Column

        Returns:
            Tuple[int, int]: Board coordinate
        """
        manpos = self.board_coords_to_manoury(row, col)
        if manpos is not None:
            return self.manoury_to_coords(manpos)
        else:
            return None
        # end if
    # end def board_coords_to_coords

    def is_case_playable(self, row: int, col: int) -> bool:
        """
        Check if a piece is playable.

        Args:
            row (int): Row
            col (int): Column

        Returns:
            bool: True if piece is playable, False otherwise
        """
        return not self.board_coords_to_manoury(row, col) is None
    # end def is_case_playable

    def set_piece(self, row: int, col: int, value: int):
        """
        Set a piece at the specified position.
        
        Args:
            row (int): Row index
            col (int): Column index
            value (int): Value to set
        
        Returns:
            bool: True if successful, False otherwise
        """
        if 0 <= row < self.size * 2 and 0 <= col < self.size:
            self.board[row, col] = value
            return True
        # end if
        return False
    # end set_piece
    
    def get_piece(self, row: int, col: int) -> int:
        """
        Get the piece at the specified position.
        
        Args:
            row (int): Row index
            col (int): Column index
        
        Returns:
            int: Value at the specified position
        """
        if 0 <= row < self.size and 0 <= col < self.size:
            return self.board[row, col]
        return -1
    
    def push_history(self):
        """Save the current board state to history."""
        self.history.append(np.copy(self.board))
    
    def is_black(self, row: int, col: int) -> bool:
        """
        Check if the piece at the specified position is black.
        
        Args:
            row (int): Row index
            col (int): Column index
        
        Returns:
            bool: True if the piece is black, False otherwise
        """
        piece = self.get_piece(row, col)
        return piece == self.black or piece == self.black_king
    
    def is_white(self, row: int, col: int) -> bool:
        """
        Check if the piece at the specified position is white.
        
        Args:
            row (int): Row index
            col (int): Column index
        
        Returns:
            bool: True if the piece is white, False otherwise
        """
        piece = self.get_piece(row, col)
        return piece == self.white or piece == self.white_king
    
    def is_king(self, row: int, col: int) -> bool:
        """
        Check if the piece at the specified position is a king.
        
        Args:
            row (int): Row index
            col (int): Column index
        
        Returns:
            bool: True if the piece is a king, False otherwise
        """
        piece = self.get_piece(row, col)
        return piece == self.black_king or piece == self.white_king
    
    def promote_to_king(self, row: int, col: int) -> bool:
        """
        Promote a piece to king if it's at the opposite end of the board.
        
        Args:
            row (int): Row index
            col (int): Column index
        
        Returns:
            bool: True if the piece was promoted, False otherwise
        """
        piece = self.get_piece(row, col)
        
        # Black piece at the bottom row
        if piece == self.black and row == self.size - 1:
            self.set_piece(row, col, self.black_king)
            return True
        
        # White piece at the top row
        if piece == self.white and row == 0:
            self.set_piece(row, col, self.white_king)
            return True
        
        return False
    # end def promote_to_king

    def print(self):
        """
        Print the current board state.
        """
        # Top border
        result = "  "
        for col in range(self.size * 2):
            result += f"{col} "
        # end for
        result += "\n"

        for b_row in range(self.size * 2):
            result += f"{b_row} "
            for b_col in range(self.size * 2):
                playable = self.is_case_playable(b_row, b_col)
                if playable:
                    row, col = self.board_coords_to_coords(b_row, b_col)
                    piece = self.board[row, col]
                else:
                    piece = self.empty
                # end if

                if not playable or piece == self.empty:
                    if (b_row + b_col) % 2 == 0:  # Light squares
                        result += "⊡ "
                    else:  # Dark squares
                        result += "⊠ "
                    # end if
                elif piece == self.black:
                    result += "[bold blue]○[/bold blue] "
                elif piece == self.white:
                    result += "[bold red]●[/bold red] "
                elif piece == self.black_king:
                    result += "[bold blue]♚[/bold blue] "
                elif piece == self.white_king:
                    result += "[bold red]♔[/bold red] "
                # end if
            # end for
            result += f"{b_row}\n"
        # end for

        # Bottom border
        result += "  "
        for col in range(self.size * 2):
            result += f"{col} "
        # end for
        console.print(result)
    # end def print
    
    def __str__(self, last_move=None):
        """
        Return a string representation of the board.
        
        Args:
            last_move (Tuple[int, int], optional): The last move made (row, col)
        
        Returns:
            str: String representation of the board
        """
        result = "  "
        for col in range(self.size * 2):
            result += f"{col} "
        # end for

        result += "\n"
        
        for b_row in range(self.size * 2):
            result += f"{b_row} "
            for b_col in range(self.size * 2):
                playable = self.is_case_playable(b_row, b_col)
                if playable:
                    row, col = self.board_coords_to_coords(b_row, b_col)
                    manpos = self.coords_to_manoury(row, col)
                    piece = self.board[row, col]
                else:
                    row, col = None, None
                    manpos = None
                    piece = self.empty
                # end if

                if not playable or piece == self.empty:
                    if (b_row + b_col) % 2 == 0:  # Light squares
                        result += "⊡ "
                    else:  # Dark squares
                        result += "⊠ "
                    # end if
                elif piece == self.black:
                    result += "○ "
                elif piece == self.white:
                    result += "● "
                elif piece == self.black_king:
                    result += "♚ "
                elif piece == self.white_king:
                    result += "♔ "
                # end if
            # end for
            result += f"{b_row}\n"
        # end for
        result += "  "
        for col in range(self.size * 2):
            result += f"{col} "
        # end for
        
        return result
    # end def __str__
    
    def __repr__(self):
        """
        Return a string representation of the board.
        
        Returns:
            str: String representation of the board
        """
        return self.__str__()


class CheckersGame(GameInterface):
    """
    Implements the checkers game logic.
    
    This class handles the game state, move validation, and game rules.
    It implements the GameInterface to ensure compatibility with the boardGPT framework.
    """

    # Board size (standard Checkers board is 10x10 (but 5x5 to play)
    SIZE = 5

    # Player constants for board representation
    EMPTY = 0  # Empty cell
    BLACK = 1  # Black piece (typically goes first)
    WHITE = 2  # White piece
    BLACK_KING = 3 # Black king
    WHITE_KING = 4 # White king
    
    def __init__(self):
        """Initialize a new checkers game."""
        # Initialize the board
        self.board = CheckersBoard(
            size=self.SIZE,
            black=self.BLACK,
            white=self.WHITE,
            black_king=self.BLACK_KING,
            white_king=self.WHITE_KING
        )
        
        # Current player (black starts)
        self.current_player = self.BLACK
        
        # Game state
        self.game_over = False
        self.winner = None
        
        # Move history
        self.moves = []
        self.moves_player = []
        
        # Forced jump flag
        self.forced_jump = True  # If True, jumps are mandatory when available
        
        # Current jump sequence (for multiple jumps)
        self.current_jump_sequence = None
    # end __init__
    
    def get_valid_moves(self) -> List[Tuple[int, int]]:
        """
        Return a list of valid moves for the current player.
        
        Returns:
            List[Tuple[int, int]]: A list of (row, col) tuples representing valid move positions
        """
        valid_moves = []
        
        # Check if we're in the middle of a jump sequence
        if self.current_jump_sequence is not None:
            row, col = self.current_jump_sequence
            jumps = self._get_valid_jumps(row, col)
            if jumps:
                return jumps
            else:
                # End of jump sequence
                self.current_jump_sequence = None
                return []
            # end if
        # end if
        
        # Check for jumps first (mandatory if available)
        jumps = []
        for row in range(self.board.size * 2):
            for col in range(self.board.size):
                if ((self.current_player == self.BLACK and self.board.is_black(row, col)) or
                    (self.current_player == self.WHITE and self.board.is_white(row, col))):
                    piece_jumps = self._get_valid_jumps(row, col)
                    if piece_jumps:
                        jumps.extend(piece_jumps)
                    # end if
                # end if
            # end for
        # end for
        
        if jumps and self.forced_jump:
            return jumps
        # end if
        
        # If no jumps or jumps are not mandatory, check for regular moves
        for row in range(self.board.size * 2):
            for col in range(self.board.size):
                if ((self.current_player == self.BLACK and self.board.is_black(row, col)) or
                    (self.current_player == self.WHITE and self.board.is_white(row, col))):
                    moves = self._get_valid_regular_moves(row, col)
                    valid_moves.extend(moves)
                # end if
            # end for
        # end for
        
        return jumps if (jumps and self.forced_jump) else valid_moves
    # end def get_valid_moves
    
    def _get_valid_regular_moves(self, row: int, col: int) -> List[Tuple[int, int]]:
        """
        Get valid regular moves for a piece at the specified position.
        
        Args:
            row (int): Row index
            col (int): Column index
        
        Returns:
            List[Tuple[int, int]]: List of valid move positions
        """
        valid_moves = []
        piece = self.board.get_piece(row, col)
        
        # Determine move directions based on piece type
        directions = []

        # Black moves down
        if piece == self.board.black or piece == self.board.black_king:
            directions.extend([(1, -1), (1, 1)])
        # end if

        # White moves down
        if piece == self.board.white or piece == self.board.white_king:
            directions.extend([(-1, -1), (-1, 1)])
        # end if

        # Is
        if self.board.is_king(row, col):
            directions = [(1, -1), (1, 1), (-1, -1), (-1, 1)]  # Kings move in all directions
        # end if
        
        # Check each direction
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if (0 <= new_row < self.board.size and 0 <= new_col < self.board.size and 
                self.board.get_piece(new_row, new_col) == self.board.empty):
                valid_moves.append((new_row, new_col))
            # end if
        # end for
        
        return valid_moves
    # end def _get_valid_regular_moves
    
    def _get_valid_jumps(self, row: int, col: int) -> List[Tuple[int, int]]:
        """
        Get valid jump moves for a piece at the specified position.
        
        Args:
            row (int): Row index
            col (int): Column index
        
        Returns:
            List[Tuple[int, int]]: List of valid jump positions
        """
        valid_jumps = []
        piece = self.board.get_piece(row, col)
        
        if piece == self.board.empty:
            return []
        
        # Determine jump directions based on piece type
        directions = []
        if piece == self.board.black or piece == self.board.black_king:
            directions.extend([(1, -1), (1, 1)])  # Black moves down
        if piece == self.board.white or piece == self.board.white_king:
            directions.extend([(-1, -1), (-1, 1)])  # White moves up
        if self.board.is_king(row, col):
            directions = [(1, -1), (1, 1), (-1, -1), (-1, 1)]  # Kings move in all directions
        
        # Check each direction for jumps
        for dr, dc in directions:
            jump_row, jump_col = row + 2*dr, col + 2*dc
            middle_row, middle_col = row + dr, col + dc
            
            if (0 <= jump_row < self.board.size and 0 <= jump_col < self.board.size and 
                self.board.get_piece(jump_row, jump_col) == self.board.empty):
                
                # Check if there's an opponent's piece to jump over
                if ((self.current_player == self.BLACK and self.board.is_white(middle_row, middle_col)) or
                    (self.current_player == self.WHITE and self.board.is_black(middle_row, middle_col))):
                    valid_jumps.append((jump_row, jump_col))
        
        return valid_jumps
    
    def is_valid_move(self, row: int, col: int) -> bool:
        """
        Check if placing a piece at (row, col) is a valid move for the current player.
        
        Args:
            row (int): Row index
            col (int): Column index
            
        Returns:
            bool: True if the move is valid, False otherwise
        """
        valid_moves = self.get_valid_moves()
        return (row, col) in valid_moves
    
    def make_move(self, row: int, col: int) -> bool:
        """
        Make a move at (row, col) for the current player.
        
        Args:
            row (int): Row index
            col (int): Column index
            
        Returns:
            bool: True if the move was successful, False if invalid
        """
        if self.game_over:
            return False
        
        if not self.is_valid_move(row, col):
            return False
        
        # Save the current board state
        self.board.push_history()
        
        # Get the source position
        source_pos = self._find_source_position(row, col)
        if source_pos is None:
            return False
        
        source_row, source_col = source_pos
        piece = self.board.get_piece(source_row, source_col)
        
        # Check if this is a jump move
        is_jump = abs(row - source_row) == 2
        
        # Move the piece
        self.board.set_piece(row, col, piece)
        self.board.set_piece(source_row, source_col, self.board.empty)
        
        # If it's a jump, remove the jumped piece
        if is_jump:
            middle_row = (row + source_row) // 2
            middle_col = (col + source_col) // 2
            self.board.set_piece(middle_row, middle_col, self.board.empty)
            
            # Check for multiple jumps
            self.current_jump_sequence = (row, col)
            if not self._get_valid_jumps(row, col):
                self.current_jump_sequence = None
        
        # Check for promotion to king
        self.board.promote_to_king(row, col)
        
        # Record the move
        move_notation = self.coords_to_notation(source_row, source_col) + "-" + self.coords_to_notation(row, col)
        self.moves.append(move_notation)
        self.moves_player.append(self.current_player)
        
        # If we're not in the middle of a jump sequence, switch players
        if self.current_jump_sequence is None:
            self.switch_player()
        
        # Check if the game is over
        self._check_game_over()
        
        return True
    
    def _find_source_position(self, target_row: int, target_col: int) -> Optional[Tuple[int, int]]:
        """
        Find the source position for a move to the target position.
        
        Args:
            target_row (int): Target row index
            target_col (int): Target column index
        
        Returns:
            Optional[Tuple[int, int]]: Source position (row, col) or None if not found
        """
        # If we're in the middle of a jump sequence, the source is the current position
        if self.current_jump_sequence is not None:
            return self.current_jump_sequence
        
        # Check for jump moves first
        for row in range(self.board.size):
            for col in range(self.board.size):
                if ((self.current_player == self.BLACK and self.board.is_black(row, col)) or
                    (self.current_player == self.WHITE and self.board.is_white(row, col))):
                    jumps = self._get_valid_jumps(row, col)
                    if (target_row, target_col) in jumps:
                        return (row, col)
        
        # If no jumps or jumps are not mandatory, check for regular moves
        for row in range(self.board.size):
            for col in range(self.board.size):
                if ((self.current_player == self.BLACK and self.board.is_black(row, col)) or
                    (self.current_player == self.WHITE and self.board.is_white(row, col))):
                    moves = self._get_valid_regular_moves(row, col)
                    if (target_row, target_col) in moves:
                        return (row, col)
        
        return None
    
    def make_random_move(self) -> Optional[Tuple[int, int]]:
        """
        Make a random valid move for the current player.
        
        Returns:
            Optional[Tuple[int, int]]: The (row, col) of the move made, or None if no move was possible
        """
        valid_moves = self.get_valid_moves()
        if not valid_moves:
            return None
        # end if
        
        move = random.choice(valid_moves)
        self.make_move(move[0], move[1])
        return move
    # end def make_random_move
    
    def has_valid_moves(self) -> bool:
        """
        Check if the current player has any valid moves.
        
        Returns:
            bool: True if the current player has at least one valid move, False otherwise
        """
        return len(self.get_valid_moves()) > 0
    
    def switch_player(self):
        """Switch the current player."""
        self.current_player = self.WHITE if self.current_player == self.BLACK else self.BLACK
    
    def _check_game_over(self):
        """Check if the game is over and update the game state accordingly."""
        # If the current player has no valid moves, the game is over
        if not self.has_valid_moves():
            self.game_over = True
            self.winner = self.BLACK if self.current_player == self.WHITE else self.WHITE
            return
        
        # Count pieces to check for a win
        black_count = 0
        white_count = 0
        
        for row in range(self.board.size):
            for col in range(self.board.size):
                piece = self.board.get_piece(row, col)
                if piece == self.board.black or piece == self.board.black_king:
                    black_count += 1
                elif piece == self.board.white or piece == self.board.white_king:
                    white_count += 1
        
        # If a player has no pieces left, the game is over
        if black_count == 0:
            self.game_over = True
            self.winner = self.WHITE
        elif white_count == 0:
            self.game_over = True
            self.winner = self.BLACK
    
    def is_game_over(self) -> bool:
        """
        Check if the game is over.
        
        Returns:
            bool: True if the game is over, False otherwise
        """
        return self.game_over
    
    def coords_to_notation(self, row: int, col: int) -> str:
        """
        Convert board coordinates to standard notation.
        
        Args:
            row (int): Row index
            col (int): Column index
            
        Returns:
            str: Position in standard notation
        """
        # Convert to algebraic notation (e.g., "a1", "b5", etc.)
        return f"{chr(97 + col)}{self.board.size - row}"
    
    def notation_to_coords(self, notation: str) -> Tuple[int, int]:
        """
        Convert standard notation to board coordinates.
        
        Args:
            notation (str): Position in standard notation
            
        Returns:
            Tuple[int, int]: (row, col) coordinates
        """
        if len(notation) != 2:
            raise ValueError(f"Invalid notation: {notation}")
        
        col = ord(notation[0].lower()) - 97
        row = self.board.size - int(notation[1])
        
        if not (0 <= row < self.board.size and 0 <= col < self.board.size):
            raise ValueError(f"Invalid notation: {notation}")
        
        return (row, col)
    
    def get_moves(self) -> List[str]:
        """
        Return the list of moves made in the game in standard notation.
        
        Returns:
            List[str]: List of moves in standard notation
        """
        return self.moves
    
    def set_moves(self, moves: List[str], moves_player: List[int] = None):
        """
        Set the list of moves for the game.
        
        Args:
            moves (List[str]): List of moves in standard notation
            moves_player (List[int], optional): List of players who made each move
        """
        self.__init__()  # Reset the game
        
        if moves_player is None:
            moves_player = []
            for i in range(len(moves)):
                moves_player.append(self.BLACK if i % 2 == 0 else self.WHITE)
        
        for i, move in enumerate(moves):
            # Parse the move notation (e.g., "a3-b4")
            parts = move.split("-")
            if len(parts) != 2:
                raise ValueError(f"Invalid move notation: {move}")
            
            from_coords = self.notation_to_coords(parts[0])
            to_coords = self.notation_to_coords(parts[1])
            
            # Set the current player
            self.current_player = moves_player[i]
            
            # Make the move
            if not self.make_move(to_coords[0], to_coords[1]):
                raise ValueError(f"Invalid move: {move}")

    def print(self):
        """
        Print the game state.
        """
        self.board.print()
    # end def print

    def show(self):
        """Display the current game state."""
        console.print(f"Current player: {'Black' if self.current_player == self.BLACK else 'White'}")
        console.print(self.board)
        
        if self.game_over:
            winner = "Black" if self.winner == self.BLACK else "White"
            console.print(f"Game over! {winner} wins!")
    
    def __str__(self):
        """
        Return a string representation of the game.
        
        Returns:
            str: String representation of the game
        """
        result = f"Current player: {'Black' if self.current_player == self.BLACK else 'White'}\n"
        result += str(self.board)
        
        if self.game_over:
            winner = "Black" if self.winner == self.BLACK else "White"
            result += f"\nGame over! {winner} wins!"
        # end if
        
        return result
    # end def __str__
    
    def __repr__(self):
        """
        Return a string representation of the game.
        
        Returns:
            str: String representation of the game
        """
        return self.__str__()
    # end __repr__

    def __len__(self):
        """
        Return the length of the game state.

        Returns:
            int: Length of the game state
        """
        return len(self.moves)
    # end __len__

# end class CheckersGame


def generate_checkers_game(seed: int = None, max_attempts: int = 100) -> CheckersGame:
    """
    Generate a random checkers game.
    
    Args:
        seed (int, optional): Random seed for reproducibility
        max_attempts (int): Maximum number of attempts to generate a valid game
    
    Returns:
        CheckersGame: A randomly generated checkers game
    """
    if seed is not None:
        random.seed(seed)
    
    game = CheckersGame()
    
    # Make random moves until the game is over or max_attempts is reached
    attempts = 0
    while not game.is_game_over() and attempts < max_attempts:
        if not game.make_random_move():
            break
        attempts += 1
    
    return game


def create_move_mapping() -> Dict[str, int]:
    """
    Create a mapping from move notation to move ID.
    
    Returns:
        Dict[str, int]: Mapping from move notation to move ID
    """
    mapping = {}
    move_id = 0
    
    # Generate all possible moves
    for from_row in range(8):
        for from_col in range(8):
            for to_row in range(8):
                for to_col in range(8):
                    # Skip invalid moves (only diagonal moves are valid in checkers)
                    if abs(from_row - to_row) != abs(from_col - to_col):
                        continue
                    
                    # Skip moves that are too far (regular moves are 1 step, jumps are 2 steps)
                    if abs(from_row - to_row) > 2:
                        continue
                    
                    from_notation = f"{chr(97 + from_col)}{8 - from_row}"
                    to_notation = f"{chr(97 + to_col)}{8 - to_row}"
                    move = f"{from_notation}-{to_notation}"
                    
                    mapping[move] = move_id
                    move_id += 1
    
    return mapping


def create_id_to_move_mapping() -> Dict[int, str]:
    """
    Create a mapping from move ID to move notation.
    
    Returns:
        Dict[int, str]: Mapping from move ID to move notation
    """
    move_to_id = create_move_mapping()
    return {v: k for k, v in move_to_id.items()}


def convert_ids_to_notation(game: List[int]) -> List[str]:
    """
    Convert a list of move IDs to move notation.
    
    Args:
        game (List[int]): List of move IDs
    
    Returns:
        List[str]: List of moves in standard notation
    """
    id_to_move = create_id_to_move_mapping()
    return [id_to_move[move_id] for move_id in game]


def load_games(input_file: str) -> List[List[int]]:
    """
    Load games from a file.
    
    Args:
        input_file (str): Path to the input file
    
    Returns:
        List[List[int]]: List of games, where each game is a list of move IDs
    """
    try:
        with open(input_file, "rb") as f:
            games = pickle.load(f)
        return games
    except Exception as e:
        error(f"Error loading games: {e}")
        return []


def extract_game_by_index(games: List[List[int]], index: int) -> List[int]:
    """
    Extract a game by index.
    
    Args:
        games (List[List[int]]): List of games
        index (int): Index of the game to extract
    
    Returns:
        List[int]: The extracted game
    """
    if 0 <= index < len(games):
        return games[index]
    else:
        error(f"Invalid game index: {index}")
        return []


def extract_games_by_length(games: List[List[int]], length: int) -> List[List[int]]:
    """
    Extract games by length.
    
    Args:
        games (List[List[int]]): List of games
        length (int): Length of games to extract
    
    Returns:
        List[List[int]]: List of games with the specified length
    """
    return [game for game in games if len(game) == length]
