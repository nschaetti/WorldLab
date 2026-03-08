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
Othello game simulator.

This module implements the Othello game rules and provides classes and functions
for simulating Othello games, validating moves, and manipulating game data.
"""

import random
import numpy as np
import pickle
from typing import List, Tuple, Set, Dict, Optional
from rich.columns import Columns
from boardGPT.games.game_interface import GameInterface


class OthelloBoard:
    """
    Mutable representation of an Othello board with helper utilities for move
    generation and history tracking.
    """

    def __init__(
            self,
            size: int = 8,
            empty: int = 0,
            black: int = 1,
            white: int = 2
    ):
        """
        Initialize an Othello board.
        
        Args:
            size (int): Size of the board (default: 8)
            empty (int): Value representing an empty cell (default: 0)
            black (int): Value representing a black piece (default: 1)
            white (int): Value representing a white piece (default: 2)
        """
        # Set storage
        self.board = [[empty for _ in range(size)] for _ in range(size)]

        # Set up the initial four pieces in the center in the standard Othello pattern
        self.board[3][3] = white  # d4 in standard notation
        self.board[3][4] = black  # e4 in standard notation
        self.board[4][3] = black  # d5 in standard notation
        self.board[4][4] = white  # e5 in standard notation

        # Possible values
        self.values = [empty, black, white]

        # Board history
        self.history: List[List[Tuple[int, int, int]]] = []
        self.current_history: List[Tuple[int, int, int]] = []
    # end __init__

    # Get flatten representation
    def get_flattened_board_state(self) -> np.array:
        """
        Get flattened board state.

        Returns:
        """
        return np.array(self.board, dtype=np.int32).flatten()
    # end get_flattened_board_state

    # Set pieces
    def set_piece(self, row: int, col: int, value: int) -> None:
        """
        Set a piece on the board.
        
        Args:
            row (int): Row index
            col (int): Column index
            value (int): Value to set (must be in self.values)
        """
        assert value in self.values, f"Expected {value} to be in {self.values}"
        self.board[row][col] = value
        self.current_history.append((row, col, value))
    # end def set_piece

    def get_piece(self, row: int, col: int) -> int:
        """
        Get the value of a piece on the board.
        
        Args:
            row (int): Row index
            col (int): Column index
            
        Returns:
            int: Value of the piece at the specified position
        """
        return self.board[row][col]
    # end def get_piece

    def push_history(self):
        """
        Push the current history to the history list and reset the current history.
        """
        self.history.append(self.current_history)
        self.current_history = []
    # end def push_history
    
    def __str__(self, last_move=None):
        """
        Return a string representation of the board.
        
        Args:
            last_move (tuple, optional): The coordinates (row, col) of the last move.
                                         If provided, this move will be marked on the board.
        
        Returns:
            str: A string representation of the board with | and - characters.
                 Black pieces are represented as 'X' and white pieces as 'O'.
                 The last move is marked with '*' on top of the square if provided.
        """
        # Create the horizontal line
        horizontal_line = "  " + "-" * (len(self.board) * 4 + 1) + "\n"
        
        # Start with column labels
        result = "  "
        for col in range(len(self.board)):
            result += f"  {chr(97 + col)} "  # 'a' through 'h' for standard 8x8 board
        # end for
        result += "\n"
        
        # Add the top border
        result += horizontal_line
        
        # Add each row with pieces
        for row in range(len(self.board)):
            # Add row label
            result += f"{row + 1} |"
            
            # Add pieces in the row
            for col in range(len(self.board[row])):
                piece = self.board[row][col]
                
                # Mark the last move with an asterisk
                if last_move and last_move == (row, col):
                    if piece == 0:  # Empty
                        result += " * |"
                    elif piece == 1:  # Black
                        result += " X*|"
                    elif piece == 2:  # White
                        result += " O*|"
                else:
                    if piece == 0:  # Empty
                        result += "   |"
                    elif piece == 1:  # Black
                        result += " X |"
                    elif piece == 2:  # White
                        result += " O |"
                    # end if
                # end if
            # end for
            
            # End the row and add horizontal line
            result += f" {row + 1}\n"
            result += horizontal_line
        # end for
        
        # Add column labels at the bottom
        result += "  "
        for col in range(len(self.board)):
            result += f"  {chr(97 + col)} "
        # end for
        
        return result
    # end __str__
    
    def __repr__(self):
        """
        Return a string representation of the board (same as __str__).
        
        Returns:
            str: A string representation of the board.
        """
        return self.__str__()
    # end __repr__

# end class OthelloBoard


class OthelloGame(GameInterface):
    """
    Represents an Othello game board and implements game rules.
    
    This class provides a complete implementation of the Othello game mechanics,
    including board representation, move validation, piece placement and flipping,
    player turn management, and game state evaluation.
    
    The board uses a 2D list to represent the 8x8 grid, with each cell containing
    one of three values: EMPTY (0), BLACK (1), or WHITE (2).
    
    Coordinates are zero-indexed, with (0,0) being the top-left corner of the board.
    Standard Othello notation is also supported, where 'a1' corresponds to (0,0).
    
    This class implements the GameInterface, providing a standard set of methods
    for interacting with the game.
    """

    # Board size (standard Othello board is 8x8)
    SIZE = 8

    # Player constants for board representation
    EMPTY = 0  # Empty cell
    BLACK = 1  # Black piece (typically goes first)
    WHITE = 2  # White piece

    # Eight directions for checking valid moves and flipping pieces
    # Format: (row_delta, column_delta)
    DIRECTIONS = [
        (-1, -1), (-1, 0), (-1, 1),  # North-west, North, North-east
        (0, -1),           (0, 1),   # West, East
        (1, -1),  (1, 0),  (1, 1)    # South-west, South, South-east
    ]

    def __init__(self):
        """
        Initialize a new Othello board with the standard starting position.
        
        Creates a 8x8 empty board and sets up the initial four pieces in the center:
        - White pieces at positions (3,3) and (4,4)
        - Black pieces at positions (3,4) and (4,3)
        
        Black player moves first, as per standard Othello rules.
        """
        # Create an empty 8x8 board filled with EMPTY (0) values
        self.board = OthelloBoard(self.SIZE, self.EMPTY)

        # Black moves first according to standard Othello rules
        self.current_player = self.BLACK

        # Keep track of moves made in standard notation (e.g., 'd3', 'e6')
        self.moves: List[str] = []
        self.moves_player: List[int] = []
    # end def __init__

    # region PUBLIC

    def flattened_board_state(self) -> np.array:
        """
        Flatten the board state into a 1D list with each cell containing
        """
        return self.board.get_flattened_board_state()
    # end def flattened_board_state

    def get_valid_moves(self) -> List[Tuple[int, int]]:
        """
        Return a list of valid moves for the current player.

        A valid move is one where placing a piece would flip at least one
        of the opponent's pieces. This method checks all board positions
        and returns coordinates of all valid moves.

        Returns:
            List[Tuple[int, int]]: A list of (row, col) tuples representing valid move positions
        """
        # Initialize an empty list to store valid moves
        valid_moves = []

        # Check every position on the board
        for row in range(self.SIZE):
            for col in range(self.SIZE):
                # If the move is valid, add it to the list
                if self.is_valid_move(row, col):
                    valid_moves.append((row, col))  # end if
                # end if
            # end for
        # end for

        return valid_moves  # end def get_valid_moves
    # end get_valid_moves

    def is_valid_move(self, row: int, col: int) -> bool:
        """
        Check if placing a piece at (row, col) is a valid move for the current player.

        A valid move in Othello must satisfy two conditions:
        1. The cell must be empty
        2. The move must flip at least one opponent's piece

        To flip an opponent's piece, there must be a straight line (horizontal,
        vertical, or diagonal) of opponent's pieces, with the current player's
        piece at the other end.

        Args:
            row (int): Row index (0-7)
            col (int): Column index (0-7)

        Returns:
            bool: True if the move is valid, False otherwise
        """
        # Rule 1: The cell must be empty
        if self.board.get_piece(row, col) != self.EMPTY:
            return False  # end if
        # end if

        # Rule 2: The move must flip at least one opponent's piece
        # Determine the opponent's piece color
        opponent = self.WHITE if self.current_player == self.BLACK else self.BLACK

        # Check in all eight directions
        for dr, dc in self.DIRECTIONS:
            # Move one step in the current direction
            r, c = row + dr, col + dc

            # Check if we have at least one opponent's piece in this direction
            if 0 <= r < self.SIZE and 0 <= c < self.SIZE and self.board.get_piece(r, c) == opponent:

                # Continue in this direction looking for our own piece
                r += dr
                c += dc
                while 0 <= r < self.SIZE and 0 <= c < self.SIZE:
                    if self.board.get_piece(r, c) == self.EMPTY:
                        # Empty cell, no flip possible in this direction
                        break
                    # end if

                    if self.board.get_piece(r, c) == self.current_player:
                        # Found our own piece at the other end, this is a valid move
                        return True
                    # end if

                    # Continue in this direction
                    r += dr
                    c += dc
                # end while
            # end if
        # end for

        # No valid flips found in any direction
        return False
    # end is_valid_move

    def make_move(self, row: int, col: int) -> bool:
        """
        Make a move at (row, col) for the current player.
        
        This method:
        1. Validates the move
        2. Places the player's piece at the specified position
        3. Flips all opponent's pieces that are captured by this move
        4. Records the move in standard notation
        5. Switches to the other player
        
        Args:
            row (int): Row index (0-7)
            col (int): Column index (0-7)
            
        Returns:
            bool: True if the move was successful, False if invalid
        """
        # First check if the move is valid
        if not self.is_valid_move(row, col):
            return False  # end if
        # end if

        # Place the current player's piece at the specified position
        self.board.set_piece(row, col, self.current_player)

        # Determine the opponent's piece color for flipping
        opponent = self.WHITE if self.current_player == self.BLACK else self.BLACK

        # Check in all eight directions for pieces to flip
        for dr, dc in self.DIRECTIONS:
            # List to store coordinates of pieces to flip in this direction
            to_flip = []

            # Move one step in the current direction
            r, c = row + dr, col + dc
            
            # Collect all opponent's pieces in this direction
            while 0 <= r < self.SIZE and 0 <= c < self.SIZE and self.board.get_piece(r, c) == opponent:
                to_flip.append((r, c))
                r += dr
                c += dc

                # If we reach the edge or an empty cell, no flip in this direction
                if not (0 <= r < self.SIZE and 0 <= c < self.SIZE) or self.board.get_piece(r, c) == self.EMPTY:
                    to_flip = []  # Clear the list as we can't flip these pieces
                    break
                # end if

                # If we reach our own piece, we can flip all pieces in between
                if self.board.get_piece(r, c) == self.current_player:
                    break
                # end if
            # end while

            # Flip all opponent's pieces that were captured
            for flip_r, flip_c in to_flip:
                self.board.set_piece(flip_r, flip_c, self.current_player)  # end for
            # end for
        # end for

        # Record the move in standard notation (e.g., 'e4')
        move_notation = self.coords_to_notation(row, col)

        # Add move and player
        self.moves.append(move_notation)
        self.moves_player.append(self.current_player)

        # Push board modification into history
        self.board.push_history()

        # Switch to the other player for the next turn
        self.switch_player()

        return True  # end def make_move
    # end make_move

    def make_random_move(self) -> Optional[Tuple[int, int]]:
        """
        Make a random valid move for the current player.
        
        If the current player has no valid moves, switches to the other player.
        If neither player has valid moves (game is over), returns None.
        
        Returns:
            Optional[Tuple[int, int]]: The (row, col) of the move made, or None if no move was possible
        """
        # Get all valid moves for the current player
        valid_moves = self.get_valid_moves()
        if not valid_moves:
            # No valid moves for current player, switch player
            self.switch_player()

            # Check if the other player also has no valid moves (game is over)
            if not self.has_valid_moves():
                return None
            # end if

            return self.make_random_move()
        # end if

        # Choose a random valid move from the available options
        row, col = random.choice(valid_moves)
        self.make_move(row, col)
        return row, col
    # end def make_random_move

    def has_valid_moves(self) -> bool:
        """
        Check if the current player has any valid moves.
        
        Returns:
            bool: True if the current player has at least one valid move, False otherwise
        """
        # Use the get_valid_moves method and check if the list is non-empty
        return len(self.get_valid_moves()) > 0  # end def has_valid_moves
    # end has_valid_moves

    def switch_player(self) -> None:
        """
        Switch to the other player.
        
        Changes the current_player from BLACK to WHITE or from WHITE to BLACK.
        """
        # Toggle between BLACK and WHITE
        self.current_player = self.WHITE if self.current_player == self.BLACK else self.BLACK  # end def switch_player
    # end switch_player

    def is_game_over(self) -> bool:
        """
        Check if the game is over.
        
        The game is over when:
        1. The board is full, or
        2. Both players pass consecutively (neither player has any valid moves)
        
        Returns:
            bool: True if the game is over, False otherwise
        """
        # Check if the board is full
        is_full = True
        for row in range(self.SIZE):
            for col in range(self.SIZE):
                if self.board.get_piece(row, col) == self.EMPTY:
                    is_full = False
                    break
            if not is_full:
                break
            # end if
        if is_full:
            return True
        # end if
        
        # Check if both players have no valid moves
        # First check if current player has valid moves
        if self.has_valid_moves():
            return False
        # end if

        # If current player has no moves, check if the opponent has moves
        # Temporarily switch player to check
        self.switch_player()
        has_moves = self.has_valid_moves()
        self.switch_player()  # Switch back to original player
        
        # Game is over if neither player has valid moves
        return not has_moves  # end def is_game_over
    # end is_game_over

    def coords_to_notation(self, row: int, col: int) -> str:
        """
        Convert board coordinates to standard Othello notation.
        
        In standard notation:
        - Columns are labeled a-h from left to right
        - Rows are labeled 1-8 from top to bottom
        
        Args:
            row (int): Row index (0-7)
            col (int): Column index (0-7)
            
        Returns:
            str: Position in standard notation (e.g., 'e4' for row=3, col=4)
        """
        # Convert column index to letter (a-h) and row index to number (1-8)
        return chr(97 + col) + str(row + 1)  # end def coords_to_notation
    # end coords_to_notation

    def notation_to_coords(self, notation: str) -> Tuple[int, int]:
        """
        Convert standard Othello notation to board coordinates.
        
        Args:
            notation (str): Position in standard notation (e.g., 'e4')
            
        Returns:
            Tuple[int, int]: (row, col) coordinates (e.g., (3, 4) for 'e4')
            
        Raises:
            ValueError: If the notation is not in the expected format (letter followed by number)
        """
        # Check if the notation is valid (a letter followed by a number)
        if len(notation) != 2 or not notation[0].isalpha() or not notation[1].isdigit():
            raise ValueError(f"Invalid move notation: '{notation}'. Expected format is a letter (a-h) followed by a number (1-8).")
        # end if
        
        # Convert letter (a-h) to column index (0-7) and number (1-8) to row index (0-7)
        col = ord(notation[0].lower()) - 97  # 'a' is ASCII 97, so 'a' -> 0, 'b' -> 1, etc.
        row = int(notation[1]) - 1   # '1' -> 0, '2' -> 1, etc.
        
        # Check if the coordinates are within the board boundaries
        if not (0 <= col < self.SIZE and 0 <= row < self.SIZE):
            raise ValueError(f"Invalid move notation: '{notation}'. Coordinates out of bounds.")
        # end if
        return row, col  # end def notation_to_coords
    # end notation_to_coords

    def get_moves(self) -> List[str]:
        """
        Return the list of moves made in the game in standard notation.
        
        Returns:
            List[str]: List of moves in standard notation (e.g., ['d3', 'c4', 'e3'])
        """
        # Return the stored list of moves
        return self.moves
    # end get_moves_notation

    def set_moves(
            self,
            moves: List[str],
            moves_player: List[int],  # end def set_moves
    ) -> None:
        """
        Set the list of moves and players.
        
        Args:
            moves (List[str]): List of moves in standard notation
            moves_player (List[int]): List of players for each move
        """
        self.moves = moves
        self.moves_player = moves_player
    # end set_moves
    
    def validate_game(self) -> Tuple[bool, str]:
        """
        Validate the current game according to Othello rules.
        
        Checks:
        - The number of moves must equal the number of players, and game length must be between 1 and 60
        - Every move must correspond to a valid square on the Othello board (from a1 to h8)
        - No square can be played more than once during the game
        - Each player entry must be either 1 (black) or 2 (white)
        - At each step, the move played must be a legal move for the current player
        - After each move, the chosen square must belong to the current player
        - After t moves, the total number of discs on the board must equal 4 + t
        - Turns should normally alternate between black and white players
        - If the same player moves twice in a row, this must correspond to a valid pass
        - If the game ends at 60 moves, the final board must be completely full
        - If the game ends before 60 moves, it must be because both players had no legal moves
        - At all times, every square on the board must be either empty, black, or white
        
        Returns:
            Tuple[bool, str]: (is_valid, error_message) - True if valid, False with error message if invalid
        """
        # Check if the game has moves
        if not self.moves:
            return False, "Game has no moves"
        # end if
        # Check game length is between 1 and 60
        if len(self.moves) < 1 or len(self.moves) > 60:
            return False, f"Game length {len(self.moves)} is not between 1 and 60"
        # end if
        # Check the number of moves equals the number of players
        if len(self.moves) != len(self.moves_player):
            return False, "Number of moves does not equal number of player entries"
        # end if
        # Check each player entry is either 1 (black) or 2 (white)
        for i, player in enumerate(self.moves_player):
            if player != self.BLACK and player != self.WHITE:
                return False, f"Invalid player value {player} at move {i+1}"
            # end if
        # Create a new board to replay and validate the game
        board = OthelloGame()
        played_squares = set()
        
        # Track the current player
        current_player = self.BLACK  # Black starts in Othello
        
        for i, move in enumerate(self.moves):
            # Check move corresponds to a valid square (a1 to h8)
            if not (len(move) == 2 and 'a' <= move[0] <= 'h' and '1' <= move[1] <= '8'):
                return False, f"Move {i+1} '{move}' is not a valid square (a1-h8)"
            # end if
            # Convert notation to coordinates
            row, col = board.notation_to_coords(move)
            
            # Check no square is played more than once
            square = (row, col)
            if square in played_squares:
                return False, f"Square {move} at move {i+1} was already played"  # end if
            played_squares.add(square)
            
            # Check the player matches the expected player
            player = self.moves_player[i]
            if player != current_player:
                # If player doesn't match, check if it's a valid pass situation
                valid_moves = []
                for r in range(self.SIZE):
                    for c in range(self.SIZE):
                        if board.is_valid_move(r, c):
                            valid_moves.append((r, c))
                        # end if
                # If current player had valid moves, this is an invalid pass
                if valid_moves:
                    return False, f"Player {current_player} had valid moves but player {player} moved at move {i+1}"
                # end if
                # Switch to the other player
                current_player = self.WHITE if current_player == self.BLACK else self.BLACK
                
                # Check if the new player matches
                if player != current_player:
                    return False, f"Invalid player sequence at move {i+1}"
                # end if
            # Check if the move is legal for the current player
            board.current_player = player
            if not board.is_valid_move(row, col):
                return False, f"Move {i+1} '{move}' is not a legal move for player {player}"
            # end if
            # Make the move
            board.make_move(row, col)
            
            # Check that after the move, the chosen square belongs to the current player
            if board.board.get_piece(row, col) != player:
                return False, f"After move {i+1}, square {move} does not belong to player {player}"
            # end if
            # Count total pieces on the board
            total_pieces = 0
            for r in range(self.SIZE):
                for c in range(self.SIZE):
                    piece = board.board.get_piece(r, c)
                    if piece != self.EMPTY:
                        total_pieces += 1  # end if
                    # Check every square is either empty, black, or white
                    if piece not in [self.EMPTY, self.BLACK, self.WHITE]:
                        return False, f"Invalid piece value {piece} at position ({r},{c})"
                    # end if
            # Check total pieces equals 4 + number of moves
            if total_pieces != 4 + (i + 1):
                return False, f"After move {i+1}, total pieces {total_pieces} does not equal {4 + (i + 1)}"
            # end if
            # Update current player for next move
            current_player = self.WHITE if player == self.BLACK else self.BLACK
        # end for
        # Check end game conditions
        if len(self.moves) == 60:
            # If game ends at 60 moves, the board must be completely full
            empty_squares = 0
            for r in range(self.SIZE):
                for c in range(self.SIZE):
                    if board.board.get_piece(r, c) == self.EMPTY:
                        empty_squares += 1
                    # end if
            if empty_squares > 0:
                return False, f"Game ended at 60 moves but board has {empty_squares} empty squares"  # end if  # end if
        else:
            # If game ends before 60 moves, it must be because both players had no legal moves
            black_has_moves = False
            white_has_moves = False
            
            # Check if black has moves
            board.current_player = self.BLACK
            for r in range(self.SIZE):
                for c in range(self.SIZE):
                    if board.is_valid_move(r, c):
                        black_has_moves = True
                        break  # end if  # end for
                if black_has_moves:
                    break
                # end if
            # Check if white has moves
            board.current_player = self.WHITE
            for r in range(self.SIZE):
                for c in range(self.SIZE):
                    if board.is_valid_move(r, c):
                        white_has_moves = True
                        break  # end if  # end for
                if white_has_moves:
                    break
                # end if
            if black_has_moves or white_has_moves:
                return False, f"Game ended at {len(self.moves)} moves but at least one player still has valid moves"
            # end if
        # All validation checks passed
        return True, "Game is valid"
    # end def validate_game
    # endregion PUBLIC

    def __len__(self):
        """
        Return the number of moves made in the game in standard notation.
        """
        return len(self.moves)  # end def __len__
    # end def __len__

    @staticmethod
    def load_moves(moves: List[str]) -> 'OthelloGame':
        """
        Load a sequence of moves and create an Othello game board.
        
        Args:
            moves (List[str]): List of moves in standard notation (e.g., ['c4', 'd3', ...])
            
        Returns:
            OthelloGame: A game board with the moves applied
        """
        from boardGPT.utils.logging import warning

        # Instance class
        board = OthelloGame()
        
        # Keep track of valid moves
        valid_moves = []

        # For each move
        for m in moves:
            try:
                # Transform to coordinates
                row, col = board.notation_to_coords(m)

                # Check if the move is valid
                if board.is_valid_move(row, col):
                    # Make the move
                    board.make_move(row, col)
                    valid_moves.append(m)
                else:
                    # Switch player
                    board.switch_player()

                    # Check that the move is valid for the other player
                    if board.is_valid_move(row, col):
                        # Make the move
                        board.make_move(row, col)
                        valid_moves.append(m)
                    else:
                        warning(f"Skipping invalid move: {m} (not valid for either player)")  # end else
                    # end if
                # end if
            except ValueError as e:
                # Skip invalid move notations
                warning(f"{str(e)}")
            # end try
        # end for

        # Update the move list to include only valid moves
        board.moves = valid_moves
        
        return board
    # end load_moves

    def show(self):
        """
        Pretty-print the sequence of board states for the recorded moves.

        Uses the project logging console to render the initial setup followed by
        each intermediate position annotated with the move number and player.
        """
        from boardGPT.utils.logging import console

        if not self.moves:
            # If no moves have been made, just show the initial board
            console.print(self.board)
        # end if

        # Create a new game and replay all moves to show the board after each move
        column_items = []
        replay_game = OthelloGame()

        # Add the initial board
        initial_board = ["\nInitial board:\n", str(replay_game.board)]
        column_items.append("\n".join(initial_board))

        # Replay each move and show the board after each move
        for i, move in enumerate(self.moves):
            # Get the player who made this move
            player = self.moves_player[i]
            player_name = "Black" if player == self.BLACK else "White"

            # Set the current player in the replay game
            replay_game.current_player = player

            # Convert move notation to coordinates
            row, col = replay_game.notation_to_coords(move)

            # Make the move
            replay_game.make_move(row, col)

            # Create a string with the move information and board state
            move_info = f"\nMove {i + 1}: {player_name} plays {move}\n"
            board_str = replay_game.board.__str__(last_move=(row, col))
            column_items.append("\n".join([move_info, board_str]))
        # end for

        # Create a Columns object with the board representations
        columns = Columns(column_items)

        # Convert the Columns object to a string and return it
        console.print(columns)
    # end show
    
    def __str__(self):
        """
        Return a string representation of the game.
        
        Shows only the board after the last move, with the last move displayed on top of the square.
        
        Returns:
            str: A string representation of the game.
        """
        if not self.moves:
            # If no moves have been made, just show the initial board
            return str(self.board)
        # end if

        # Create a new game and replay all moves to get to the final state
        replay_game = OthelloGame()

        # Replay each move to reach the final state
        for i, move in enumerate(self.moves):
            # Get the player who made this move
            player = self.moves_player[i]

            # Set the current player in the replay game
            replay_game.current_player = player
            
            # Convert move notation to coordinates
            row, col = replay_game.notation_to_coords(move)
            
            # Make the move
            replay_game.make_move(row, col)
        # end for
        
        # Get the last move coordinates to mark on the board
        last_move_row, last_move_col = replay_game.notation_to_coords(self.moves[-1])
        
        # Return the string representation of the final board state with the last move marked
        return replay_game.board.__str__(last_move=(last_move_row, last_move_col))
    # end def __str__
    
    def __repr__(self):
        """
        Return a string representation of the game (same as __str__).
        
        Returns:
            str: A string representation of the game.
        """
        return self.__str__()
    # end __repr__

# end class OthelloGame


def generate_othello_game(
        seed: int = None,
        max_attempts: int = 100
) -> List[str]:
    """
    Generate a single valid Othello game and return the list of moves.
    
    This function simulates an Othello game by making random valid moves
    until either the game is over (no player has valid moves), or the maximum
    number of moves is reached. It validates the generated game against all
    Othello rules and retries if the game is invalid.
    
    Args:
        seed (int, optional): Random seed for reproducibility. If None, no seed is set.
        max_attempts (int): Maximum number of attempts to generate a valid game (default: 100)
    
    Returns:
        List[str]: List of moves in standard notation (e.g., ['d3', 'c4', 'e3'])
    """
    from boardGPT.utils.logging import warning

    # Set random seed if provided
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)  # end if
    # end if
    
    attempt = 0
    while attempt < max_attempts:
        # Create a new Othello board with standard starting position
        board = OthelloGame()

        # Continue making valid moves
        move_done = False
        while not move_done:
            # Make a random move
            rnd_move = board.make_random_move()
            move_done = rnd_move is None
        # end while

        # Validate the generated game
        is_valid, error_message = board.validate_game()
        if is_valid:
            # Return the list of moves made during the game
            return board.get_moves()
        else:
            warning("Invalid Othello game! Retrying...")
            # Try again with a different seed
            attempt += 1
            if seed is not None:
                # Use a different seed for each attempt
                random.seed(seed + attempt)
                np.random.seed(seed + attempt)  # end if
            # end if
        # end if
    # end while
    
    # If we've reached the maximum number of attempts, raise an exception
    raise RuntimeError(f"Failed to generate a valid Othello game after {max_attempts} attempts")  # end def generate_game
# end generate_game


# Create a mapping from move notation to ID
# The vocabulary size is 61: BOS token (0) + 60 possible moves (1-60)
def create_move_mapping() -> Dict[str, int]:
    """
    Create a mapping from move notation to ID.
    
    The mapping includes:
    - BOS (Beginning of Sequence) token with ID 0
    - All possible moves on an 8x8 board with IDs 1-60
    
    Returns:
        Dict[str, int]: Dictionary mapping move notation to ID
    """
    # Create a dictionary to store the mapping
    move_to_id = {"BOS": 0}  # BOS token has ID 0
    
    # Generate all possible move notations (a1-h8)
    id_counter = 1
    for col in range(8):  # a-h
        for row in range(8):  # 1-8
            # Move not possible on the centered square
            if (3 <= col <= 4) and (3 <= row <= 4):
                continue  # end if
            # end if
            notation = chr(97 + col) + str(row + 1)
            move_to_id[notation] = id_counter
            id_counter += 1
        # end for
    # end for
    
    return move_to_id  # end def create_move_mapping
# end create_move_mapping


def create_id_to_move_mapping() -> Dict[int, str]:
    """
    Create a mapping from ID to move notation.
    
    The mapping includes:
    - BOS (Beginning of Sequence) token with ID 0
    - All possible moves on an 8x8 board with IDs 1-60
    
    Returns:
        Dict[int, str]: Dictionary mapping ID to move notation
    """
    # Get the move_to_id mapping
    move_to_id = create_move_mapping()
    
    # Create the reverse mapping
    id_to_move = {id: move for move, id in move_to_id.items()}
    
    return id_to_move  # end def create_id_to_move_mapping
# end create_id_to_move_mapping


def convert_ids_to_notation(game: List[int]) -> List[str]:
    """
    Convert a game represented as a list of move IDs to a list of move notations.
    
    Args:
        game (List[int]): Game as a list of move IDs
        
    Returns:
        List[str]: Game as a list of move notations
    """
    # Get the id_to_move mapping
    id_to_move = create_id_to_move_mapping()
    
    # Convert move IDs to move notations, skipping the BOS token (ID 0) if present
    return [id_to_move[move_id] for move_id in game if move_id != 0]  # end def convert_ids_to_notation
# end convert_ids_to_notation


def load_games(input_file: str) -> List[List[str]]:
    """
    Load games from a binary file.
    
    Handles all formats:
    - Original format (lists of integers)
    - Numpy uint8 arrays
    - Python array.array('B') objects
    
    Args:
        input_file (str): Path to the input file
        
    Returns:
        List[List[str]]: List of games as moves
    """
    with open(input_file, 'rb') as f:
        game_sequences = pickle.load(f)  # end with
    # end with
    
    # Convert arrays back to lists for compatibility with existing code
    result = []
    for game in game_sequences:
        if isinstance(game, np.ndarray):
            # Convert numpy array to regular Python list
            result.append(game.tolist())  # end if
        elif hasattr(game, 'typecode') and game.typecode == 'B':
            # Convert array.array to regular Python list
            result.append(list(game))  # end elif
        else:
            # Already a list, no conversion needed
            result.append(game)  # end else
        # end if
    # end for
    
    return result  # end def load_games
# end def load_games


def extract_game_by_index(games: List[List[int]], index: int) -> List[int]:
    """
    Extract a game by its index.
    
    Args:
        games (List[List[int]]): List of games
        index (int): Index of the game to extract
        
    Returns:
        List[int]: The extracted game as a list of move IDs
    """
    if index < 0 or index >= len(games):
        raise ValueError(f"Index {index} out of range. There are {len(games)} games.")  # end if
    return games[index]  # end def extract_game_by_index
# end def extract_game_by_index


def extract_games_by_length(games: List[List[int]], length: int) -> List[Tuple[int, List[int]]]:
    """
    Extract games with a specific length.
    
    Args:
        games (List[List[int]]): List of games
        length (int): Length of games to extract (including BOS token)
        
    Returns:
        List[Tuple[int, List[int]]]: List of tuples (index, game) with the specified length
    """
    return [(i, game) for i, game in enumerate(games) if len(game) - 1 == length]  # end def extract_games_by_length
# end def extract_games_by_length
