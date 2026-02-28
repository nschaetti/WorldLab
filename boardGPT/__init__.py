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
from typing import Union, List, Tuple, Optional, Any, Dict
import random
import numpy as np

from .games.othello.othello_simulator import OthelloGame
from .games.othello.othello_utils import verify_game as verify_othello_game
try:
    from .games.checkers.checkers_simulator import CheckersGame
    from .games.checkers.checkers_utils import verify_game as verify_checkers_game
except ModuleNotFoundError:
    CheckersGame = None
    verify_checkers_game = None
from .games.game_interface import GameInterface


def othello(moves_str: Union[str, List[str]]) -> OthelloGame:
    """
    Create an OthelloGame object from a string of moves.
    
    Args:
        moves_str (Union[str, List[str]]): list of moves as a string or list of strings
    
    Returns:
        OthelloGame: An OthelloGame object with the moves applied
        
    Raises:
        ValueError: If the move sequence is not possible (contains invalid moves)
    """
    # Parse the input string into a list of moves
    if isinstance(moves_str, str):
        if ',' in moves_str:
            # Handle comma-separated moves
            moves = [move.strip() for move in moves_str.split(',')]
        else:
            # Handle space-separated moves
            moves = [move.strip() for move in moves_str.split()]
        # end if
    else:
        moves = moves_str
    # end if
    
    # Remove any empty moves
    moves = [move for move in moves if move]
    
    # Verify that the move sequence is valid
    is_valid, invalid_moves = verify_othello_game(moves)
    
    if not is_valid:
        raise ValueError(f"Invalid move sequence. The following moves are invalid: {', '.join(invalid_moves)}")
    # end if
    
    # Create an OthelloGame object with the moves applied
    game = OthelloGame.load_moves(moves)
    
    return game
# end othello


def checkers(moves_str: Union[str, List[str]]) -> Any:
    """
    Create a CheckersGame object from a string of moves.
    
    Args:
        moves_str (Union[str, List[str]]): list of moves as a string or list of strings
    
    Returns:
        CheckersGame: A CheckersGame object with the moves applied
        
    Raises:
        ValueError: If the move sequence is not possible (contains invalid moves)
    """
    # Parse the input string into a list of moves
    if isinstance(moves_str, str):
        if ',' in moves_str:
            # Handle comma-separated moves
            moves = [move.strip() for move in moves_str.split(',')]
        else:
            # Handle space-separated moves
            moves = [move.strip() for move in moves_str.split()]
        # end if
    else:
        moves = moves_str
    # end if
    
    # Remove any empty moves
    moves = [move for move in moves if move]
    
    # Verify that the move sequence is valid
    if CheckersGame is None or verify_checkers_game is None:
        raise ModuleNotFoundError(
            "Checkers dependencies are not available in this environment. "
            "Install optional visualization dependencies to enable checkers APIs."
        )

    is_valid, invalid_moves = verify_checkers_game(moves)
    
    if not is_valid:
        raise ValueError(f"Invalid move sequence. The following moves are invalid: {', '.join(invalid_moves)}")
    # end if
    
    # Create a CheckersGame object with the moves applied
    game = CheckersGame()
    game.set_moves(moves)
    
    return game
# end checkers


def valid(game: GameInterface, move: Union[str, Tuple[int, int]]) -> bool:
    """
    Check if a move is valid for the given game.
    
    Args:
        game (GameInterface): A game object implementing the GameInterface
        move (Union[str, Tuple[int, int]]): Move in standard notation (e.g., 'e4') or as coordinates (row, col)
    
    Returns:
        bool: True if the move is valid, False otherwise
    """
    # Convert move to coordinates if it's in notation format
    if isinstance(move, str):
        row, col = game.notation_to_coords(move)
    else:
        row, col = move
    
    return game.is_valid_move(row, col)
# end valid


def next(game: GameInterface, move: Union[str, Tuple[int, int]]) -> GameInterface:
    """
    Apply the move to the game and return the same game object.
    
    Args:
        game (GameInterface): A game object implementing the GameInterface
        move (Union[str, Tuple[int, int]]): Move in standard notation (e.g., 'e4') or as coordinates (row, col)
    
    Returns:
        GameInterface: The same game object with the move applied
        
    Raises:
        ValueError: If the move is invalid
    """
    # Convert move to coordinates if it's in notation format
    if isinstance(move, str):
        row, col = game.notation_to_coords(move)
    else:
        row, col = move
    
    # Make the move
    success = game.make_move(row, col)
    
    if not success:
        raise ValueError(f"Invalid move: {move}")
    
    return game
# end next


def rnext(game: GameInterface) -> GameInterface:
    """
    Apply a random valid move to the game and return the same game object.
    
    Args:
        game (GameInterface): A game object implementing the GameInterface
    
    Returns:
        GameInterface: The same game object with a random move applied
        
    Raises:
        ValueError: If there are no valid moves
    """
    move = game.make_random_move()
    
    if move is None:
        raise ValueError("No valid moves available")
    
    return game
# end rnext


def valid_moves(game: GameInterface) -> List[Tuple[int, int]]:
    """
    Get all valid moves for the current player.
    
    Args:
        game (GameInterface): A game object implementing the GameInterface
    
    Returns:
        List[Tuple[int, int]]: A list of (row, col) tuples representing valid move positions
    """
    return game.get_valid_moves()
# end valid_moves


def valid_moves_notation(game: GameInterface) -> List[str]:
    """
    Get all valid moves for the current player in standard notation.
    
    Args:
        game (GameInterface): A game object implementing the GameInterface
    
    Returns:
        List[str]: A list of moves in standard notation
    """
    valid_moves_coords = game.get_valid_moves()
    return [game.coords_to_notation(row, col) for row, col in valid_moves_coords]
# end valid_moves_notation


def is_over(game: GameInterface) -> bool:
    """
    Check if the game is over.
    
    Args:
        game (GameInterface): A game object implementing the GameInterface
    
    Returns:
        bool: True if the game is over, False otherwise
    """
    return game.is_game_over()
# end is_over


def show(game: GameInterface) -> None:
    """
    Display the current game state.
    
    Args:
        game (GameInterface): A game object implementing the GameInterface
    """
    game.show()
# end show


def has_moves(game: GameInterface) -> bool:
    """
    Check if the current player has any valid moves.
    
    Args:
        game (GameInterface): A game object implementing the GameInterface
    
    Returns:
        bool: True if the current player has at least one valid move, False otherwise
    """
    return game.has_valid_moves()
# end has_moves


def get_moves(game: GameInterface) -> List[str]:
    """
    Get the list of moves made in the game in standard notation.
    
    Args:
        game (GameInterface): A game object implementing the GameInterface
    
    Returns:
        List[str]: List of moves in standard notation
    """
    return game.get_moves()
# end get_moves


def notation_to_moves(game: GameInterface, notation: Union[str, List[str]]) -> Union[Tuple[int, int], List[Tuple[int, int]]]:
    """
    Convert standard notation to move coordinates.
    
    Args:
        game (GameInterface): A game object implementing the GameInterface
        notation (Union[str, List[str]]): Move(s) in standard notation (e.g., 'e4' or ['e4', 'd3'])
    
    Returns:
        Union[Tuple[int, int], List[Tuple[int, int]]]: Move coordinates as (row, col) or list of coordinates
        
    Raises:
        ValueError: If the notation is invalid
    """
    # Handle single notation
    if isinstance(notation, str):
        return game.notation_to_coords(notation)
    
    # Handle list of notations
    return [game.notation_to_coords(move) for move in notation]
# end notation_to_moves


def moves_to_notation(game: GameInterface, moves: Union[Tuple[int, int], List[Tuple[int, int]]]) -> Union[str, List[str]]:
    """
    Convert move coordinates to standard notation.
    
    Args:
        game (GameInterface): A game object implementing the GameInterface
        moves (Union[Tuple[int, int], List[Tuple[int, int]]]): Move coordinates as (row, col) or list of coordinates
    
    Returns:
        Union[str, List[str]]: Move(s) in standard notation (e.g., 'e4' or ['e4', 'd3'])
        
    Raises:
        ValueError: If the coordinates are invalid
    """
    # Handle single move
    if isinstance(moves, tuple) and len(moves) == 2:
        row, col = moves
        return game.coords_to_notation(row, col)
    
    # Handle list of moves
    return [game.coords_to_notation(row, col) for row, col in moves]
# end moves_to_notation


def generate_game(
    game: str = "othello",
    starting_moves: Optional[Union[str, List[str]]] = None,
    full_game: bool = True,
    max_length: Optional[int] = None,
    seed: Optional[int] = None,
    max_attempts: int = 100
) -> GameInterface:
    """
    Generate a board game with optional parameters.
    
    Args:
        game (str): Type of game to generate (default: "othello")
        starting_moves (Optional[Union[str, List[str]]]): Optional starting moves in standard notation
        full_game (bool): Whether to generate a full game (default: True)
        max_length (Optional[int]): Maximum number of moves to generate (ignored if full_game is True)
        seed (Optional[int]): Random seed for reproducibility
        max_attempts (int): Maximum number of attempts to generate a valid game
    
    Returns:
        GameInterface: A game object implementing the GameInterface
        
    Raises:
        ValueError: If the starting moves are invalid or if the game type is not supported
        RuntimeError: If failed to generate a valid game after max_attempts
    """
    # Map of game types to their respective game classes
    from boardGPT.games.othello.othello_simulator import OthelloGame
    
    game_classes: Dict[str, Any] = {
        "othello": OthelloGame,
    }

    if CheckersGame is not None:
        game_classes["checkers"] = CheckersGame
    
    # Check if the game type is supported
    if game not in game_classes:
        raise ValueError(f"Unsupported game type: {game}. Supported types are: {', '.join(game_classes.keys())}")
    # end if
    
    # Get the game class for the specified type
    GameClass = game_classes[game]
    
    # Parse starting moves if provided
    if starting_moves is not None:
        if isinstance(starting_moves, str):
            if ',' in starting_moves:
                # Handle comma-separated moves
                moves = [move.strip() for move in starting_moves.split(',')]
            else:
                # Handle space-separated moves
                moves = [move.strip() for move in starting_moves.split()]
            # end if
        else:
            moves = starting_moves
        # end if
        
        # Remove any empty moves
        moves = [move for move in moves if move]
        
        # Create a game with the starting moves
        try:
            game = GameClass.load_moves(moves)
        except ValueError as e:
            raise ValueError(f"Invalid starting moves: {e}")
        # end try
    else:
        # Create a new game with standard starting position
        game = GameClass()
        moves = []
    # end if
    
    # Set random seed if provided
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    # end if
    
    # If full_game is True or max_length is not specified, generate moves until the game is over
    if full_game or max_length is None:
        print(f"generating full game")
        # Continue making valid moves until the game is over
        attempt = 0
        while attempt < max_attempts:
            print(f"attempt {attempt}/{max_attempts}")
            # Make random moves until the game is over
            move_done = True
            while move_done:
                # Make a random move
                rnd_move = game.make_random_move()
                print(f"random move {rnd_move}")
                move_done = not rnd_move is None
            # end while
            print(f"Finished")
            # Return the game object
            return game
        # end while
        
        # If we've reached the maximum number of attempts, raise an exception
        raise RuntimeError(f"Failed to generate a valid game after {max_attempts} attempts")
    else:
        # Generate moves up to max_length
        moves_generated = 0
        while moves_generated < max_length and not game.is_game_over():
            # Make a random move
            rnd_move = game.make_random_move()
            if rnd_move is not None:
                moves_generated += 1
            else:
                # No more valid moves
                break
            # end if
        # end while
        
        # Return the game object
        return game
    # end if
# end generate_game
