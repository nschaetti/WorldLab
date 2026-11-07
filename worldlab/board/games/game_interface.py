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
Common interface for all board games.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Any, Optional


class GameInterface(ABC):
    """
    Common interface for all board games.
    
    This abstract class defines the methods that all game implementations must provide.
    It ensures consistency across different game types and facilitates code reuse.
    
    Game implementations should inherit from this class and implement all abstract methods.
    """
    
    @abstractmethod
    def get_valid_moves(self) -> List[Tuple[int, int]]:
        """
        Return a list of valid moves for the current player.
        
        Returns:
            List[Tuple[int, int]]: A list of (row, col) tuples representing valid move positions
        """
        pass
    # end def get_valid_moves
    
    @abstractmethod
    def is_valid_move(self, row: int, col: int) -> bool:
        """
        Check if placing a piece at (row, col) is a valid move for the current player.
        
        Args:
            row (int): Row index
            col (int): Column index
            
        Returns:
            bool: True if the move is valid, False otherwise
        """
        pass
    
    @abstractmethod
    def make_move(self, row: int, col: int) -> bool:
        """
        Make a move at (row, col) for the current player.
        
        Args:
            row (int): Row index
            col (int): Column index
            
        Returns:
            bool: True if the move was successful, False if invalid
        """
        pass
    
    @abstractmethod
    def make_random_move(self) -> Optional[Tuple[int, int]]:
        """
        Make a random valid move for the current player. If there is no valid move, return None.
        
        Returns:
            Optional[Tuple[int, int]]: The (row, col) of the move made, or None if no move was possible
        """
        pass
    
    @abstractmethod
    def has_valid_moves(self) -> bool:
        """
        Check if the current player has any valid moves.
        
        Returns:
            bool: True if the current player has at least one valid move, False otherwise
        """
        pass
    
    @abstractmethod
    def is_game_over(self) -> bool:
        """
        Check if the game is over.
        
        Returns:
            bool: True if the game is over, False otherwise
        """
        pass
    
    @abstractmethod
    def coords_to_notation(self, row: int, col: int) -> str:
        """
        Convert board coordinates to standard notation.
        
        Args:
            row (int): Row index
            col (int): Column index
            
        Returns:
            str: Position in standard notation
        """
        pass
    # end def coords_to_notation
    
    @abstractmethod
    def notation_to_coords(self, notation: str) -> Tuple[int, int]:
        """
        Convert standard notation to board coordinates.
        
        Args:
            notation (str): Position in standard notation
            
        Returns:
            Tuple[int, int]: (row, col) coordinates
        """
        pass
    # end def notation_to_coords
    
    @abstractmethod
    def get_moves(self) -> List[str]:
        """
        Return the list of moves made in the game in standard notation.
        
        Returns:
            List[str]: List of moves in standard notation
        """
        pass
    # end def get_moves
    
    @abstractmethod
    def show(self) -> None:
        """
        Display the current game state.
        """
        pass
    # end show

    @abstractmethod
    def __len__(self) -> int:
        """
        Return the length of the current game state.

        Returns:
            int: Length of the current game state
        """
        pass
    # end def __len__

# end class GameInterface

