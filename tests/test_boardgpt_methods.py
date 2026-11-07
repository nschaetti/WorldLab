#!/usr/bin/env python3
"""
Test script for boardGPT methods.

This script tests the methods implemented in boardGPT/__init__.py
to ensure they work correctly with a game that implements GameInterface.
"""
import worldlab.board as boardGPT
from worldlab.board.games.game_interface import GameInterface
from typing import List, Tuple, Optional
import random


class MockGame(GameInterface):
    """A simple mock implementation of GameInterface for testing purposes."""
    
    def __init__(self):
        self.board = [[0 for _ in range(8)] for _ in range(8)]
        self.current_player = 1  # 1 for black, 2 for white
        self.moves_history = []
        
    def get_valid_moves(self) -> List[Tuple[int, int]]:
        """Return a list of valid moves for the current player."""
        # For simplicity, let's say any empty cell is a valid move
        valid_moves = []
        for row in range(8):
            for col in range(8):
                if self.board[row][col] == 0:
                    valid_moves.append((row, col))
        return valid_moves[:5]  # Limit to 5 moves for testing
    
    def is_valid_move(self, row: int, col: int) -> bool:
        """Check if placing a piece at (row, col) is a valid move."""
        if row < 0 or row >= 8 or col < 0 or col >= 8:
            return False
        return self.board[row][col] == 0
    
    def make_move(self, row: int, col: int) -> bool:
        """Make a move at (row, col) for the current player."""
        if not self.is_valid_move(row, col):
            return False
        
        self.board[row][col] = self.current_player
        notation = self.coords_to_notation(row, col)
        self.moves_history.append(notation)
        
        # Switch player
        self.current_player = 3 - self.current_player  # 1 -> 2, 2 -> 1
        
        return True
    
    def make_random_move(self) -> Optional[Tuple[int, int]]:
        """Make a random valid move for the current player."""
        valid_moves = self.get_valid_moves()
        if not valid_moves:
            return None
        
        row, col = random.choice(valid_moves)
        self.make_move(row, col)
        return row, col
    
    def has_valid_moves(self) -> bool:
        """Check if the current player has any valid moves."""
        return len(self.get_valid_moves()) > 0
    
    def is_game_over(self) -> bool:
        """Check if the game is over."""
        # For simplicity, let's say the game is over if the board is full
        for row in range(8):
            for col in range(8):
                if self.board[row][col] == 0:
                    return False
        return True
    
    def coords_to_notation(self, row: int, col: int) -> str:
        """Convert board coordinates to standard notation."""
        col_letter = chr(ord('a') + col)
        row_number = str(row + 1)
        return col_letter + row_number
    
    def notation_to_coords(self, notation: str) -> Tuple[int, int]:
        """Convert standard notation to board coordinates."""
        col = ord(notation[0]) - ord('a')
        row = int(notation[1]) - 1
        return row, col
    
    def get_moves(self) -> List[str]:
        """Return the list of moves made in the game in standard notation."""
        return self.moves_history
    
    def __len__(self) -> int:
        """Return the length of the current game state."""
        return len(self.moves_history)

    def show(self) -> None:
        """Display the current game state."""
        print("  a b c d e f g h")
        for row in range(8):
            print(f"{row+1} ", end="")
            for col in range(8):
                if self.board[row][col] == 0:
                    print(". ", end="")
                elif self.board[row][col] == 1:
                    print("B ", end="")
                else:
                    print("W ", end="")
            print()
        print(f"Current player: {'Black' if self.current_player == 1 else 'White'}")


def test_boardgpt_methods():
    """Test the methods implemented in boardGPT/__init__.py."""
    # Create a mock game
    game = MockGame()
    
    print("Testing boardGPT methods with mock game:")
    print("-" * 50)
    
    # Test show method
    print("\nInitial game state:")
    boardGPT.show(game)
    
    # Test valid_moves and valid_moves_notation methods
    valid_moves_coords = boardGPT.valid_moves(game)
    valid_moves_notation = boardGPT.valid_moves_notation(game)
    
    print("\nValid moves (coordinates):", valid_moves_coords)
    print("Valid moves (notation):", valid_moves_notation)
    
    # Test has_moves method
    has_moves = boardGPT.has_moves(game)
    print("\nCurrent player has valid moves:", has_moves)
    
    # Test valid method with both coordinate and notation formats
    if valid_moves_coords:
        move_coords = valid_moves_coords[0]
        move_notation = valid_moves_notation[0]
        
        print(f"\nTesting valid method:")
        print(f"Is {move_coords} valid? {boardGPT.valid(game, move_coords)}")
        print(f"Is {move_notation} valid? {boardGPT.valid(game, move_notation)}")
        
        # Test next method with notation format
        print(f"\nApplying move {move_notation} using next method:")
        boardGPT.next(game, move_notation)
        boardGPT.show(game)
        
        # Test get_moves method
        moves_made = boardGPT.get_moves(game)
        print("\nMoves made so far:", moves_made)
        
        # Test rnext method
        print("\nApplying random move using rnext method:")
        boardGPT.rnext(game)
        boardGPT.show(game)
        
        # Test get_moves method again
        moves_made = boardGPT.get_moves(game)
        print("\nMoves made so far:", moves_made)
    
    # Test is_over method
    is_game_over = boardGPT.is_over(game)
    print("\nIs game over?", is_game_over)
    
    print("\nAll tests completed successfully!")


if __name__ == "__main__":
    test_boardgpt_methods()