#!/usr/bin/env python3
# Test script for Othello display

# Import directly from the file to avoid dependency issues
import sys
import os

# Add the directory containing the file to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the classes directly from the file
from worldlab.board.games.othello.othello_simulator import OthelloBoard, OthelloGame

def test_board_display():
    """Test the display of an OthelloBoard"""
    print("Testing OthelloBoard display:")
    board = OthelloBoard()
    print(board)
    print("\nTesting OthelloBoard display with last move:")
    print(board.__str__(last_move=(3, 3)))  # Mark one of the initial pieces as the last move

def test_game_display():
    """Test the display of an OthelloGame"""
    print("\nTesting OthelloGame display:")
    game = OthelloGame()
    
    # Make a few moves
    moves = [
        (2, 3),  # Black plays d3
        (2, 2),  # White plays c3
        (1, 2),  # Black plays c2
        (3, 2),  # White plays c4
    ]
    
    for row, col in moves:
        game.make_move(row, col)
    
    # Print the game
    print(game)

if __name__ == "__main__":
    test_board_display()
    test_game_display()