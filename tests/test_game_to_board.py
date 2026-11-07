#!/usr/bin/env python3
"""
Test script for the game_to_board function in boardGPT/utils/othello_simulator.py
"""

from worldlab.board.games.othello.othello_utils import game_to_board
from worldlab.board.games.othello import OthelloGame

def test_valid_moves():
    """Test with valid moves"""
    print("Testing with valid moves...")
    # Standard opening moves in Othello
    moves = ["d3", "c3", "c4", "e3"]
    try:
        board = game_to_board(moves)
        print(f"Success! Board has {len(board)} cells")
        # Print the board in a more readable format
        game = OthelloGame()
        for i, move in enumerate(moves):
            row, col = game.notation_to_coords(move)
            print(f"Move {i+1}: {move} -> ({row}, {col})")
            game.make_move(row, col)
        print("Final board state:")
        game.print_board()
    except Exception as e:
        print(f"Error: {e}")

def test_invalid_move_for_current_player():
    """Test with a move that's invalid for the current player but valid for the other"""
    print("\nTesting with a move invalid for current player but valid for other...")
    # Create a specific board state where a move is invalid for current player but valid for other
    moves = ["d3", "c3", "c4", "e3", "f4"]  # First few moves are valid
    try:
        board = game_to_board(moves)
        print(f"Success! Player was switched correctly")
    except Exception as e:
        print(f"Error: {e}")

def test_invalid_move_for_both_players():
    """Test with a move that's invalid for both players"""
    print("\nTesting with a move invalid for both players...")
    # Add an invalid move (a1 is typically invalid in early game)
    moves = ["d3", "c3", "c4", "a1"]
    try:
        board = game_to_board(moves)
        print("Error: Should have raised ValueError")
    except ValueError as e:
        print(f"Success! Caught expected error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

def test_invalid_notation():
    """Test with invalid notation"""
    print("\nTesting with invalid notation...")
    moves = ["d3", "c3", "z9"]  # z9 is invalid notation
    try:
        board = game_to_board(moves)
        print(f"Success! Invalid notation was skipped")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    print("Testing game_to_board function...\n")
    test_valid_moves()
    test_invalid_move_for_current_player()
    test_invalid_move_for_both_players()
    test_invalid_notation()
    print("\nAll tests completed.")