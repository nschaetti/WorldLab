#!/usr/bin/env python3
"""
Test script for the metrics module.
"""

import os
import sys
import random

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from worldlab.board.validation.metrics import is_valid, invalid_move_rate
from simulators.othello import create_move_mapping, create_id_to_move_mapping, OthelloGame


def test_is_valid():
    """
    Test the is_valid function with some simple examples.
    """
    print("Testing is_valid function...")
    
    # Create a new game
    game = OthelloGame()
    
    # Get valid moves for the initial board
    valid_moves = game.get_valid_moves()
    
    # Convert coordinates to move IDs
    move_to_id = create_move_mapping()
    id_to_move = create_id_to_move_mapping()
    
    # BOS token
    bos_id = 0
    
    # Test with empty previous moves (just BOS)
    for row, col in valid_moves:
        move_notation = game.coords_to_notation(row, col)
        move_id = move_to_id[move_notation]
        
        # Check if the move is valid
        is_valid_result = is_valid(move_id, [bos_id])
        print(f"Move {move_notation} (ID: {move_id}) is valid: {is_valid_result}")
        
        # It should be valid
        assert is_valid_result, f"Move {move_notation} should be valid but is_valid returned False"
    
    # Test with an invalid move
    invalid_move_id = 1  # Assuming ID 1 corresponds to a1, which is not valid in the initial board
    is_valid_result = is_valid(invalid_move_id, [bos_id])
    print(f"Invalid move (ID: {invalid_move_id}) is valid: {is_valid_result}")
    
    # It should be invalid
    assert not is_valid_result, f"Move with ID {invalid_move_id} should be invalid but is_valid returned True"
    
    print("is_valid function tests passed!")


def test_invalid_move_rate():
    """
    Test the invalid_move_rate function with a small sample.
    """
    print("\nTesting invalid_move_rate function...")
    
    # Set a small number of samples for testing
    num_samples = 10
    
    # Test with the validation set
    data_dir = "../data"
    split = "val"
    data_filename = "val.pkl"
    
    try:
        rate = invalid_move_rate(data_dir, split, data_filename, num_samples)
        print(f"Invalid move rate on {split} set: {rate:.2%}")
        print("invalid_move_rate function test passed!")
    except Exception as e:
        print(f"Error testing invalid_move_rate: {e}")
        print("Make sure the dataset is available at the specified path.")


if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    
    # Run tests
    test_is_valid()
    test_invalid_move_rate()