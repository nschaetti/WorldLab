import os
import sys

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from worldlab.board.validation.metrics import is_valid_game_sequence
from simulators.othello import OthelloGame

def test_is_valid_game_sequence():
    """
    Test the is_valid_game_sequence function with different scenarios.
    """
    print("Testing is_valid_game_sequence function...")
    
    # Test case 1: Valid game sequence
    valid_game = ["d3", "c3", "c4", "e3"]
    try:
        result = is_valid_game_sequence(valid_game)
        print(f"Test case 1 (Valid game): {'PASS' if result else 'FAIL'}")
    except Exception as e:
        print(f"Test case 1 (Valid game): FAIL - Unexpected exception: {e}")
    
    # Test case 2: Invalid last move
    invalid_last_move = ["d3", "c3", "c4", "d4"]  # d4 should be invalid after these moves
    try:
        result = is_valid_game_sequence(invalid_last_move)
        print(f"Test case 2 (Invalid last move): {'PASS' if not result else 'FAIL'}")
    except Exception as e:
        print(f"Test case 2 (Invalid last move): FAIL - Unexpected exception: {e}")
    
    # Test case 3: Invalid move before the last one
    invalid_earlier_move = ["d3", "c3", "d4", "e3"]  # d4 should be invalid after d3, c3
    try:
        is_valid_game_sequence(invalid_earlier_move)
        print("Test case 3 (Invalid earlier move): FAIL - Expected ValueError not raised")
    except ValueError as e:
        print(f"Test case 3 (Invalid earlier move): PASS - Correctly raised ValueError: {e}")
    except Exception as e:
        print(f"Test case 3 (Invalid earlier move): FAIL - Unexpected exception type: {e}")
    
    # Test case 4: Empty game sequence
    empty_game = []
    try:
        result = is_valid_game_sequence(empty_game)
        print(f"Test case 4 (Empty game): {'PASS' if result else 'FAIL'}")
    except Exception as e:
        print(f"Test case 4 (Empty game): FAIL - Unexpected exception: {e}")
    
    # Test case 5: Sequence where a player has no valid moves and must pass
    pass_sequence = ['d3', 'e3', 'f5', 'c5', 'b5', 'd2', 'd1', 'c1', 'f3', 'e1', 'f4']
    try:
        result = is_valid_game_sequence(pass_sequence)
        print(f"Test case 5 (Player must pass): {'PASS' if result else 'FAIL'}")
    except Exception as e:
        print(f"Test case 5 (Player must pass): FAIL - Unexpected exception: {e}")

if __name__ == "__main__":
    test_is_valid_game_sequence()