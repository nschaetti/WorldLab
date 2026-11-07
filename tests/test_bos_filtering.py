#!/usr/bin/env python3
"""
Test script to verify that BOS tokens are correctly filtered out in the to_moves method.
"""

import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from worldlab.nn.board.gpt import GPT

def test_to_moves_filtering():
    """Test that the to_moves method correctly filters out BOS tokens."""
    # Create a list of token IDs including BOS (0)
    token_ids = [0, 1, 2, 0, 3, 4]
    
    # Convert to moves using the to_moves method
    moves = GPT.to_moves(token_ids)
    
    # Get the mapping to check the expected output
    _, id_to_move = GPT.create_move_mapping()
    expected_moves = [id_to_move[i] for i in token_ids if i != 0]
    
    # Print results
    print(f"Input token IDs: {token_ids}")
    print(f"Expected moves (without BOS): {expected_moves}")
    print(f"Actual moves returned: {moves}")
    
    # Check if BOS tokens were filtered out
    if 'BOS' not in moves:
        print("SUCCESS: BOS tokens were correctly filtered out!")
    else:
        print("FAILURE: BOS tokens were not filtered out!")
    
    # Verify the result matches the expected output
    if moves == expected_moves:
        print("SUCCESS: Output matches expected result!")
    else:
        print("FAILURE: Output does not match expected result!")

if __name__ == "__main__":
    test_to_moves_filtering()