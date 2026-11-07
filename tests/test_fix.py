#!/usr/bin/env python
# Test script to verify the fix for the ValueError in notation_to_coords

from worldlab.board.utils import show_othello

# Test with valid moves
print("Testing with valid moves:")
valid_moves = ['c4', 'd3', 'e3', 'f4']
show_othello(valid_moves)

# Test with invalid moves (including 'BOS')
print("\nTesting with invalid moves:")
invalid_moves = ['c4', 'd3', 'BOS', 'e3', 'f4']
show_othello(invalid_moves)

# Test with out-of-bounds moves
print("\nTesting with out-of-bounds moves:")
out_of_bounds_moves = ['c4', 'd3', 'j9', 'e3', 'f4']
show_othello(out_of_bounds_moves)

print("\nAll tests completed successfully!")