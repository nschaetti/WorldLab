#!/usr/bin/env python3
"""
Test script for the show_othello visualization function.
"""

from worldlab.board.utils.viz import show_othello

# Test with no moves (empty board with initial setup)
print("Testing show_othello with no moves...")
show_othello()

# Test with a predefined sequence of moves
# Using a valid sequence of moves for Othello
print("Testing show_othello with a predefined sequence of moves...")
show_othello(['d3', 'c3', 'c4', 'e3', 'f4', 'c5', 'b4'])