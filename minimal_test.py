#!/usr/bin/env python3
# Minimal test script for OthelloGame.__str__ method

# Direct import of the module
import sys
import os

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath('.'))

# Import the OthelloGame class directly
from worldlab.board.games.othello.othello_simulator import OthelloGame

# Create a new Othello game
game = OthelloGame()

# Make some moves
moves = ["d3", "c3", "c4", "e3", "f4", "d2"]

for move in moves:
    row, col = game.notation_to_coords(move)
    game.make_move(row, col)

# Print the game to see the string representation
print("String representation of the game:")
print(str(game))

print("\nRepr representation of the game:")
print(repr(game))