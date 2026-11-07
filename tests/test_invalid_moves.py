#!/usr/bin/env python3
# Test script for invalid moves in plot_othello_game

from worldlab.board.utils import plot_othello_game
import matplotlib.pyplot as plt

# Create a list of moves with some invalid ones
# Standard opening moves: d3, c3, c4, e3, f4
# Adding some invalid moves: d4 (occupied), z9 (out of bounds)
moves = ['d3', 'c3', 'c4', 'd4', 'e3', 'z9', 'f4']

# Plot the game with invalid moves
animation = plot_othello_game(moves)

# Show the plot
plt.show()