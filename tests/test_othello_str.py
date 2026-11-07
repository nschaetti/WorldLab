#!/usr/bin/env python3
# Test script for OthelloGame.__str__ method

from worldlab.board.games.othello.othello_simulator import OthelloGame

def main():
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

if __name__ == "__main__":
    main()