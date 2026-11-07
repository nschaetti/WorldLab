#!/usr/bin/env python3
"""
Debug script to trace through the execution of is_valid_game_sequence with the problematic sequence.
"""

import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from worldlab.board.validation.metrics import is_valid_game_sequence
from worldlab.board.games.othello import OthelloGame

def debug_sequence(sequence):
    """
    Debug the execution of is_valid_game_sequence with the given sequence.
    """
    print(f"Debugging sequence: {sequence}")
    
    # Create a new Othello game
    othello_game = OthelloGame()
    
    # Apply moves one by one and check validity
    for i, move in enumerate(sequence):
        print(f"\nChecking move {i+1}: {move}")
        
        # Convert move notation to coordinates
        row, col = othello_game.notation_to_coords(move)
        print(f"  Coordinates: row={row}, col={col}")
        
        # Check if the move is valid
        is_valid = othello_game.is_valid_move(row, col)
        print(f"  Is valid move: {is_valid}")
        
        if not is_valid:
            print(f"  ERROR: Move {move} is invalid!")
            
            # Print the current board state
            print("\nCurrent board state:")
            for r in range(8):
                row_str = ""
                for c in range(8):
                    piece = othello_game.board.get_piece(r, c)
                    if piece == 0:
                        row_str += ". "
                    elif piece == 1:
                        row_str += "B "
                    else:
                        row_str += "W "
                print(f"  {row_str}")
            
            # Check valid moves
            valid_moves = othello_game.get_valid_moves()
            valid_notations = [othello_game.coords_to_notation(r, c) for r, c in valid_moves]
            print(f"\nValid moves: {valid_notations}")
            
            return False
        
        # Make the move
        success = othello_game.make_move(row, col)
        if not success:
            print(f"  ERROR: Failed to make move {move}!")
            return False
        
        print(f"  Move {move} successfully applied.")
        print(f"  Current player: {'Black' if othello_game.current_player == 1 else 'White'}")
    
    print("\nAll moves in the sequence are valid!")
    return True

def main():
    # The problematic sequence
    sequence = ['d3', 'e3', 'f5', 'c5', 'b5', 'd2', 'd1', 'c1', 'f3', 'e1', 'f4']
    
    # Debug the sequence
    debug_result = debug_sequence(sequence)
    print(f"\nDebug result: {'Success' if debug_result else 'Failure'}")
    
    # Also check with is_valid_game_sequence
    try:
        is_valid = is_valid_game_sequence(sequence)
        print(f"\nis_valid_game_sequence result: {is_valid}")
    except ValueError as e:
        print(f"\nis_valid_game_sequence raised an error: {e}")

if __name__ == "__main__":
    main()