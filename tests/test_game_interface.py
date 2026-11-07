#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script to verify that OthelloGame works correctly with the new GameInterface.
"""

from worldlab.board.games import GameInterface, OthelloGame

def test_othello_game():
    """Test that OthelloGame implements GameInterface correctly."""
    # Create a new Othello game
    game = OthelloGame()
    
    # Verify that game is an instance of GameInterface
    assert isinstance(game, GameInterface), "OthelloGame should be an instance of GameInterface"
    
    # Test get_valid_moves
    valid_moves = game.get_valid_moves()
    print(f"Valid moves: {valid_moves}")
    
    # Test is_valid_move
    if valid_moves:
        row, col = valid_moves[0]
        assert game.is_valid_move(row, col), f"Move {row}, {col} should be valid"
        
    # Test coords_to_notation and notation_to_coords
    notation = game.coords_to_notation(3, 4)
    print(f"Notation for (3, 4): {notation}")
    row, col = game.notation_to_coords(notation)
    assert (row, col) == (3, 4), f"Conversion failed: {notation} should convert to (3, 4)"
    
    # Test make_move
    if valid_moves:
        row, col = valid_moves[0]
        notation = game.coords_to_notation(row, col)
        print(f"Making move at {notation} ({row}, {col})")
        result = game.make_move(row, col)
        assert result, f"Make move should return True for valid move {notation}"
        
    # Test get_moves
    moves = game.get_moves()
    print(f"Moves made: {moves}")
    
    # Test make_random_move
    random_move = game.make_random_move()
    print(f"Random move: {random_move}")
    
    # Test has_valid_moves
    has_moves = game.has_valid_moves()
    print(f"Has valid moves: {has_moves}")
    
    # Test is_game_over
    is_over = game.is_game_over()
    print(f"Game is over: {is_over}")
    
    print("All tests passed!")

if __name__ == "__main__":
    test_othello_game()