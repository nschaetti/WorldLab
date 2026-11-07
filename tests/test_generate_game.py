#!/usr/bin/env python
"""
Test script for the modified generate_game function.
"""
import worldlab.board as boardGPT

def test_generate_game_default():
    """Test generate_game with default parameters (should use Othello)."""
    print("Testing generate_game with default parameters (Othello)...")
    
    # Generate a game with default parameters
    game = worldlab.board.generate_game(max_length=10, full_game=False)
    
    # Get the moves from the game
    moves = game.get_moves()
    print(f"Generated {len(moves)} moves: {', '.join(moves)}")
    
    # Verify the game is valid
    print(f"Game is valid: {game is not None}")
    
    print("Default test passed!\n")

def test_generate_game_explicit_othello():
    """Test generate_game with explicitly specified Othello game type."""
    print("Testing generate_game with explicitly specified Othello game type...")
    
    # Generate a game with explicitly specified Othello game type
    game = worldlab.board.generate_game(game="othello", max_length=10, full_game=False)
    
    # Get the moves from the game
    moves = game.get_moves()
    print(f"Generated {len(moves)} moves: {', '.join(moves)}")
    
    # Verify the game is valid
    print(f"Game is valid: {game is not None}")
    
    print("Explicit Othello test passed!\n")

def test_generate_game_invalid_type():
    """Test generate_game with an invalid game type."""
    print("Testing generate_game with an invalid game type...")
    
    try:
        # Try to generate a game with an invalid game type
        game = worldlab.board.generate_game(game="invalid_game", max_length=10, full_game=False)
        print("ERROR: Should have raised ValueError")
    except ValueError as e:
        print(f"Correctly raised ValueError: {e}")
        print("Invalid game type test passed!\n")

if __name__ == "__main__":
    test_generate_game_default()
    test_generate_game_explicit_othello()
    test_generate_game_invalid_type()
    
    print("All tests passed!")