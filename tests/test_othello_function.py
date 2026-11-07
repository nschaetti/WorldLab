#!/usr/bin/env python3
# Test script for the othello function
import worldlab.board as bgpt

def test_valid_moves():
    """Test the othello function with valid move sequences."""
    print("Testing with valid move sequences:")
    
    # Test with space-separated moves
    print("\nTest 1: Space-separated moves")
    try:
        game = bgpt.othello("d3 c4 e3 f4")
        print(f"Success! Game created with {len(game.get_moves())} moves.")
        print(f"Moves: {game.get_moves()}")
    except ValueError as e:
        print(f"Error: {e}")
    
    # Test with comma-separated moves
    print("\nTest 2: Comma-separated moves")
    try:
        game = bgpt.othello("d3,c4,e3,f4")
        print(f"Success! Game created with {len(game.get_moves())} moves.")
        print(f"Moves: {game.get_moves()}")
    except ValueError as e:
        print(f"Error: {e}")
    
    # Test with mixed separators
    print("\nTest 3: Mixed separators (should handle this case)")
    try:
        game = bgpt.othello("d3, c4, e3, f4")
        print(f"Success! Game created with {len(game.get_moves())} moves.")
        print(f"Moves: {game.get_moves()}")
    except ValueError as e:
        print(f"Error: {e}")

def test_invalid_moves():
    """Test the othello function with invalid move sequences."""
    print("\nTesting with invalid move sequences:")
    
    # Test with invalid notation
    print("\nTest 4: Invalid notation")
    try:
        game = bgpt.othello("d3 z9 e3 f4")
        print(f"Success! Game created with {len(game.get_moves())} moves.")
        print(f"Moves: {game.get_moves()}")
    except ValueError as e:
        print(f"Error: {e}")
    
    # Test with invalid move sequence
    print("\nTest 5: Invalid move sequence")
    try:
        game = bgpt.othello("a1 a2 a3 a4")  # These moves are not valid in Othello
        print(f"Success! Game created with {len(game.get_moves())} moves.")
        print(f"Moves: {game.get_moves()}")
    except ValueError as e:
        print(f"Error: {e}")
    
    # Test with empty string
    print("\nTest 6: Empty string")
    try:
        game = bgpt.othello("")
        print(f"Success! Game created with {len(game.get_moves())} moves.")
        print(f"Moves: {game.get_moves()}")
    except ValueError as e:
        print(f"Error: {e}")

def main():
    """Run all tests."""
    test_valid_moves()
    test_invalid_moves()
    
    print("\nAll tests completed.")

if __name__ == "__main__":
    main()