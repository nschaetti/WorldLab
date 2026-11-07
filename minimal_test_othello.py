#!/usr/bin/env python3
# Minimal test script for the othello function

# Direct imports to avoid dependency issues
from worldlab.board.games.othello.othello_simulator import OthelloGame
from worldlab.board.games.othello.othello_utils import verify_game

def othello(moves_str: str) -> OthelloGame:
    """
    Create an OthelloGame object from a string of moves.
    
    Args:
        moves_str (str): A string of moves in standard notation, separated by spaces or commas
                         (e.g., "d3 c4 e3" or "d3,c4,e3")
    
    Returns:
        OthelloGame: An OthelloGame object with the moves applied
        
    Raises:
        ValueError: If the move sequence is not possible (contains invalid moves)
    """
    # Parse the input string into a list of moves
    if ',' in moves_str:
        # Handle comma-separated moves
        moves = [move.strip() for move in moves_str.split(',')]
    else:
        # Handle space-separated moves
        moves = [move.strip() for move in moves_str.split()]
    
    # Remove any empty moves
    moves = [move for move in moves if move]
    
    # Verify that the move sequence is valid
    is_valid, invalid_moves = verify_game(moves)
    
    if not is_valid:
        raise ValueError(f"Invalid move sequence. The following moves are invalid: {', '.join(invalid_moves)}")
    
    # Create an OthelloGame object with the moves applied
    game = OthelloGame.load_moves(moves)
    
    return game

def test_valid_moves():
    """Test the othello function with valid move sequences."""
    print("Testing with valid move sequences:")
    
    # Test with space-separated moves
    print("\nTest 1: Space-separated moves")
    try:
        game = othello("d3 c4 e3 f4")
        print(f"Success! Game created with {len(game.get_moves())} moves.")
        print(f"Moves: {game.get_moves()}")
    except ValueError as e:
        print(f"Error: {e}")
    
    # Test with comma-separated moves
    print("\nTest 2: Comma-separated moves")
    try:
        game = othello("d3,c4,e3,f4")
        print(f"Success! Game created with {len(game.get_moves())} moves.")
        print(f"Moves: {game.get_moves()}")
    except ValueError as e:
        print(f"Error: {e}")
    
    # Test with mixed separators
    print("\nTest 3: Mixed separators (should handle this case)")
    try:
        game = othello("d3, c4, e3, f4")
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
        game = othello("d3 z9 e3 f4")
        print(f"Success! Game created with {len(game.get_moves())} moves.")
        print(f"Moves: {game.get_moves()}")
    except ValueError as e:
        print(f"Error: {e}")
    
    # Test with invalid move sequence
    print("\nTest 5: Invalid move sequence")
    try:
        game = othello("a1 a2 a3 a4")  # These moves are not valid in Othello
        print(f"Success! Game created with {len(game.get_moves())} moves.")
        print(f"Moves: {game.get_moves()}")
    except ValueError as e:
        print(f"Error: {e}")
    
    # Test with empty string
    print("\nTest 6: Empty string")
    try:
        game = othello("")
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