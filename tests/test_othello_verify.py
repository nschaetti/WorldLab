#!/usr/bin/env python3

from worldlab.board.games.othello.othello_utils import verify_game

def test_verify_game():
    # Test case 1: Valid game sequence (first few moves of a standard Othello game)
    # Standard opening moves in Othello: Black plays d3, White plays c3, etc.
    valid_moves = ['d3', 'c3', 'c4', 'e3', 'f4', 'e6', 'f5']
    is_valid, invalid_moves = verify_game(valid_moves)
    print(f"Test case 1 (valid game):")
    print(f"  Is valid: {is_valid}")
    print(f"  Invalid moves: {invalid_moves}")
    assert is_valid is True
    assert len(invalid_moves) == 0
    
    # Test case 2: Invalid game sequence with some invalid moves
    invalid_moves_list = ['d3', 'c3', 'a1', 'd6', 'z9', 'f4', 'e6']
    is_valid, invalid_moves = verify_game(invalid_moves_list)
    print(f"\nTest case 2 (invalid game):")
    print(f"  Is valid: {is_valid}")
    print(f"  Invalid moves: {invalid_moves}")
    assert is_valid is False
    assert 'a1' in invalid_moves
    assert 'z9' in invalid_moves
    
    # Test case 3: Empty game
    empty_game = []
    is_valid, invalid_moves = verify_game(empty_game)
    print(f"\nTest case 3 (empty game):")
    print(f"  Is valid: {is_valid}")
    print(f"  Invalid moves: {invalid_moves}")
    assert is_valid is True
    assert len(invalid_moves) == 0
    
    print("\nAll tests passed!")

if __name__ == "__main__":
    test_verify_game()