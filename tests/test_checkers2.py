"""
Test script for the checkers game implementation.
"""

import sys

# Simple console output functions to avoid dependency on rich
def print_bold(text):
    print(f"\033[1m{text}\033[0m")

def print_blue(text):
    print(f"\033[1;34m{text}\033[0m")

def print_green(text):
    print(f"\033[1;32m{text}\033[0m")

def print_red(text):
    print(f"\033[1;31m{text}\033[0m")

def print_yellow(text):
    print(f"\033[1;33m{text}\033[0m")

# Import the CheckersGame class directly to avoid dependencies
sys.path.append('/')
from worldlab.board.games.checkers.checkers_simulator import CheckersGame

def test_initialization():
    """Test game initialization."""
    print_blue("Testing game initialization...")
    game = CheckersGame()
    print("Initial board state:")
    print(game)
    print()

def test_valid_moves():
    """Test valid moves."""
    print_blue("Testing valid moves...")
    game = CheckersGame()
    
    # Get valid moves for the initial board
    valid_moves = game.get_valid_moves()
    print(f"Valid moves for Black: {[game.coords_to_notation(row, col) for row, col in valid_moves]}")
    
    # Make a move
    move_row, move_col = valid_moves[0]
    move_notation = game.coords_to_notation(move_row, move_col)
    print(f"Making move to {move_notation}...")
    game.make_move(move_row, move_col)
    
    # Show the board after the move
    print("Board after move:")
    print(game)
    
    # Get valid moves for White
    valid_moves = game.get_valid_moves()
    print(f"Valid moves for White: {[game.coords_to_notation(row, col) for row, col in valid_moves]}")
    print()

def test_game_completion():
    """Test game completion with a simple sequence of moves."""
    print_blue("Testing game completion...")
    game = CheckersGame()
    
    # Play a sequence of moves
    moves = [
        "a3-b4", "c5-a3", "b2-c3", "a3-c5", 
        "c3-d4", "e5-c3", "d2-e3", "c3-e5",
        "e3-f4", "g5-e3", "f2-g3", "e3-g5"
    ]
    
    print(f"Playing sequence of moves: {moves}")
    
    for i, move in enumerate(moves):
        parts = move.split("-")
        from_notation, to_notation = parts
        from_row, from_col = game.notation_to_coords(from_notation)
        to_row, to_col = game.notation_to_coords(to_notation)
        
        # Make the move
        result = game.make_move(to_row, to_col)
        
        if not result:
            print_red(f"Move {i+1}: {move} failed!")
            break
        
        print(f"Move {i+1}: {move} - {'Black' if game.current_player == game.WHITE else 'White'} to play")
    
    # Show the final board state
    print("Final board state:")
    print(game)
    
    # Check if the game is over
    if game.is_game_over():
        winner = "Black" if game.winner == game.BLACK else "White"
        print_green(f"Game over! {winner} wins!")
    else:
        print_yellow("Game not over yet.")
    print()

def test_random_game():
    """Test a random game."""
    print_blue("Testing random game...")
    game = CheckersGame()
    
    # Play a random game
    move_count = 0
    while not game.is_game_over() and move_count < 100:
        move = game.make_random_move()
        if move is None:
            print_red(f"No valid moves for {'Black' if game.current_player == game.BLACK else 'White'}!")
            break
        
        move_count += 1
        if move_count % 10 == 0:
            print(f"Made {move_count} moves...")
    
    # Show the final board state
    print("Final board state after random game:")
    print(game)
    
    # Check if the game is over
    if game.is_game_over():
        winner = "Black" if game.winner == game.BLACK else "White"
        print_green(f"Game over! {winner} wins!")
    else:
        print_yellow(f"Game not over yet after {move_count} moves.")
    print()

if __name__ == "__main__":
    print_bold("Testing Checkers Game Implementation")
    print("=" * 50)
    
    test_initialization()
    test_valid_moves()
    test_game_completion()
    test_random_game()
    
    print_green("All tests completed!")