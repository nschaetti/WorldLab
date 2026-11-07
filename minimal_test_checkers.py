"""
Minimal test script for the checkers game implementation.
This script directly imports the necessary files without going through module imports.
"""

import sys
import os

# Simple console output functions
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

# Get the absolute path to the checkers_simulator.py file
repo_path = os.path.dirname(os.path.abspath(__file__))
checkers_simulator_path = os.path.join(repo_path, 'boardGPT', 'games', 'checkers', 'checkers_simulator.py')

# Check if the file exists
if not os.path.exists(checkers_simulator_path):
    print_red(f"Error: {checkers_simulator_path} does not exist!")
    sys.exit(1)

# Add the repository path to sys.path
sys.path.insert(0, repo_path)

# Get the path to the game_interface.py file
game_interface_path = os.path.join(repo_path, 'boardGPT', 'games', 'game_interface.py')

# Check if the file exists
if not os.path.exists(game_interface_path):
    print_red(f"Error: {game_interface_path} does not exist!")
    sys.exit(1)

# Import the GameInterface class using exec
game_interface_code = {}
with open(game_interface_path, 'r') as f:
    exec(f.read(), game_interface_code)

# Get the GameInterface class from the exec result
GameInterface = game_interface_code['GameInterface']

# Create a mock class for matplotlib.pyplot
class MockPlt:
    def __init__(self):
        pass
    
    def __getattr__(self, name):
        return lambda *args, **kwargs: None

# Create a mock class for matplotlib.widgets
class MockWidgets:
    class Button:
        def __init__(self, *args, **kwargs):
            pass

# Create a mock class for PIL.Image
class MockImage:
    def __init__(self):
        pass
    
    def __getattr__(self, name):
        return lambda *args, **kwargs: None

# Create a mock class for rich.text
class MockText:
    def __init__(self, *args, **kwargs):
        pass

# Create a mock class for rich.columns
class MockColumns:
    def __init__(self, *args, **kwargs):
        pass

# Create a mock class for worldlab.board.utils
class MockUtils:
    def __init__(self):
        pass
    
    def __getattr__(self, name):
        return lambda *args, **kwargs: None

# Import the CheckersBoard class first
checkers_board_code = {}
with open(checkers_simulator_path, 'r') as f:
    # Read the code
    code = f.read()
    
    # Replace import statements
    code = code.replace('from worldlab.board.games.game_interface import GameInterface', '')
    code = code.replace('import matplotlib.pyplot as plt', '')
    code = code.replace('from matplotlib.widgets import Button', '')
    code = code.replace('from PIL import Image', '')
    code = code.replace('from rich.text import Text', '')
    code = code.replace('from rich.columns import Columns', '')
    code = code.replace('from worldlab.board.utils import console, warning, info, error', '')
    
    # Add required modules and mocks to the globals
    checkers_board_code['GameInterface'] = GameInterface
    import random
    import numpy as np
    from typing import List, Tuple, Set, Dict, Optional
    from collections import Counter
    checkers_board_code['random'] = random
    checkers_board_code['np'] = np
    checkers_board_code['List'] = List
    checkers_board_code['Tuple'] = Tuple
    checkers_board_code['Set'] = Set
    checkers_board_code['Dict'] = Dict
    checkers_board_code['Optional'] = Optional
    checkers_board_code['Counter'] = Counter
    
    # Add mocks
    checkers_board_code['plt'] = MockPlt()
    checkers_board_code['Button'] = MockWidgets.Button
    checkers_board_code['Image'] = MockImage()
    checkers_board_code['Text'] = MockText
    checkers_board_code['Columns'] = MockColumns
    checkers_board_code['console'] = MockUtils()
    checkers_board_code['warning'] = MockUtils()
    checkers_board_code['info'] = MockUtils()
    checkers_board_code['error'] = MockUtils()
    
    # Execute the code
    exec(code, checkers_board_code)

# Get the CheckersGame class from the exec result
CheckersGame = checkers_board_code['CheckersGame']

def test_basic_functionality():
    """Test basic functionality of the CheckersGame class."""
    print_blue("Testing basic functionality...")
    
    # Create a new game
    game = CheckersGame()
    
    # Print the initial board state
    print("Initial board state:")
    print(str(game))
    
    # Get valid moves for the initial board
    valid_moves = game.get_valid_moves()
    print(f"Valid moves for Black: {[game.coords_to_notation(row, col) for row, col in valid_moves]}")
    
    # Make a move
    if valid_moves:
        move_row, move_col = valid_moves[0]
        move_notation = game.coords_to_notation(move_row, move_col)
        print(f"Making move to {move_notation}...")
        result = game.make_move(move_row, move_col)
        
        if result:
            print_green("Move successful!")
            print("Board after move:")
            print(str(game))
            
            # Get valid moves for White
            valid_moves = game.get_valid_moves()
            print(f"Valid moves for White: {[game.coords_to_notation(row, col) for row, col in valid_moves]}")
        else:
            print_red("Move failed!")
    else:
        print_red("No valid moves available!")
    
    print_green("Basic functionality test completed!")

if __name__ == "__main__":
    print_bold("Minimal Test for Checkers Game Implementation")
    print("=" * 50)
    
    test_basic_functionality()
    
    print_green("All tests completed!")