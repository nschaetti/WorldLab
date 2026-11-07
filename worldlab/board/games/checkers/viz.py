"""
Copyright (C) 2025 boardGPT Contributors

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

"""
Visualization utilities for the checkers game.
"""

import random
import numpy as np
from typing import List, Tuple, Optional, Union
import sys
import os
import seaborn as sns

import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.animation import FuncAnimation

from .checkers_simulator import CheckersGame


# Function to detect if code is running in Jupyter notebook
def is_jupyter() -> bool:
    """
    Check if the code is running in a Jupyter notebook.

    Returns:
        bool: True if running in Jupyter notebook, False otherwise
    """
    try:
        from IPython import get_ipython
        if get_ipython() is None:
            return False
        if 'IPKernelApp' not in get_ipython().config:
            return False
        return True
    except ImportError:
        return False


def show_checkers(
        moves: Optional[List[str]] = None
) -> Union[plt.Figure, None]:
    """
    Display an interactive visualization of a Checkers game using matplotlib.

    Args:
        moves (Optional[List[str]]): List of moves in standard notation (e.g., ['a3-b4', 'c5-d4', ...])
                                    If None, starts with an empty board

    Returns:
        Union[plt.Figure, None]: In Jupyter notebooks, returns the figure object for inline display.
                                In regular Python scripts, returns None after displaying the figure.
    """
    # Create a board to replay the game
    game = CheckersGame()
    
    # Initialize moves list if None
    if moves is None:
        moves = []
    else:
        try:
            game.set_moves(moves)
        except ValueError as e:
            print(f"Error loading moves: {e}")
            return None
    
    # Current move index (start at -1 to show initial board)
    current_move = -1
    
    # Track attempted illegal moves for highlighting
    illegal_move = None
    
    # Create the figure and axes for the board
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.canvas.manager.set_window_title("Checkers Game Viewer")
    
    # Create button axes
    prev_button_ax = plt.axes([0.2, 0.05, 0.15, 0.05])
    next_button_ax = plt.axes([0.65, 0.05, 0.15, 0.05])
    
    # Create buttons
    prev_button = Button(prev_button_ax, 'Previous')
    next_button = Button(next_button_ax, 'Next')
    
    def draw_board():
        """
        Draw the current state of the board.
        """
        ax.clear()
        
        # Draw the checkerboard pattern
        for row in range(8):
            for col in range(8):
                color = 'wheat' if (row + col) % 2 == 0 else 'saddlebrown'
                ax.add_patch(plt.Rectangle((col, row), 1, 1, color=color))
        
        # If there's an illegal move, highlight it with red background
        if illegal_move:
            row, col = illegal_move
            ax.add_patch(plt.Rectangle((col, row), 1, 1, color='red', alpha=0.5))
        
        # Create a temporary game to replay moves
        temp_game = CheckersGame()
        
        if current_move >= 0 and moves:
            # Replay moves up to the current move
            for i in range(current_move + 1):
                move = moves[i]
                parts = move.split('-')
                if len(parts) != 2:
                    continue
                
                from_notation, to_notation = parts
                try:
                    from_row, from_col = temp_game.notation_to_coords(from_notation)
                    to_row, to_col = temp_game.notation_to_coords(to_notation)
                    temp_game.make_move(to_row, to_col)
                except ValueError:
                    continue
        
        # Draw the pieces based on the current board state
        for row in range(8):
            for col in range(8):
                piece = temp_game.board.get_piece(row, col)
                if piece == temp_game.board.black:
                    ax.add_patch(plt.Circle((col + 0.5, row + 0.5), 0.4, color='black'))
                elif piece == temp_game.board.white:
                    ax.add_patch(plt.Circle((col + 0.5, row + 0.5), 0.4, color='white', edgecolor='black'))
                elif piece == temp_game.board.black_king:
                    ax.add_patch(plt.Circle((col + 0.5, row + 0.5), 0.4, color='black'))
                    ax.text(col + 0.5, row + 0.5, '♚', fontsize=20, ha='center', va='center', color='gold')
                elif piece == temp_game.board.white_king:
                    ax.add_patch(plt.Circle((col + 0.5, row + 0.5), 0.4, color='white', edgecolor='black'))
                    ax.text(col + 0.5, row + 0.5, '♔', fontsize=20, ha='center', va='center', color='gold')
        
        # Set title based on current move
        if current_move == -1:
            ax.set_title("Initial Board")
        else:
            move_text = moves[current_move]
            player = "Black" if temp_game.current_player == temp_game.WHITE else "White"  # Player who just moved
            ax.set_title(f"Move {current_move + 1}: {player} plays {move_text}")
        
        # Add column and row labels
        ax.set_xticks([i + 0.5 for i in range(8)])
        ax.set_yticks([i + 0.5 for i in range(8)])
        ax.set_xticklabels(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])
        ax.set_yticklabels(['8', '7', '6', '5', '4', '3', '2', '1'])
        
        # Set limits and aspect
        ax.set_xlim(0, 8)
        ax.set_ylim(0, 8)
        ax.set_aspect('equal')
        
        # Update the figure
        fig.canvas.draw_idle()
    
    def on_prev_click(event):
        """
        Handle click on Previous button.
        """
        nonlocal current_move, illegal_move
        
        if current_move > -1:
            current_move -= 1
            illegal_move = None
            draw_board()
    
    def on_next_click(event):
        """
        Handle click on Next button.
        """
        nonlocal current_move, illegal_move
        
        if moves and current_move < len(moves) - 1:
            current_move += 1
            illegal_move = None
            draw_board()
    
    def on_board_click(event):
        """
        Handle click on the board.
        """
        nonlocal current_move, illegal_move, moves
        
        # Check if click is within the board
        if not (0 <= event.xdata < 8 and 0 <= event.ydata < 8):
            return
        
        # Convert click coordinates to board indices
        col = int(event.xdata)
        row = int(event.ydata)
        
        # If we're at the end of the moves list or no moves provided, allow making a new move
        if not moves or current_move == len(moves) - 1:
            # Create a temporary game to replay moves
            temp_game = CheckersGame()
            
            if moves:
                # Replay moves up to the current move
                for i in range(current_move + 1):
                    move = moves[i]
                    parts = move.split('-')
                    if len(parts) != 2:
                        continue
                    
                    from_notation, to_notation = parts
                    try:
                        from_row, from_col = temp_game.notation_to_coords(from_notation)
                        to_row, to_col = temp_game.notation_to_coords(to_notation)
                        temp_game.make_move(to_row, to_col)
                    except ValueError:
                        continue
            
            # Check if the clicked position is a valid move
            if temp_game.is_valid_move(row, col):
                # Make the move
                temp_game.make_move(row, col)
                
                # Get the move in notation form
                move_notation = temp_game.moves[-1]
                
                # Add the move to the moves list
                if not moves:
                    moves = [move_notation]
                else:
                    moves.append(move_notation)
                
                # Update current move index
                current_move += 1
                
                # Clear illegal move highlight
                illegal_move = None
            else:
                # Highlight illegal move
                illegal_move = (row, col)
        
        # Redraw the board
        draw_board()
    
    # Connect event handlers
    prev_button.on_clicked(on_prev_click)
    next_button.on_clicked(on_next_click)
    fig.canvas.mpl_connect('button_press_event', on_board_click)
    
    # Draw the initial board
    draw_board()
    
    # Show the figure
    if is_jupyter():
        plt.close()  # Prevent duplicate display in Jupyter
        return fig
    else:
        plt.show()
        return None


def plot_checkers_game(moves: Optional[List[str]] = None) -> plt.Figure:
    """
    Create a static visualization of a Checkers game.
    
    Args:
        moves (Optional[List[str]]): List of moves in standard notation (e.g., ['a3-b4', 'c5-d4', ...])
                                    If None, shows the initial board
    
    Returns:
        plt.Figure: The figure object containing the visualization
    """
    # Create a game to replay the moves
    game = CheckersGame()
    
    if moves:
        try:
            game.set_moves(moves)
        except ValueError as e:
            print(f"Error loading moves: {e}")
            return plt.figure()
    
    # Number of moves to display
    num_moves = len(moves) if moves else 0
    
    # Calculate grid dimensions
    grid_size = int(np.ceil(np.sqrt(num_moves + 1)))
    
    # Create figure and axes
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(3 * grid_size, 3 * grid_size))
    fig.suptitle("Checkers Game Progression", fontsize=16)
    
    # Flatten axes for easier indexing
    axes = axes.flatten()
    
    # Function to draw a board state
    def draw_board(move_idx):
        ax = axes[move_idx]
        
        # Create a temporary game to replay moves
        temp_game = CheckersGame()
        
        if move_idx > 0 and moves:
            # Replay moves up to the current move
            for i in range(move_idx):
                move = moves[i]
                parts = move.split('-')
                if len(parts) != 2:
                    continue
                
                from_notation, to_notation = parts
                try:
                    from_row, from_col = temp_game.notation_to_coords(from_notation)
                    to_row, to_col = temp_game.notation_to_coords(to_notation)
                    temp_game.make_move(to_row, to_col)
                except ValueError:
                    continue
        
        # Draw the checkerboard pattern
        for row in range(8):
            for col in range(8):
                color = 'wheat' if (row + col) % 2 == 0 else 'saddlebrown'
                ax.add_patch(plt.Rectangle((col, row), 1, 1, color=color))
        
        # Draw the pieces based on the current board state
        for row in range(8):
            for col in range(8):
                piece = temp_game.board.get_piece(row, col)
                if piece == temp_game.board.black:
                    ax.add_patch(plt.Circle((col + 0.5, row + 0.5), 0.4, color='black'))
                elif piece == temp_game.board.white:
                    ax.add_patch(plt.Circle((col + 0.5, row + 0.5), 0.4, color='white', edgecolor='black'))
                elif piece == temp_game.board.black_king:
                    ax.add_patch(plt.Circle((col + 0.5, row + 0.5), 0.4, color='black'))
                    ax.text(col + 0.5, row + 0.5, '♚', fontsize=12, ha='center', va='center', color='gold')
                elif piece == temp_game.board.white_king:
                    ax.add_patch(plt.Circle((col + 0.5, row + 0.5), 0.4, color='white', edgecolor='black'))
                    ax.text(col + 0.5, row + 0.5, '♔', fontsize=12, ha='center', va='center', color='gold')
        
        # Set title based on current move
        if move_idx == 0:
            ax.set_title("Initial Board")
        else:
            move_text = moves[move_idx - 1]
            player = "Black" if move_idx % 2 == 1 else "White"
            ax.set_title(f"Move {move_idx}: {player} plays {move_text}")
        
        # Add column and row labels
        ax.set_xticks([i + 0.5 for i in range(8)])
        ax.set_yticks([i + 0.5 for i in range(8)])
        ax.set_xticklabels(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])
        ax.set_yticklabels(['8', '7', '6', '5', '4', '3', '2', '1'])
        
        # Set limits and aspect
        ax.set_xlim(0, 8)
        ax.set_ylim(0, 8)
        ax.set_aspect('equal')
    
    # Draw each board state
    for i in range(num_moves + 1):
        draw_board(i)
    
    # Hide unused subplots
    for i in range(num_moves + 1, len(axes)):
        axes[i].axis('off')
    
    # Adjust layout
    plt.tight_layout()
    
    return fig