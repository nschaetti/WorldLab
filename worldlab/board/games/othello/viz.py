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

"""


# Imports
import random
import numpy as np
from typing import List, Tuple, Optional, Union
import sys
import os
import seaborn as sns

import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.animation import FuncAnimation

from .othello_simulator import OthelloGame


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
            return False  # end if
        if 'IPKernelApp' not in get_ipython().config:
            return False  # end if
        return True  # end try
    except ImportError:
        return False

    # end except
def show_othello(
        moves: Optional[List[str]] = None  # end def show_othello
) -> Union[plt.Figure, None]:
    """
    Display an interactive visualization of an Othello game using matplotlib.

    Args:
        moves (Optional[List[str]]): List of moves in standard notation (e.g., ['c4', 'd3', ...])
                                    If None, starts with an empty board

    Returns:
        Union[plt.Figure, None]: In Jupyter notebooks, returns the figure object for inline display.
                                In regular Python scripts, returns None after displaying the figure.
    """
    # Create a board to replay the game
    if moves:
        board = OthelloGame.load_moves(moves)  # end if
    else:
        board = OthelloGame()
    # end else
    # Current move index (start at -1 to show initial board)
    current_move = -1

    # Track attempted illegal moves for highlighting
    illegal_move = None

    # Create the figure and axes for the board
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.canvas.manager.set_window_title("Othello Game Viewer")

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

        # Draw the green background
        ax.add_patch(plt.Rectangle((0, 0), 8, 8, color='green'))

        # Draw the grid lines
        for i in range(9):
            ax.plot([i, i], [0, 8], 'k-', lw=1)
            ax.plot([0, 8], [i, i], 'k-', lw=1)
        # end for
        # If there's an illegal move, highlight it with red background
        if illegal_move:
            row, col = illegal_move
            ax.add_patch(plt.Rectangle((col, row), 1, 1, color='red', alpha=0.5))
        # end if
        # Draw the initial board state
        if current_move == -1:
            # Draw the four initial pieces
            ax.add_patch(plt.Circle((3.5, 3.5), 0.4, color='white'))
            ax.add_patch(plt.Circle((4.5, 4.5), 0.4, color='white'))
            ax.add_patch(plt.Circle((3.5, 4.5), 0.4, color='black'))
            ax.add_patch(plt.Circle((4.5, 3.5), 0.4, color='black'))
            ax.set_title("Initial Board")  # end if
        else:
            # Draw each move until current_move
            for m_i in range(current_move +1):
                # Get board modifications
                m: List[Tuple[int, int, int]] = board.board.history[m_i]

                # For each modification
                for row, col, p in m:
                    # Draw the piece
                    if p == board.BLACK:
                        ax.add_patch(plt.Circle((col + 0.5, row + 0.5), 0.4, color='black'))  # end if
                    else:
                        ax.add_patch(plt.Circle((col + 0.5, row + 0.5), 0.4, color='white'))
                    # end else
            # Get current move
            move_text = board.moves[current_move]

            # Use the stored player information
            player = "Black" if board.moves_player[current_move] == board.BLACK else "White"

            # Check if there was a pass before this move
            if current_move > 0 and board.moves_player[current_move] == board.moves_player[current_move -1]:
                # If the same player made two consecutive moves, it means the other player passed
                opposite_player = "White" if player == "Black" else "Black"
                ax.set_title(f"Move {current_move + 1}: {opposite_player} passed, {player} plays {move_text}")  # end if
            else:
                ax.set_title(f"Move {current_move + 1}: {player} plays {move_text}")
            # end else
        # Add column and row labels
        ax.set_xticks([i + 0.5 for i in range(8)])
        ax.set_yticks([i + 0.5 for i in range(8)])
        ax.set_xticklabels(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'])
        ax.set_yticklabels(['1', '2', '3', '4', '5', '6', '7', '8'])

        # Set limits and aspect
        ax.set_xlim(0, 8)
        ax.set_ylim(0, 8)
        ax.set_aspect('equal')

        # Update the figure
        fig.canvas.draw_idle()
    # end def draw_board
    def on_prev_click(event):
        """
        Handle click on Previous button.
        """
        nonlocal current_move, illegal_move

        # Clear any illegal move highlighting
        illegal_move = None

        if current_move > -1:
            # Reset the board and replay up to the previous move
            current_move -= 1
            draw_board()
        # end if
    def on_next_click(event):
        """
        Handle click on Next button.
        """
        nonlocal current_move, illegal_move

        # Clear any illegal move highlighting
        illegal_move = None

        # Check that we are not at the end
        if current_move < len(board) - 1:
            current_move += 1
            draw_board()
        # end if
    def on_board_click(event):
        """
        Handle click on the board to make a move.
        """
        nonlocal current_move, illegal_move

        # Only process clicks within the board area
        if event.xdata is None or event.ydata is None:
            return
        # end if
        # Convert click coordinates to board indices
        col = int(event.xdata)
        row = int(event.ydata)

        # Check if the click is within the board boundaries
        if 0 <= row < 8 and 0 <= col < 8:
            # Create a temporary game state to check if the move is valid
            temp_game = OthelloGame()

            # Replay all moves up to the current point
            if current_move >= 0:
                for i in range(current_move + 1):
                    move_notation = board.moves[i]
                    move_row, move_col = temp_game.notation_to_coords(move_notation)
                    temp_game.make_move(move_row, move_col)
                # end for
            # Check if the clicked position is a valid move
            if temp_game.is_valid_move(row, col):
                # Convert the move to notation
                move_notation = temp_game.coords_to_notation(row, col)

                # If we're not at the end of the existing moves list
                if current_move < len(board) - 1:
                    # Check if this move matches the next recorded move
                    next_move = board.moves[current_move + 1]
                    if move_notation == next_move:
                        # This is the correct next move, advance
                        current_move += 1
                        illegal_move = None
                        draw_board()  # end if  # end if
                    else:
                        # This is not the next recorded move, highlight as illegal
                        illegal_move = (row, col)
                        draw_board()  # end else  # end if
                else:
                    # We're at the end of the recorded moves, add a new move
                    temp_game.make_move(row, col)
                    board.moves.append(move_notation)
                    board.moves_player.append(temp_game.current_player)
                    board.board.history.append([(row, col, temp_game.current_player)])
                    current_move += 1
                    illegal_move = None
                    draw_board()  # end else  # end if
            else:
                # Highlight the illegal move
                illegal_move = (row, col)
                draw_board()
            # end else
    # Connect button events
    prev_button.on_clicked(on_prev_click)
    next_button.on_clicked(on_next_click)

    # Connect board click event
    fig.canvas.mpl_connect('button_press_event', on_board_click)

    # Initial draw
    draw_board()

    # Apply tight layout to the figure
    plt.tight_layout()

    # Check if running in Jupyter notebook
    if is_jupyter():
        # In Jupyter, return the figure for inline display
        return fig  # end if
    else:
        # In regular Python scripts, show the figure and return None
        plt.show()

    # end else
def plot_othello_game(moves: Optional[List[str]] = None) -> Union[FuncAnimation, None]:
    """
    Create an animation of an Othello game with 1 second per move.

    Args:
        moves (Optional[List[str]]): List of moves in standard notation (e.g., ['c4', 'd3', ...'])
                                    If None, starts with an empty board

    Returns:
        Union[FuncAnimation, None]: In Jupyter notebooks, returns the animation object for inline display.
                                   In regular Python scripts, returns None after displaying the animation.
    """
    # Create a board to replay the game
    if moves:
        # Create a new game
        game = OthelloGame()

        # Keep track of all moves (valid and invalid)
        all_moves = []
        valid_moves = []
        invalid_moves = []

        # Process each move
        for move in moves:
            try:
                # Convert to coordinates
                row, col = game.notation_to_coords(move)

                # Check if the move is valid
                if game.is_valid_move(row, col):
                    # Make the move
                    game.make_move(row, col)
                    valid_moves.append(move)
                    all_moves.append((move, True))  # (move, is_valid)  # end if
                else:
                    # Try for the other player
                    game.switch_player()
                    if game.is_valid_move(row, col):
                        # Make the move
                        game.make_move(row, col)
                        valid_moves.append(move)
                        all_moves.append((move, True))  # (move, is_valid)  # end if
                    else:
                        # Invalid move for both players
                        game.switch_player()  # Switch back
                        invalid_moves.append((move, row, col))
                        all_moves.append((move, False))  # (move, is_valid)  # end else  # end else  # end try
            except ValueError:
                # Skip invalid move notations
                pass
            # end except
        # Use the board with only valid moves for the animation
        board = OthelloGame.load_moves(valid_moves)  # end if
    else:
        board = OthelloGame()
        all_moves = []
        invalid_moves = []
    # end else
    # Current move index (start at -1 to show initial board)
    current_move = -1

    # Create the figure and axes for the board
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.canvas.manager.set_window_title("Othello Game Animation")

    # Create button axes
    prev_button_ax = plt.axes([0.2, 0.05, 0.15, 0.05])
    next_button_ax = plt.axes([0.65, 0.05, 0.15, 0.05])

    # Create buttons
    prev_button = Button(prev_button_ax, 'Previous')
    next_button = Button(next_button_ax, 'Next')

    # Function to draw the board at a specific move index
    def draw_board(move_idx):
        """
        Draw the board state at the given move index.

        Args:
            move_idx (int): The move index to display (-1 for initial board)
        """
        ax.clear()

        # Draw the green background
        ax.add_patch(plt.Rectangle((0, 0), 8, 8, color='green'))

        # Draw the grid lines
        for i in range(9):
            ax.plot([i, i], [0, 8], 'k-', lw=1)
            ax.plot([0, 8], [i, i], 'k-', lw=1)
        # end for
        # Check if we're showing an invalid move
        is_invalid_move = False
        invalid_move_coords = None

        if move_idx >= 0 and move_idx < len(all_moves):
            move_text, is_valid = all_moves[move_idx]
            if not is_valid:
                is_invalid_move = True
                # Find the coordinates for this invalid move
                for inv_move, row, col in invalid_moves:
                    if inv_move == move_text:
                        invalid_move_coords = (row, col)
                        break
                    # end if
        # Draw the initial board state
        if move_idx == -1:
            # Draw the four initial pieces
            ax.add_patch(plt.Circle((3.5, 3.5), 0.4, color='white'))
            ax.add_patch(plt.Circle((4.5, 4.5), 0.4, color='white'))
            ax.add_patch(plt.Circle((3.5, 4.5), 0.4, color='black'))
            ax.add_patch(plt.Circle((4.5, 3.5), 0.4, color='black'))
            ax.set_title("Initial Board")  # end if
        elif is_invalid_move:
            # For invalid moves, show the board state before the invalid move
            # and highlight the invalid move with a red background

            # Create a board state representation
            board_state = [[0 for _ in range(8)] for _ in range(8)]

            # Set initial pieces
            board_state[3][3] = board.WHITE  # d4 in standard notation
            board_state[3][4] = board.BLACK  # e4 in standard notation
            board_state[4][3] = board.BLACK  # d5 in standard notation
            board_state[4][4] = board.WHITE  # e5 in standard notation

            # Count valid moves up to this point
            valid_count = 0
            for i in range(move_idx):
                if all_moves[i][1]:  # If move is valid
                    valid_count += 1
            # end for
            # Apply all valid moves up to this point
            for m_i in range(valid_count):
                # Get board modifications
                m: List[Tuple[int, int, int]] = board.board.history[m_i]

                # For each modification
                for row, col, p in m:
                    # Update the board state
                    board_state[row][col] = p
                # end for
            # Draw all pieces based on the current board state
            for row in range(8):
                for col in range(8):
                    p = board_state[row][col]
                    if p == board.BLACK:
                        ax.add_patch(plt.Circle((col + 0.5, row + 0.5), 0.4, color='black'))  # end if
                    elif p == board.WHITE:
                        ax.add_patch(plt.Circle((col + 0.5, row + 0.5), 0.4, color='white'))
                    # end elif
            # Highlight the invalid move with a red background
            if invalid_move_coords:
                row, col = invalid_move_coords
                ax.add_patch(plt.Rectangle((col, row), 1, 1, color='red', alpha=0.5))
            # end if
            # Get current move
            move_text = all_moves[move_idx][0]

            # Determine whose turn it would be
            if valid_count > 0:
                last_player = board.moves_player[valid_count -1]
                current_player = board.BLACK if last_player == board.WHITE else board.WHITE
                player = "Black" if current_player == board.BLACK else "White"  # end if  # end elif
            else:
                # First move is always black
                player = "Black"
            # end else
            ax.set_title(f"Invalid Move: {player} tries {move_text}")
        else:
            # Count valid moves up to this point
            valid_count = 0
            for i in range(move_idx + 1):
                if all_moves[i][1]:  # If move is valid
                    valid_count += 1
            # end for
            # Create a board state representation
            board_state = [[0 for _ in range(8)] for _ in range(8)]

            # Set initial pieces
            board_state[3][3] = board.WHITE  # d4 in standard notation
            board_state[3][4] = board.BLACK  # e4 in standard notation
            board_state[4][3] = board.BLACK  # d5 in standard notation
            board_state[4][4] = board.WHITE  # e5 in standard notation

            # Apply all valid moves up to valid_count
            for m_i in range(valid_count):
                # Get board modifications
                m: List[Tuple[int, int, int]] = board.board.history[m_i]

                # For each modification
                for row, col, p in m:
                    # Update the board state
                    board_state[row][col] = p
                # end for
            # Draw all pieces based on the current board state
            for row in range(8):
                for col in range(8):
                    p = board_state[row][col]
                    if p == board.BLACK:
                        ax.add_patch(plt.Circle((col + 0.5, row + 0.5), 0.4, color='black'))  # end if
                    elif p == board.WHITE:
                        ax.add_patch(plt.Circle((col + 0.5, row + 0.5), 0.4, color='white'))
                    # end elif
            # Get current move
            move_text = all_moves[move_idx][0]

            # Use the stored player information
            player = "Black" if board.moves_player[valid_count -1] == board.BLACK else "White"

            # Check if there was a pass before this move
            if valid_count > 1 and board.moves_player[valid_count -1] == board.moves_player[valid_count -2]:
                # If the same player made two consecutive moves, it means the other player passed
                opposite_player = "White" if player == "Black" else "Black"
                ax.set_title(f"Move {valid_count}: {opposite_player} passed, {player} plays {move_text}")  # end if
            else:
                ax.set_title(f"Move {valid_count}: {player} plays {move_text}")
            # end else
        # Add column and row labels
        ax.set_xticks([i + 0.5 for i in range(8)])
        ax.set_yticks([i + 0.5 for i in range(8)])
        ax.set_xticklabels(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'])
        ax.set_yticklabels(['1', '2', '3', '4', '5', '6', '7', '8'])

        # Set limits and aspect
        ax.set_xlim(0, 8)
        ax.set_ylim(0, 8)
        ax.set_aspect('equal')

        return ax
    # end def draw_board
    # Define button click handlers
    def on_prev_click(event):
        nonlocal current_move
        if current_move > -1:
            current_move -= 1
            draw_board(current_move)
            fig.canvas.draw_idle()
        # end if
    def on_next_click(event):
        nonlocal current_move
        if current_move < len(all_moves) - 1:
            # Store the current move to check if we're showing an invalid move
            prev_move = current_move

            # Move to the next move
            current_move += 1

            # If we were showing an invalid move and clicked next, skip to the next valid move
            if prev_move >= 0 and prev_move < len(all_moves) and not all_moves[prev_move][1]:
                # Find the next valid move
                while current_move < len(all_moves) and not all_moves[current_move][1]:
                    current_move += 1
                # end while
                # If we reached the end, stay at the last invalid move
                if current_move >= len(all_moves):
                    current_move = prev_move
                # end if
            # Draw the board
            draw_board(current_move)
            fig.canvas.draw_idle()
        # end if
    # Connect button events
    prev_button.on_clicked(on_prev_click)
    next_button.on_clicked(on_next_click)

    # Create frames for animation (from initial board to final move)
    frames = range(-1, len(all_moves))

    # Create the animation with 1 second per frame
    animation = FuncAnimation(
        fig,
        draw_board,
        frames=frames,
        interval=1000,  # 1000 ms = 1 second
        blit=False,
        repeat=False
    )

    # Apply tight layout to the figure
    plt.tight_layout()

    # Check if running in Jupyter notebook
    if is_jupyter():
        # In Jupyter, return the animation for inline display
        return animation  # end if
    else:
        # In regular Python scripts, show the animation and return None
        plt.show()
        return animation



    # end else  # end def plot_othello_game