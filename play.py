#!/usr/bin/env python3
"""
play.py - Interactive Othello game against a GPT model

This script allows you to play Othello against a GPT model. You can select
which model to use, which config file to use, and whether you or the model
starts first. Moves are made by clicking on the board.

Usage:
    python play.py --model MODEL_PATH --config CONFIG_PATH [--model-starts]

Arguments:
    --model MODEL_PATH    Path to the safetensors model file
    --config CONFIG_PATH  Path to the model config file (JSON)
    --model-starts        If provided, the model makes the first move
                          Otherwise, you (the human player) start
"""

import argparse
import sys
import os
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np
import time
from typing import List, Optional

# Add the project root to the path to import from boardGPT
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from worldlab.board.utils import load_safetensors
from worldlab.board.games.othello import OthelloGame


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Play Othello against a GPT model")
    parser.add_argument(
        "--model", 
        type=str, 
        required=True,
        help="Path to the safetensors model file"
    )
    parser.add_argument(
        "--config", 
        type=str, 
        required=False,
        help="Path to the model config file (JSON). If not provided, assumes it's the same as model but with .json extension."
    )
    parser.add_argument(
        "--model-starts", 
        action="store_true",
        help="If provided, the model makes the first move. Otherwise, you (the human player) start."
    )
    return parser.parse_args()


def play_othello(model_path: str, config_path: Optional[str] = None, model_starts: bool = False):
    """
    Play Othello against a GPT model.
    
    Args:
        model_path: Path to the safetensors model file
        config_path: Path to the model config file (JSON)
        model_starts: If True, the model makes the first move
    """
    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return
    
    # If config_path is not provided, try to find it
    if config_path is None:
        default_config = model_path.rsplit('.', 1)[0] + '.json'
        if os.path.exists(default_config):
            config_path = default_config
        else:
            # Check if there's a config.json in the same directory
            dir_path = os.path.dirname(model_path)
            alt_config = os.path.join(dir_path, 'config.json')
            if os.path.exists(alt_config):
                config_path = alt_config
                print(f"Using config file found at {config_path}")
            else:
                print(f"Error: No config file found. Please specify the config file path using --config.")
                return
    elif not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        return
    
    # Load the model
    try:
        model = load_safetensors(model_path, config_path)
        print(f"Model loaded from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Initialize the game
    game = OthelloGame()
    moves = []
    game_over = False
    
    # Determine who plays first
    human_player = game.WHITE if model_starts else game.BLACK
    model_player = game.BLACK if model_starts else game.WHITE
    current_player = game.BLACK  # Black always starts in Othello
    
    # If model starts, get its first move
    if model_starts:
        print("Model is thinking...")
        model_moves = model.generate_moves(
            sequence=moves,
            max_new_tokens=1,
            temperature=0.7
        )
        if len(model_moves) > 0:
            move = model_moves[0]
            moves.append(move)
            row, col = game.notation_to_coords(move)
            game.make_move(row, col)
            current_player = game.WHITE
            print(f"Model played: {move}")
    
    # Set up the figure and axes
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.subplots_adjust(bottom=0.2)
    
    # Add buttons for control
    ax_reset = plt.axes([0.15, 0.05, 0.2, 0.075])
    ax_quit = plt.axes([0.65, 0.05, 0.2, 0.075])
    btn_reset = Button(ax_reset, 'Reset Game')
    btn_quit = Button(ax_quit, 'Quit')
    
    # Function to draw the board
    def draw_board():
        ax.clear()
        # Draw the green background
        ax.add_patch(plt.Rectangle((0, 0), 8, 8, color='green'))
        
        # Draw the grid
        for i in range(9):
            ax.axhline(i, color='black', lw=2)
            ax.axvline(i, color='black', lw=2)
        
        # Add labels
        for i in range(8):
            ax.text(-0.5, i + 0.5, chr(97 + i), fontsize=14, ha='center', va='center')
            ax.text(i + 0.5, -0.5, str(i + 1), fontsize=14, ha='center', va='center')
        
        # Draw the pieces
        for row in range(8):
            for col in range(8):
                piece = game.board.get_piece(row, col)
                if piece == game.BLACK:
                    circle = plt.Circle((col + 0.5, row + 0.5), 0.4, color='black')
                    ax.add_patch(circle)
                elif piece == game.WHITE:
                    circle = plt.Circle((col + 0.5, row + 0.5), 0.4, color='white')
                    ax.add_patch(circle)
        
        # Highlight valid moves for the current player
        if not game_over:
            game.current_player = current_player
            valid_moves = game.get_valid_moves()
            for row, col in valid_moves:
                rect = plt.Rectangle((col, row), 1, 1, color='green', alpha=0.3)
                ax.add_patch(rect)
        
        # Set limits and title
        ax.set_xlim(-1, 8)
        ax.set_ylim(-1, 8)
        ax.invert_yaxis()  # Invert y-axis to match board coordinates
        
        # Update title with current player and score
        black_count = sum(1 for row in range(8) for col in range(8) if game.board.get_piece(row, col) == game.BLACK)
        white_count = sum(1 for row in range(8) for col in range(8) if game.board.get_piece(row, col) == game.WHITE)
        
        player_str = "Your turn" if current_player == human_player else "Model's turn"
        if game_over:
            if black_count > white_count:
                winner = "Black" if human_player == game.BLACK else "Model"
                player_str = f"Game over! {winner} wins!"
            elif white_count > black_count:
                winner = "White" if human_player == game.WHITE else "Model"
                player_str = f"Game over! {winner} wins!"
            else:
                player_str = "Game over! It's a tie!"
        
        ax.set_title(f"{player_str} (Black: {black_count}, White: {white_count})")
        
        plt.draw()
    
    # Function to handle board clicks
    def on_board_click(event):
        nonlocal current_player, game_over
        
        if game_over or current_player != human_player:
            return
        
        # Check if click is within the board
        if event.xdata is None or event.ydata is None:
            return
        
        # Convert click coordinates to board coordinates
        col = int(event.xdata)
        row = int(event.ydata)
        
        # Check if the move is valid
        game.current_player = current_player
        if 0 <= row < 8 and 0 <= col < 8 and game.is_valid_move(row, col):
            # Make the move
            game.make_move(row, col)
            move = game.coords_to_notation(row, col)
            moves.append(move)
            print(f"You played: {move}")
            
            # Switch player
            current_player = model_player
            
            # Redraw the board
            draw_board()
            
            # Check if game is over
            if game.is_game_over():
                game_over = True
                draw_board()
                return
            
            # Model's turn
            print("Model is thinking...")
            model_moves = model.generate_moves(
                sequence=moves,
                max_new_tokens=1,
                temperature=0.7
            )
            
            if len(model_moves) > len(moves):
                move = model_moves[len(moves)]
                try:
                    row, col = game.notation_to_coords(move)
                    if game.is_valid_move(row, col):
                        game.make_move(row, col)
                        moves.append(move)
                        print(f"Model played: {move}")
                    else:
                        print(f"Model tried to play invalid move: {move}.")
                        # Display a red square at the invalid move location
                        try:
                            invalid_row, invalid_col = game.notation_to_coords(move)
                            # Add a red rectangle to highlight the invalid move
                            invalid_rect = plt.Rectangle((invalid_col, invalid_row), 1, 1, color='red', alpha=0.7)
                            ax.add_patch(invalid_rect)
                            plt.draw()
                            # Wait for 1 second
                            time.sleep(1)
                            # Remove the red rectangle
                            invalid_rect.remove()
                            plt.draw()
                        except ValueError:
                            # If the notation is completely invalid, we can't show a red square
                            pass
                        
                        # Ask the model for a new move
                        print("Asking model for a new move...")
                        model_moves = model.generate_moves(
                            sequence=moves,
                            max_new_tokens=1,
                            temperature=0.7
                        )
                        
                        if len(model_moves) > len(moves):
                            move = model_moves[len(moves)]
                            try:
                                row, col = game.notation_to_coords(move)
                                if game.is_valid_move(row, col):
                                    game.make_move(row, col)
                                    moves.append(move)
                                    print(f"Model played: {move}")
                                else:
                                    # If still invalid, then choose a random valid move as fallback
                                    print(f"Model tried to play another invalid move: {move}. Choosing random valid move.")
                                    valid_moves = game.get_valid_moves()
                                    if valid_moves:
                                        row, col = valid_moves[np.random.randint(len(valid_moves))]
                                        game.make_move(row, col)
                                        move = game.coords_to_notation(row, col)
                                        moves.append(move)
                                        print(f"Model played: {move}")
                            except ValueError:
                                # If still invalid notation, choose a random valid move
                                print(f"Model generated invalid notation again: {move}. Choosing random valid move.")
                                valid_moves = game.get_valid_moves()
                                if valid_moves:
                                    row, col = valid_moves[np.random.randint(len(valid_moves))]
                                    game.make_move(row, col)
                                    move = game.coords_to_notation(row, col)
                                    moves.append(move)
                                    print(f"Model played: {move}")
                        else:
                            # Model couldn't generate a valid move, choose randomly
                            valid_moves = game.get_valid_moves()
                            if valid_moves:
                                row, col = valid_moves[np.random.randint(len(valid_moves))]
                                game.make_move(row, col)
                                move = game.coords_to_notation(row, col)
                                moves.append(move)
                                print(f"Model played: {move}")
                except ValueError:
                    print(f"Model generated invalid notation: {move}.")
                    # We can't display a red square for invalid notation
                    # since we can't convert it to coordinates
                    
                    # Ask the model for a new move
                    print("Asking model for a new move...")
                    model_moves = model.generate_moves(
                        sequence=moves,
                        max_new_tokens=1,
                        temperature=0.7
                    )
                    
                    if len(model_moves) > len(moves):
                        move = model_moves[len(moves)]
                        try:
                            row, col = game.notation_to_coords(move)
                            if game.is_valid_move(row, col):
                                game.make_move(row, col)
                                moves.append(move)
                                print(f"Model played: {move}")
                            else:
                                # If still invalid, then choose a random valid move as fallback
                                print(f"Model tried to play another invalid move: {move}. Choosing random valid move.")
                                valid_moves = game.get_valid_moves()
                                if valid_moves:
                                    row, col = valid_moves[np.random.randint(len(valid_moves))]
                                    game.make_move(row, col)
                                    move = game.coords_to_notation(row, col)
                                    moves.append(move)
                                    print(f"Model played: {move}")
                        except ValueError:
                            # If still invalid notation, choose a random valid move
                            print(f"Model generated invalid notation again: {move}. Choosing random valid move.")
                            valid_moves = game.get_valid_moves()
                            if valid_moves:
                                row, col = valid_moves[np.random.randint(len(valid_moves))]
                                game.make_move(row, col)
                                move = game.coords_to_notation(row, col)
                                moves.append(move)
                                print(f"Model played: {move}")
                    else:
                        # Model couldn't generate a valid move, choose randomly
                        valid_moves = game.get_valid_moves()
                        if valid_moves:
                            row, col = valid_moves[np.random.randint(len(valid_moves))]
                            game.make_move(row, col)
                            move = game.coords_to_notation(row, col)
                            moves.append(move)
                            print(f"Model played: {move}")
            else:
                # Model couldn't generate a valid move, choose randomly
                valid_moves = game.get_valid_moves()
                if valid_moves:
                    row, col = valid_moves[np.random.randint(len(valid_moves))]
                    game.make_move(row, col)
                    move = game.coords_to_notation(row, col)
                    moves.append(move)
                    print(f"Model played: {move}")
            
            # Switch back to human player
            current_player = human_player
            
            # Check if game is over after model's move
            if game.is_game_over():
                game_over = True
            
            # Redraw the board
            draw_board()
    
    # Function to reset the game
    def on_reset_click(event):
        nonlocal game, moves, game_over, current_player
        game = OthelloGame()
        moves = []
        game_over = False
        current_player = game.BLACK
        
        # If model starts, get its first move
        if model_starts:
            print("Model is thinking...")
            model_moves = model.generate_moves(
                sequence=moves,
                max_new_tokens=1,
                temperature=0.7
            )
            if len(model_moves) > 0:
                move = model_moves[0]
                moves.append(move)
                row, col = game.notation_to_coords(move)
                game.make_move(row, col)
                current_player = game.WHITE
                print(f"Model played: {move}")
        
        draw_board()
    
    # Function to quit the game
    def on_quit_click(event):
        plt.close(fig)
    
    # Connect event handlers
    fig.canvas.mpl_connect('button_press_event', on_board_click)
    btn_reset.on_clicked(on_reset_click)
    btn_quit.on_clicked(on_quit_click)
    
    # Initial draw
    draw_board()
    
    # Show the plot
    plt.show()


def main():
    """Main function."""
    args = parse_args()
    play_othello(args.model, args.config, args.model_starts)


if __name__ == "__main__":
    main()