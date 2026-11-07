#!/usr/bin/env python3
"""
Generate valid Othello games.

This script generates valid Othello games and outputs them to a binary file.
Each game is represented as a sequence of move IDs, starting with a BOS token (ID 0).
Moves are mapped from standard notation (e.g., "d3", "e6") to IDs (1-60).

The script implements the complete Othello game rules, including:
- Standard 8x8 board setup with initial four pieces in the center
- Legal move validation (must flip at least one opponent's piece)
- Piece flipping in all eight directions
- Game termination when neither player has valid moves
- Move notation conversion between board coordinates and standard notation

Usage:
    python othello_simulator.py --num-games 100 --output games.bin
"""

import argparse
import random
import numpy as np
import pickle
import sys
import os
import array
from collections import Counter
from typing import List, Tuple, Set, Dict
from rich.console import Console
from rich.text import Text
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from PIL import Image

# Import the Othello simulator from the boardGPT package
from worldlab.board.games.othello import (
    OthelloGame, OthelloBoard, generate_othello_game, create_move_mapping,
    create_id_to_move_mapping, convert_ids_to_notation, load_games,
    extract_game_by_index, extract_games_by_length
)

# Create a Rich console
console = Console()


def generate_games(
        num_games: int,
        seed: int = None,
        output_file: str = None,
        chunk_size: int = None
) -> List[List[str]]:
    """
    Generate multiple Othello games.
    
    If output_file is provided, games will be saved in chunks as they are generated to save memory.
    Otherwise, all games will be kept in memory and returned.
    
    Args:
        num_games (int): Number of games to generate
        seed (int, optional): Random seed for reproducibility. If None, no seed is set.
        output_file (str, optional): Path to save the generated games. If None, games are only returned.
        chunk_size (int, optional): Number of games per file. If None or 0, all games are saved to a single file.
        
    Returns:
        List[List[str]]: List of games if output_file is None, otherwise an empty list
    """
    # Set random seed if provided
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # Initialize empty list to store the generated games
    games = []
    
    # If output_file is provided and chunk_size is specified, prepare for chunking
    save_in_chunks = output_file is not None and chunk_size is not None and chunk_size > 0
    chunk_counter = 0
    
    # Create a progress bar
    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("[bold green]{task.completed}/{task.total} games"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ) as progress:
        # Create a task for game generation
        task = progress.add_task("[bold green]Generating games...", total=num_games)
        
        # Generate the specified number of games
        for i in range(num_games):
            # Generate a single game and add it to the list
            # If seed is provided, use a different seed for each game to ensure variety
            game_seed = None if seed is None else seed + i
            game = generate_othello_game(game_seed)
            games.append(game)
            
            # Update the progress bar
            progress.update(task, advance=1)
            
            # If we're saving in chunks and have reached the chunk size, save and clear the games list
            if save_in_chunks and len(games) >= chunk_size:
                # Get the base filename and extension
                base_name, ext = os.path.splitext(output_file)
                
                # Create the chunk filename
                chunk_counter += 1
                chunk_filename = f"{base_name}_{chunk_counter}{ext}"
                
                # Save the chunk
                save_games(games, chunk_filename)
                
                # Clear the games list to free memory
                games = []
            # end if
        # end for
    
    # If there are remaining games and we're saving to a file
    if output_file is not None and games:
        if save_in_chunks:
            # Save the final chunk
            chunk_counter += 1
            base_name, ext = os.path.splitext(output_file)
            chunk_filename = f"{base_name}_{chunk_counter}{ext}"
            save_games(games, chunk_filename)
            
            # Return an empty list since games were saved to files
            return []
        else:
            # Save all games to a single file
            save_games(games, output_file)
            
            # Return an empty list since games were saved to a file
            return []
    
    # Return the games if they weren't saved to a file
    return games
# end generate_games


def save_games(
        games: List[List[str]],
        output_file: str,
        chunk_size: int = None
) -> None:
    """
    Save games to one or more binary files.
    
    Each game is saved as a sequence of move IDs, starting with a BOS token (ID 0).
    Token indexes are saved as 8-bit integers to save memory.
    
    Args:
        games (List[List[str]]): List of games to save
        output_file (str): Path to the output file
        chunk_size (int, optional): Number of games per file. If None, all games are saved to a single file.
    """
    # Create the move mapping
    move_to_id = create_move_mapping()
    
    # Convert games to sequences of IDs using byte arrays to save memory
    game_sequences = []
    for game in games:
        # Create a byte array for this game
        sequence = array.array('B')  # 'B' is unsigned char (8-bit)
        
        # Start with BOS token
        sequence.append(move_to_id["BOS"])
        
        # Add move IDs
        for move in game:
            if move != "pass":
                sequence.append(move_to_id[move])
            # end if
        # end for
        
        game_sequences.append(sequence)
    # end for
    
    # If chunk_size is None or 0, save all games to a single file
    if chunk_size is None or chunk_size <= 0:
        # Save to binary file using pickle
        with open(output_file, 'wb') as f:
            pickle.dump(game_sequences, f)
        # end with
    else:
        # Split games into chunks and save each chunk to a separate file
        num_chunks = (len(game_sequences) + chunk_size - 1) // chunk_size  # Ceiling division
        
        # Get the base filename and extension
        base_name, ext = os.path.splitext(output_file)
        
        # Save each chunk to a separate file
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, len(game_sequences))
            chunk = game_sequences[start_idx:end_idx]
            
            # Create the chunk filename
            chunk_filename = f"{base_name}_{i+1}{ext}"
            
            # Save the chunk
            with open(chunk_filename, 'wb') as f:
                pickle.dump(chunk, f)
            # end with
        # end for
    # end if
# end save_games


def crop_to_square(image_path: str) -> None:
    """
    Crop an image to make it square based on the smaller dimension.
    
    Args:
        image_path (str): Path to the image file
    """
    # Open the image
    img = Image.open(image_path)
    
    # Get image dimensions
    width, height = img.size
    
    # Determine the smaller dimension
    min_dim = min(width, height)
    
    # Calculate cropping box (centered)
    left = (width - min_dim) // 2
    top = (height - min_dim) // 2
    right = left + min_dim
    bottom = top + min_dim
    
    # Crop the image
    cropped_img = img.crop((left, top, right, bottom))
    
    # Save the cropped image back to the same path
    cropped_img.save(image_path)


def extract_game_as_images(game_file: str, game_index: int, output_dir: str, image_size: int = 8) -> None:
    """
    Extract each state of an Othello game as images without borders and title.
    The images are cropped to be square based on the smaller dimension.
    
    Args:
        game_file (str): Path to the binary file containing games
        game_index (int): Index of the game to extract
        output_dir (str): Directory to save the images
        image_size (int): Size of the output images in inches (default: 8)
    """
    # Create a Rich console
    console = Console()
    
    # Load games from the input file
    console.print(f"Loading games from {game_file}...", style="blue")
    games = load_games(game_file)
    console.print(f"Loaded {len(games)} games.")
    
    # Extract the game at the specified index
    try:
        game_moves = extract_game_by_index(games, game_index)
        console.print(f"Extracting game at index {game_index} with {len(game_moves) - 1} moves.", style="green")
    except ValueError as e:
        console.print(f"Error: {e}", style="bold red")
        return
    
    # Convert move IDs to notations (skipping BOS token)
    move_notations = convert_ids_to_notation(game_moves)
    
    # Log the game moves
    console.print(f"Game {game_index} has moves {move_notations}.", style="cyan")
    
    # Create a board to replay the game
    board = OthelloGame.load_moves(move_notations)
    
    # Create output directory if it doesn't exist
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Save initial board state
    console.print(f"Saving board states as images to {output_dir}...", style="blue")
    
    # Create a figure for the initial board
    fig, ax = plt.subplots(figsize=(image_size, image_size))
    
    # Draw the green background
    ax.add_patch(plt.Rectangle((0, 0), 8, 8, color='green'))
    
    # Draw the grid lines
    for i in range(9):
        ax.plot([i, i], [0, 8], 'k-', lw=1)
        ax.plot([0, 8], [i, i], 'k-', lw=1)
    
    # Draw the four initial pieces
    ax.add_patch(plt.Circle((3.5, 3.5), 0.4, color='white'))
    ax.add_patch(plt.Circle((4.5, 4.5), 0.4, color='white'))
    ax.add_patch(plt.Circle((3.5, 4.5), 0.4, color='black'))
    ax.add_patch(plt.Circle((4.5, 3.5), 0.4, color='black'))
    
    # Remove borders, ticks, and labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Save the initial board
    initial_board_path = f"{output_dir}/board_initial.png"
    plt.savefig(initial_board_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    
    # Crop the image to make it square
    crop_to_square(initial_board_path)
    
    # For each move, draw and save the board state
    for move_idx in range(len(board.moves)):
        # Create a new figure for each move
        fig, ax = plt.subplots(figsize=(image_size, image_size))
        
        # Draw the green background
        ax.add_patch(plt.Rectangle((0, 0), 8, 8, color='green'))
        
        # Draw the grid lines
        for i in range(9):
            ax.plot([i, i], [0, 8], 'k-', lw=1)
            ax.plot([0, 8], [i, i], 'k-', lw=1)
        
        # Create a temporary game to replay up to this move
        temp_game = OthelloGame()
        
        # Replay the game up to the current move
        for m_i in range(move_idx + 1):
            move = board.moves[m_i]
            row, col = temp_game.notation_to_coords(move)
            temp_game.make_move(row, col)
        
        # Draw the complete board state
        for row in range(8):
            for col in range(8):
                piece = temp_game.board.get_piece(row, col)
                if piece == temp_game.BLACK:
                    ax.add_patch(plt.Circle((col + 0.5, row + 0.5), 0.4, color='black'))
                elif piece == temp_game.WHITE:
                    ax.add_patch(plt.Circle((col + 0.5, row + 0.5), 0.4, color='white'))
        
        # Remove borders, ticks, and labels
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        # Save the board state
        move_board_path = f"{output_dir}/board_move_{move_idx+1:03d}.png"
        plt.savefig(move_board_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        
        # Crop the image to make it square
        crop_to_square(move_board_path)
    
    console.print(f"Saved {len(board.moves) + 1} board states as images.", style="green")


def view_game(game_file: str, game_index: int) -> None:
    """
    Display an interactive visualization of an Othello game using matplotlib.
    
    Args:
        game_file (str): Path to the binary file containing games
        game_index (int): Index of the game to view
    """
    # Create a Rich console
    console = Console()
    
    # Load games from the input file
    console.print(f"Loading games from {game_file}...", style="blue")
    games = load_games(game_file)
    console.print(f"Loaded {len(games)} games.")
    
    # Extract the game at the specified index
    try:
        game_moves = extract_game_by_index(games, game_index)
        console.print(f"Viewing game at index {game_index} with {len(game_moves) - 1} moves.", style="green")
    except ValueError as e:
        console.print(f"Error: {e}", style="bold red")
        return
    # end try

    # Convert move IDs to notations (skipping BOS token)
    move_notations = convert_ids_to_notation(game_moves)

    # Log the game moves
    console.print(f"Game {game_index} has moves {move_notations}.", style="cyan")
    
    # Create a board to replay the game
    board = OthelloGame.load_moves(move_notations)
    
    # Current move index (start at -1 to show initial board)
    current_move = -1
    
    # Create the figure and axes for the board
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.canvas.manager.set_window_title(f"Othello Game Viewer - Game {game_index}")
    
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

        # Draw the four initial pieces
        ax.add_patch(plt.Circle((3.5, 3.5), 0.4, color='white'))
        ax.add_patch(plt.Circle((4.5, 4.5), 0.4, color='white'))
        ax.add_patch(plt.Circle((3.5, 4.5), 0.4, color='black'))
        ax.add_patch(plt.Circle((4.5, 3.5), 0.4, color='black'))
        
        # Add column and row labels
        ax.set_xticks([i + 0.5 for i in range(8)])
        ax.set_yticks([i + 0.5 for i in range(8)])
        ax.set_xticklabels(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'])
        ax.set_yticklabels(['1', '2', '3', '4', '5', '6', '7', '8'])

        # Set title with current move information
        if current_move == -1:
            ax.set_title("Initial Board")
        else:
            # Draw each move until current_move
            for m_i in range(current_move+1):
                # Get board modifications
                m: List[Tuple[int, int, int]] = board.board.history[m_i]

                # For each modification
                for row, col, p in m:
                    # Draw the piece
                    if p == board.BLACK:
                        ax.add_patch(plt.Circle((col + 0.5, row + 0.5), 0.4, color='black'))
                    else:
                        ax.add_patch(plt.Circle((col + 0.5, row + 0.5), 0.4, color='white'))
                    # end if
                # end for
            # end for

            # Get current move
            move_text = board.moves[current_move]

            # Use the stored player information
            player = "Black" if board.moves_player[current_move] == board.BLACK else "White"

            # Check if there was a pass before this move
            if current_move > 0 and player == board.moves_player[current_move-1]:
                # If the same player made two consecutive moves, it means the other player passed
                opposite_player = "White" if player == "Black" else "Black"
                ax.set_title(f"Move {current_move + 1}: {opposite_player} passed, {player} plays {move_text}")
            else:
                ax.set_title(f"Move {current_move + 1}: {player} plays {move_text}")
            # end if
        # end if

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
        nonlocal current_move, board
        
        if current_move > -1:
            # Reset the board and replay up to the previous move
            current_move -= 1

            # Update disply
            draw_board()
        # end if
    # end on_prev_click
    
    def on_next_click(event):
        """
        Handle click on Next button.
        """
        nonlocal current_move, board

        # Check that we are not at the end
        if current_move < len(board) - 1:
            # Update display
            current_move += 1
            draw_board()
        # end if
    # end def on_next_click
    
    # Connect button events
    prev_button.on_clicked(on_prev_click)
    next_button.on_clicked(on_next_click)
    
    # Initial draw
    draw_board()
    
    # Show the plot
    plt.tight_layout()
    plt.show()
# end def view_game


def main():
    """
    Main function to parse arguments and generate or extract games.
    
    Command-line arguments for generating games:
        --num-games: Number of games to generate (default: 10)
        --max-moves: Maximum number of moves per game (default: 60)
        --output: Output file path (default: othello_games.bin)
        --seed: Random seed for reproducibility
        --chunk-size: Number of games per file (default: None, all games in one file)
        
    Command-line arguments for extracting games:
        --extract: Extract games from a binary file
        --input: Input file path
        --index: Index of the game to extract
        --length: Length of games to extract
        --output: Output file path for extracted games
        
    Command-line arguments for viewing games:
        --input: Input file path containing games
        --index: Index of the game to view
        
    Command-line arguments for extracting games as images:
        --input: Input file path containing games
        --index: Index of the game to extract
        --output-dir: Directory to save the images
    """
    global console
    import os

    # Set up command-line argument parser
    parser = argparse.ArgumentParser(description='Generate or extract valid Othello games.')
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Parser for generate command
    generate_parser = subparsers.add_parser('generate', help='Generate Othello games')
    generate_parser.add_argument(
        '--num-games',
        type=int,
        required=True,
        help='Number of games to generate'
    )

    generate_parser.add_argument(
        '--output',
        type=str,
        default='othello_games.bin',
        help='Output file path (default: othello_games.bin)'
    )
    
    generate_parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility'
    )
    
    generate_parser.add_argument(
        '--chunk-size',
        type=int,
        default=None,
        help='Number of games per file (default: None, all games in one file)'
    )
    
    # Parser for view command
    view_parser = subparsers.add_parser('view', help='View an Othello game')
    view_parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input file path containing games'
    )
    
    view_parser.add_argument(
        '--index',
        type=int,
        required=True,
        help='Index of the game to view'
    )
    
    # Parser for extract-images command
    extract_images_parser = subparsers.add_parser('extract-images', help='Extract game states as images')
    extract_images_parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input file path containing games'
    )
    
    extract_images_parser.add_argument(
        '--index',
        type=int,
        required=True,
        help='Index of the game to extract'
    )
    
    extract_images_parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Directory to save the images'
    )
    
    extract_images_parser.add_argument(
        '--image-size',
        type=int,
        default=8,
        help='Size of the output images in inches (default: 8)'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Execute the appropriate command
    if args.command == 'generate':
        # Generate games
        console.print(f"Generating {args.num_games} Othello games...", style="bold green")
        
        # Generate the games
        generate_games(
            num_games=args.num_games,
            seed=args.seed,
            output_file=args.output,
            chunk_size=args.chunk_size
        )
        
        # Print success message
        console.print(f"Successfully generated {args.num_games} games and saved to {args.output}", style="bold green")
    
    elif args.command == 'view':
        # View a game
        view_game(args.input, args.index)
    
    elif args.command == 'extract-images':
        # Extract game states as images
        extract_game_as_images(
            game_file=args.input,
            game_index=args.index,
            output_dir=args.output_dir,
            image_size=args.image_size
        )
    
    else:
        # No command specified, print help
        parser.print_help()
# end def main


if __name__ == "__main__":
    main()