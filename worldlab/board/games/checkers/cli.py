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
Command-line interface for the checkers game.
"""

# Imports
import argparse
import sys
import importlib
from typing import List, Optional
import random
import numpy as np
import pickle
import sys
import os
import array
from collections import Counter
from typing import List, Tuple, Set, Dict
from rich.text import Text
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from PIL import Image
from worldlab.board.utils import console, info, warning, error, success, debug

from .checkers_simulator import generate_checkers_game, CheckersGame


def save_games(games: List[List[str]], output_file: str):
    """
    Save games to a file.
    
    Args:
        games (List[List[str]]): List of games, where each game is a list of moves
        output_file (str): Path to the output file
    """
    try:
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        
        # Save the games
        with open(output_file, "wb") as f:
            pickle.dump(games, f)
        
        info(f"Saved {len(games)} games to {output_file}")
    except Exception as e:
        error(f"Error saving games: {e}")


def _chunk_file_path(
        output: str,
        chunk_counter: int
):
    """
    Chunk file path.
    """
    chunk_filename = f"checkers-synthetic-train_{chunk_counter:05d}.bin"
    return os.path.join(output, chunk_filename)


def checkers_generate(
        args: argparse.Namespace,
):
    """
    Generate multiple Checkers games.

    If output_file is provided, games will be saved in chunks as they are generated to save memory.
    Otherwise, all games will be kept in memory and returned.

    Args:
        args (argparse.Namespace): Arguments parsed by argparse.
    """
    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
    
    # Initialize an empty list to store the generated games
    games = []
    game_lengths = []
    
    # If output is provided and chunk_size is specified, prepare for chunking
    save_in_chunks = args.output is not None and args.chunk_size is not None and args.chunk_size > 0
    chunk_counter = 0
    
    # Create a progress bar
    with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=None),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("[bold green]{task.completed}/{task.total} games"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
    ) as progress:
        # Create a task for game generation
        task = progress.add_task(
            "[bold green]Generating games...",
            total=args.num_games
        )
        
        # Generate the specified number of games
        for i in range(args.num_games):
            # Generate a single game and add it to the list
            #  If a seed is provided, use a different seed for each game to ensure variety
            game_seed = None if args.seed is None else args.seed + i
            game = generate_checkers_game(game_seed)
            games.append(game.get_moves())
            game_lengths.append(len(game.get_moves()))
            
            # Update the progress bar
            progress.update(task, advance=1)
            
            # If we're saving in chunks and have reached the chunk size, save and clear the games list
            if save_in_chunks and len(games) >= args.chunk_size:
                # Create the chunk filename
                chunk_counter += 1
                
                # Save the chunk
                save_games(games, _chunk_file_path(args.output, chunk_counter))
                
                # Clear the game list to free memory
                games = []
    
    # If there are remaining games, and we're saving to a file
    if args.output is not None and games:
        if save_in_chunks:
            # Save the final chunk
            chunk_counter += 1
            save_games(games, _chunk_file_path(args.output, chunk_counter))
        else:
            # Save all games to a single file
            save_games(games, args.output)
    
    # Count for lengths
    counts = np.bincount(np.array(game_lengths))
    
    # Print count by length
    console.log(f"Generation ended, {len(games)} games generated:")
    for i in range(counts.shape[0]):
        if counts[i] > 0:
            console.log(f"\t- [bold green]{counts[i]:>5}[/] games with length [bold blue]{i}[/]")
    
    # Statistics
    console.log(f"Generation statistics")
    console.log(f"\tNumber of games: {counts.sum()}")
    console.log(f"\tTotal token generated: {sum(game_lengths)}")
    console.log(f"\tAverage tokens per game: {np.array(game_lengths).mean()}")
    console.log(f"\tStd tokens per game: {np.array(game_lengths).std()}")


def checkers_play(args):
    """
    Start an interactive Checkers game.
    """
    try:
        from worldlab.board.games.checkers import CheckersGame
        
        game = CheckersGame()
        info("Starting a new Checkers game...")
        info("Use the 'visualize' command to play interactively with a GUI.")
        info("This is a placeholder for future implementation of a text-based game.")
    except ImportError as e:
        error(f"Checkers functionality is not available due to missing dependencies: {e}")


def checkers_visualize(args):
    """
    Visualize a Checkers game from a list of moves.
    """
    try:
        # First try to import the basic Checkers functionality
        from worldlab.board.games.checkers import CheckersGame
        
        # Then try to import the visualization functionality
        try:
            from worldlab.board.games.checkers.viz import show_checkers
            
            moves = args.moves.split(',') if args.moves else None
            
            if args.file:
                try:
                    with open(args.file, 'r') as f:
                        moves = [move.strip() for move in f.readlines()]
                except FileNotFoundError:
                    error(f"File {args.file} not found.")
                    return
            
            if not moves:
                info("Starting visualization with an empty board...")
            else:
                info(f"Visualizing game with {len(moves)} moves...")
            
            # Use the interactive visualization function
            show_checkers(moves)
        except ImportError as e:
            error(f"Checkers visualization is not available due to missing dependencies: {e}")
    except ImportError as e:
        error(f"Checkers functionality is not available due to missing dependencies: {e}")


def checkers_verify(args):
    """
    Verify if a sequence of moves forms a valid Checkers game.
    """
    try:
        from worldlab.board.games.checkers.checkers_utils import verify_game
        
        moves = args.moves.split(',') if args.moves else []
        
        if args.file:
            try:
                with open(args.file, 'r') as f:
                    moves = [move.strip() for move in f.readlines()]
            except FileNotFoundError:
                error(f"File {args.file} not found.")
                return
        
        if not moves:
            error("No moves provided. Use --moves or --file to provide moves.")
            return
        
        is_valid, invalid_moves = verify_game(moves)
        
        if is_valid:
            success("The game is valid.")
        else:
            error("The game is invalid. Invalid moves:")
            for move in invalid_moves:
                error(f"  - {move}")
    except ImportError as e:
        error(f"Checkers functionality is not available due to missing dependencies: {e}")


def checkers_convert(args):
    """
    Convert between different representations of Checkers games.
    """
    try:
        from worldlab.board.games.checkers.checkers_utils import game_to_board, board_to_string
        
        moves = args.moves.split(',') if args.moves else []
        
        if args.file:
            try:
                with open(args.file, 'r') as f:
                    moves = [move.strip() for move in f.readlines()]
            except FileNotFoundError:
                error(f"File {args.file} not found.")
                return
        
        if not moves:
            error("No moves provided. Use --moves or --file to provide moves.")
            return
        
        if args.to_board:
            try:
                board = game_to_board(moves)
                info("Board representation:")
                console.print(board_to_string(board))
            except ValueError as e:
                error(f"{e}")
        else:
            info("Moves representation:")
            for i, move in enumerate(moves):
                info(f"{i+1}. {move}")
    except ImportError as e:
        error(f"Checkers functionality is not available due to missing dependencies: {e}")


def checkers_load(args):
    """
    Load Checkers games from a file.
    """
    try:
        from worldlab.board.games.checkers import load_games, convert_ids_to_notation
        
        if not args.file:
            error("No file provided. Use --file to provide a file.")
            return
        
        try:
            games = load_games(args.file)
            info(f"Loaded {len(games)} games from {args.file}.")
            
            if args.index is not None:
                if 0 <= args.index < len(games):
                    game_moves = convert_ids_to_notation(games[args.index])
                    info(f"Game {args.index} has {len(game_moves)} moves:")
                    
                    game = CheckersGame()
                    game.set_moves(game_moves)
                    game.show()
                else:
                    error(f"Index {args.index} is out of range. There are {len(games)} games.")
        except Exception as e:
            error(f"Error loading games: {e}")
    except ImportError as e:
        error(f"Checkers functionality is not available due to missing dependencies: {e}")