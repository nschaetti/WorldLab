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

#!/usr/bin/env python3
"""
Command Line Interface for boardGPT games.
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
from boardGPT.utils import console, info, warning, error, success, debug

from .othello_simulator import generate_othello_game, OthelloGame


def save_games(
        games: List[List[str]],
        output_file: str
) -> None:
    """
    Save games to one or more binary files.

    Each game is saved as a sequence of move IDs, starting with a BOS token (ID 0).
    Token indexes are saved as 8-bit integers to save memory.

    Args:
        games (List[List[str]]): List of games to save
        output_file (str): Path to the output file
    """
    # Convert games to sequences of IDs using byte arrays to save memory
    game_sequences = [np.array(game, dtype="S2") for game in games]

    # Save to a binary file using pickle
    with open(output_file, 'wb') as f:
        pickle.dump(game_sequences, f)
    # end with
# end save_games


def _chunk_file_path(
        output:str,
        chunk_counter: int
):
    """
    Chunk file path.
    """
    chunk_filename = f"othello-synthetic-train_{chunk_counter:05d}.bin"
    return os.path.join(output, chunk_filename)
# end def _chunk_file_path


def othello_generate(
        args: argparse.Namespace,
):
    """
    Generate multiple Othello games.

    If output_file is provided, games will be saved in chunks as they are generated to save memory.
    Otherwise, all games will be kept in memory and returned.

    Args:
        args (argparse.Namespace): Arguments parsed by argparse.
    """
    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
    # end if

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
            game = generate_othello_game(game_seed)
            games.append(game)
            game_lengths.append(len(game))

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
            # end if
        # end for
    # end with

    # If there are remaining games, and we're saving to a file
    if args.output is not None and games:
        if save_in_chunks:
            # Save the final chunk
            chunk_counter += 1
            save_games(games, _chunk_file_path(args.output, chunk_counter))
        else:
            # Save all games to a single file
            save_games(games, args.output)
        # end if
    # end if

    # Count for lengths
    counts = np.bincount(np.array(game_lengths))

    # Print count by length
    console.log(f"Generation ended, {len(games)} games generated:")
    for i in range(counts.shape[0]):
        if counts[i] > 0:
            console.log(f"\t- [bold green]{counts[i]:>5}[/] games with length [bold blue]{i}[/]")
        # end if
    # end for

    # Statistics
    console.log(f"Generation statistics")
    console.log(f"\tNumber of games: {counts.sum()}")
    console.log(f"\tTotal token generated: {sum(game_lengths)}")
    console.log(f"\tAverage tokens per game: {np.array(game_lengths).mean()}")
    console.log(f"\tStd tokens per game: {np.array(game_lengths).std()}")
# end othello_generate


def othello_play(args):
    """
    Start an interactive Othello game.
    """
    try:
        from boardGPT.games.othello import OthelloGame

        game = OthelloGame()
        info("Starting a new Othello game...")
        info("Use the 'visualize' command to play interactively with a GUI.")
        info("This is a placeholder for future implementation of a text-based game.")  # end try
    except ImportError as e:
        error(f"Othello functionality is not available due to missing dependencies: {e}")
    # end except

def othello_visualize(args):
    """
    Visualize an Othello game from a list of moves.
    """
    try:
        # First try to import the basic Othello functionality
        from boardGPT.games.othello import OthelloGame

        # Then try to import the visualization functionality
        try:
            from boardGPT.games.othello.viz import show_othello

            moves = args.moves.split(',') if args.moves else None

            if args.file:
                try:
                    with open(args.file, 'r') as f:
                        moves = [move.strip() for move in f.readlines()]  # end with  # end try
                except FileNotFoundError:
                    error(f"File {args.file} not found.")
                    return
                # end except
            if not moves:
                info("Starting visualization with an empty board...")  # end if
            else:
                info(f"Visualizing game with {len(moves)} moves...")
            # end else
            # Use the interactive visualization function
            show_othello(moves)  # end try  # end try
        except ImportError as e:
            error(f"Othello visualization is not available due to missing dependencies: {e}")  # end except
    except ImportError as e:
        error(f"Othello functionality is not available due to missing dependencies: {e}")

    # end except
def othello_verify(args):
    """
    Verify if a sequence of moves forms a valid Othello game.
    """
    try:
        from boardGPT.games.othello.othello_utils import verify_game

        moves = args.moves.split(',') if args.moves else []

        if args.file:
            try:
                with open(args.file, 'r') as f:
                    moves = [move.strip() for move in f.readlines()]  # end with  # end try
            except FileNotFoundError:
                error(f"File {args.file} not found.")
                return
            # end except
        if not moves:
            error("No moves provided. Use --moves or --file to provide moves.")
            return
        # end if
        is_valid, invalid_moves = verify_game(moves)

        if is_valid:
            success("The game is valid.")  # end if
        else:
            error("The game is invalid. Invalid moves:")
            for move in invalid_moves:
                error(f"  - {move}")  # end for  # end else  # end try
    except ImportError as e:
        error(f"Othello functionality is not available due to missing dependencies: {e}")

    # end except
def othello_convert(args):
    """
    Convert between different representations of Othello games.
    """
    try:
        from boardGPT.games.othello.othello_utils import game_to_board

        moves = args.moves.split(',') if args.moves else []

        if args.file:
            try:
                with open(args.file, 'r') as f:
                    moves = [move.strip() for move in f.readlines()]  # end with  # end try
            except FileNotFoundError:
                error(f"File {args.file} not found.")
                return
            # end except
        if not moves:
            error("No moves provided. Use --moves or --file to provide moves.")
            return
        # end if
        if args.to_board:
            try:
                board = game_to_board(moves)
                info("Board representation:")
                for i, piece in enumerate(board):
                    if i % 8 == 0 and i > 0:
                        console.print()  # end if
                    console.print(f"{piece} ", end="")  # end for
                console.print()  # end try
            except ValueError as e:
                error(f"{e}")  # end except  # end if
        else:
            info("Moves representation:")
            for i, move in enumerate(moves):
                info(f"{ i +1}. {move}")  # end for  # end else  # end try
    except ImportError as e:
        error(f"Othello functionality is not available due to missing dependencies: {e}")


def othello_load(args):
    """
    Load Othello games from a file.
    """
    try:
        from boardGPT.games.othello import load_games, convert_ids_to_notation
        if not args.file:
            print("Error: No file provided. Use --file to provide a file.")
            return
        # end if
        try:
            games = load_games(args.file)
            print(f"Loaded {len(games)} games from {args.file}.")
            if args.index is not None:
                if 0 <= args.index < len(games):
                    moves: List[str] = games[args.index]
                    print(f"Game {args.index} has {len(moves)} moves:")
                    moves = [m.decode("utf-8") for m in moves]
                    game = OthelloGame.load_moves(moves)
                    game.show()
                else:
                    print \
                        (f"Error: Index {args.index} is out of range. There are {len(games)} games.")  # end else  # end try  # end try
                # end if
            # end if
        except Exception as e:
            print(f"Error loading games: {e}")
        # end except
    except ImportError as e:
        print(f"Error: Othello functionality is not available due to missing dependencies: {e}")
    # end try
# end def othello_load


def othello_interpretability(args: argparse.Namespace):
    """
    Delegate interpretability subcommands to `boardGPT.interpretability.cli`.

    Usage from the games CLI:
        boardgpt othello interp smoke --seed 0
        boardgpt othello interp b2-metrics --layers 0,1
    """
    from boardGPT.interpretability.cli import main as interp_main

    interp_args = getattr(args, "interp_args", None)
    if interp_args is None:
        interp_args = []

    if len(interp_args) > 0 and interp_args[0] == "--":
        interp_args = interp_args[1:]

    return interp_main(interp_args)
