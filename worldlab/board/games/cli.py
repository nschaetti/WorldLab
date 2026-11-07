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

import argparse
import sys
import importlib
from typing import List, Optional


def _othello_generate(args):
    from .othello.cli import othello_generate
    return othello_generate(args)


def _othello_play(args):
    from .othello.cli import othello_play
    return othello_play(args)


def _othello_visualize(args):
    from .othello.cli import othello_visualize
    return othello_visualize(args)


def _othello_verify(args):
    from .othello.cli import othello_verify
    return othello_verify(args)


def _othello_convert(args):
    from .othello.cli import othello_convert
    return othello_convert(args)


def _othello_load(args):
    from .othello.cli import othello_load
    return othello_load(args)


def _othello_interpretability(args):
    from worldlab.board.interpretability.cli import main as interp_main

    interp_args = getattr(args, "interp_args", [])
    if len(interp_args) > 0 and interp_args[0] == "--":
        interp_args = interp_args[1:]
    return interp_main(interp_args)


    # end except
def checkers_info(args):
    """
    Display information about the Checkers implementation status.
    """
    print("Checkers implementation is not yet available.")
    print("This is a placeholder for future implementation.")
# def checkers_info

# end def checkers_info
def main(argv: Optional[List[str]] = None):
    """
    Main entry point for the CLI.
    """
    parser = argparse.ArgumentParser(description="Command Line Interface for boardGPT games.")
    subparsers = parser.add_subparsers(dest="game", help="Game to interact with")
    
    # Othello commands
    othello_parser = subparsers.add_parser("othello", help="Commands for Othello game")
    othello_subparsers = othello_parser.add_subparsers(dest="command", help="Command to execute")

    # Othello generate
    othello_generate_parser = othello_subparsers.add_parser("generate", help="Generate Othello game sequences")
    othello_generate_parser.add_argument("--num-games", required=True, help="Othello game to generate")
    othello_generate_parser.add_argument("--seed", default=42, help="Othello game seed")
    othello_generate_parser.add_argument("--output", required=True, help="Path to the output .bin file")
    othello_generate_parser.add_argument("--chunk-size", default=None, help="Size of chunks (how many samples)")
    othello_generate_parser.set_defaults(func=_othello_generate)
    
    # Othello play
    othello_play_parser = othello_subparsers.add_parser("play", help="Start an interactive Othello game")
    othello_play_parser.set_defaults(func=_othello_play)
    
    # Othello visualize
    othello_visualize_parser = othello_subparsers.add_parser("visualize", help="Visualize an Othello game")
    othello_visualize_parser.add_argument("--moves", help="Comma-separated list of moves (e.g., 'd3,c4,e3')")
    othello_visualize_parser.add_argument("--file", help="File containing moves (one per line)")
    othello_visualize_parser.set_defaults(func=_othello_visualize)
    
    # Othello verify
    othello_verify_parser = othello_subparsers.add_parser("verify", help="Verify if a sequence of moves forms a valid Othello game")
    othello_verify_parser.add_argument("--moves", help="Comma-separated list of moves (e.g., 'd3,c4,e3')")
    othello_verify_parser.add_argument("--file", help="File containing moves (one per line)")
    othello_verify_parser.set_defaults(func=_othello_verify)
    
    # Othello convert
    othello_convert_parser = othello_subparsers.add_parser("convert", help="Convert between different representations of Othello games")
    othello_convert_parser.add_argument("--moves", help="Comma-separated list of moves (e.g., 'd3,c4,e3')")
    othello_convert_parser.add_argument("--file", help="File containing moves (one per line)")
    othello_convert_parser.add_argument("--to-board", action="store_true", help="Convert moves to board representation")
    othello_convert_parser.set_defaults(func=_othello_convert)
    
    # Othello load
    othello_load_parser = othello_subparsers.add_parser("load", help="Load Othello games from a file")
    othello_load_parser.add_argument("--file", required=True, help="File containing games")
    othello_load_parser.add_argument("--index", type=int, help="Index of the game to display")
    othello_load_parser.set_defaults(func=_othello_load)

    # Othello interpretability bridge
    othello_interp_parser = othello_subparsers.add_parser(
        "interp",
        help="Run interpretability CLI (smoke, b2-metrics, b2-attrib, report)",
    )
    othello_interp_parser.add_argument(
        "interp_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to worldlab.board.interpretability.cli",
    )
    othello_interp_parser.set_defaults(func=_othello_interpretability)
    
    # Checkers commands
    checkers_parser = subparsers.add_parser("checkers", help="Commands for Checkers game")
    checkers_subparsers = checkers_parser.add_subparsers(dest="command", help="Command to execute")
    
    # Checkers info
    checkers_info_parser = checkers_subparsers.add_parser("info", help="Display information about the Checkers implementation status")
    checkers_info_parser.set_defaults(func=checkers_info)
    
    # Parse arguments
    args = parser.parse_args(argv)
    
    # Execute the appropriate function
    if hasattr(args, "func"):
        result = args.func(args)
        if isinstance(result, int):
            return result
        return 0
    else:
        parser.print_help()
        return 1
     # end if
# end def main

    # end else
if __name__ == "__main__":
    raise SystemExit(main())
# end if
