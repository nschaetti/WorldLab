"""Command-line options shared by the board sequence trainers."""

import argparse


def parse_args():
    """
    Parse command line arguments for the training script.

    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description='Train a GPT model for board games')
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to the YAML configuration file'
    )

    parser.add_argument(
        '--data-dir',
        type=str,
        required=True,
        help='Path to the data directory. This directory must contain "train" and "val" folders with bin files for each split.'
    )

    parser.add_argument(
        '--ckpt',
        type=str,
        default=None,
        help='Path to a checkpoint file (.pt) to load weights from. If not specified, a new model will '
             'be initialized from scratch.'
    )

    parser.add_argument(
        '--num-iter',
        type=int,
        default=None,
        help='Iteration number to start from when resuming from a checkpoint. Used with --ckpt.'
    )

    parser.add_argument(
        '--run-id',
        type=str,
        default=None,
        help="Run ID if the run must be resumed."
    )

    return parser.parse_args()
# end def parse_args
