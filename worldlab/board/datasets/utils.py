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

import os
import pickle
from typing import Tuple, List, Union

import numpy as np
import torch
try:
    from transformers import PreTrainedTokenizerFast
except ModuleNotFoundError:  # pragma: no cover
    from typing import Any as PreTrainedTokenizerFast

from .game_dataset import GameDataset


def load_othello_data_files(
        data_dir: str,
        split: str,
        data_filename: str,
        flatten: bool = True,
        log: bool = True,  # end def load_othello_dataset
) -> Union[List[int], List[List[int]]]:
    """
    ...
    """
    if log: print(f"Loading {split} data into memory...")

    # Data dir for the specified split (train or val)
    data_dir = os.path.join(data_dir, split)

    # Pattern for bin files
    pattern = "*.bin"

    # Find all matching bin files
    import glob
    bin_files = glob.glob(os.path.join(data_dir, pattern))

    if not bin_files:
        # If no bin files found in the specified directory, print an error message
        print(
            f"Error: No bin files found in {data_dir}. Make sure the data directory contains 'train' and "
            f"'val' folders with bin files."
        )

        # Fallback to old method if no matching files found
        fallback_data_dir = os.path.join("data", "othello")

        print(f"Falling back to {os.path.join(fallback_data_dir, data_filename)}")

        with open(os.path.join(fallback_data_dir, data_filename), 'rb') as f:
            game_sequences = pickle.load(f)  # end with
        # end with
    else:
        # Load all bin files and combine their data
        if log: print(f"Found {len(bin_files)} bin files for {split} split")
        game_sequences = []
        for bin_file in bin_files:
            if log: print(f"Loading {bin_file}...")
            with open(bin_file, 'rb') as f:
                sequences = pickle.load(f)
                game_sequences.extend(sequences)  # end with
            # end with
        # end for
    # end if

    # Concatenate game sequences
    if flatten:
        return [x for sublist in game_sequences for x in sublist.tolist()]
    else:
        return game_sequences
    # end if
# end def load_othello_dataset


def collate_fn(batch, tokenizer: PreTrainedTokenizerFast):
    """
    Convert a batch of raw strings into padded tensors.

    Args:
        batch (list): A batch of raw strings.
        tokenizer (PreTrainedTokenizerFast): Tokenizer object used to tokenize the batch.
        max_length (int, optional): Maximum length of the padded tensors.

    Returns:
        tuple: padded tensors and padded lengths.
    """
    enc = tokenizer(
        batch,
        return_tensors="pt"
    )

    # Sequence length
    seq_len = enc["input_ids"].shape[-1] // 2

    # Split into X and Y
    X = enc["input_ids"][:, :seq_len]
    Y = enc["input_ids"][:, seq_len:]

    return X, Y
# end def collate_fn


def collate_fn_board(batch, tokenizer: PreTrainedTokenizerFast):
    """
    Convert a batch of raw strings into padded tensors.
    """
    moves = [m for (m, _) in batch]
    enc = tokenizer(
        moves,
        return_tensors="pt"
    )

    # Board states
    board_states = [bs for (_, bs) in batch]
    board_states = np.array(board_states)
    Y = torch.LongTensor(board_states)

    # Split into X and Y
    X = enc["input_ids"]

    return X, Y
# end def collate_fn_board


def get_dataloader(
        split: str,
        config,
        tokenizer: PreTrainedTokenizerFast,
) -> torch.utils.data.DataLoader:
    """
    Get dataloaders for training and validation.

    Args:
        split (str): 'train' or 'val' to specify which data split to use
        config (TrainingConfig): Configuration object containing 'data_dir' which points to a directory
        with 'train' and 'val' folders containing bin files for each split
        tokenizer (PreTrainedTokenizerFast): Tokenizer object used to tokenize the batch
    """
    dataset = GameDataset(
        data_dir=config.data_dir,
        split=split,
        block_size=config.block_size,
        ood_perc=config.ood_perc,
        num_samples=config.num_samples
    )

    # Create a dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        shuffle=config.shuffle,
        drop_last=config.drop_last,
        collate_fn=lambda b: collate_fn(b, tokenizer)
    )

    return dataloader
# end def get_dataloader


def infinite_loader(
        dataloader: torch.utils.data.DataLoader
):
    """
    Infinite loader that returns batches from dataloader.
    """
    while True:
        for batch in dataloader:
            X, Y = batch
            yield X, Y
        # end for
    # end while
# end def infinite_loader
