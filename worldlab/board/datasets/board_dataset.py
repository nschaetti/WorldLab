"""
Copyright (C) 2025 Nils Schaetti

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

# Imports
import os
import pickle
import random
import glob
from typing import List, Callable

import numpy as np
from torch.utils.data import Dataset

from .game_dataset import GameDataset


MAP_START_INDEX = 1


# Board Dataset
class BoardDataset(GameDataset):
    """
    Game Dataset (to learn board state)
    """

    # Constructor
    def __init__(
            self,
            data_dir: str,
            board_func: Callable,
            max_len: int = 60,
            block_size: int = 60,
            split: str = 'train',
            ood_perc: float = 0.,
            num_samples: int = -1,
            padding_int: int = 0
    ):
        """
        Othello Dataset

        Args:
             data_dir: path to Othello dataset
             board_func: function that takes board state as input and returns board state as output
             max_len (int, optional): max length of game sequence
             block_size (int, optional): block size of game sequence
             split: train or test
             ood_perc: percentage of Othello dataset to use
             num_samples: number of samples to use
             padding_int (int): padding integer for Othello dataset
        """
        # Super
        super().__init__(
            data_dir=data_dir,
            max_len=max_len,
            block_size=block_size,
            split=split,
            ood_perc=ood_perc,
            num_samples=num_samples,
            padding_int=padding_int
        )

        # Properties
        self.board_func = board_func
    # end def __init__

    # region PUBLIC

    # endregion PUBLIC

    # region OVERWRITE

    def __getitem__(self, idx):
        """
        Get item from dataset
        """
        # Get a game sequence
        game_sequence: np.bytes_ = self.data[idx]

        # Get a random position
        ix = random.randint(1, len(game_sequence) - 1)

        # Get subsequence and pad
        x = game_sequence[:ix]

        # Decode
        x = [s.decode() for s in x]

        # Transform into board
        # Create an OthelloGame object with the moves applied
        board_state = self.board_func(x)

        x = ["<pad>"] * (self.block_size - ix) + x

        # Transform in text
        x = " ".join(x)

        return x, board_state
    # end __getitem__

    # endregion OVERWRITE

# end class BoardDataset

