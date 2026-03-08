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

from .board_dataset import BoardDataset
from .game_dataset import GameDataset
from .utils import (
    load_othello_data_files,
    get_dataloader,
    collate_fn,
    infinite_loader,
    collate_fn_board
)
from .symbolic_stats import (
    load_symbolic_pickle,
    compute_cooccurrence_matrix,
    top_cooccurring_pairs,
    grouped_label_order,
    plot_grouped_cooccurrence_matrix,
)

__all__ = [
    "BoardDataset",
    "GameDataset",
    "load_othello_data_files",
    "get_dataloader",
    "collate_fn",
    "collate_fn_board",
    "infinite_loader",
    "load_symbolic_pickle",
    "compute_cooccurrence_matrix",
    "top_cooccurring_pairs",
    "grouped_label_order",
    "plot_grouped_cooccurrence_matrix",
]
