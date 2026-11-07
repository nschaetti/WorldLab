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
Visualization utilities for worldlab.board.

This module provides visualization functions for attention matrices, board states, and other
visualization utilities for the boardGPT project.
"""

import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.animation import FuncAnimation
import numpy as np
from typing import List, Tuple, Optional, Union
import sys
import os
import seaborn as sns
import random


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
        return False  # end except
    # end try-except
# end def is_jupyter


def plot_attention_matrix(
        qk_matrix,
        tokens,
        head_idx=0,
        normalize=False  # end def plot_attention_matrix
):
    """
    Display the QK similarity matrix (before softmax).
    
    This function visualizes the attention matrix for a specific attention head.
    
    Args:
        qk_matrix (torch.Tensor): Tensor of shape (batch, n_heads, seq_len, seq_len)
        tokens (List[str]): List of tokens associated with the sequence
        head_idx (int): Index of the head to display
        normalize (bool): If True, applies softmax to convert to attention distribution
        
    Returns:
        None: Displays the plot directly
    """
    # Select the requested head
    att = qk_matrix[0, head_idx].detach().cpu().numpy()

    # Optional: softmax normalization (otherwise it's just raw QK^T)
    if normalize:
        import torch.nn.functional as F
        import torch
        att = torch.softmax(torch.tensor(att), dim=-1).numpy()  # end if
    # end if

    # White → red gradient
    cmap = plt.cm.Reds
    cmap.set_under("white")  # for exact 0

    plt.figure(figsize=(8, 6))
    plt.imshow(att, cmap=cmap, interpolation="nearest", vmin=0, vmax=1)
    plt.title(f"Attention (Head {head_idx})")
    plt.xlabel("Key tokens")
    plt.ylabel("Query tokens")
    plt.xticks(range(len(tokens)), tokens, rotation=90)
    plt.yticks(range(len(tokens)), tokens)
    plt.grid(False)
    plt.colorbar().remove()  # no side bar
    plt.show()
# end def plot_attention_matrix


def plot_heads_attention(qk_matrix, tokens, layer_idx=0):
    """
    Affiche toutes les têtes d'une couche dans une seule figure,
    avec un dégradé blanc (0.0) → rouge (1.0).

    Args:
        qk_matrix: torch.Tensor (batch, n_heads, seq_len, seq_len)
        tokens: liste de str (tokens associés)
        layer_idx: index de la couche
    """
    att = qk_matrix[0].detach().cpu().numpy()  # shape: (n_heads, seq_len, seq_len)
    n_heads = att.shape[0]

    # Grille auto (2 colonnes pour la lisibilité)
    ncols = 4 if n_heads >= 4 else n_heads
    nrows = int(np.ceil(n_heads / ncols))

    cmap = plt.cm.Reds
    cmap.set_under("white")

    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
    axes = axes.flatten()

    for h in range(n_heads):
        ax = axes[h]
        ax.imshow(att[h], cmap=cmap, interpolation="nearest", vmin=0, vmax=1)
        ax.set_title(f"Layer {layer_idx}, Head {h}")
        ax.set_xticks(range(len(tokens)))
        ax.set_yticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=90, fontsize=6)
        ax.set_yticklabels(tokens, fontsize=6)
        ax.grid(False)  # end for
    # cacher les cases vides si n_heads < nrows*ncols

    for h in range(n_heads, len(axes)):
        axes[h].axis("off")
    # end for
    plt.tight_layout()
    plt.show()  # end def plot_heads_attention
# end def plot_heads_attention


def show_linear_probe_samples(data, num_samples=3, random_seed=None):
    """
    Display random samples from the linear probe training data.
    
    This function visualizes the relationship between move sequences (x) and 
    board representations (y) for random samples from the linear probe training data.
    
    Args:
        data (Dict[int, Tuple[torch.Tensor, torch.LongTensor]]): Dictionary where keys are sequence lengths
            and values are tuples of (x, y) where:
            - x: tensor of move sequences (shape: [num_sequences, sequence_length])
            - y: tensor of board representations (shape: [num_sequences, 64])
        num_samples (int): Number of random samples to display (default: 3)
        random_seed (int, optional): Random seed for reproducibility
    
    Returns:
        Union[plt.Figure, None]: In Jupyter notebooks, returns the figure object for inline display.
                                In regular Python scripts, returns None after displaying the figure.
    """
    # Import here to avoid circular imports
    from worldlab.board.games.othello import OthelloGame
    from worldlab.board.games import create_id_to_move_mapping
    
    if random_seed is not None:
        random.seed(random_seed)
    # end if
    # Get the mapping from move IDs to move notations
    id_to_move = create_id_to_move_mapping()
    
    # Create a figure with subplots for each sample
    fig = plt.figure(figsize=(15, 5 * num_samples))
    
    # Get all available sequence lengths
    lengths = list(data.keys())
    
    # Counter for subplot positioning
    subplot_idx = 1
    
    # For each sample
    for i in range(num_samples):
        # Randomly select a sequence length
        length_idx = random.randint(0, len(lengths) - 1)
        length = lengths[length_idx]
        
        # Get the data for this length
        x, y = data[length]
        
        # Randomly select a sample from this length
        sample_idx = random.randint(0, x.shape[0] - 1)
        
        # Get the move sequence and board representation for this sample
        move_sequence = x[sample_idx].cpu().numpy()
        board_repr = y[sample_idx].cpu().numpy()
        
        # Convert move IDs to move notations
        move_notations = [id_to_move[move_id.item()] for move_id in x[sample_idx] if move_id.item() != 0]
        
        # Create subplot for the move sequence
        ax1 = fig.add_subplot(num_samples, 2, subplot_idx)
        subplot_idx += 1
        
        # Display the move sequence
        ax1.axis('off')
        ax1.set_title(f"Sample {i+1}: Move Sequence (length {length})")
        
        # Format the move sequence as a readable string
        move_text = "Moves: " + " → ".join(move_notations)
        ax1.text(0.5, 0.5, move_text, ha='center', va='center', wrap=True, fontsize=12)
        
        # Create subplot for the board representation
        ax2 = fig.add_subplot(num_samples, 2, subplot_idx)
        subplot_idx += 1
        
        # Display the board representation as a visual board
        ax2.set_title(f"Sample {i+1}: Board Representation")
        
        # Draw the green background
        ax2.add_patch(plt.Rectangle((0, 0), 8, 8, color='green'))
        
        # Draw the grid lines
        for j in range(9):
            ax2.plot([j, j], [0, 8], 'k-', lw=1)
            ax2.plot([0, 8], [j, j], 'k-', lw=1)
        # end for
        # Draw the pieces based on the board representation
        for row in range(8):
            for col in range(8):
                # Calculate the index in the 1D board representation
                # The board representation is ordered by columns then rows (a1, a2, ..., h8)
                idx = col * 8 + row
                
                piece = board_repr[idx]
                
                if piece == 1:  # White
                    ax2.add_patch(plt.Circle((col + 0.5, row + 0.5), 0.4, color='white'))
                elif piece == 2:  # Black
                    ax2.add_patch(plt.Circle((col + 0.5, row + 0.5), 0.4, color='black'))
            # end for
        # Add column and row labels
        ax2.set_xticks([i + 0.5 for i in range(8)])
        ax2.set_yticks([i + 0.5 for i in range(8)])
        ax2.set_xticklabels(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'])
        ax2.set_yticklabels(['1', '2', '3', '4', '5', '6', '7', '8'])
        
        # Set limits and aspect
        ax2.set_xlim(0, 8)
        ax2.set_ylim(0, 8)
        ax2.set_aspect('equal')
    # end for
    # Apply tight layout to the figure
    plt.tight_layout()
    
    # Check if running in Jupyter notebook
    if is_jupyter():
        # In Jupyter, return the figure for inline display
        return fig  # end if
    else:
        # In regular Python scripts, show the figure and return None
        plt.show()
        return None  # end else  # end def show_linear_probe_samples
# end def show_linear_probe_samples
