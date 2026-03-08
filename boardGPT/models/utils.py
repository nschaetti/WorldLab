"""
Utility helpers for saving boardGPT model checkpoints and related metadata.
"""

# Imports
import os
import torch

from boardGPT.utils.logging import info


def save_checkpoint(
        model,
        optimizer,
        iter_num,
        best_val_loss,
        config,
        model_args
):
    """
    Save a checkpoint of the model and optimizer state.

    Args:
        model: The model to save
        optimizer: The optimizer to save
        iter_num (int): Current iteration number
        best_val_loss (float): Best validation loss so far
        config (TrainingConfig): Configuration object
        model_args (dict): Model arguments
    """
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'model_args': model_args,
        'iter_num': iter_num,
        'best_val_loss': best_val_loss,
        'config': config.to_dict(),
    }

    # Save with standard name for backward compatibility
    iter_filename = f'ckpt_iter{iter_num:04d}.pt'
    info(f"Saving checkpoint to {os.path.join(config['out_dir'], iter_filename)}")
    torch.save(checkpoint, os.path.join(config['out_dir'], iter_filename))
    info(f"Also saved checkpoint as {iter_filename}")
# end save_checkpoint

