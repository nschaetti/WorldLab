"""Save board training state using the established checkpoint schema."""

from pathlib import Path

import torch

from worldlab.board.utils.logging import info
from worldlab.training.runtime import atomic_write


def save_board_checkpoint(out_dir, model, optimizer, iter_num, best_val_loss, config, model_args):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "model_args": model_args,
        "iter_num": iter_num,
        "best_val_loss": best_val_loss,
        "config": config,
    }
    path = Path(out_dir) / f"ckpt_iter{iter_num:04d}.pt"
    info(f"Saving checkpoint to {path}")
    atomic_write(path, lambda temporary: torch.save(checkpoint, temporary))


def save_checkpoint(model, optimizer, iter_num, best_val_loss, config, model_args):
    save_board_checkpoint(
        config["out_dir"], model, optimizer, iter_num, best_val_loss, config.to_dict(), model_args,
    )
