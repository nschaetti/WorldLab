

# Imports
import wandb
import os
import time
import json
import random
import math
import pickle
import argparse
import yaml
from contextlib import nullcontext
from typing import List, Dict, Any, Optional

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from worldlab.training import optimizer_step

from transformers import PreTrainedTokenizerFast

# BoardGPT
from boardGPT.datasets import (
    get_dataloader,
    infinite_loader
)

from boardGPT.models import GameAutoEncoder, build_vocab, build_tokenizer, save_checkpoint
from boardGPT.nn import GPTConfig, GPT
# from boardGPT.validation.metrics import is_valid_game_sequence, invalid_move_rate
from boardGPT.utils import (
    info,
    train_log,
    eval_log,
    TrainingConfig,
    setup_training_environment,
    get_lr,
    setup_optimizer
)


# -----------------------------------------------------------------------------
# Configuration handling
# -----------------------------------------------------------------------------

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


def initialize_model(
        config: TrainingConfig,
        device: str,
        ckpt_path: Optional[str] = None,
        start_iter: Optional[int] = None
):
    """
    Initialize the model based on checkpoint arguments or from scratch.

    Args:
        config (TrainingConfig): Configuration object
        device (str): Device to use for the model
        ckpt_path (str, optional): Path to a checkpoint file to load weights from.
        start_iter (int, optional): Iteration number to start from when resuming from a checkpoint.
                                   If provided with ckpt_path, overrides the iteration number in the checkpoint.

    Returns:
        tuple: (model, iter_num, best_val_loss, model_args) where model is the initialized GPT model,
               iter_num is the starting iteration number, best_val_loss is the best
               validation loss (used for checkpointing), and model_args are the model arguments.
    """
    # Initialize iteration counter and best validation loss
    iter_num = 0
    best_val_loss = 1e9

    # Set up model arguments from configuration
    model_args = dict(
        n_layer=config.n_layer,
        n_head=config.n_head,
        n_embd=config.n_embd,
        block_size=config.block_size,
        bias=config.bias,
        vocab_size=None,
        dropout=config.dropout,
        n_latent_token=config.n_latent_token,
        n_latent=config.n_latent,
    )

    # Vocab size
    vocab_size = config.vocab_size

    # Check if a specific checkpoint path is provided via command line
    if ckpt_path is not None:
        # Load the checkpoint
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        checkpoint_model_args = checkpoint['model_args']

        # Force these config attributes to be equal otherwise we can't resume training
        # The rest of the attributes (e.g. dropout) can stay as desired from command line
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
            model_args[k] = checkpoint_model_args[k]
        # end for

        # Create the model
        gptconf = GPTConfig(**model_args)
        model = GameAutoEncoder(gptconf)

        # Load the state dict
        state_dict = checkpoint['model']

        # Fix the keys of the state dictionary if needed
        unwanted_prefix = '_orig_mod.'
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            # end if
        # end for

        # Load weights
        model.load_state_dict(state_dict)

        # Set iteration number - use command line value if provided, otherwise use checkpoint value
        if start_iter is not None:
            iter_num = start_iter
        else:
            iter_num = checkpoint['iter_num']
        # end if

        best_val_loss = checkpoint['best_val_loss']
    else:
        # Create the model
        model_args['vocab_size'] = vocab_size if vocab_size is not None else 50304
        gptconf = GPTConfig(**model_args)
        model = GameAutoEncoder(gptconf)
    # end if

    # Crop down the model block size if desired, using model surgery
    if config.block_size < model.config.block_size:
        model.crop_block_size(config.block_size)
        model_args['block_size'] = config.block_size  # So that the checkpoint will have the right value
    # end if

    # Move model to the specified device
    model.to(device)

    return model, iter_num, best_val_loss, model_args
# end initialize_model


@torch.no_grad()
def estimate_loss(
        model,
        val_iter_data,
        train_iter_data,
        ctx,
        config,
        device
):
    """
    Estimate loss overtrain and validation splits using multiple batches.
    Also calculates an invalid move ratio for validation split if board_game is enabled.

    Args:
        model: The model to evaluate
        train_iter_data: The training data iterator
        val_iter_data: The iteration data to use
        dataset: The dataset to use
        tokenizer: The tokenizer to use
        ctx: Context manager for mixed precision
        config (TrainingConfig): Configuration object
        device (str): Device to use for tensors
        device_type (str): Type of device ('cuda' or 'cpu')

    Returns:
        dict: Contains average loss for 'train' and 'val' splits, and invalid_move_ratio for 'val' split if board_game is enabled
    """
    out = {}
    model.eval()

    for split in ['train', 'val']:
        losses = torch.zeros(config.eval_iters)

        # Iter data
        iter_data = train_iter_data if split == 'train' else val_iter_data

        # Error
        error_count = 0
        error_total = 0

        # For each eval iterations
        for k in range(config.eval_iters):
            # Use the appropriate data loading function based on the configuration
            X, _ = next(iter_data)
            X = X.to(device)

            with ctx:
                logits, loss = model(idx=X, targets=X)
            # end with

            # Count reconstruction error
            pred = logits.argmax(dim=-1)
            error_count += (pred != X).sum().item()
            error_total += X.nelement()

            # Keep loss
            losses[k] = loss.item()
        # end for

        out[split] = losses.mean()
        out[f"{split}_error_rate"] = error_count / error_total
    # end for

    model.train()
    return out
# end def estimate_loss


def main():
    """
    Main training function that orchestrates the entire training process.
    """
    # Parse command line arguments
    args = parse_args()

    # Load configuration from YAML file
    # config = load_config(args.config)
    info(f"Loading config file {args.config}")
    config = TrainingConfig.from_yaml(args.config)

    # Add data directory from command line arguments to the configuration
    config.data_dir = args.data_dir
    info(f"Using data directory: {config.data_dir}")

    # Set up the training environment
    info(f"Initialize environment")
    ddp, master_process, seed_offset, ddp_world_size, ddp_local_rank, device, device_type, ctx, gradient_accumulation_steps = setup_training_environment(
        config)

    # Initialize the model
    info(f"Initializing model")
    model, last_iter_num, best_val_loss, model_args = initialize_model(config, device, args.ckpt, args.num_iter)

    # Set up the optimizer and gradient scaler
    info(f"Initializing optimizer")
    optimizer, scaler, num_decay_params, num_nodecay_params = setup_optimizer(model, config, device_type)
    info(f"# decay params: {num_decay_params}")
    info(f"# nodecay: {num_nodecay_params}")

    # Compile the model if requested (requires PyTorch 2.0+)
    if config['compile']:
        info("compiling the model... (takes a ~minute)")
        model = torch.compile(model)
    # end if

    # Wrap model into DDP container for distributed training
    if ddp:
        model = DDP(
            module=model,
            device_ids=[ddp_local_rank] if ddp_local_rank is not None else None
        )
    # end if

    # Set up wandb logging if enabled
    if config['wandb_log'] and master_process:
        if args.run_id:
            wandb.init(
                project=config['wandb_project'],
                name=config['wandb_run_name'],
                id=args.run_id,
                resume='allow',
                config=config.to_dict()
            )
        else:
            wandb.init(
                project=config['wandb_project'],
                name=config['wandb_run_name'],
                config=config.to_dict()
            )
        # end if
    # end if

    # Training loop initialization
    # Use the appropriate data loading function based on the configuration
    vocab = build_vocab(output=config.out_dir)
    tokenizer = build_tokenizer(vocab=vocab, output=config.out_dir)

    dataloader = get_dataloader(split="train", config=config, tokenizer=tokenizer)
    val_dataloader = get_dataloader(split="val", config=config, tokenizer=tokenizer)
    val_data_iter = infinite_loader(val_dataloader)
    data_iter = infinite_loader(dataloader)

    # Get first batch
    X, _ = next(data_iter)
    X = X.to(device)

    t0 = time.time()
    local_iter_num = 0  # Number of iterations in the lifetime of this process
    raw_model = model.module if ddp else model  # Unwrap DDP container if needed
    running_mfu = -1.0

    # Main training loop
    for iter_num in range(last_iter_num, last_iter_num + args.num_iter + 1):
        # Determine and set the learning rate for this iteration
        lr = get_lr(iter_num, config) if config['decay_lr'] else config['learning_rate']
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        # end for

        # Evaluate the loss on train/val sets and write checkpoints
        if iter_num % config['eval_interval'] == 0 and master_process:
            # Estimate loss
            losses = estimate_loss(
                model=model,
                train_iter_data=data_iter,
                val_iter_data=val_data_iter,
                ctx=ctx,
                config=config,
                device=device
            )

            # Log eval
            eval_log(
                f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, "
                f"train error rate: {losses['train_error_rate']*100}, "
                f"val error rate: {losses['val_error_rate']*100}"
            )

            # Log to wandb if enabled
            if config['wandb_log']:
                log_data = {
                    "iter": iter_num,
                    "val/loss": losses['val']
                }

                # Log invalid move ratio if available
                log_data["train/error_rate"] = losses['train_error_rate']
                log_data["val/error_rate"] = losses['val_error_rate']
                wandb.log(log_data)
            # end if

            # Save checkpoint if validation loss improved or if always_save_checkpoint is True
            if losses['val'] < best_val_loss or config['always_save_checkpoint']:
                best_val_loss = losses['val']
                if iter_num > 0:
                    save_checkpoint(raw_model, optimizer, iter_num, best_val_loss, config, model_args)
                # end if
            # end if
        # end if

        # Exit if eval_only flag is set
        if iter_num == 0 and config['eval_only']:
            break
        # end if

        # Forward backward update, with optional gradient accumulation to simulate larger batch size
        # and using the GradScaler if data type is float16
        for micro_step in range(gradient_accumulation_steps):
            if ddp:
                # In DDP training, we only need to sync gradients at the last micro step.
                # The official way to do this is with model.no_sync() context manager, but
                # looking at the source of that context manager, it just toggles this variable
                model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
            # end if

            # Forward pass
            with ctx:
                logits, loss = model(idx=X, targets=X)

                # Scale the loss to account for gradient accumulation
                loss = loss / gradient_accumulation_steps
            # end with

            # Immediately async prefetch next batch while model is doing the forward pass on the GPU
            # Use the appropriate data loading function based on the configuration
            X, _ = next(data_iter)
            X = X.to(device)

            # Backward pass, with gradient scaling if training in fp16
            scaler.scale(loss).backward()
        # end for

        optimizer_step(
            optimizer, scaler=scaler, parameters=model.parameters(), grad_clip=config['grad_clip']
        )

        # Timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1

        if iter_num % config['log_interval'] == 0 and master_process:
            # Get loss as float. Note: this is a CPU-GPU sync point
            # Scale up to undo the division above, approximating the true total loss
            lossf = loss.item() * gradient_accumulation_steps

            # Calculate model flops utilization (MFU)
            if local_iter_num >= 5:  # Let the training loop settle a bit
                mfu = raw_model.estimate_mfu(
                    n_layer=config.n_layer,
                    n_head=config.n_head,
                    n_embd=config.n_embd,
                    n_latent=config.n_latent,
                    n_latent_token=config.n_latent_token,
                    block_size=config.block_size,
                    fwdbwd_per_iter=config['batch_size'] * gradient_accumulation_steps,
                    dt=dt
                )
                running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
            # end if

            train_log(
                f"iter {iter_num:04d}: loss {lossf:02.4f}, time {dt * 1000:05.2f}ms, mfu {running_mfu * 100:04.2f}%"
            )

            if config['wandb_log'] and master_process:
                wandb.log({
                    "iter": iter_num,
                    "train/loss": lossf,
                    "lr": lr,
                    "mfu": running_mfu * 100,  # Convert to percentage
                })
            # end if
        # end if

        # Increment iteration counters
        iter_num += 1
        local_iter_num += 1

        # Check termination condition
        if iter_num > config['max_iters']:
            break
        # end if
    # end for epochs

    # Clean up a distributed process group if using DDP
    if ddp:
        destroy_process_group()
    # end if


# end def main

# Execute the main function if a script is run directly
if __name__ == "__main__":
    main()
# end if

