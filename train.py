"""
BoardGPT Training Script
========================

This script trains a GPT model for board games. It supports both single-GPU and
distributed data parallel (DDP) training configurations.

The script handles:
- Model initialization (from scratch, resume from checkpoint, or pretrained GPT-2)
- Data loading and batching
- Training loop with gradient accumulation
- Learning rate scheduling
- Evaluation and checkpointing
- Logging (console and optional wandb integration)

Examples:
---------
To run on a single GPU:
$ python train.py --data_dir=/path/to/data --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node:
$ torchrun --standalone --nproc_per_node=4 train.py --data_dir=/path/to/data

To run with DDP on 4 gpus across 2 nodes:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py --data_dir=/path/to/data
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py --data_dir=/path/to/data
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)

Note: The data directory must contain 'train' and 'val' folders with bin files for each split.
"""
import json

# Imports
import wandb
import os
import time
import random
import pickle
import argparse
import yaml
from contextlib import nullcontext
from typing import List, Dict, Any

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from worldlab.training import optimizer_step
from boardGPT.utils.training import get_lr

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast
from tokenizers.pre_tokenizers import Whitespace

# BoardGPT
from boardGPT.datasets import load_othello_data_files, GameDataset
from boardGPT.models import GameGPT
from boardGPT.nn import GPTConfig, GPT
from boardGPT.validation.metrics import is_valid_game_sequence, invalid_move_rate
from boardGPT.utils import info, error, warning, train_log, eval_log

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

class TrainingConfig:
    """
    Configuration class for training that loads from a YAML file and provides
    property-based access to configuration values.
    """
    def __init__(self, config_dict=None):
        """
        Initialize the training configuration with a dictionary.
        
        Args:
            config_dict (dict, optional): Dictionary containing configuration values.
                                         If None, an empty dictionary is used.
        """
        self._config = config_dict or {}

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the configuration to a dictionary.
        """
        return self._config
    # end def to_dict
        
    @classmethod
    def from_yaml(cls, config_path):
        """
        Load configuration from a YAML file.
        
        Args:
            config_path (str): Path to the YAML configuration file
            
        Returns:
            TrainingConfig: Configuration object
        """
        try:
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
            # end with
            return cls(config_dict)
        except FileNotFoundError:
            error(f"Configuration file {config_path} not found. Using default configuration.")
            return cls({})
        except yaml.YAMLError as e:
            error(f"Error parsing YAML configuration file: {e}")
            return cls({})
        # end try
    
    def __getattr__(self, name):
        """
        Get a configuration value by attribute name.
        
        Args:
            name (str): Name of the configuration property
            
        Returns:
            Any: Value of the configuration property
            
        Raises:
            AttributeError: If the property doesn't exist in the configuration
        """
        if name in self._config:
            return self._config[name]
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute "
            f"'{name}' (available attributes: {self._config.keys()})"
        )
    
    def __getitem__(self, key):
        """
        Get a configuration value by dictionary-style access.
        
        Args:
            key (str): Name of the configuration property
            
        Returns:
            Any: Value of the configuration property
            
        Raises:
            KeyError: If the property doesn't exist in the configuration
        """
        if key in self._config:
            return self._config[key]
        # end if
        raise KeyError(key)
    
    def __setattr__(self, name, value):
        """
        Set a configuration value by attribute name.
        
        Args:
            name (str): Name of the configuration property
            value (Any): Value to set
        """
        if name == '_config':
            super().__setattr__(name, value)
        else:
            self._config[name] = value
    
    def __setitem__(self, key, value):
        """
        Set a configuration value by dictionary-style access.
        
        Args:
            key (str): Name of the configuration property
            value (Any): Value to set
        """
        self._config[key] = value
    
    def get(self, name, default=None):
        """
        Get a configuration value with a default fallback.
        
        Args:
            name (str): Name of the configuration property
            default (Any, optional): Default value if property doesn't exist
            
        Returns:
            Any: Value of the configuration property or default
        """
        return self._config.get(name, default)
    
    def __contains__(self, name):
        """
        Check if a configuration property exists.
        
        Args:
            name (str): Name of the configuration property
            
        Returns:
            bool: True if property exists, False otherwise
        """
        return name in self._config

def load_config(config_path):
    """
    Load configuration from a YAML file.
    
    Args:
        config_path (str): Path to the YAML configuration file
        
    Returns:
        TrainingConfig: Configuration object
    """
    return TrainingConfig.from_yaml(config_path)
# end def load_config

def setup_training_environment(config):
    """
    Set up the training environment including distributed training, 
    random seeds, and device configuration.
    
    Args:
        config (TrainingConfig): Configuration object
        
    Returns:
        tuple: Contains various setup parameters including master_process flag,
               device_type, context manager for mixed precision, etc.
    """
    # Check if this is a distributed data parallel (DDP) run
    ddp = int(os.environ.get('RANK', -1)) != -1

    # Initialize variables
    ddp_rank = None
    ddp_local_rank = None
    ddp_world_size = None
    master_process = True
    seed_offset = 0
    gradient_accumulation_steps = config['gradient_accumulation_steps']
    device = config.device

    if ddp:
        # Initialize the distributed process group
        init_process_group(backend=config['backend'])
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0  # This process will do logging, checkpointing etc.
        seed_offset = ddp_rank  # Each process gets a different seed

        # Scale down gradient accumulation steps proportionally to world size
        assert gradient_accumulation_steps % ddp_world_size == 0
        gradient_accumulation_steps //= ddp_world_size
    else:
        # If not DDP, we are running on a single GPU with one process
        ddp_world_size = 1
        # Use the device specified in the configuration
    # end if
    
    # Calculate tokens per iteration for logging
    tokens_per_iter = gradient_accumulation_steps * ddp_world_size * config['batch_size'] * config['block_size']
    info(f"tokens per iteration will be: {tokens_per_iter:,}")
    
    # Create output directory if needed (only on master process)
    if master_process:
        os.makedirs(config['out_dir'], exist_ok=True)
    # end if
    
    # Set random seed for reproducibility
    torch.manual_seed(1337 + seed_offset)
    
    # Enable TF32 precision on CUDA devices (faster and usually sufficient precision)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # torch.backends.cudnn.conv.fp32_precision = 'tf32'
    # torch.backends.cuda.matmul.fp32_precision = 'ieee'
    
    # Determine device type for later use
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    
    # Set up precision type for training
    dtype = config['dtype']
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    
    # Create context manager for mixed precision training
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    
    return ddp, master_process, seed_offset, ddp_world_size, ddp_local_rank, device, device_type, ctx, gradient_accumulation_steps
# end setup_training_environment

# Global variables to store loaded data
_train_data = None
_val_data = None

def get_board_batch(split, config, device, device_type):
    """
    Get a random batch of board game data from the specified split.
    
    Args:
        split (str): 'train' or 'val' to specify which data split to use
        config (TrainingConfig): Configuration object containing 'data_dir' which points to a directory
                                with 'train' and 'val' folders containing bin files for each split
        device (str): Device to use for tensors
        device_type (str): Type of device ('cuda' or 'cpu')
        
    Returns:
        tuple: (x, y) where x is input tokens and y is target tokens
    """
    global _train_data, _val_data
    
    # Load all data into memory if not already loaded
    if (split == 'train' and _train_data is None) or (split == 'val' and _val_data is None):
        info(f"Loading {split} data into memory...")
        
        # Data dir for the specified split (train or val)
        data_dir = os.path.join(config.data_dir, split)
        
        # Pattern for bin files
        pattern = "*.bin"
        
        # Find all matching bin files
        import glob
        bin_files = glob.glob(os.path.join(data_dir, pattern))
        
        if not bin_files:
            # If no bin files found in the specified directory, print an error message
            info(f"Error: No bin files found in {data_dir}. Make sure the data directory contains 'train' and 'val' folders with bin files.")
            
            # Fallback to old method if no matching files found
            fallback_data_dir = os.path.join("data", config['board_game'])
            data_filename = config['train_data_filename'] if split == 'train' else config['val_data_filename']
            
            info(f"Falling back to {os.path.join(fallback_data_dir, data_filename)}")
            
            with open(os.path.join(fallback_data_dir, data_filename), 'rb') as f:
                game_sequences = pickle.load(f)
            # end with
        else:
            # Load all bin files and combine their data
            info(f"Found {len(bin_files)} bin files for {split} split")
            game_sequences = []
            for bin_file in bin_files:
                info(f"Loading {bin_file}...")
                with open(bin_file, 'rb') as f:
                    sequences = pickle.load(f)
                    game_sequences.extend(sequences)
                # end with
            # end for
        # end if
        
        # Concatenate game sequences
        all_sequences = [x for sublist in game_sequences for x in sublist.tolist()]
        info(f"Total sequences for {split}: {len(all_sequences)}")
        
        # Store in the appropriate global variable
        if split == 'train':
            _train_data = all_sequences
            info(f"Train data loaded into memory")
        else:
            _val_data = all_sequences
            info(f"Validation data loaded into memory")
        # end if
    # end if
    
    # Get the appropriate data based on the split
    game_sequences = _train_data if split == 'train' else _val_data
    
    # Sample random sequence start position
    ix = [random.randint(0, len(game_sequences)-config['block_size']-1) for _ in range(config['batch_size'])]
    
    # Select subsequences
    x = torch.stack([torch.tensor(game_sequences[i:i+config['block_size']], dtype=torch.long) for i in ix])
    y = torch.stack([torch.tensor(game_sequences[i+1:i+1+config['block_size']], dtype=torch.long) for i in ix])
    
    # Device type
    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    # end if
    
    return x, y
# end get_board_batch

def get_batch(split, config, device, device_type):
    """
    Get a random batch of data from the specified split.
    
    Args:
        split (str): 'train' or 'val' to specify which data split to use
        config (TrainingConfig): Configuration object containing 'data_dir' which points to a directory
                                with 'train' and 'val' folders containing bin files for each split
        device (str): Device to use for tensors
        device_type (str): Type of device ('cuda' or 'cpu')
        
    Returns:
        tuple: (x, y) where x is input tokens and y is target tokens
    """
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    
    # Use the data directory from the configuration, with the appropriate split folder
    data_dir = os.path.join(config.data_dir, split)
    bin_file = os.path.join(data_dir, f'{split}.bin')
    
    # Check if the bin file exists
    if not os.path.exists(bin_file):
        error(f"Error: {bin_file} not found. Make sure the data directory contains 'train' and 'val' folders with bin files.")
        # Fallback to old method
        fallback_data_dir = os.path.join('data', config['dataset'])
        bin_file = os.path.join(fallback_data_dir, f'{split}.bin')
        error(f"Falling back to {bin_file}")
    
    # Load the data
    data = np.memmap(bin_file, dtype=np.uint16, mode='r')
    # end if
    
    # Sample random indices for batch
    ix = torch.randint(len(data) - config['block_size'], (config['batch_size'],))
    
    # Create input and target sequences
    x = torch.stack([torch.from_numpy((data[i:i+config['block_size']]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+config['block_size']]).astype(np.int64)) for i in ix])
    
    if device_type == 'cuda':
        # Pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    # end if
    
    return x, y
# end get_batch


def build_vocab(output: str):
    """
    Build vocabulary for BoardGPT:
    - 60 positions (8x8 board minus 4 starting squares for Othello)
    - Special tokens <pad>

    Args:
        output (str): Path to output file

    Returns:
        dict: Vocabulary dictionary
    """
    # Generate all positions
    positions = [f"{c}{r}" for c in "abcdefgh" for r in range(1, 9)]

    # Remove the 4 starting squares (Othello)
    for start in ["d4", "e5", "d5", "e4"]:
        positions.remove(start)
    # end for

    specials = ["<pad>"]
    tokens = specials + positions

    # Map tokens to integer IDs
    vocab = {tok: i for i, tok in enumerate(tokens)}

    # Save to vocab.json
    vocab_file = os.path.join(output, "vocab.json")
    with open(vocab_file, "w") as f:
        info(f"Writing vocab to {vocab_file}")
        json.dump(
            obj=vocab,
            fp=f,
            indent=2
        )
    # end with

    return vocab
# end def build_vocab


def build_tokenizer(vocab: dict, output: str, save_path="tokenizer.json"):
    """
    Build a HuggingFace-compatible tokenizer from a fixed vocab.
    Saves a tokenizer.json file that can be reloaded later.
    """
    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="<pad>"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.save(save_path)

    fast_tok = PreTrainedTokenizerFast(
        tokenizer_file=save_path,
        pad_token="<pad>",
        unk_token="<pad>",
    )

    # Save tokenizer.json (already exists inside tokenizer)
    info(f"Writing tokenizer to {output}/tokenizer")
    fast_tok.save_pretrained(save_directory=os.path.join(output, "tokenizer"))

    return fast_tok
# end build_tokenizer


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


def get_dataloader(
        split: str,
        config: TrainingConfig,
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


def initialize_model(config, device, ckpt_path=None, start_iter=None):
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
        dropout=config.dropout
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
        model = GameGPT(gptconf)
        
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
        model = GameGPT(gptconf)
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
        iter_data,
        dataset,
        tokenizer: PreTrainedTokenizerFast,
        ctx,
        config,
        device
):
    """
    Estimate loss overtrain and validation splits using multiple batches.
    Also calculates an invalid move ratio for validation split if board_game is enabled.
    
    Args:
        model: The model to evaluate
        iter_data: The iteration data to use
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

        # For each eval iterations
        for k in range(config.eval_iters):
            # Use the appropriate data loading function based on the configuration
            X, Y = next(iter_data)
            X, Y = X.to(device), Y.to(device)

            with ctx:
                _, logits, loss, _ = model(X, Y)
            # end with

            # Keep loss
            losses[k] = loss.item()
        # end for

        if split == 'val':
            eval_log(f"Computing invalid move ratio on {split} split...")
            inv_rate_ratio = invalid_move_rate(
                model=model,
                iter=iter_data,
                dataset=dataset,
                tokenizer=tokenizer,
                device=device,
                num_samples=config.imr_iters
            )
            out['IMR'] = inv_rate_ratio
        # end if
        
        out[split] = losses.mean()
    # end for
    
    model.train()
    return out
# end def estimate_loss


def setup_optimizer(
        model: GPT,
        config,
        device_type,
        checkpoint=None
):
    """
    Set up the optimizer for training.
    
    Args:
        model: The model to optimize
        config (TrainingConfig): Configuration object
        device_type (str): Type of device ('cuda' or 'cpu')
        checkpoint: Optional checkpoint dictionary for resuming training
        
    Returns:
        tuple: (optimizer, scaler) where optimizer is the configured optimizer and
               scaler is the GradScaler for mixed precision training
    """
    # Initialize a GradScaler for mixed precision training
    # If enabled=False (not using float16), scaler is a no-op
    scaler = torch.amp.GradScaler(device=device_type)

    # Set up the optimizer
    optimizer, num_decay_params, num_nodecay_params = model.configure_optimizers(
        config['weight_decay'], 
        config['learning_rate'], 
        (config['beta1'], config['beta2']), 
        device_type
    )

    # Load optimizer state if checkpoint is provided
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    # end if
    
    return optimizer, scaler, num_decay_params, num_nodecay_params
# end def setup_optimizer

def save_checkpoint(
        model,
        optimizer,
        iter_num,
        best_val_loss,
        config: TrainingConfig,
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


def infinite_loader(
        dataloader: torch.utils.data.DataLoader
):
    """
    Infinite loader iterator.
    """
    while True:
        for batch in dataloader:
            X, Y = batch
            yield X, Y
        # end for
    # end while
# end def infinite_loader


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
    ddp, master_process, seed_offset, ddp_world_size, ddp_local_rank, device, device_type, ctx, gradient_accumulation_steps = setup_training_environment(config)
    
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
    X, Y = next(data_iter)
    X, Y = X.to(device), Y.to(device)

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
                iter_data=val_data_iter,
                dataset=val_dataloader.dataset,
                tokenizer=tokenizer,
                ctx=ctx,
                config=config,
                device=device
            )

            # Log eval
            eval_log(
                f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, imr {losses['IMR']:.4f}"
            )
            
            # Log to wandb if enabled
            if config['wandb_log']:
                log_data = {
                    "iter": iter_num,
                    "val/loss": losses['val']
                }

                # Log invalid move ratio if available
                log_data["val/IMR"] = losses['IMR']
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
                _, logits, loss, _ = model(X, Y)

                # Scale the loss to account for gradient accumulation
                loss = loss / gradient_accumulation_steps
            # end with
            
            # Immediately async prefetch next batch while model is doing the forward pass on the GPU
            # Use the appropriate data loading function based on the configuration
            X, Y = next(data_iter)
            X, Y = X.to(device), Y.to(device)
            
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
                mfu = raw_model.estimate_mfu(config['batch_size'] * gradient_accumulation_steps, dt)
                running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
            # end if
            
            train_log(f"iter {iter_num:04d}: loss {lossf:02.4f}, time {dt*1000:05.2f}ms, mfu {running_mfu*100:04.2f}%")

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

