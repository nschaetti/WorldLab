"""
Training utilities for configuring optimizers, distributed setup, and schedules.
"""

import inspect
import json

# Imports
import os
import math
import yaml
from contextlib import nullcontext
from typing import List, Dict, Any

import torch
from torch.distributed import init_process_group

# BoardGPT
from boardGPT.nn import GPT

from .logging import info, error


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
    # end def __contains__

# end class TrainingConfig


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
    optimizer, num_decay_params, num_nodecay_params = configure_optimizers(
        module=model,
        weight_decay=config['weight_decay'],
        learning_rate=config['learning_rate'],
        betas=(config['beta1'], config['beta2']),
        device_type=device_type
    )

    # Load optimizer state if checkpoint is provided
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    # end if

    return optimizer, scaler, num_decay_params, num_nodecay_params


# end def setup_optimizer


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

    # Create output directory if needed (only on a master process)
    if master_process:
        os.makedirs(config['out_dir'], exist_ok=True)
    # end if

    # Set random seed for reproducibility
    torch.manual_seed(1337 + seed_offset)

    # Enable TF32 precision on CUDA devices (faster and usually sufficient precision)
    # torch.backends.cudnn.conv.fp32_precision = 'tf32'
    # torch.backends.cuda.matmul.fp32_precision = 'ieee'
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Determine a device type for later use
    device_type = 'cuda' if 'cuda' in device else 'cpu'

    # Set up precision type for training
    dtype = config['dtype']
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]

    # Create a context manager for mixed precision training
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    return ddp, master_process, seed_offset, ddp_world_size, ddp_local_rank, device, device_type, ctx, gradient_accumulation_steps
# end setup_training_environment


def get_lr(it, config):
    """
    Get learning rate for the current iteration according to the schedule.

    Args:
        it (int): Current iteration number
        config (TrainingConfig): Configuration object

    Returns:
        float: Learning rate for the current iteration
    """
    # 1) Linear warmup for warmup_iters steps
    if it < config['warmup_iters']:
        return config['learning_rate'] * (it + 1) / (config['warmup_iters'] + 1)
    # end if

    # 2) If it > lr_decay_iters, return min learning rate
    if it > config['lr_decay_iters']:
        return config['min_lr']
    # end if

    # 3) In between, use cosine decay down to min learning rate
    decay_ratio = (it - config['warmup_iters']) / (config['lr_decay_iters'] - config['warmup_iters'])
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # Coeff ranges 0..1
    return config['min_lr'] + coeff * (config['learning_rate'] - config['min_lr'])
# end def get_lr


def configure_optimizers(
        module,
        weight_decay,
        learning_rate,
        betas,
        device_type
):
    """
    Configure the optimizer for training.

    This method sets up an AdamW optimizer with weight decay applied only to
    appropriate parameters (weights but not biases or layer norms).

    Args:
        module (nn.Module): Module whose parameters to configure
        weight_decay (float): Weight decay coefficient
        learning_rate (float): Learning rate
        betas (tuple): Adam betas parameters (β1, β2)
        device_type (str): Device type ('cuda' or 'cpu')

    Returns:
        torch.optim.AdamW: Configured optimizer
    """
    # Start with all of the candidate parameters
    param_dict = {pn: p for pn, p in module.named_parameters()}

    # Filter out those that do not require grad
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

    # Create optim groups. Any parameter that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]

    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    # print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    # print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

    # Create AdamW optimizer and use the fused version if it is available
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == 'cuda'
    extra_args = dict(fused=True) if use_fused else dict()
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
    # print(f"using fused AdamW: {use_fused}")

    return optimizer, num_decay_params, num_nodecay_params
# end def configure_optimizers
