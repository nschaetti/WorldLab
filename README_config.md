# BoardGPT Configuration Guide

This document explains how to use the YAML configuration system for training BoardGPT models.

## Overview

The training script (`train.py`) now supports configuration via YAML files. This allows for easier management of training parameters and better reproducibility of experiments.

## Usage

### Basic Usage

To train a model using the default configuration file:

```bash
python train.py
```

This will use the default configuration file `config.yaml` in the project root directory.

### Specifying a Custom Configuration File

To use a custom configuration file:

```bash
worldlab-board-train --config path/to/your/config.yaml
```

## Configuration Parameters

The configuration file supports the following parameters:

### I/O Settings

```yaml
# I/O settings
out_dir: 'out'                  # Output directory for checkpoints and logs
eval_interval: 2000             # How often to evaluate the model
log_interval: 1                 # How often to log training progress
eval_iters: 200                 # Number of batches to use for evaluation
eval_only: false                # If True, script exits right after the first eval
always_save_checkpoint: true    # If True, always save a checkpoint after each eval
init_from: 'scratch'            # 'scratch', 'resume', or 'gpt2*'
```

### Weights & Biases Logging Settings

```yaml
# wandb logging settings
wandb_log: false                # Whether to use wandb logging (disabled by default)
wandb_project: 'owt'            # Project name for wandb
wandb_run_name: 'gpt2'          # Run name for wandb
```

### Data Settings

```yaml
# Data settings
board_game: "othello"           # Board game to train on
data_filename: "synthetic.bin"  # Filename for board game data
dataset: 'openwebtext'          # Dataset name for text training
gradient_accumulation_steps: 40 # Used to simulate larger batch sizes
batch_size: 12                  # If gradient_accumulation_steps > 1, this is the micro-batch size
block_size: 61                  # Context size for the model
```

### Model Architecture Settings

```yaml
# Model architecture settings
n_layer: 8                      # Number of transformer layers
n_head: 8                       # Number of attention heads
n_embd: 512                     # Embedding dimension
dropout: 0.0                    # Dropout rate (0 for pretraining, try 0.1+ for finetuning)
bias: false                     # Whether to use bias in LayerNorm and Linear layers
```

### Optimizer Settings

```yaml
# Optimizer settings (AdamW)
learning_rate: 6.0e-4           # Maximum learning rate
max_iters: 600000               # Total number of training iterations
weight_decay: 1.0e-1            # Weight decay coefficient
beta1: 0.9                      # AdamW beta1 parameter
beta2: 0.95                     # AdamW beta2 parameter
grad_clip: 1.0                  # Clip gradients at this value (disable if == 0.0)
```

### Learning Rate Decay Settings

```yaml
# Learning rate decay settings
decay_lr: true                  # Whether to decay the learning rate
warmup_iters: 2000              # Number of warmup steps
lr_decay_iters: 600000          # Should be ~= max_iters per Chinchilla
min_lr: 6.0e-5                  # Minimum learning rate (~= learning_rate/10 per Chinchilla)
```

### Distributed Data Parallel Settings

```yaml
# DDP settings
backend: 'nccl'                 # Backend for distributed training ('nccl', 'gloo', etc.)
```

### System Settings

```yaml
# System settings
device: 'cuda'                  # Device to use ('cpu', 'cuda', 'cuda:0', 'cuda:1', 'mps' on macbooks)
dtype: 'bfloat16'               # Data type for training (will fall back to float16 if bfloat16 not supported)
compile: true                   # Whether to use PyTorch 2.0 compilation for speed
```

## Example Configurations

### Training Othello Model

```yaml
# Othello training configuration
board_game: "othello"
data_filename: "synthetic.bin"
n_layer: 8
n_head: 8
n_embd: 512
block_size: 61
batch_size: 12
learning_rate: 6.0e-4
max_iters: 100000
```

### Fine-tuning with Higher Dropout

```yaml
# Fine-tuning configuration
init_from: 'resume'
out_dir: 'out/finetuned'
dropout: 0.1
learning_rate: 1.0e-4
max_iters: 10000
```

### Evaluation Only

```yaml
# Evaluation configuration
eval_only: true
init_from: 'resume'
out_dir: 'out/eval'
```

## Creating Custom Configurations

You can create custom configuration files by copying the default `config.yaml` and modifying the parameters as needed. Only specify the parameters you want to change; any parameters not specified will use their default values.