

# Imports
import argparse
import torch
from transformers import AutoTokenizer, PreTrainedTokenizerFast

# BoardGPT
from worldlab.board.datasets import load_othello_data_files, GameDataset
from worldlab.nn import GameGPT
from worldlab.nn.board import GPTConfig, GPT
from worldlab.board.utils import info, error, warning, train_log, eval_log
from worldlab.board.validation import evaluate_IMR


def parse_args():
    """
    Parse command line arguments for the training script.

    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description='Evaluate IMR of a model on a dataset.'
    )

    parser.add_argument(
        '--data-dir',
        type=str,
        required=True,
        help='Path to the data directory. This directory must contain "train" and '
             '"val" folders with bin files for each split.'
    )

    parser.add_argument(
        '--num-iter',
        type=int,
        help='Number of evaluation iterations.'
    )

    parser.add_argument(
        '--repo-id',
        type=str,
        default=None,
        help='Path to model repository'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        required=True,
        help='Batch size for evaluation',
    )

    parser.add_argument(
        '--num-workers',
        type=int,
        required=True,
        help='Number of workers for data loading',
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility',
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device'
    )

    return parser.parse_args()
# end def parse_args


def load_model(
        repo_id: str,
        device
):
    """
    Initialize the model based on checkpoint arguments or from scratch.

    Args:
        repo_id (str): Repo ID
        device (str): Device to use for the model

    Returns:
        tuple: (model, iter_num, best_val_loss, model_args) where model is the initialized GPT model,
               iter_num is the starting iteration number, best_val_loss is the best
               validation loss (used for checkpointing), and model_args are the model arguments.
    """
    # Load model from repo
    model, config = GameGPT.from_pretrained(repo_id=repo_id)
    model.to(device)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(repo_id, subfolder='tokenizer')

    return model, config, tokenizer
# end load_model


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
        data_dir: str,
        block_size: int,
        batch_size: int,
        num_workers: int,
        tokenizer: PreTrainedTokenizerFast,
) -> torch.utils.data.DataLoader:
    """
    Get dataloaders for training and validation.

    Args:
        data_dir (str): Path to the data directory
        config (TrainingConfig): Configuration object containing 'data_dir' which points to a directory
        with 'train' and 'val' folders containing bin files for each split
        tokenizer (PreTrainedTokenizerFast): Tokenizer object used to tokenize the batch
    """
    dataset = GameDataset(
        data_dir=data_dir,
        split="val",
        block_size=block_size,
        ood_perc=0.0,
        num_samples=-1
    )

    # Create a dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
        collate_fn=lambda b: collate_fn(b, tokenizer)
    )

    return dataloader
# end def get_dataloader


def infinite_loader(
        dataloader: torch.utils.data.DataLoader
):
    """
    Infinite loader function for evaluating a batch of data.
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
    Main evaluation function.
    """
    # Parse command line arguments
    args = parse_args()

    # Set random seed for reproducibility
    torch.manual_seed(args.seed)

    # Initialize the model
    info(f"Initializing model")
    model, config, tokenizer = load_model(
        repo_id=args.repo_id,
        device=args.device
    )

    # Create validation dataloader
    val_dataloader = get_dataloader(
        data_dir=args.data_dir,
        block_size=config.block_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        tokenizer=tokenizer
    )
    val_data_iter = infinite_loader(val_dataloader)

    # Compute the IMR
    info(f"Evaluating model")
    im_rate = evaluate_IMR(
        model=model,
        iter=val_data_iter,
        tokenizer=tokenizer,
        num_samples=args.num_iter,
        device=args.device
    )

    info(f"IM RATIO: {im_rate * 100:.4f}%")
# end def main


# Execute the main function if a script is run directly
if __name__ == "__main__":
    main()
# end if

