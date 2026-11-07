#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from worldlab.board.validation.metrics import invalid_move_rate
from worldlab.nn.board.gpt import GPT, GPTConfig


def main():
    """
    Test the invalid_move_rate function with a model.
    """
    parser = argparse.ArgumentParser(description="Test invalid_move_rate function")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory containing the dataset")
    parser.add_argument("--split", type=str, default="val", help="Dataset split to use ('train' or 'val')")
    parser.add_argument("--data_filename", type=str, default="val.pkl", help="Filename for the dataset")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples to evaluate")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for sampling")
    parser.add_argument("--top_k", type=int, default=None, help="Top-k sampling parameter")
    args = parser.parse_args()

    # Load the model
    print(f"Loading model from {args.model_path}...")
    model = GPT.from_pretrained('othello')
    model.load_safetensors(args.model_path)
    model.eval()

    # Compute the invalid move rate
    print(f"Computing invalid move rate for {args.num_samples} samples...")
    rate = invalid_move_rate(
        model=model,
        data_dir=args.data_dir,
        split=args.split,
        data_filename=args.data_filename,
        num_samples=args.num_samples,
        temperature=args.temperature,
        top_k=args.top_k
    )

    print(f"Invalid move rate: {rate:.4f}")


if __name__ == "__main__":
    main()