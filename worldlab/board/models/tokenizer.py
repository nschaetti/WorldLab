"""
Tokenizer construction utilities for the boardGPT project.

The helpers generate a deterministic vocabulary of board positions and export a
Hugging Face compatible tokenizer that can be reused across experiments.
"""

import os
import json
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast
from tokenizers.pre_tokenizers import Whitespace

from worldlab.board.utils.logging import info


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

