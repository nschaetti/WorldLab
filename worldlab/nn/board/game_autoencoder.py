from __future__ import annotations

"""
Copyright (C) 2025 Nils Schaetti <n.schaetti@gmail.com>

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


# Imports
import math
import inspect
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional, Any, TYPE_CHECKING
import torch
from torch.nn import functional as F

from .register import ActivationRecorder
from .gptae import GPTAE

if TYPE_CHECKING:
    from transformers import AutoTokenizer


class GameAutoEncoder(GPTAE):
     """
     Autoencoder specialized for board game move sequences with helpers for
     encoding and decoding token indexes or human readable move strings.
     """

     # Encode a list of indices
     def encode_indices(self, idx: torch.LongTensor) -> torch.Tensor:
          """
          Encode a list of indices.
          """
          if idx.dim() == 1:
               idx = idx.unsqueeze(0)
          # end if

          # Encode idx
          with torch.no_grad():
               enc = self.encode(idx)
          # end with

          # Encode idx
          return enc
     # end def encode_indices

     # Encode a list of moves
     def encode_moves(
             self,
             moves: str,
             tokenizer: AutoTokenizer,
             padding: bool = True,
             pad_token: str = "<pad>",
     ):
          """
          Encode moves

          Args:
               moves (str): A list of moves as a string
               tokenizer (AutoTokenizer): Tokenizer to use to encode moves
               padding (bool, optional): If True, pad moves before encoding
               pad_token (str, optional): The token to use for padding
          """
          block_size = self.config.block_size
          if padding and len(moves.split()) != block_size:
               seq_len = len(moves.split())
               moves = ' '.join([pad_token] * (block_size - seq_len)) + ' ' + moves
          # end if

          # To idx
          idx = tokenizer(moves)['input_ids']

          # To tensors
          idx = torch.LongTensor(idx)
          idx = torch.unsqueeze(idx, 0)

          # Encode idx
          with torch.no_grad():
               enc = self.encode(idx)
          # end with

          # Encode idx
          return enc
     # end encode_moves

     # Decode a game embedding
     def decode_moves(
             self,
             emb: torch.Tensor,
             tokenizer: AutoTokenizer
     ):
          """
          Decode an embedding
          """
          if emb.ndim == 1:
               emb = emb.unsqueeze(0)
          # end if

          # Decode tensor
          with torch.no_grad():
               logits = self.decode(emb)
          # end with

          # Get highest prob. token
          idx = torch.argmax(logits, dim=-1)

          # To move tokens
          tokens = tokenizer.decode(idx[0].cpu().tolist())

          return tokens
     # end def decode_moves

     # Decode embedding to indices
     def decode_indices(
             self,
             emb: torch.Tensor
     ) -> torch.LongTensor:
          """
          Decode an embedding
          """
          if emb.ndim == 1:
               emb = emb.unsqueeze(0)
          # end if

          # Decode tensor
          with torch.no_grad():
               logits = self.decode(emb)
          # end with

          # Get highest prob. token
          return torch.argmax(logits, dim=-1).long()
     # end def decode_indices

     def forward(
             self,
             idx: torch.LongTensor,
             targets: torch.Tensor = None
     ):
          """
          Forward pass through the model.

          Args:
              idx (torch.Tensor): Input token indices of shape (batch_size, seq_len)
              targets (torch.Tensor, optional): Target token indices of shape (batch_size, seq_len)

          Returns:
              tuple: (logits, loss) where logits is the output predictions and loss is the
                    cross-entropy loss if targets are provided, otherwise None
          """
          logits = super().forward(
               idx=idx
          )

          if targets is not None:
               # If we are given some desired targets also calculate the loss
               loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1)  # end if
          else:
               # Inference-time mini-optimization: only forward the lm_head on the very last position
               # output is (b, 1, voc_size)
               logits = self.lm_head(x[:, [-1], :])  # note: using list [-1] to preserve the time dim
               loss = None
          # end if

          return logits, loss
     # end def forward

# end class GameAutoEncoder
