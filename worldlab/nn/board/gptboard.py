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
from typing import Dict, Tuple, List, Optional, Any, Union
import torch
import torch.nn as nn
from torch.nn import functional as F
from .register import ActivationRecorder
from .gpt import GPT


class GameGPT(GPT):
     """
     GPT language model wrapper tailored to board game token streams.

     Provides convenience helpers for autoregressive generation and
     compatibility with activation recorders used in interpretability tooling.
     """

     # Predict next
     def predict_next(
             self,
             idx: Union[List[int], torch.LongTensor],
             device: str = 'cuda'
     ):
          """
          Predicts the next token from a sequence of tokens.

          Args:
               idx (torch.LongTensor): The sequence of tokens.
               device (str, optional): The device to use. Defaults to 'cuda'.

          Returns:
               torch.LongTensor: The predicted next token.
          """
          # To tensor
          if type(idx) is int:
               idx = torch.LongTensor(idx)
          # end if

          # Device
          idx = idx.to(device)

          # Forward the model to get the logits for the index in the sequence
          with torch.no_grad():
               x, logits, loss, ret_list = self(
                    idx=idx,
                    recorder=None,
                    to_return=None,
               )
          # end with

          return logits
     # end def predict_next

     # Generate moves
     def generate_moves(
             self,
             idx: Union[List[int], torch.LongTensor],
             max_new_tokens: int,
             temperature: float = 1.0,
             top_k: int = None,
             recorder: ActivationRecorder = None,
             to_return: List[str] = None,
             device: str = 'cuda' ,
     ) -> torch.Tensor:
          """
          Generate moves from a sequence.

          Args:
              idx (List[int]): List of integer token
              max_new_tokens (int): Maximum number of tokens to generate
              device (torch.device): Device to use
              temperature (float): Temperature parameter
              top_k (int): If specified, only generate tokens with this many tokens
              device (torch.device): Device to use
              recorder (ActivationRecorder): Recorder to record moves
              to_return (List[str]): List of information to return.
          """
          # To tensor
          if type(idx) is int:
               idx = torch.LongTensor(idx)
          # end if

          # Device
          idx = idx.to(device)

          # Generate tokens
          # gen_seq is (seq_len + max_new_token)
          gen_seq, ret_list = self.generate(
               idx=idx,
               max_new_tokens=max_new_tokens,
               temperature=temperature,
               top_k=top_k,
               recorder=recorder,
               to_return=to_return
          )
          gen_seq = gen_seq[0]

          # Transform into str sequence
          return gen_seq
     # end generate_tokens

     def forward(
             self,
             idx: torch.LongTensor,
             targets: torch.Tensor = None,
             recorder: ActivationRecorder = None,
             to_return: Optional[List[str]] = None,  # end def forward
     ):
          """
          Forward pass through the model.

          Args:
              idx (torch.Tensor): Input token indices of shape (batch_size, seq_len)
              targets (torch.Tensor, optional): Target token indices of shape (batch_size, seq_len)
              recorder (ActivationRecorder, optional): Activation recorder
              to_return (List[str]): List of object to return

          Returns:
              tuple: (logits, loss) where logits is the output predictions and loss is the
                    cross-entropy loss if targets are provided, otherwise None
          """
          x, logits, obj_to_return = super().forward(
               idx=idx,
               recorder=recorder,
               to_return=to_return
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

          return x, logits, loss, obj_to_return
     # end def forward

# end class GameGPT
