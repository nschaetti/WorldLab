"""
Copyright (C) 2025 boardGPT Contributors

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

"""
Transformer block implementation for the boardGPT model.

This module provides a standard transformer block implementation with self-attention and MLP.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layer_norm import LayerNorm
from .hooks import HookPoint
from .register import ActivationRecorder


class MLP(nn.Module):
    """
    Multi-Layer Perceptron module used in transformer blocks.

    This implements the feed-forward network component of a transformer block,
    consisting of two linear transformations with a GELU activation in between.

    Args:
        config: Configuration object containing MLP parameters
    """

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
    # end def __init__
    def forward(self, x):
        """
        Apply MLP to the input.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, n_embd)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, n_embd)
        """
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
    # end def forward
# end class MLP


class CausalSelfAttention(nn.Module):
    """
    Causal self-attention module with optional Flash Attention support.

    This module implements multi-head causal self-attention, where each token can only
    attend to itself and previous tokens in the sequence. It supports both a standard
    implementation and an optimized Flash Attention implementation when available.

    Args:
        config: Configuration object containing attention parameters
    """

    def __init__(
            self,
            config,
            use_flash: bool = True  # end def __init__
    ):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # Hooks
        self.qk_hook = HookPoint()
        self.v_hook = HookPoint()

        # key, query, value projections for all heads, but in a batch
        # n_embd => 3 * n_embd
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)

        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and use_flash
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer(
                name="bias",
                tensor=torch.tril(
                    torch.ones(config.block_size, config.block_size)
                ).view(1, 1, config.block_size, config.block_size)
            )  # end if
        # end if
    # end __init__

    def forward(
            self,
            x,
            recorder: ActivationRecorder = None,
            recorder_prefix: str = ""  # end def forward
    ):
        """
        Apply causal self-attention to the input.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, n_embd)
            recorder: Activation recorder
            recorder_prefix: str

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, n_embd)
        """
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # Linear projection, then split in 3 for q, k, v
        # q, k, v are [b, t, 512]
        # Each token (moves) has 3 representations
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        # Change to (B, n_head, T, head_size)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # V hook
        self.v_hook(v)

        # Q (query) : ce que ce token cherche à savoir des autres.
        # K (key) : ce que chaque autre token offre comme information.
        # V (value) : l’information transportée si un token est sélectionné.

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # QK^t = [i, h, t_q, t_k] = similarité entre le token t_q et le token t_k vue par la tête h.
            # QK^t c'est la matrice de similarité
            # softmax => C’est une matrice de poids entre tokens (chaque ligne somme à 1).

            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(
                query=q,
                key=k,
                value=v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True
            )  # end if
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            # att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            # build causal mask on the fly (triangular lower mask)
            causal_mask = torch.tril(torch.ones(T, T, device=q.device, dtype=torch.bool))
            att = att.masked_fill(~causal_mask, float('-inf'))
            att = F.softmax(att, dim=-1)
            self.qk_hook(att)
            if recorder is not None:
                recorder.save(f"{recorder_prefix}.QK", att)  # end if
            # end if
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)  # end else
        # end if

        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y
    # end def forward
# end class CausalSelfAttention
# end class CausalSelfAttention


class Block(nn.Module):
    """
    Transformer block combining self-attention and MLP with residual connections.

    This implements a standard transformer block with pre-layer normalization,
    consisting of a multi-head self-attention layer followed by an MLP,
    with residual connections around each.

    Args:
        config: Configuration object containing block parameters
    """

    def __init__(self, config, use_flash: bool = True):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config, use_flash=use_flash)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)  # end def __init__
    # end __init__

    def forward(
            self,
            x,
            recorder: ActivationRecorder = None,
            recorder_prefix: str = ""  # end def forward
    ):
        """
        Apply transformer block to the input.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, n_embd)
            recorder (ActivationRecorder): Activation recorder
            recorder_prefix (str): Prefix to add to recorder

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, n_embd)
        """
        # Block.x: torch.Size([b, t, 512])
        # Block.attn: torch.Size([b, t, 512])
        attn_out = self.attn(
            x=self.ln_1(x),
            recorder=recorder,
            recorder_prefix=f"{recorder_prefix}.residuals.attn"
        )

        # Block.x: torch.Size([b, t, 512])
        x = x + attn_out  # residual connection around attention

        # Save activation to recorder
        if recorder is not None:
            recorder.save(f"{recorder_prefix}.residuals.attn", x)  # end if
        # end if

        # Block.mlp: torch.Size([b, t, 512])
        mlp_out = self.mlp(self.ln_2(x))

        # Block.x: torch.Size([b, t, 512])
        x = x + mlp_out  # residual connection around MLP

        # Save activation to recorder
        if recorder is not None:
            recorder.save(f"{recorder_prefix}.residuals.mlp", x)  # end if
        # end if

        return x
    # end def forward
# end class Block
# end class Block