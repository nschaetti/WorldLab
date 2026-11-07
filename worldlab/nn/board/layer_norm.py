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
Custom LayerNorm implementation for the boardGPT model.

This module provides a custom implementation of LayerNorm with optional bias parameter.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F


class LayerNorm(nn.Module):
    """
    Custom LayerNorm implementation with optional bias parameter.

    This implementation differs from PyTorch's built-in LayerNorm by allowing
    the bias parameter to be disabled, which can improve performance in some cases.

    Args:
        ndim (int): The feature dimension to normalize over
        bias (bool): Whether to include a bias parameter
    """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None  # end def __init__
    # end def __init__

    def forward(self, input):
        """
        Apply layer normalization to the input.

        Args:
            input (torch.Tensor): Input tensor of shape (..., ndim)

        Returns:
            torch.Tensor: Normalized tensor of the same shape as input
        """
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)  # end def forward
    # end forward
# end class LayerNorm
# end class LayerNorm