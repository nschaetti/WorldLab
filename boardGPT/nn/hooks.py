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

import torch.nn as nn


class HookPoint(nn.Module):
    """
    Torch module that forwards its input while optionally applying a list of
    user registered hook functions.

    Hook functions receive the current tensor and may either return a modified
    tensor or ``None`` to leave the stream unchanged.
    """

    def __init__(self):
        super().__init__()
        self.fns = []  # end def __init__
    # end def __init__

    def add_hook(self, fn):
        """
        Register a hook function that will be invoked during the forward pass.

        Args:
            fn (Callable[[torch.Tensor], Optional[torch.Tensor]]): Hook to run.
        """
        self.fns.append(fn)  # end def add_hook
    # end def add_hook

    def clear(self):
        """
        Remove all registered hook functions.
        """
        self.fns = []  # end def clear
    # end def clear

    def forward(self, x):
        """
        Apply registered hooks sequentially to the incoming tensor.

        Each hook can optionally return a replacement tensor. If ``None`` is
        returned the current tensor is left untouched.
        """
        for fn in self.fns:
            rx = fn(x)
            if rx is not None:
                x = rx  # end if
            # end if
        # end for
        return x  # end def forward
    # end def forward
# end class HookPoint
# end class HookPoint

