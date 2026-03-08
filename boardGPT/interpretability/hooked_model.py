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
import torch
import torch.nn as nn
from collections import defaultdict


class HookedModel:
    """
    Thin wrapper around a PyTorch module that records intermediate activations
    and optionally patches them during the forward pass.

    The wrapper manages a set of named forward hooks whose outputs are cached
    on CPU for later inspection. Custom patch functions can be registered per
    hook name to modify activations in-flight.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self._hooks = []
        self.activations = defaultdict(list)
        self.patches = {}
    # end def __init__

    def _make_hook(self, name, store=True):
        def hook(module, input, output):
            # 1. Stockage
            if store:
                self.activations[name].append(output.detach().cpu())
            # 2. Patching
            if name in self.patches:
                return self.patches[name](output)
            return output
        return hook
    # end def _make_hook

    def add_hook(self, module_name, store=True):
        """
        Add a hook to submodule identified by its name.
        """
        module = dict(self.model.named_modules())[module_name]
        h = module.register_forward_hook(self._make_hook(module_name, store))
        self._hooks.append(h)

    def clear_hooks(self):
        """
        Supprime tous les hooks
        """
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def clear_activations(self):
        """
        Vide le buffer d’activations
        """
        self.activations.clear()

    def add_patch(self, module_name, fn):
        """
        Définit une fonction de patch pour modifier l’output
        """
        self.patches[module_name] = fn
    # end def add_patch

    def clear_patches(self):
        """
        Remove all registered activation patch functions.
        """
        self.patches.clear()
    # end def clear_patches

    def forward(self, *args, **kwargs):
        """
        Execute the wrapped model while keeping hooks active.
        """
        return self.model(*args, **kwargs)
    # end def forward

# end class HookedModel
