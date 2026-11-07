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

from typing import Union, Tuple
import json
from safetensors.torch import safe_open
from worldlab.nn.board import GPT, GPTConfig


def load_safetensors(
        path: str,
        config_path: str = None  # end def load_safetensors
) -> Tuple[GPT, GPTConfig]:
    """
    Load a model from a safe tensor file.

    Args:
        path (str): Path to the safe tensor file.
        config_path (str, optional): Path to the JSON configuration file.
            If not provided, assumes it's the same as path but with .json extension.

    Returns:
        A GPT model loaded from a safe tensor file.
    """
    # Determine the config path if not provided
    if config_path is None:
        config_path = path.rsplit('.', 1)[0] + '.json'  # end if
    # end if
    
    # Load the model configuration from JSON
    with open(config_path, 'r') as f:
        model_args = json.load(f)  # end with
    # end with
    
    # Create a GPTConfig object from the configuration
    gptconf = GPTConfig(**model_args)
    
    # Initialize a GPT model with this config
    model = GPT(gptconf)
    
    # Load the weights from the safetensors file
    model.load_safetensors(path)
    
    return model, model_args
# end load_safetensors

