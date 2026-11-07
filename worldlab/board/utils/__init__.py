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

from typing import Optional

_models_import_error: Optional[ModuleNotFoundError] = None

try:
    from .models import load_safetensors as _load_safetensors
except ModuleNotFoundError as exc:
    _models_import_error = exc
    _load_safetensors = None


def load_safetensors(*args, **kwargs):
    if _load_safetensors is None:
        raise ModuleNotFoundError(
            "load_safetensors requires optional dependency 'safetensors'. "
            "Install it with `pip install safetensors` to enable checkpoint loading."
        ) from _models_import_error
    return _load_safetensors(*args, **kwargs)
from .training import (
    TrainingConfig,
    load_config,
    setup_training_environment,
    get_lr,
    setup_optimizer,
    configure_optimizers
)
from .viz import plot_attention_matrix, plot_heads_attention, show_linear_probe_samples
from ..games.othello.viz import show_othello, plot_othello_game
from .logging import (
    info, warning, error, debug, success,
    print_exception, log_exception, console, train_log, eval_log
)

__all__ = [
    "TrainingConfig",
    "load_config",
    "setup_training_environment",
    "get_lr",
    "setup_optimizer",
    "load_safetensors",
    "plot_attention_matrix",
    "plot_heads_attention",
    "show_linear_probe_samples",
    "show_othello",
    "plot_othello_game",
    "info", "warning", "error", "debug", "success",
    "print_exception", "log_exception", "console", "train_log",
    "eval_log"
]
