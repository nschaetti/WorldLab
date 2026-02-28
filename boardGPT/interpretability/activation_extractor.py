"""
Activation extraction helpers for interpretability experiments.
"""

from typing import Iterable, List, Optional, Sequence, Tuple, Union

import torch

from boardGPT.nn import GPT


PositionPolicy = Union[str, int]


def _normalize_layer_index(model: GPT, layer: int) -> int:
    if layer < 0 or layer >= model.config.n_layer:
        raise ValueError(f"Layer index {layer} out of range [0, {model.config.n_layer - 1}]")
    return layer


def _select_positions(residuals: torch.Tensor, position_policy: PositionPolicy) -> torch.Tensor:
    if position_policy == "all":
        return residuals
    if position_policy == "last":
        return residuals[:, -1, :]
    if isinstance(position_policy, int):
        if position_policy < 0 or position_policy >= residuals.size(1):
            raise ValueError(
                f"Position {position_policy} out of range [0, {residuals.size(1) - 1}]"
            )
        return residuals[:, position_policy, :]
    raise ValueError("position_policy must be 'all', 'last' or an integer index")


def _get_tokens_from_batch(batch: Union[torch.Tensor, Sequence[torch.Tensor]]) -> torch.Tensor:
    if torch.is_tensor(batch):
        return batch
    if isinstance(batch, (list, tuple)) and len(batch) > 0 and torch.is_tensor(batch[0]):
        return batch[0]
    raise ValueError("Unsupported batch format: expected Tensor or tuple/list where first element is Tensor")


@torch.no_grad()
def extract_residuals(
    model: GPT,
    token_batch: torch.Tensor,
    layer: int,
    device: Optional[Union[str, torch.device]] = None,
) -> torch.Tensor:
    """
    Extract residual activations after a given transformer layer.

    Returns:
        Residual tensor with shape [batch, seq, d_model].
    """
    layer = _normalize_layer_index(model, layer)
    token_batch = token_batch.long()

    if device is None:
        device = next(model.parameters()).device
    device = torch.device(device)

    was_training = model.training
    model.eval()

    _, _, obj = model(idx=token_batch.to(device), to_return=[f"residuals{layer}"])
    if len(obj) != 1:
        raise RuntimeError(
            f"Model returned {len(obj)} tensors for layer extraction, expected exactly one"
        )

    residuals = obj[0]
    if was_training:
        model.train()
    return residuals


@torch.no_grad()
def extract_dataset_residuals(
    model: GPT,
    batches: Iterable[Union[torch.Tensor, Sequence[torch.Tensor]]],
    layer: int,
    position_policy: PositionPolicy = "all",
    device: Optional[Union[str, torch.device]] = None,
    return_cpu: bool = True,
) -> torch.Tensor:
    """
    Extract residuals on an iterable of mini-batches.

    Args:
        model: GPT model.
        batches: Iterable that yields a token tensor or tuple/list with token tensor first.
        layer: Layer index in [0, n_layer-1].
        position_policy: "all", "last", or a fixed position index.
        device: Compute device.
        return_cpu: If True, concatenated tensor is moved to CPU.

    Returns:
        Concatenated activations.
        - "all": [N, T, D]
        - "last" or int: [N, D]
    """
    container: List[torch.Tensor] = []
    for batch in batches:
        tokens = _get_tokens_from_batch(batch)
        residuals = extract_residuals(model=model, token_batch=tokens, layer=layer, device=device)
        selected = _select_positions(residuals, position_policy=position_policy)
        if return_cpu:
            selected = selected.detach().cpu()
        container.append(selected)

    if not container:
        raise ValueError("No batches provided to extract_dataset_residuals")

    return torch.cat(container, dim=0)
