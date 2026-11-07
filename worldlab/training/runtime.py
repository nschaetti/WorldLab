"""Runtime utilities shared by experiment runners."""

import os
import random

import torch


def get_device(requested, local_rank=0):
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if requested == "auto" else requested)
    if device.type == "cuda":
        device = torch.device("cuda", local_rank if device.index is None else device.index)
        torch.cuda.set_device(device)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Only CPU and CUDA are supported by the experiment runner")
    return device


def autocast(device, precision):
    return torch.autocast(device.type, enabled=precision != "fp32",
                          dtype=torch.float16 if precision == "fp16" else torch.bfloat16)


def seed_epoch(seed, epoch, rank):
    epoch_seed = seed + epoch * 100003 + rank
    torch.manual_seed(epoch_seed)
    random.seed(epoch_seed)


def atomic_write(path, write):
    temporary = path.with_suffix(path.suffix + ".tmp")
    write(temporary)
    os.replace(temporary, path)
