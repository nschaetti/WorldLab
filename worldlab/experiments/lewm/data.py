"""Episode-safe frame/action windows from stable-worldmodel HDF5 files."""

from bisect import bisect_right
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401  # Register compression filters used by the public datasets.
import numpy as np
import torch
from torch.utils.data import Dataset, Subset, random_split
from torchvision.transforms import v2

IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)


def image_transform(resolution):
    return v2.Compose([
        v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(IMAGE_MEAN, IMAGE_STD), v2.Resize((resolution, resolution), antialias=True),
    ])


class Trajectories(Dataset):
    """Read frame t and the block of actions starting at t; windows stay inside an episode."""

    def __init__(self, path, resolution, history, frame_skip, columns=(), stride=1, action_stats=None):
        self.path = Path(path).expanduser()
        if min(history, frame_skip, stride) < 1:
            raise ValueError("History, frameskip and stride must be positive")
        self.history, self.frame_skip, self.stride = history, frame_skip, stride
        self.span = (history + 1) * frame_skip
        self.transform = image_transform(resolution)
        self.columns = tuple(columns)
        self._file = None
        with h5py.File(self.path, "r") as source:
            self.offsets = source["ep_offset"][:].astype(np.int64)
            self.lengths = source["ep_len"][:].astype(np.int64)
            count = len(source["pixels"])
            if (len(self.offsets) != len(self.lengths) or np.any(self.offsets < 0)
                    or np.any(self.lengths < 0) or np.any(self.offsets + self.lengths > count)
                    or np.any(self.offsets[1:] < self.offsets[:-1] + self.lengths[:-1])):
                raise ValueError("Invalid episode offsets or lengths")
            if any(len(source[name]) != count for name in ("action", *self.columns)):
                raise ValueError("Pixels, actions and per-step probe targets must be aligned")
            actions = torch.from_numpy(source["action"][:]).float().reshape(count, -1)
        if action_stats is None:
            valid = actions[actions.isfinite().all(dim=1)]
            if len(valid) < 2:
                raise ValueError("Need at least two finite action rows to compute normalization")
            self.mean = valid.mean(0)
            std = valid.std(0)
            self.std = torch.where(std > 0, std, torch.ones_like(std))
        else:
            self.mean = torch.as_tensor(action_stats["mean"], dtype=torch.float32)
            self.std = torch.as_tensor(action_stats["std"], dtype=torch.float32)
            if self.mean.shape != actions.shape[1:] or self.std.shape != actions.shape[1:]:
                raise ValueError("Checkpoint action statistics do not match the dataset")
        self.actions = torch.nan_to_num((actions - self.mean) / self.std, posinf=0, neginf=0)
        self.action_dim = actions.size(1) * frame_skip
        windows = np.maximum((self.lengths - self.span) // stride + 1, 0)
        self.ends = windows.cumsum().tolist()
        if not self.ends or self.ends[-1] == 0:
            raise ValueError("No episodes are long enough for a complete window")

    def __len__(self):
        return self.ends[-1]

    def __getstate__(self):
        return {**self.__dict__, "_file": None}

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        episode = bisect_right(self.ends, index)
        previous = self.ends[episode - 1] if episode else 0
        start = int(self.offsets[episode]) + (index - previous) * self.stride
        end = start + self.span
        if self._file is None:
            self._file = h5py.File(self.path, "r")
        pixels = torch.from_numpy(self._file["pixels"][start:end:self.frame_skip])
        if pixels.ndim != 4:
            raise ValueError("Pixels must have shape (N,H,W,3) or (N,3,H,W)")
        if pixels.size(-1) == 3:
            pixels = pixels.permute(0, 3, 1, 2)
        if pixels.size(1) != 3:
            raise ValueError("LeWM expects RGB observations")
        batch = {
            "pixels": self.transform(pixels.contiguous()),
            "action": self.actions[start:end].reshape(self.history + 1, self.action_dim),
        }
        for column in self.columns:
            batch[column] = torch.as_tensor(self._file[column][start:end:self.frame_skip])
        return batch


def split_dataset(dataset, fraction, seed, split):
    generator = torch.Generator().manual_seed(seed)
    if split == "windows":
        train, val = random_split(dataset, [1 - fraction, fraction], generator=generator)
    else:
        usable = [i for i, end in enumerate(dataset.ends) if end > (dataset.ends[i - 1] if i else 0)]
        if len(usable) < 2:
            raise ValueError("An episode split requires at least two usable episodes")
        order = torch.randperm(len(usable), generator=generator).tolist()
        cutoff = min(len(usable) - 1, max(1, round((1 - fraction) * len(usable))))

        def subset(indices):
            return Subset(dataset, [row for i in indices for ep in (usable[i],)
                                    for row in range(dataset.ends[ep - 1] if ep else 0, dataset.ends[ep])])

        train, val = subset(order[:cutoff]), subset(order[cutoff:])
    if not len(train) or not len(val):
        raise ValueError("The dataset is too small to split into train and validation")
    return train, val
