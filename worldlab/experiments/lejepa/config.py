"""Configuration for LeJEPA runs and experiment matrices."""

from dataclasses import dataclass, fields
from pathlib import Path

import yaml


@dataclass
class RunConfig:
    dataset: str = "imagenette"
    data_root: str = "data/lejepa"
    download: bool = False
    backbone: str = "resnet50"
    epochs: int = 400
    batch_size: int = 512  # global, across all ranks
    workers: int = 8
    seed: int = 0
    precision: str = "bf16"
    lr: float = 0.0005
    weight_decay: float = 0.0005
    min_lr_ratio: float = 0.001
    min_lr: float | None = None
    warmup_epochs: int = 1
    grad_clip: float = 0.0
    probe_lr: float = 0.001
    probe_weight_decay: float = 0.000001
    probe_layers: int = 2
    probe_train_views: str = "global"
    resolution: int = 224
    local_resolution: int = 96
    global_views: int = 8
    local_views: int = 0
    augment: str = "paper"
    weight: float = 0.05
    embedding_dim: int = 512
    projection_dim: int = 128
    projector_depth: int = 3
    projector_width: int = 2048
    num_slices: int = 1024
    num_points: int = 17
    t_max: float = 5.0
    symmetric_grid: bool = False
    chunk_size: int | None = 64
    drop_path_rate: float = 0.0
    register_tokens: int = 0
    patch_mask_ratio: float = 0.0
    predictor: bool = False
    swa: bool = False
    activation_checkpointing: bool = False
    sync_batchnorm: bool = False
    eval_every: int = 1
    pretrained: bool = False

    def __post_init__(self):
        positive = (
            "epochs", "batch_size", "resolution", "local_resolution", "global_views", "eval_every",
            "embedding_dim", "projection_dim", "projector_depth", "projector_width",
        )
        if any(getattr(self, key) < 1 for key in positive):
            raise ValueError(f"Expected positive values for {', '.join(positive)}")
        if min(self.local_views, self.workers, self.register_tokens, self.warmup_epochs) < 0:
            raise ValueError("View, worker, register and warmup counts must be non-negative")
        if self.global_views + self.local_views < 2:
            raise ValueError("At least two views are required for pretraining")
        if not 0 <= self.patch_mask_ratio < 1:
            raise ValueError("patch_mask_ratio must lie in [0, 1)")
        if self.precision not in {"fp32", "bf16", "fp16"}:
            raise ValueError("precision must be fp32, bf16 or fp16")
        if self.augment not in {"paper", "minimal"}:
            raise ValueError("augment must be paper or minimal")
        if self.probe_layers not in {0, 1, 2} or self.probe_train_views not in {"global", "all"}:
            raise ValueError("Invalid probe_layers or probe_train_views")


def read_yaml(path):
    with open(path, encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return value


def presets():
    return read_yaml(Path(__file__).with_name("presets.yaml"))
# end def presets


def load_config(
        preset=None,
        path=None,
        overrides=()
) -> RunConfig:
    values = dict(presets()[preset]) if preset else {}
    if path:
        values.update(read_yaml(path))
    # end if
    for override in overrides:
        key, separator, value = override.partition("=")
        if not separator:
            raise ValueError(f"Expected KEY=VALUE, got {override}")
        # end if
        values[key] = yaml.safe_load(value)
    # end for
    unknown = values.keys() - {field.name for field in fields(RunConfig)}
    if unknown:
        raise ValueError(f"Unknown configuration keys: {', '.join(sorted(unknown))}")
    # end if
    return RunConfig(**values)
# end def load_config
