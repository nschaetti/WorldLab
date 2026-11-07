"""One configuration for all unimodal training arms."""

from dataclasses import dataclass, fields
from pathlib import Path

import yaml

from worldlab.nn import NextTokenPrediction, PlatonicJEPA, TextEncoder
from worldlab.nn.platonic import TEXT_ARCHITECTURES


@dataclass
class RunConfig:
    modality: str = "text"
    objective: str = "lejepa"
    good_views: float = 1.0
    arch: str = "small"
    tokenizer: str = "bert-base-uncased"
    max_len: int = 64
    resolution: int = 224
    local_resolution: int = 96
    global_views: int = 2
    local_views: int = 6
    global_scale: tuple[float, float] = (0.5, 1.0)
    local_scale: tuple[float, float] = (0.15, 0.5)
    mask: float = 0.15
    weight: float = 0.02
    slices: int = 1024
    projection_dim: int = 512
    projection_width: int = 2048
    epochs: int = 200
    batch_size: int = 256
    workers: int = 8
    lr: float = 0.0005
    weight_decay: float = 0.05
    warmup_epochs: int = 10
    grad_clip: float = 0.0
    precision: str = "bf16"
    seed: int = 0

    def __post_init__(self):
        if self.modality not in {"text", "image"} or self.objective not in {"lejepa", "ntp"}:
            raise ValueError("Invalid modality or objective")
        if self.modality == "image" and self.objective != "lejepa":
            raise ValueError("Next-token prediction is text-only")
        if self.arch not in TEXT_ARCHITECTURES or self.precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError("Invalid architecture or precision")
        if self.max_len < 4 or self.max_len * (2 if self.objective == "ntp" and self.good_views else 1) > 512:
            raise ValueError("Token length, including the second caption when enabled, must fit in BERT")
        if min(self.global_views, self.epochs, self.batch_size, self.resolution, self.local_resolution,
               self.projection_dim, self.projection_width, self.slices) < 1:
            raise ValueError("Model and training sizes must be positive")
        if self.modality == "image":
            patch = 8 if self.arch == "smoke" else 16
            if self.resolution % patch or (self.local_views and self.local_resolution % patch):
                raise ValueError("Crop sizes must be divisible by the vision patch size")
        if min(self.local_views, self.workers, self.warmup_epochs) < 0 or self.lr <= 0:
            raise ValueError("Invalid number of local views, workers, warmup or learning rate")
        if self.objective == "lejepa" and self.global_views + self.local_views < 2:
            raise ValueError("JEPA needs at least two views")
        if not all(0 <= value <= 1 for value in (self.good_views, self.mask, self.weight)):
            raise ValueError("Good-view proportion, mask and SIGReg weight must lie in [0,1]")
        for scale in (self.global_scale, self.local_scale):
            if len(scale) != 2 or not 0 < scale[0] <= scale[1] <= 1:
                raise ValueError("Expected span bounds 0 < low <= high <= 1")


PRESETS = {
    "good-text": {},
    "bad-text": {"good_views": 0.0},
    "next-token": {"good_views": 0.0, "objective": "ntp"},
    "paired-next-token": {"objective": "ntp"},
    "image": {"modality": "image", "good_views": 0.0},
    "good-image": {"modality": "image"},
}
SMOKE = {
    "arch": "smoke", "max_len": 16, "resolution": 32, "local_resolution": 16,
    "global_views": 2, "local_views": 1, "slices": 8, "projection_dim": 8,
    "projection_width": 16, "epochs": 1, "batch_size": 4, "workers": 0,
    "precision": "fp32", "warmup_epochs": 0,
}


def load_config(preset, path=None, overrides=(), smoke=False):
    values = dict(PRESETS[preset])
    if smoke:
        values.update(SMOKE)
    if path:
        settings = yaml.safe_load(Path(path).read_text())
        if not isinstance(settings, dict):
            raise ValueError("Training config must be a YAML mapping")
        values.update(settings)
    for option in overrides:
        key, separator, value = option.partition("=")
        if not separator:
            raise ValueError(f"Expected KEY=VALUE, got {option}")
        values[key] = yaml.safe_load(value)
    unknown = values.keys() - {field.name for field in fields(RunConfig)}
    if unknown:
        raise ValueError(f"Unknown settings: {', '.join(sorted(unknown))}")
    return RunConfig(**values)


def load_tokenizer(config):
    if config.modality == "image":
        return None
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(config.tokenizer)


def build_model(config, tokenizer=None):
    if config.modality == "text":
        encoder = TextEncoder(len(tokenizer), tokenizer.pad_token_id, arch=config.arch,
                              causal=config.objective == "ntp")
        if config.objective == "ntp":
            return NextTokenPrediction(encoder)
        width = encoder.embed_dim
    elif config.arch == "smoke":
        from timm.models.vision_transformer import VisionTransformer

        width = 24
        encoder = VisionTransformer(img_size=config.resolution, patch_size=8, embed_dim=width, depth=2,
                                    num_heads=3, num_classes=0, dynamic_img_size=True)
    else:
        import timm

        encoder = timm.create_model(f"vit_{config.arch}_patch16_224", pretrained=False, num_classes=0,
                                    img_size=config.resolution, dynamic_img_size=True, drop_path_rate=0.1)
        width = encoder.num_features
    return PlatonicJEPA(encoder, width, weight=config.weight, projection_dim=config.projection_dim,
                        projection_width=config.projection_width, num_slices=config.slices)
