"""Resolved training recipes and construction of native or published checkpoints."""

import json
from dataclasses import dataclass, fields
from pathlib import Path

import torch
import yaml
from torch import nn

from worldlab.nn import ActionConditionedPredictor, ActionEmbedder, LeWMProjection, LeWorldModel, SIGReg


@dataclass
class RunConfig:
    encoder: str = "vit-tiny"
    resolution: int = 224
    patch_size: int = 14
    latent_dim: int = 192
    projection_width: int = 2048
    history: int = 3
    frame_skip: int = 5
    predictor_depth: int = 6
    predictor_dim: int = 192
    heads: int = 16
    head_dim: int = 64
    predictor_width: int = 2048
    dropout: float = 0.1
    weight: float = 0.09
    slices: int = 1024
    knots: int = 17
    batch_size: int = 128
    epochs: int = 10
    lr: float = 0.00005
    weight_decay: float = 0.001
    warmup_fraction: float = 0.01
    grad_clip: float = 1.0
    precision: str = "bf16"
    workers: int = 6
    seed: int = 3072
    val_fraction: float = 0.1
    split: str = "windows"
    reconstruction_weight: float = 0.0

    def __post_init__(self):
        if min(self.history, self.frame_skip, self.resolution, self.patch_size,
               self.latent_dim, self.batch_size, self.epochs) < 1:
            raise ValueError("Model and training sizes must be positive")
        if not 0 < self.val_fraction < 1 or self.split not in {"windows", "episodes"}:
            raise ValueError("Expected val_fraction in (0, 1) and split=windows or episodes")
        if self.precision not in {"fp32", "bf16", "fp16"} or self.workers < 0:
            raise ValueError("Invalid precision or worker count")
        if not 0 <= self.warmup_fraction < 1:
            raise ValueError("warmup_fraction must lie in [0, 1)")
        if self.encoder == "vit-tiny" and self.resolution % self.patch_size:
            raise ValueError("Resolution must be divisible by the ViT patch size")
        if self.reconstruction_weight < 0:
            raise ValueError("Reconstruction weight must be non-negative")


PRESETS = {
    "pusht": {},
    "tworoom": {"history": 1},
    "cube": {},
    "reacher": {},
    "smoke": {"encoder": "small-cnn", "resolution": 24, "patch_size": 6, "latent_dim": 16,
              "projection_width": 32, "history": 2, "frame_skip": 2, "predictor_depth": 2,
              "predictor_dim": 16, "heads": 2, "head_dim": 8, "predictor_width": 32,
              "slices": 8, "knots": 5, "batch_size": 4, "epochs": 1, "precision": "fp32",
              "workers": 0, "split": "episodes", "warmup_fraction": 0.0},
}


def load_config(preset, path=None, overrides=()):
    settings = dict(PRESETS[preset])
    if path:
        config = yaml.safe_load(Path(path).read_text())
        if not isinstance(config, dict):
            raise ValueError("Training config must be a YAML mapping")
        settings.update(config)
    for option in overrides:
        key, separator, value = option.partition("=")
        if not separator:
            raise ValueError(f"Expected KEY=VALUE, got {option}")
        settings[key] = yaml.safe_load(value)
    unknown = settings.keys() - {field.name for field in fields(RunConfig)}
    if unknown:
        raise ValueError(f"Unknown configuration keys: {', '.join(sorted(unknown))}")
    return RunConfig(**settings)


def vit_tiny(resolution, patch_size):
    from transformers import ViTConfig, ViTModel

    return ViTModel(ViTConfig(
        image_size=resolution, patch_size=patch_size, hidden_size=192, intermediate_size=768,
        num_hidden_layers=12, num_attention_heads=3, layer_norm_eps=1e-12,
    ), add_pooling_layer=False, use_mask_token=False)


def build_model(config, action_dim):
    is_vit = config.encoder == "vit-tiny"
    if is_vit:
        encoder, encoder_dim = vit_tiny(config.resolution, config.patch_size), 192
    elif config.encoder == "resnet18":
        from torchvision.models import resnet18

        encoder = resnet18()
        encoder_dim = encoder.fc.in_features
        encoder.fc = nn.Identity()
    elif config.encoder == "small-cnn":
        encoder_dim = 16
        encoder = nn.Sequential(nn.Conv2d(3, encoder_dim, 3, padding=1), nn.GELU(),
                                nn.AdaptiveAvgPool2d(1), nn.Flatten())
    else:
        raise ValueError(f"Unknown encoder: {config.encoder}")
    return LeWorldModel(
        encoder=encoder,
        predictor=ActionConditionedPredictor(
            num_frames=config.history, input_dim=config.latent_dim, hidden_dim=config.predictor_dim,
            depth=config.predictor_depth, heads=config.heads, mlp_dim=config.predictor_width,
            dim_head=config.head_dim, dropout=config.dropout,
        ),
        action_encoder=ActionEmbedder(action_dim, config.latent_dim),
        projector=LeWMProjection(encoder_dim, config.projection_width, config.latent_dim),
        pred_proj=LeWMProjection(config.latent_dim, config.projection_width, config.latent_dim),
        sigreg=SIGReg(config.slices, config.knots, t_max=3, seed=config.seed),
        weight=config.weight, hf_encoder=is_vit,
    )


def _without_target(settings):
    return {key: value for key, value in settings.items() if not key.startswith("_")}


def load_checkpoint(path, device="cpu"):
    """Return a model, image resolution and optional training action statistics."""
    path = Path(path).expanduser()
    action_stats = None
    if path.is_dir() and (path / "config.json").exists():
        settings = json.loads((path / "config.json").read_text())
        encoder_cfg = settings["encoder"]
        if encoder_cfg["size"] != "tiny" or encoder_cfg.get("use_mask_token", False):
            raise ValueError("Expected an official LeWM ViT-Tiny checkpoint")

        def projection(key):
            options = _without_target(settings[key])
            options.pop("norm_fn", None)
            return LeWMProjection(**options)

        model = LeWorldModel(
            vit_tiny(encoder_cfg["image_size"], encoder_cfg["patch_size"]),
            ActionConditionedPredictor(**_without_target(settings["predictor"])),
            ActionEmbedder(**_without_target(settings["action_encoder"])),
            projection("projector"), projection("pred_proj"), hf_encoder=True,
        )
        state = torch.load(path / "weights.pt", map_location="cpu", weights_only=True)
        resolution = encoder_cfg["image_size"]
    else:
        path = path / "last.pt" if path.is_dir() else path
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        config = RunConfig(**checkpoint["config"])
        model = build_model(config, checkpoint["action_dim"])
        state, resolution = checkpoint["model"], config.resolution
        action_stats = checkpoint.get("action_stats")
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), resolution, action_stats
