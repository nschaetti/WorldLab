"""Vision backbones and the online classification probe."""

import timm
import torch
from timm.models.vision_transformer import VisionTransformer
from torch import nn
from torch.optim.swa_utils import AveragedModel
from torch.utils.checkpoint import checkpoint

from worldlab.nn import LeJEPA, LeJEPAObjective, ProjectionMLP, SIGReg


def pool_tokens(tokens, backbone):
    if backbone.cls_token is not None:
        return tokens[:, 0]
    return tokens[:, backbone.num_prefix_tokens:].mean(dim=1)


class VisionEncoder(nn.Module):
    """Expose backbone features for probing, with final-layer features last."""

    def __init__(self, config):
        super().__init__()
        kwargs = {}
        name = config.backbone
        if name == "vit_tiny_patch8_224":
            if config.pretrained:
                raise ValueError("The ViT-Tiny/8 variant has no pretrained timm weights")
            name = "vit_tiny_patch16_224"
            kwargs["patch_size"] = 8
        if name.startswith(("vit_", "deit_")) and not name.startswith("vit_pe_core_"):
            kwargs.update(img_size=config.resolution, dynamic_img_size=bool(config.local_views),
                          patch_drop_rate=config.patch_mask_ratio)
            if config.register_tokens:
                kwargs["reg_tokens"] = config.register_tokens
        elif config.patch_mask_ratio or config.register_tokens:
            raise ValueError("Patch masking and register tokens require a ViT backbone")
        elif config.backbone.startswith("swin"):
            kwargs["img_size"] = config.resolution
        self.backbone = timm.create_model(
            name, pretrained=config.pretrained, num_classes=0,
            drop_path_rate=config.drop_path_rate, **kwargs,
        )
        if isinstance(self.backbone, VisionTransformer):
            sizes = [config.resolution]
            if config.local_views:
                sizes.append(config.local_resolution)
            patch_height, patch_width = self.backbone.patch_embed.patch_size
            if any(size % patch_height or size % patch_width for size in sizes):
                raise ValueError(f"Crop sizes {sizes} must be divisible by the ViT patch size")
        self.last_dim = self.backbone.num_features
        if not isinstance(self.backbone, VisionTransformer):
            self.last_dim = getattr(self.backbone, "head_hidden_size", self.last_dim)
        self.layers = 1
        self.checkpoint_blocks = config.activation_checkpointing
        if isinstance(self.backbone, VisionTransformer):
            self.layers = min(max(1, config.probe_layers), len(self.backbone.blocks))
        elif self.checkpoint_blocks and hasattr(self.backbone, "set_grad_checkpointing"):
            self.backbone.set_grad_checkpointing()
        self.readout = nn.Identity()
        self.output_dim = self.last_dim * self.layers
        if config.probe_layers == 0:
            self.readout = nn.Linear(self.last_dim, config.embedding_dim)
            self.output_dim = self.last_dim = config.embedding_dim

    def forward(self, images):
        backbone = self.backbone
        if not isinstance(backbone, VisionTransformer) or self.layers == 1:
            if isinstance(backbone, VisionTransformer) and self.checkpoint_blocks:
                tokens = self._vit_blocks(images)[-1]
                return self.readout(backbone.forward_head(tokens, pre_logits=True))
            return self.readout(backbone(images))
        return torch.cat([pool_tokens(tokens, backbone) for tokens in self._vit_blocks(images)], dim=-1)

    def _vit_blocks(self, images):
        backbone = self.backbone
        tokens = backbone.patch_embed(images)
        tokens = backbone.norm_pre(backbone.patch_drop(backbone._pos_embed(tokens)))
        last_layers = []
        for index, block in enumerate(backbone.blocks):
            if self.checkpoint_blocks and self.training and torch.is_grad_enabled():
                tokens = checkpoint(block, tokens, use_reentrant=False)
            else:
                tokens = block(tokens)
            if index >= len(backbone.blocks) - self.layers:
                last_layers.append(backbone.norm(tokens))
        return last_layers


class ImageProjector(nn.Module):
    def __init__(self, encoder, config):
        super().__init__()
        self.last_dim = encoder.last_dim
        self.embedding = (nn.Linear(self.last_dim, config.embedding_dim)
                          if config.probe_layers else nn.Identity())
        self.projector = ProjectionMLP(
            config.embedding_dim, config.projection_dim,
            (config.projector_width,) * (config.projector_depth - 1),
        )

    def forward(self, features):
        return self.projector(self.embedding(features[..., -self.last_dim:]))


class LeJEPAClassifier(nn.Module):
    """LeJEPA with a label-isolated linear probe and optional Table 4 variants."""

    def __init__(self, config, num_classes):
        super().__init__()
        encoder = VisionEncoder(config)
        objective = LeJEPAObjective(config.weight, SIGReg(
            config.num_slices, config.num_points, config.t_max, seed=config.seed,
            chunk_size=config.chunk_size, symmetric_grid=config.symmetric_grid,
        ))
        self.jepa = LeJEPA(encoder, ImageProjector(encoder, config), objective)
        self.probe = nn.Sequential(nn.LayerNorm(encoder.output_dim), nn.Linear(encoder.output_dim, num_classes))
        self.predictor = None
        if config.predictor:
            self.predictor = ProjectionMLP(config.projection_dim, config.projection_dim, (config.projector_width,))
        self.teacher = AveragedModel(self.jepa) if config.swa else None
        if self.teacher is not None:
            self.teacher.requires_grad_(False)
        self.global_views = config.global_views
        self.probe_train_views = config.probe_train_views

    def forward(self, views, step):
        features, projections = self.jepa.encode_views(views)
        predictions = None
        if self.predictor is not None:
            predictions = self.predictor(projections.flatten(0, 1)).reshape_as(projections)
        target = None
        if self.teacher is not None:
            self.teacher.eval()
            with torch.no_grad():
                _, teacher_projections = self.teacher.module.encode_views(views[:self.global_views])
                target = teacher_projections.mean(0)
        ssl = self.jepa.objective(
            projections, self.global_views, step=step, predictions=predictions, target=target,
        )
        if self.probe_train_views == "global":
            features = features[:self.global_views]
        return ssl, self.probe(features.detach())

    def classify(self, images):
        return self.probe(self.jepa.encode(images))

    def ssl_parameters(self):
        yield from self.jepa.parameters()
        if self.predictor is not None:
            yield from self.predictor.parameters()

    def update_teacher(self):
        if self.teacher is not None:
            self.teacher.update_parameters(self.jepa)


def set_unfused_attention(backbone):
    """Capture the last attention softmax via its dropout module in eval mode."""
    attention = backbone.blocks[-1].attn
    attention.fused_attn = False
    return attention.attn_drop
