"""Vision backbones and the online classification probe."""

from collections.abc import Sequence
from typing import Iterator

import timm
import torch
from timm.models.vision_transformer import VisionTransformer
from torch import Tensor, nn
from torch.optim.swa_utils import AveragedModel
from torch.utils.checkpoint import checkpoint

from worldlab.experiments.lejepa.config import RunConfig
from worldlab.nn import LeJEPA, LeJEPAObjective, LeJEPAOutput, ProjectionMLP, SIGReg


def pool_tokens(tokens: Tensor, backbone: nn.Module) -> Tensor:
    """Pool per-token features into a single feature vector per sample.

    Args:
        tokens (Tensor): Token features shaped (batch, tokens, features),
            the output of a vision backbone.
        backbone (nn.Module): The vision backbone that produced the
            tokens; read for ``cls_token`` and ``num_prefix_tokens``.

    Returns:
        Tensor: Pooled features shaped (batch, features). The class token
        is returned directly when present, otherwise the patch tokens are
        averaged.
    """
    if backbone.cls_token is not None:
        # (batch, tokens, features) -> (batch, features), keep the CLS token
        return tokens[:, 0]
    # end if
    # (batch, tokens - prefix, features) -> (batch, features)
    return tokens[:, backbone.num_prefix_tokens:].mean(dim=1)
    # end return
# end def pool_tokens


class VisionEncoder(nn.Module):
    """Expose backbone features for probing, with final-layer features last.

    Wraps a timm backbone, handles the ViT-Tiny/8 special case without
    pretrained weights, validates resolution divisibility and optionally
    reads the last ``probe_layers`` transformer blocks.

    Attributes:
        backbone (nn.Module): The timm backbone with ``num_classes=0``.
        last_dim (int): Size of the final (or pooled) feature dimension.
        layers (int): Number of transformer blocks read for probing;
            1 for non-ViT backbones.
        checkpoint_blocks (bool): Whether ViT blocks run under gradient
            checkpointing during training.
        readout (nn.Module): Identity or the linear embedding head when
            ``probe_layers == 0``.
        output_dim (int): Size of the encoder output feature dimension.
    """

    def __init__(
            self,
            config: RunConfig
    ):
        """Instantiate the timm backbone and the probing readout.

        Args:
            config (RunConfig): Run configuration supplying the backbone
                name, resolution, probe settings, patch masking, register
                tokens, drop path rate and activation checkpointing flag.

        Raises:
            ValueError: If a pretrained ViT-Tiny/8 is requested, if patch
                masking or register tokens are requested with a non-ViT
                backbone, or if crop sizes are not divisible by the ViT
                patch size.
        """
        super().__init__()

        kwargs = {}

        name = config.backbone

        # ViT Tiny
        if name == "vit_tiny_patch8_224":
            if config.pretrained:
                raise ValueError("The ViT-Tiny/8 variant has no pretrained timm weights")
            # end if
            # Fall back to the patch16 architecture and force patch size 8
            name = "vit_tiny_patch16_224"
            kwargs["patch_size"] = 8
        # end if

        # vit_
        if name.startswith(("vit_", "deit_")) and not name.startswith("vit_pe_core_"):
            kwargs.update(
                img_size=config.resolution,
                dynamic_img_size=bool(config.local_views),
                patch_drop_rate=config.patch_mask_ratio
            )
            if config.register_tokens:
                kwargs["reg_tokens"] = config.register_tokens
            # end if
        elif config.patch_mask_ratio or config.register_tokens:
            raise ValueError("Patch masking and register tokens require a ViT backbone")
        elif config.backbone.startswith("swin"):
            kwargs["img_size"] = config.resolution
        # end if

        self.backbone = timm.create_model(
            name,
            pretrained=config.pretrained,
            num_classes=0,
            drop_path_rate=config.drop_path_rate,
            **kwargs,
        )

        if isinstance(self.backbone, VisionTransformer):
            sizes = [config.resolution]
            if config.local_views:
                sizes.append(config.local_resolution)
            # end if
            patch_height, patch_width = self.backbone.patch_embed.patch_size
            if any(size % patch_height or size % patch_width for size in sizes):
                raise ValueError(f"Crop sizes {sizes} must be divisible by the ViT patch size")
            # end if
        # end if

        self.last_dim = self.backbone.num_features
        if not isinstance(self.backbone, VisionTransformer):
            self.last_dim = getattr(self.backbone, "head_hidden_size", self.last_dim)
        # end if

        self.layers = 1
        self.checkpoint_blocks = config.activation_checkpointing

        if isinstance(self.backbone, VisionTransformer):
            # Probe reads the last max(1, probe_layers) blocks
            self.layers = min(max(1, config.probe_layers), len(self.backbone.blocks))
        elif self.checkpoint_blocks and hasattr(self.backbone, "set_grad_checkpointing"):
            self.backbone.set_grad_checkpointing()
        # end if

        self.readout = nn.Identity()
        self.output_dim = self.last_dim * self.layers

        if config.probe_layers == 0:
            # Embedding head replaces the probing readout, shape (batch, embedding_dim)
            self.readout = nn.Linear(self.last_dim, config.embedding_dim)
            self.output_dim = self.last_dim = config.embedding_dim
        # end if
    # end def __init__

    def forward(self, images: Tensor) -> Tensor:
        """Encode image batches into probe-ready feature vectors.

        Args:
            images (Tensor): Images shaped (batch, 3, resolution,
                resolution), or mixed-resolution views batched per group.

        Returns:
            Tensor: Features shaped (batch, output_dim). With a multi-layer
            ViT probe, final-layer features of read blocks are concatenated
            along the feature dimension.
        """
        backbone = self.backbone
        if not isinstance(backbone, VisionTransformer) or self.layers == 1:
            if isinstance(backbone, VisionTransformer) and self.checkpoint_blocks:
                # Last probed block tokens, shape (batch, tokens, features)
                tokens = self._vit_blocks(images)[-1]
                # Backbone head pooling, -> (batch, last_dim)
                return self.readout(backbone.forward_head(tokens, pre_logits=True))
            # end if
            # Backbone output, shape (batch, last_dim)
            return self.readout(backbone(images))
        # end if
        # Concatenate the pooled tokens of probed layers, -> (batch, last_dim * layers)
        return torch.cat(
            [
                pool_tokens(tokens, backbone)
                for tokens in self._vit_blocks(images)
            ],
            dim=-1
        )
        # end return
    # end def forward

    def _vit_blocks(
            self,
            images: Tensor
    ) -> list[Tensor]:
        """Run the ViT patch embedding plus blocks, keeping probed layers.

        Args:
            images (Tensor): Images shaped (batch, 3, resolution,
                resolution), local or global crops share the batch axis.

        Returns:
            list[Tensor]: Pre-normed token tensors of each probed block,
            shaped (batch, tokens, features), the last element being the
            final transformer block.

        Raises:
            ValueError: If ``images`` has an invalid shape or when gradient
                checkpointing fails on the backbone.
        """
        backbone = self.backbone
        # Patch embedding, (batch, num_patches, embed_dim)
        tokens = backbone.patch_embed(images)
        # Pre-norm: positional embedding + patch drop + pre-norm, (batch, tokens, features)
        tokens = backbone.norm_pre(
            backbone.patch_drop(backbone._pos_embed(tokens))
        )
        last_layers = []
        for index, block in enumerate(backbone.blocks):
            if self.checkpoint_blocks and self.training and torch.is_grad_enabled():
                tokens = checkpoint(block, tokens, use_reentrant=False)
            else:
                tokens = block(tokens)
            # end if
            if index >= len(backbone.blocks) - self.layers:
                last_layers.append(backbone.norm(tokens))
            # end if
        # end for
        # Return the probed layers, each (batch, tokens, features)
        return last_layers
        # end return
    # end def _vit_blocks
# end class VisionEncoder


class ImageProjector(nn.Module):
    """Project encoder features to LeJEPA embeddings and projections.

    Attributes:
        last_dim (int): Size of the encoder's final feature dimension.
        embedding (nn.Module): Linear embedding or Identity depending on
            the ``probe_layers`` setting of the run configuration.
        projector (ProjectionMLP): Convex projection head mapping the
            embedding to the LeJEPA projection space.
    """

    def __init__(self, encoder: VisionEncoder, config: RunConfig):
        """Build the embedding head and the projection MLP.

        Args:
            encoder (VisionEncoder): The encoder whose ``last_dim`` sets
                the input feature size of the embedding head.
            config (RunConfig): Run configuration supplying the embedding
                size, projection size, projector width and depth.
        """
        super().__init__()
        self.last_dim = encoder.last_dim
        self.embedding = (nn.Linear(self.last_dim, config.embedding_dim)
                          if config.probe_layers else nn.Identity())
        self.projector = ProjectionMLP(
            config.embedding_dim, config.projection_dim,
            (config.projector_width,) * (config.projector_depth - 1),
        )
    # end def __init__

    def forward(self, features: Tensor) -> Tensor:
        """Project encoder features into the LeJEPA projection space.

        Args:
            features (Tensor): Backbone features shaped (..., features),
                possibly mixing views of different feature sizes along the
                last dimension.

        Returns:
            Tensor: Projections shaped (..., projection_dim), matching the
            input shape recovered by the projector head.
        """
        return self.projector(
            self.embedding(features[..., -self.last_dim:]))
        # end return
    # end def forward
# end class ImageProjector


class LeJEPAClassifier(nn.Module):
    """LeJEPA with a label-isolated linear probe and optional Table 4 variants.

    Combines the LeJEPA objective, an ImageProjector head on top of a
    vision encoder and a label-linear probe. Optional variants of the
    paper's Table 4: predictor head, SWA teacher, per-view probe training
    policy.

    Attributes:
        jepa (LeJEPA): The encoder + projector + objective pipeline.
        probe (nn.Module): LayerNorm + Linear classification head.
        predictor (ProjectionMLP | None): Optional predictor head applied
            on projections before the prediction term.
        teacher (AveragedModel | None): Optional SWA teacher copy of the
            JEPA pipeline used to produce the prediction target.
        global_views (int): Number of leading global views.
        probe_train_views (str): Whether the probe trains on "global" or
            "all" views.
    """

    def __init__(
            self,
            config: RunConfig,
            num_classes: int,
    ):
        """Assemble the JEPA pipeline, the probe and the optional variants.

        Args:
            config (RunConfig): Run configuration supplying the encoder,
                objective (SIGReg) and the optional variant flags.
            num_classes (int): Number of classes the probe classifies.

        Raises:
            ValueError: Raised indirectly from ``VisionEncoder`` when the
                backbone or resolution configuration is inconsistent.
        """
        super().__init__()

        # Initialize the VisionEncoder
        encoder = VisionEncoder(config=config)
        # encoder : (batch, output_dim)


        objective = LeJEPAObjective(
            config.weight,
            SIGReg(
                config.num_slices,
                config.num_points,
                config.t_max,
                seed=config.seed,
                chunk_size=config.chunk_size,
                symmetric_grid=config.symmetric_grid
            )
        )

        self.jepa = LeJEPA(
            encoder=encoder,
            projector=ImageProjector(encoder, config),
            objective=objective
        )

        self.probe = nn.Sequential(
            nn.LayerNorm(encoder.output_dim),
            nn.Linear(encoder.output_dim, num_classes)
        )

        self.predictor = None
        if config.predictor:
            self.predictor = ProjectionMLP(
                config.projection_dim,
                config.projection_dim,
                (config.projector_width,)
            )
        # end if

        self.teacher = AveragedModel(self.jepa) if config.swa else None
        if self.teacher is not None:
            self.teacher.requires_grad_(False)
        # end if

        self.global_views = config.global_views
        self.probe_train_views = config.probe_train_views
    # end def __init__

    def forward(
            self,
            views: Tensor | Sequence[Tensor],
            step: int,
    ) -> tuple[LeJEPAOutput, Tensor]:
        """Compute the SSL loss and probe logits for a view batch.

        Args:
            views (Tensor | Sequence[Tensor]): Multi-crop views shaped
                (views, batch, 3, resolution, resolution) or a sequence of
                (batch, 3, resolution, resolution) tensors of mixed
                resolutions, global views first.
            step (int): Optimization step forwarded to the objective for
                deterministic SIGReg direction sampling.

        Returns:
            tuple[LeJEPAOutput, Tensor]: The SSL loss container and the
            probe logits shaped (probe_views, batch, num_classes) where
            ``probe_views`` is ``global_views`` for the "global" policy
            and ``views`` for the "all" policy.
        """
        # Encode all views and project them, stacking views
        # per sample.
        features, projections = self.jepa.encode_views(
            views=views
        )
        # features : (views, batch, F)
        # projections : (views, batch, D)

        predictions = None
        if self.predictor is not None:
            # (views * batch, projection_dim), reshaped to (views, batch, projection_dim)
            predictions = self.predictor(
                projections.flatten(0, 1)
            ).reshape_as(projections)
        # end if

        target = None
        if self.teacher is not None:
            self.teacher.eval()
            with torch.no_grad():
                _, teacher_projections = self.teacher.module.encode_views(
                    views[:self.global_views]
                )
                # (global_views, batch, projection_dim) -> (batch, projection_dim)
                target = teacher_projections.mean(0)
            # end with
        # end if

        ssl = self.jepa.objective(
            projections,
            self.global_views,
            step=step,
            predictions=predictions,
            target=target,
        )

        if self.probe_train_views == "global":
            # (global_views, batch, features)
            features = features[:self.global_views]
        # end if

        return ssl, self.probe(features.detach())
    # end def forward

    def classify(self, images: Tensor) -> Tensor:
        """Classify single-view image batches, with the probe only.

        Args:
            images (Tensor): Images shaped (batch, 3, resolution,
                resolution), matching the evaluation transform output.

        Returns:
            Tensor: Probe logits shaped (batch, num_classes).
        """
        return self.probe(self.jepa.encode(images))
    # end def classify

    def ssl_parameters(self) -> Iterator[nn.Parameter]:
        """Yield the parameters trained by the SSL objective.

        Yields:
            Parameter: All the JEPA pipeline parameters, followed by the
            predictor parameters when a predictor head exists.
        """
        yield from self.jepa.parameters()
        if self.predictor is not None:
            yield from self.predictor.parameters()
        # end if
    # end def ssl_parameters

    def update_teacher(self) -> None:
        """Fold the current JEPA parameters into the SWA teacher."""
        if self.teacher is not None:
            self.teacher.update_parameters(self.jepa)
        # end if
    # end def update_teacher
# end class LeJEPAClassifier


def set_unfused_attention(backbone: nn.Module) -> nn.Dropout:
    """Capture the last attention softmax via its dropout module in eval mode.

    Args:
        backbone (nn.Module): A ViT backbone whose ``blocks[-1].attn``
            exposes ``fused_attn`` and ``attn_drop``.

    Returns:
        nn.Dropout: The attention dropout module, whose forward now runs
        on the attention weights after re-enabling the unfused path.
    """
    attention = backbone.blocks[-1].attn
    attention.fused_attn = False
    return attention.attn_drop
    # end return
# end def set_unfused_attention
