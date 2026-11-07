"""Action-conditioned LeWorldModel, arXiv:2603.19312v3.

The predictor architecture follows the MIT-licensed reference by Lucas Maes.
See licenses/le-wm.txt. Parameter names match the public weights.
"""

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .sigreg import SIGReg, _working_dtype


class ActionEmbedder(nn.Module):
    """Embed each flattened block of consecutive environment actions."""

    def __init__(self, input_dim: int, emb_dim: int, smoothed_dim: int = 10, mlp_scale: int = 4):
        super().__init__()
        self.patch_embed = nn.Conv1d(input_dim, smoothed_dim, kernel_size=1)
        self.embed = nn.Sequential(
            nn.Linear(smoothed_dim, mlp_scale * emb_dim), nn.SiLU(),
            nn.Linear(mlp_scale * emb_dim, emb_dim),
        )

    def forward(self, actions: Tensor) -> Tensor:
        return self.embed(self.patch_embed(actions.transpose(1, 2)).transpose(1, 2))


class _Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout):
        super().__init__()
        self.heads = heads
        self.dropout = dropout
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, 3 * heads * dim_head, bias=False)
        self.to_out = (
            nn.Identity() if heads == 1 and dim_head == dim
            else nn.Sequential(nn.Linear(heads * dim_head, dim), nn.Dropout(dropout))
        )

    def forward(self, x):
        batch, time, _ = x.shape
        qkv = self.to_qkv(self.norm(x)).reshape(batch, time, 3, self.heads, -1)
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        out = F.scaled_dot_product_attention(
            query, key, value, is_causal=True, dropout_p=self.dropout if self.training else 0.0,
        )
        return self.to_out(out.transpose(1, 2).reshape(batch, time, -1))


class _FeedForward(nn.Module):
    def __init__(self, dim, width, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, width), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(width, dim), nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class _ConditionalBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, width, dropout):
        super().__init__()
        self.attn = _Attention(dim, heads, dim_head, dropout)
        self.mlp = _FeedForward(dim, width, dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x, condition):
        attn_shift, attn_scale, attn_gate, mlp_shift, mlp_scale, mlp_gate = (
            self.adaLN_modulation(condition).chunk(6, dim=-1)
        )
        x = x + attn_gate * self.attn(self.norm1(x) * (1 + attn_scale) + attn_shift)
        return x + mlp_gate * self.mlp(self.norm2(x) * (1 + mlp_scale) + mlp_shift)


def _linear_or_identity(input_dim, output_dim):
    return nn.Identity() if input_dim == output_dim else nn.Linear(input_dim, output_dim)


class _ActionTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, depth, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.input_proj = _linear_or_identity(input_dim, hidden_dim)
        self.cond_proj = _linear_or_identity(input_dim, hidden_dim)
        self.output_proj = _linear_or_identity(hidden_dim, output_dim)
        self.layers = nn.ModuleList(
            _ConditionalBlock(hidden_dim, heads, dim_head, mlp_dim, dropout) for _ in range(depth)
        )

    def forward(self, x, condition):
        x, condition = self.input_proj(x), self.cond_proj(condition)
        for block in self.layers:
            x = block(x, condition)
        return self.output_proj(self.norm(x))


class ActionConditionedPredictor(nn.Module):
    """Causal temporal transformer with zero-initialized AdaLN action conditioning."""

    def __init__(
        self, *, num_frames: int, input_dim: int, hidden_dim: int, depth: int = 6,
        heads: int = 16, mlp_dim: int = 2048, dim_head: int = 64, output_dim: int | None = None,
        dropout: float = 0.1, emb_dropout: float = 0.0,
    ):
        super().__init__()
        if min(num_frames, input_dim, hidden_dim, depth, heads, mlp_dim, dim_head) < 1:
            raise ValueError("Predictor dimensions must be positive")
        self.num_frames = num_frames
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, input_dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = _ActionTransformer(
            input_dim, hidden_dim, output_dim or input_dim, depth, heads, dim_head, mlp_dim, dropout,
        )

    def forward(self, latents: Tensor, action_embeddings: Tensor) -> Tensor:
        if latents.ndim != 3 or latents.shape != action_embeddings.shape:
            raise ValueError("Expected aligned latents and action embeddings shaped (B, T, D)")
        if not 1 <= latents.size(1) <= self.num_frames:
            raise ValueError("Sequence length exceeds the predictor context")
        return self.transformer(
            self.dropout(latents + self.pos_embedding[:, :latents.size(1)]), action_embeddings,
        )


class LeWMProjection(nn.Module):
    """GELU projection head with batch normalization as used in LeWM."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


@dataclass
class LeWMOutput:
    loss: Tensor
    prediction: Tensor
    sigreg: Tensor
    embeddings: Tensor


class LeWorldModel(nn.Module):
    """LeWM with a pluggable frame encoder and explicit action timing.

    An action block at position t leaves frame t. The encoder must return a
    vector for each frame, or a Hugging Face ViT output with a CLS token.
    Training targets use the same encoder and remain in the gradient graph.
    """

    def __init__(
        self, encoder: nn.Module, predictor: ActionConditionedPredictor, action_encoder: nn.Module,
        projector: nn.Module | None = None, pred_proj: nn.Module | None = None,
        *, sigreg: SIGReg | None = None, weight: float = 0.09, hf_encoder: bool = False,
    ):
        super().__init__()
        if not 0 <= weight < float("inf"):
            raise ValueError("SIGReg weight must be finite and non-negative")
        self.encoder = encoder
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.projector = projector if projector is not None else nn.Identity()
        self.pred_proj = pred_proj if pred_proj is not None else nn.Identity()
        self.sigreg = sigreg if sigreg is not None else SIGReg(t_max=3.0)
        self.weight = weight
        self.hf_encoder = hf_encoder

    @property
    def history(self) -> int:
        return self.predictor.num_frames

    def encode(self, frames: Tensor) -> Tensor:
        if frames.ndim < 3 or min(frames.shape[:2]) < 1:
            raise ValueError("Expected frames shaped (B, T, ...)")
        flat = frames.flatten(0, 1)
        if self.hf_encoder:
            features = self.encoder(flat, interpolate_pos_encoding=True).last_hidden_state[:, 0]
        else:
            features = self.encoder(flat)
        if features.ndim != 2:
            raise ValueError("The frame encoder must return one vector per frame")
        return self.projector(features).reshape(*frames.shape[:2], -1)

    def predict(self, latents: Tensor, actions: Tensor) -> Tensor:
        """Predict from a context and the action leaving each context frame."""
        return self._predict(latents, self.action_encoder(actions))

    def _predict(self, latents: Tensor, action_embeddings: Tensor) -> Tensor:
        predicted = self.predictor(latents, action_embeddings)
        return self.pred_proj(predicted.flatten(0, 1)).reshape_as(predicted)

    def forward(self, frames: Tensor, actions: Tensor, *, step: int | None = None) -> LeWMOutput:
        if frames.ndim < 3 or frames.size(1) != self.history + 1:
            raise ValueError("Training clips must contain history + 1 frames")
        if actions.ndim != 3 or actions.size(0) != frames.size(0) or actions.size(1) not in (
            self.history, self.history + 1,
        ):
            raise ValueError("Provide one action block per transition (or per frame)")
        embeddings = self.encode(frames)
        predicted = self.predict(embeddings[:, :-1], torch.nan_to_num(actions[:, :self.history], posinf=0, neginf=0))
        target = embeddings[:, 1:].to(_working_dtype(embeddings))
        prediction = (predicted.to(target) - target).square().mean()
        regularization = self.sigreg(embeddings.transpose(0, 1), step=step)
        return LeWMOutput(prediction + self.weight * regularization, prediction, regularization, embeddings)

    def rollout(self, context: Tensor, future_actions: Tensor, past_actions: Tensor | None = None) -> Tensor:
        """Return context followed by predicted latents, preserving action gradients.

        Context is (B, H, D). Future actions are (B, S, T, A) for S plans.
        Past actions are (B, H-1, A) and shared by all plans from a context.
        Use eval mode for planning to keep dropout and batch statistics fixed.
        """
        if self.training:
            raise ValueError("Use model.eval() for autoregressive planning")
        if context.ndim != 3 or future_actions.ndim != 4 or context.size(0) != future_actions.size(0):
            raise ValueError("Expected context (B,H,D) and future actions (B,S,T,A)")
        batch, samples, horizon, action_dim = future_actions.shape
        history = context.size(1)
        if not 1 <= history <= self.history or min(batch, samples, horizon, action_dim) < 1:
            raise ValueError("Invalid rollout context or horizon")
        if past_actions is None:
            if history != 1:
                raise ValueError("Multi-frame rollout needs the executed action blocks between frames")
            past_actions = future_actions.new_empty(batch, 0, action_dim)
        if past_actions.shape != (batch, history - 1, action_dim):
            raise ValueError("Past actions must have shape (B,H-1,A)")
        all_actions = torch.cat((past_actions[:, None].expand(-1, samples, -1, -1), future_actions), dim=2)
        action_embeddings = self.action_encoder(all_actions.flatten(0, 1))
        frames = list(context[:, None].expand(-1, samples, -1, -1).flatten(0, 1).unbind(1))
        for time in range(horizon):
            end = history + time
            start = max(0, end - self.history)
            latent_history = torch.stack(frames[start:], dim=1)
            frames.append(self._predict(latent_history, action_embeddings[:, start:end])[:, -1])
        return torch.stack(frames, dim=1).reshape(batch, samples, history + horizon, -1)

    def goal_cost(
        self, context: Tensor, goal: Tensor, future_actions: Tensor, past_actions: Tensor | None = None,
    ) -> Tensor:
        """Squared Euclidean distance from each terminal latent to a goal (B, D)."""
        terminal = self.rollout(context, future_actions, past_actions)[:, :, -1]
        return (terminal - goal[:, None]).square().sum(dim=-1)
