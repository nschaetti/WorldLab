"""Unimodal networks used in the Platonic Representation paper (arXiv:2610.39302)."""

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .lejepa import LeJEPA, LeJEPAObjective, ProjectionMLP
from .sigreg import SIGReg

TEXT_ARCHITECTURES = {
    "tiny": (192, 3, 12),
    "small": (384, 6, 12),
    "base": (768, 12, 12),
    "large": (1024, 16, 24),
    "smoke": (24, 3, 2),
}


class TextEncoder(nn.Module):
    """Randomly initialized BERT with CLS pooling or causal mean pooling."""

    def __init__(
            self,
            vocab_size: int,
            pad_id: int, *,
            arch: str = "small",
            causal: bool = False
    ):
        super().__init__()
        from transformers import BertConfig, BertModel

        width, heads, depth = TEXT_ARCHITECTURES[arch]
        self.bert = BertModel(
            BertConfig(
                vocab_size=vocab_size, pad_token_id=pad_id, hidden_size=width,
                num_hidden_layers=depth, num_attention_heads=heads, intermediate_size=4 * width,
                is_decoder=causal, use_cache=False,
            ),
            add_pooling_layer=False,
        )
        self.embed_dim = width
        self.pad_id = pad_id
        self.causal = causal

    def hidden(self, tokens: Tensor, *, all_layers: bool = False):
        return self.bert(input_ids=tokens, attention_mask=tokens.ne(self.pad_id),
                         output_hidden_states=all_layers)

    def mean_pool(self, features: Tensor, tokens: Tensor) -> Tensor:
        valid = tokens.ne(self.pad_id).unsqueeze(-1)
        return (features * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)

    def forward(self, tokens: Tensor) -> Tensor:
        features = self.hidden(tokens).last_hidden_state
        return self.mean_pool(features, tokens) if self.causal else features[:, 0]


class NextTokenPrediction(nn.Module):
    """Tied next-token head on the causal counterpart of the text JEPA encoder."""

    def __init__(self, encoder: TextEncoder):
        super().__init__()
        if not encoder.causal:
            raise ValueError("Next-token prediction requires a causal encoder")
        self.encoder = encoder

    def forward(self, tokens: Tensor) -> Tensor:
        # A causal position cannot see its label or any later token.
        hidden = self.encoder.hidden(tokens[:, :-1]).last_hidden_state
        logits = hidden @ self.encoder.bert.embeddings.word_embeddings.weight.T
        return F.cross_entropy(logits.flatten(0, 1).float(), tokens[:, 1:].flatten(),
                               ignore_index=self.encoder.pad_id)


class PlatonicJEPA(LeJEPA):
    """The paper's identity-predictor JEPA, instantiated separately per modality."""

    def __init__(
        self, encoder: nn.Module, feature_dim: int, *, weight: float = 0.02,
        projection_dim: int = 512, projection_width: int = 2048, num_slices: int = 1024,
    ):
        projector = nn.Sequential(
            nn.Linear(feature_dim, projection_dim),
            ProjectionMLP(projection_dim, projection_dim, (projection_width, projection_width)),
        )
        objective = LeJEPAObjective(weight, SIGReg(num_slices, t_max=3),
                                   pool_views=True, additive=True)
        super().__init__(encoder, projector, objective)

    def encode_views(self, views):
        # The same projector batch statistics are shared by global and local views.
        features = torch.cat([
            encoded.reshape(count, batch, -1) for encoded, count, batch in self._encoded_groups(views)
        ])
        projections = self.projector(features.flatten(0, 1))
        return features, projections.reshape(*features.shape[:2], -1)
