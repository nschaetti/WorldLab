"""Patch-query decoder for frozen-encoder diagnostics and the reconstruction ablation."""

import torch
from torch import nn


class _DecoderBlock(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.mlp = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 4 * width),
                                 nn.GELU(), nn.Linear(4 * width, width))

    def forward(self, queries, latent):
        attended, _ = self.attention(self.norm(queries), latent, latent, need_weights=False)
        queries = queries + attended
        return queries + self.mlp(queries)


class PatchDecoder(nn.Module):
    def __init__(self, latent_dim, resolution, patch_size=16, width=128, depth=3):
        super().__init__()
        if resolution % patch_size:
            raise ValueError("Resolution must be divisible by the decoder patch size")
        self.grid = resolution // patch_size
        self.patch = patch_size
        self.queries = nn.Parameter(torch.randn(1, self.grid**2, width) * 0.02)
        self.latent = nn.Linear(latent_dim, width)
        self.layers = nn.ModuleList(_DecoderBlock(width, 4) for _ in range(depth))
        self.rgb = nn.Linear(width, 3 * patch_size**2)

    def forward(self, embeddings):
        latent = self.latent(embeddings.flatten(0, 1)).unsqueeze(1)
        queries = self.queries.expand(len(latent), -1, -1)
        for layer in self.layers:
            queries = layer(queries, latent)
        patches = self.rgb(queries).reshape(-1, self.grid, self.grid, 3, self.patch, self.patch)
        image_size = self.grid * self.patch
        return patches.permute(0, 3, 1, 4, 2, 5).reshape(*embeddings.shape[:2], 3, image_size, image_size)
