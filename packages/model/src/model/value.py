from __future__ import annotations

import torch
from torch import Tensor, nn


class ValueHead(nn.Module):
    """Training-only value estimate from the causal recurrent representation."""

    def __init__(self, model_dim: int, latent_dim: int, heads: int) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.zeros(1, model_dim))
        self.query_attention = nn.MultiheadAttention(
            model_dim, heads, batch_first=True
        )
        self.query_norm = nn.LayerNorm(model_dim)
        self.latent_projection = nn.Linear(latent_dim, model_dim)
        self.network = nn.Sequential(
            nn.LayerNorm(model_dim * 2),
            nn.Linear(model_dim * 2, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, 1),
        )
        nn.init.normal_(self.query, std=0.02)

    def forward(self, hidden: Tensor, latent: Tensor) -> Tensor:
        query = self.query[None].expand(hidden.shape[0], -1, -1)
        state_query, _ = self.query_attention(query, hidden, hidden, need_weights=False)
        state_query = self.query_norm(state_query[:, 0])
        pooled_latent = self.latent_projection(latent.mean(dim=1))
        return self.network(torch.cat((state_query, pooled_latent), dim=-1)).squeeze(-1)


__all__ = ["ValueHead"]
