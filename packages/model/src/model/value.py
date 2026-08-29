from __future__ import annotations

import torch
from torch import Tensor, nn


class ValueHead(nn.Module):
    """Training-only value estimate from the causal recurrent representation."""

    def __init__(self, model_dim: int, heads: int) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.zeros(1, model_dim))
        self.query_attention = nn.MultiheadAttention(
            model_dim, heads, batch_first=True
        )
        self.query_norm = nn.LayerNorm(model_dim)
        self.network = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, 1),
        )
        nn.init.normal_(self.query, std=0.02)

    def forward(self, hidden: Tensor) -> Tensor:
        query = self.query[None].expand(hidden.shape[0], -1, -1)
        state_query, _ = self.query_attention(query, hidden, hidden, need_weights=False)
        state_query = self.query_norm(state_query[:, 0])
        return self.network(state_query).squeeze(-1)


__all__ = ["ValueHead"]
