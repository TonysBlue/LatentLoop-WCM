from __future__ import annotations

import torch
from torch import Tensor, nn


class _PerceiverBlock(nn.Module):
    def __init__(self, dim: int, heads: int, ffn_dim: int, dropout: float) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(dim)
        self.cross_attention = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True
        )
        self.self_norm = nn.LayerNorm(dim)
        self.self_attention = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True
        )
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, slots: Tensor, inputs: Tensor) -> Tensor:
        normalized = self.input_norm(slots)
        crossed, _ = self.cross_attention(normalized, inputs, inputs, need_weights=False)
        slots = slots + self.dropout(crossed)
        normalized = self.self_norm(slots)
        attended, _ = self.self_attention(normalized, normalized, normalized, need_weights=False)
        slots = slots + self.dropout(attended)
        return slots + self.dropout(self.ffn(self.ffn_norm(slots)))


class Perceiver(nn.Module):
    """Compress all observation stems into fixed, ordered multimodal slots."""

    def __init__(
        self, dim: int, heads: int, ffn_dim: int, slots: int, layers: int, dropout: float
    ) -> None:
        super().__init__()
        self.queries = nn.Parameter(torch.zeros(slots, dim))
        self.layers = nn.ModuleList(
            _PerceiverBlock(dim, heads, ffn_dim, dropout) for _ in range(layers)
        )
        self.final_norm = nn.LayerNorm(dim)
        nn.init.normal_(self.queries, std=0.02)

    def forward(self, inputs: Tensor) -> Tensor:
        slots = self.queries[None].expand(inputs.shape[0], -1, -1)
        for layer in self.layers:
            slots = layer(slots, inputs)
        return self.final_norm(slots)


class _PredictorBlock(nn.Module):
    def __init__(self, dim: int, heads: int, ffn_dim: int, dropout: float) -> None:
        super().__init__()
        self.self_norm = nn.LayerNorm(dim)
        self.self_attention = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True
        )
        self.world_norm = nn.LayerNorm(dim)
        self.world_attention = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True
        )
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, slots: Tensor, world: Tensor) -> Tensor:
        normalized = self.self_norm(slots)
        attended, _ = self.self_attention(normalized, normalized, normalized, need_weights=False)
        slots = slots + self.dropout(attended)
        normalized = self.world_norm(slots)
        crossed, _ = self.world_attention(normalized, world, world, need_weights=False)
        slots = slots + self.dropout(crossed)
        return slots + self.dropout(self.ffn(self.ffn_norm(slots)))


class JEPAHead(nn.Module):
    """Predict the next Perceiver representation from the causal state."""

    def __init__(
        self,
        dim: int,
        latent_dim: int,
        heads: int,
        ffn_dim: int,
        slots: int,
        layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.slot_identity = nn.Parameter(torch.zeros(slots, dim))
        self.world_projection = nn.Linear(latent_dim, dim)
        self.layers = nn.ModuleList(
            _PredictorBlock(dim, heads, ffn_dim, dropout) for _ in range(layers)
        )
        self.final_norm = nn.LayerNorm(dim)
        nn.init.normal_(self.slot_identity, std=0.02)

    def forward(self, hidden: Tensor, semantic_memory: Tensor) -> Tensor:
        slots = hidden + self.slot_identity[None]
        projected_world = self.world_projection(semantic_memory)
        for layer in self.layers:
            slots = layer(slots, projected_world)
        return self.final_norm(slots)


class PredictionAdapter(nn.Module):
    """Map detached predicted slots into the Backbone's future context space."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.projection = nn.Linear(dim, dim)
        with torch.no_grad():
            self.projection.weight.copy_(torch.eye(dim))
            self.projection.bias.zero_()

    def forward(self, predicted: Tensor) -> Tensor:
        return self.projection(self.norm(predicted))


Predictor = JEPAHead

__all__ = ["Perceiver", "PredictionAdapter", "JEPAHead", "Predictor"]
