from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from model.types import LayerKV


class CachedSelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.dropout = dropout

    def forward(
        self,
        x: Tensor,
        cache: LayerKV,
        max_tokens: int,
    ) -> tuple[Tensor, LayerKV]:
        batch, current_tokens, dim = x.shape
        qkv = self.qkv(x).view(batch, current_tokens, 3, self.heads, self.head_dim)
        query, key, value = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        all_key = torch.cat((cache.key, key), dim=2)
        all_value = torch.cat((cache.value, value), dim=2)
        scores = torch.matmul(query, all_key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        weights = torch.softmax(scores.float(), dim=-1).to(scores.dtype)
        weights = torch.dropout(weights, self.dropout, self.training)
        output = (
            torch.matmul(weights, all_value).transpose(1, 2).reshape(batch, current_tokens, dim)
        )

        new_cache = LayerKV(
            key=all_key[:, :, -max_tokens:],
            value=all_value[:, :, -max_tokens:],
        )
        return self.out(output), new_cache


class StreamingTransformerLayer(nn.Module):
    def __init__(
        self, dim: int, heads: int, ffn_dim: int, dropout: float, cross_world: bool
    ) -> None:
        super().__init__()
        self.self_norm = nn.LayerNorm(dim)
        self.self_attention = CachedSelfAttention(dim, heads, dropout)
        self.world_norm = nn.LayerNorm(dim) if cross_world else None
        self.world_attention = (
            nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
            if cross_world
            else None
        )
        self.future_norm = nn.LayerNorm(dim)
        self.future_attention = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True
        )
        self.future_gate = nn.Linear(dim * 2, 1)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim),
        )
        self.dropout = nn.Dropout(dropout)
        nn.init.zeros_(self.future_gate.weight)
        nn.init.constant_(self.future_gate.bias, -4.0)

    def forward(
        self,
        hidden: Tensor,
        world: Tensor,
        future: Tensor,
        cache: LayerKV,
        max_tokens: int,
    ) -> tuple[Tensor, LayerKV, Tensor]:
        attended, new_cache = self.self_attention(
            self.self_norm(hidden),
            cache,
            max_tokens,
        )
        hidden = hidden + self.dropout(attended)
        if self.world_attention is not None and self.world_norm is not None:
            normalized = self.world_norm(hidden)
            crossed, _ = self.world_attention(normalized, world, world, need_weights=False)
            hidden = hidden + self.dropout(crossed)
        normalized = self.future_norm(hidden)
        future_context, _ = self.future_attention(
            normalized, future, future, need_weights=False
        )
        gate = torch.sigmoid(self.future_gate(torch.cat((normalized, future_context), dim=-1)))
        hidden = hidden + self.dropout(gate * future_context)
        hidden = hidden + self.dropout(self.ffn(self.ffn_norm(hidden)))
        return hidden, new_cache, gate
