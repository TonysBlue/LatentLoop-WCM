from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from model.types import LayerKV, SlowMemoryState


class SlowMemory(nn.Module):
    """Fixed-size linear associative memory with gated delta writes."""

    def __init__(self, heads: int, head_dim: int) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.write_gate = nn.Linear(1, 1)
        self.decay_logit = nn.Parameter(torch.tensor(4.0))

    def _phi(self, value: Tensor) -> Tensor:
        return torch.nn.functional.elu(value.float()) + 1.0

    def read(self, query: Tensor, state: SlowMemoryState) -> Tensor:
        original_shape = query.shape
        if query.ndim == 3:
            batch, tokens, dim = query.shape
            query = query.view(batch, tokens, self.heads, self.head_dim).transpose(1, 2)
            flatten = True
        else:
            flatten = False
        features = self._phi(query)
        numerator = torch.einsum("bhqd,bhde->bhqe", features, state.matrix.float())
        denominator = torch.einsum("bhqd,bhd->bhq", features, state.normalizer.float())
        output = (numerator / denominator.clamp_min(1e-4).unsqueeze(-1)).to(query.dtype)
        if flatten:
            output = output.transpose(1, 2).reshape(original_shape)
        return output

    def write(self, state: SlowMemoryState, expired: LayerKV) -> SlowMemoryState:
        if expired.key.shape[2] == 0:
            return state
        keys = self._phi(expired.key)
        values = expired.value.float()
        features = keys
        numerator = torch.einsum("bhqd,bhde->bhqe", features, state.matrix.float())
        denominator = torch.einsum("bhqd,bhd->bhq", features, state.normalizer.float())
        predicted = (numerator / denominator.clamp_min(1e-4).unsqueeze(-1)).float()
        error = values - predicted
        surprise = error.square().mean(dim=(1, 2, 3), keepdim=True).sqrt()
        gate_input = surprise.reshape(-1, 1).to(self.write_gate.weight.dtype)
        gate = torch.sigmoid(self.write_gate(gate_input)).float().reshape(-1, 1, 1, 1)
        decay = torch.sigmoid(self.decay_logit)
        matrix = state.matrix.float() * decay
        normalizer = state.normalizer.float() * decay
        matrix = matrix + gate * torch.einsum("bhqd,bhqe->bhde", keys, error)
        normalizer = normalizer + gate.squeeze(-1) * keys.sum(dim=2)
        return SlowMemoryState(matrix.to(state.matrix.dtype), normalizer.to(state.normalizer.dtype))


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
    ) -> tuple[Tensor, LayerKV, LayerKV]:
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
        evicted = LayerKV(
            key=all_key[:, :, :-max_tokens], value=all_value[:, :, :-max_tokens]
        )
        return self.out(output), new_cache, evicted


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
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden: Tensor,
        cache: LayerKV,
        max_tokens: int,
        slow_memory: SlowMemory,
        slow_state: SlowMemoryState,
        semantic_memory: Tensor,
    ) -> tuple[Tensor, LayerKV, SlowMemoryState, Tensor]:
        attended, new_cache, evicted = self.self_attention(
            self.self_norm(hidden),
            cache,
            max_tokens,
        )
        slow_context = slow_memory.read(self.self_norm(hidden), slow_state).mean(dim=1)
        attended = attended + slow_context
        hidden = hidden + self.dropout(attended)
        if self.world_attention is not None and self.world_norm is not None:
            normalized = self.world_norm(hidden)
            crossed, _ = self.world_attention(
                normalized, semantic_memory, semantic_memory, need_weights=False
            )
            hidden = hidden + self.dropout(crossed)
        hidden = hidden + self.dropout(self.ffn(self.ffn_norm(hidden)))
        return (
            hidden,
            new_cache,
            slow_memory.write(slow_state, evicted),
            torch.zeros_like(hidden[..., :1]),
        )
