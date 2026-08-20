from __future__ import annotations

import torch
from runtime.config import ModelConfig
from torch import Tensor, nn

from model.types import SpeechLocalState, SpeechSamplingConfig


class FactorizedSpeechHead(nn.Module):
    """Speech mode plus causal residual-codec prediction from one unit hidden."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        dim = config.model_dim
        self.config = config
        self.query = nn.Parameter(torch.zeros(1, dim))
        self.query_attention = nn.MultiheadAttention(
            dim, config.speech_depth_heads, dropout=config.dropout, batch_first=True
        )
        self.query_norm = nn.LayerNorm(dim)
        self.temporal = nn.GRUCell(dim, dim)
        self.mode = nn.Linear(dim, 2)
        self.depth_embeddings = nn.ModuleList(
            nn.Embedding(config.speech_codebook_size, dim) for _ in range(config.speech_codebooks)
        )
        self.bos = nn.Parameter(torch.zeros(dim))
        self.positions = nn.Parameter(torch.zeros(config.speech_codebooks, dim))
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=config.speech_depth_heads,
            dim_feedforward=config.speech_depth_ffn_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.depth = nn.TransformerEncoder(
            layer, config.speech_depth_layers, enable_nested_tensor=False
        )
        self.depth_norm = nn.LayerNorm(dim)
        self.output_bias = nn.Parameter(
            torch.zeros(config.speech_codebooks, config.speech_codebook_size)
        )
        nn.init.normal_(self.bos, std=0.02)
        nn.init.normal_(self.positions, std=0.02)
        nn.init.normal_(self.query, std=0.02)

    def context(self, hidden: Tensor) -> Tensor:
        query = self.query[None].expand(hidden.shape[0], -1, -1)
        context, _ = self.query_attention(query, hidden, hidden, need_weights=False)
        return self.query_norm(context[:, 0])

    def update_temporal(self, context: Tensor, state: SpeechLocalState) -> Tensor:
        active = state.previous_codes.ne(0).any(dim=-1, keepdim=True)
        previous = torch.stack(
            [
                embedding(state.previous_codes[:, index])
                for index, embedding in enumerate(self.depth_embeddings)
            ],
            dim=1,
        ).mean(dim=1)
        previous = torch.where(active, previous, torch.zeros_like(previous))
        previous_state = torch.where(active, state.temporal, torch.zeros_like(state.temporal))
        return self.temporal(context + previous, previous_state)

    def mode_logits(self, context: Tensor) -> Tensor:
        return self.mode(context)

    def _inputs(self, temporal: Tensor, previous: list[Tensor]) -> Tensor:
        batch = temporal.shape[0]
        values = [self.bos.expand(batch, -1)]
        values.extend(
            embedding(code)
            for embedding, code in zip(self.depth_embeddings, previous, strict=False)
        )
        return torch.stack(values, dim=1) + temporal[:, None] + self.positions[: len(values)][None]

    def _hidden(self, temporal: Tensor, previous: list[Tensor]) -> Tensor:
        values = self._inputs(temporal, previous)
        length = values.shape[1]
        mask = torch.triu(
            torch.ones(length, length, device=values.device, dtype=torch.bool), diagonal=1
        )
        return self.depth_norm(self.depth(values, mask=mask))

    def _logits(self, hidden: Tensor, codebook: int) -> Tensor:
        return torch.nn.functional.linear(
            hidden, self.depth_embeddings[codebook].weight, self.output_bias[codebook]
        )

    def teacher_logits(self, temporal: Tensor, codes: Tensor) -> Tensor:
        previous = [codes[:, 0, index] for index in range(self.config.speech_codebooks - 1)]
        hidden = self._hidden(temporal, previous)
        logits = [
            self._logits(hidden[:, index], index) for index in range(self.config.speech_codebooks)
        ]
        return torch.stack(logits, dim=1)[:, None]

    def generate(self, temporal: Tensor, sampling: SpeechSamplingConfig) -> tuple[Tensor, Tensor]:
        selected: list[Tensor] = []
        logits: list[Tensor] = []
        for codebook in range(self.config.speech_codebooks):
            hidden = self._hidden(temporal, selected)[:, -1]
            current = self._logits(hidden, codebook)
            logits.append(current)
            selected.append(self._sample(current, sampling))
        return torch.stack(logits, dim=1)[:, None], torch.stack(selected, dim=1)[:, None]

    @staticmethod
    def sampled_logprob(logits: Tensor, tokens: Tensor) -> Tensor:
        """Return log-probability for sampled codec tokens without a new head."""
        return torch.log_softmax(logits, dim=-1).gather(-1, tokens.unsqueeze(-1)).squeeze(-1)

    @staticmethod
    def _sample(logits: Tensor, sampling: SpeechSamplingConfig) -> Tensor:
        # A physical rollout must never emit an invalid probability tensor. In
        # addition to making the failure observable in training metrics, this
        # keeps a transient non-finite activation from crossing the codec
        # boundary as an arbitrary device-side error.
        logits = torch.nan_to_num(logits, nan=0.0, posinf=30.0, neginf=-30.0)
        if sampling.greedy or sampling.temperature <= 0:
            return logits.argmax(dim=-1)
        scaled = logits / sampling.temperature
        if 0 < sampling.top_k < scaled.shape[-1]:
            values, indices = torch.topk(scaled, sampling.top_k, dim=-1)
            choice = torch.multinomial(torch.softmax(values, dim=-1), 1)
            return indices.gather(-1, choice).squeeze(-1)
        return torch.multinomial(torch.softmax(scaled, dim=-1), 1).squeeze(-1)
