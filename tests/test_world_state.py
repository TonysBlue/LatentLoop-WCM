from __future__ import annotations

import torch
from data import SyntheticEpisodeDataset
from model import StreamingLatentLoop
from model.latentloop import DeltaTimeEncoder


def test_semantic_memory_is_updated_inside_backbone(smoke_config) -> None:
    model = StreamingLatentLoop(smoke_config.model)
    unit = SyntheticEpisodeDataset(smoke_config.data, smoke_config.model).make_episode(0).units[0]
    state = model.initial_state(1, "cpu")
    output = model(unit, state)
    assert output.state.semantic_memory.shape == (
        1, smoke_config.model.semantic_memory_slots, smoke_config.model.model_dim
    )
    assert not torch.equal(output.state.semantic_memory, state.semantic_memory)


def test_delta_time_encoder_ignores_absolute_timestamp(smoke_config) -> None:
    encoder = DeltaTimeEncoder(
        smoke_config.model.model_dim,
        bands=smoke_config.model.delta_time_fourier_bands,
        base_period_ms=smoke_config.model.delta_time_base_period_ms,
    )
    delta = torch.tensor([80.0, 500.0])

    first = encoder(delta)
    second = encoder(delta)

    assert first.shape == (2, 1, smoke_config.model.model_dim)
    assert torch.equal(first, second)


def test_delta_time_changes_backbone_and_semantic_state(smoke_config) -> None:
    model = StreamingLatentLoop(smoke_config.model)
    episode = SyntheticEpisodeDataset(smoke_config.data, smoke_config.model).make_episode(0)
    unit = episode.units[0]
    state = model.initial_state(1, "cpu")
    state.hidden = torch.randn_like(state.hidden)
    state.semantic_memory = torch.randn_like(state.semantic_memory)

    short = unit.to("cpu")
    long = unit.to("cpu")
    short.delta_ms = torch.tensor([80])
    long.delta_ms = torch.tensor([1_000])

    short_output = model(short, state)
    long_output = model(long, state)

    assert not torch.equal(short_output.state.semantic_memory, long_output.state.semantic_memory)
    assert not torch.equal(short_output.hidden, long_output.hidden)
