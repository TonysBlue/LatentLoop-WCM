from __future__ import annotations

import torch
from data import SyntheticEpisodeDataset
from model import StreamingLatentLoop, compute_jepa_loss
from model.attention import SlowMemory
from model.losses import compute_losses
from model.types import LayerKV, SlowMemoryState


def test_jepa_target_is_detached_and_single_sample_skips_variance() -> None:
    predicted = torch.randn(1, 16, 8, requires_grad=True)
    source = torch.randn(1, 16, 8, requires_grad=True)
    target = torch.randn(1, 16, 8, requires_grad=True)

    losses = compute_jepa_loss(predicted, source, target)
    losses["total"].backward()

    assert predicted.grad is not None
    assert source.grad is not None
    assert target.grad is None
    assert losses["variance"].item() == 0.0


def test_jepa_variance_uses_batch_time_samples() -> None:
    predicted = torch.randn(3, 2, 16, 8, requires_grad=True)
    source = torch.zeros(3, 2, 16, 8, requires_grad=True)
    target = torch.randn(3, 2, 16, 8)

    losses = compute_jepa_loss(predicted, source, target)

    assert torch.isfinite(losses["total"])
    assert losses["variance"] > 0.9


def test_behavior_and_jepa_gradients_follow_separate_jepa_paths(smoke_config) -> None:
    model = StreamingLatentLoop(smoke_config.model)
    units = SyntheticEpisodeDataset(smoke_config.data, smoke_config.model).make_episode(0).units
    first = model(units[0], model.initial_state(1, "cpu"), units[0].speech_codes)

    compute_losses(first, units[0])["total"].backward()

    assert model.jepa_head.final_norm.weight.grad is None
    assert model.perceiver.queries.grad is not None

    model.zero_grad(set_to_none=True)
    first = model(units[0], model.initial_state(1, "cpu"), units[0].speech_codes)
    second = model(units[1], first.state, units[1].speech_codes)
    jepa = compute_jepa_loss(
        first.jepa_prediction,
        first.perceiver_slots,
        second.perceiver_slots,
    )
    jepa["total"].backward()

    assert model.jepa_head.final_norm.weight.grad is not None
    assert model.perceiver.queries.grad is not None
    assert model.semantic_slot_identity.grad is not None


def test_perceiver_only_lookahead_does_not_advance_recurrent_state(smoke_config) -> None:
    model = StreamingLatentLoop(smoke_config.model)
    units = SyntheticEpisodeDataset(smoke_config.data, smoke_config.model).make_episode(0).units
    output = model(units[0], model.initial_state(1, "cpu"))
    state = output.state
    before = state.detach()

    target, returned_cache = model.encode_observation(units[1], state.audio_cache.clone())

    assert target.shape == (1, 16, smoke_config.model.model_dim)
    assert returned_cache.shape == state.audio_cache.shape
    assert torch.equal(state.semantic_memory, before.semantic_memory)
    assert torch.equal(state.hidden, before.hidden)
    assert torch.equal(state.unit_index, before.unit_index)
    assert all(
        torch.equal(actual.key, expected.key)
        for actual, expected in zip(state.layer_kv, before.layer_kv, strict=True)
    )


def test_slow_memory_is_fixed_capacity(smoke_config) -> None:
    model = StreamingLatentLoop(smoke_config.model)
    unit = SyntheticEpisodeDataset(smoke_config.data, smoke_config.model).make_episode(0).units[0]

    output = model(unit, model.initial_state(1, "cpu"))

    assert len(output.state.slow_memory) == smoke_config.model.num_layers
    assert len(model.slow_memories) == smoke_config.model.num_layers


def test_slow_memory_no_eviction_is_identity_and_reads_are_finite() -> None:
    memory = SlowMemory(heads=2, head_dim=4)
    state = SlowMemoryState(torch.zeros(1, 2, 4, 4), torch.zeros(1, 2, 4))
    empty = LayerKV(torch.empty(1, 2, 0, 4), torch.empty(1, 2, 0, 4))

    unchanged = memory.write(state, empty)
    read = memory.read(torch.randn(1, 3, 8), unchanged)

    assert unchanged is state
    assert torch.isfinite(read).all()
    assert torch.equal(read, torch.zeros_like(read))


def test_slow_memory_delta_write_corrects_conflicting_value_and_decays() -> None:
    memory = SlowMemory(heads=1, head_dim=2)
    with torch.no_grad():
        memory.write_gate.weight.zero_()
        memory.write_gate.bias.fill_(10.0)
        memory.decay_logit.fill_(0.0)
    state = SlowMemoryState(torch.zeros(1, 1, 2, 2), torch.zeros(1, 1, 2))
    key = torch.tensor([[[[0.5, -0.5]]]])
    first = LayerKV(key, torch.tensor([[[[1.0, 0.0]]]]))
    second = LayerKV(key, torch.tensor([[[[0.0, 1.0]]]]))

    after_first = memory.write(state, first)
    before_correction = memory.read(key, after_first)
    after_second = memory.write(after_first, second)
    after_correction = memory.read(key, after_second)

    assert after_second.matrix.shape == state.matrix.shape
    assert after_second.normalizer.shape == state.normalizer.shape
    assert after_second.normalizer.sum() < after_first.normalizer.sum() * 1.6
    assert after_correction[..., 1].item() > before_correction[..., 1].item()


def test_long_stream_keeps_recent_and_slow_memory_bounded(smoke_config) -> None:
    model = StreamingLatentLoop(smoke_config.model).eval()
    unit = SyntheticEpisodeDataset(smoke_config.data, smoke_config.model).make_episode(0).units[0]
    state = model.initial_state(1, "cpu")
    expected_kv = smoke_config.model.kv_units * smoke_config.model.perceiver_slots
    expected_slow = (
        1,
        smoke_config.model.num_heads,
        smoke_config.model.model_dim // smoke_config.model.num_heads,
    )

    with torch.no_grad():
        for _ in range(smoke_config.model.kv_units * 3):
            state = model(unit, state, unit.speech_codes).state

    assert all(cache.key.shape[2] == expected_kv for cache in state.layer_kv)
    assert all(memory.normalizer.shape == expected_slow for memory in state.slow_memory)
    assert all(torch.isfinite(memory.matrix).all() for memory in state.slow_memory)
