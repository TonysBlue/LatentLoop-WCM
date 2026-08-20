from __future__ import annotations

import torch
from data import SyntheticEpisodeDataset
from model import StreamingLatentLoop, compute_jepa_loss
from model.losses import compute_losses


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


def test_behavior_and_jepa_gradients_follow_separate_predictor_paths(smoke_config) -> None:
    model = StreamingLatentLoop(smoke_config.model)
    units = SyntheticEpisodeDataset(smoke_config.data, smoke_config.model).make_episode(0).units
    first = model(units[0], model.initial_state(1, "cpu"), units[0].speech_codes)

    compute_losses(first, units[0])["total"].backward()

    assert model.predictor.final_norm.weight.grad is None
    assert model.prediction_adapter.projection.weight.grad is not None
    assert model.layers[0].future_gate.weight.grad is not None
    assert model.perceiver.queries.grad is not None

    model.zero_grad(set_to_none=True)
    first = model(units[0], model.initial_state(1, "cpu"), units[0].speech_codes)
    second = model(units[1], first.state, units[1].speech_codes)
    jepa = compute_jepa_loss(
        first.predicted_next_slots,
        first.perceiver_slots,
        second.perceiver_slots,
    )
    jepa["total"].backward()

    assert model.predictor.final_norm.weight.grad is not None
    assert model.perceiver.queries.grad is not None
    assert model.world_state_update.gate.weight.grad is not None


def test_perceiver_only_lookahead_does_not_advance_recurrent_state(smoke_config) -> None:
    model = StreamingLatentLoop(smoke_config.model)
    units = SyntheticEpisodeDataset(smoke_config.data, smoke_config.model).make_episode(0).units
    output = model(units[0], model.initial_state(1, "cpu"))
    state = output.state
    before = state.detach()

    target, returned_cache = model.encode_observation(units[1], state.audio_cache.clone())

    assert target.shape == (1, 16, smoke_config.model.model_dim)
    assert returned_cache.shape == state.audio_cache.shape
    assert torch.equal(state.latent, before.latent)
    assert torch.equal(state.hidden, before.hidden)
    assert torch.equal(state.unit_index, before.unit_index)
    assert all(
        torch.equal(actual.key, expected.key)
        for actual, expected in zip(state.layer_kv, before.layer_kv, strict=True)
    )


def test_future_gate_starts_as_small_per_slot_residual(smoke_config) -> None:
    model = StreamingLatentLoop(smoke_config.model)
    unit = SyntheticEpisodeDataset(smoke_config.data, smoke_config.model).make_episode(0).units[0]

    output = model(unit, model.initial_state(1, "cpu"))

    expected = torch.sigmoid(torch.tensor(-4.0))
    assert torch.isclose(output.future_gate_mean, expected, atol=1e-6)
    assert torch.isclose(output.future_gate_max, expected, atol=1e-6)
    assert all(layer.future_gate.out_features == 1 for layer in model.layers)
