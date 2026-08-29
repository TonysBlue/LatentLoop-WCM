from __future__ import annotations

import math

import torch
from data import SyntheticEpisodeDataset
from model import StreamingLatentLoop
from model.losses import compute_losses
from runtime.config import ProjectConfig


def test_recurrent_state_is_bounded_and_heads_receive_gradients(
    smoke_config: ProjectConfig,
) -> None:
    model = StreamingLatentLoop(smoke_config.model)
    episode = SyntheticEpisodeDataset(smoke_config.data, smoke_config.model).make_episode(0)
    state = model.initial_state(1, "cpu")
    total = torch.tensor(0.0)
    output = None
    for unit in episode.units:
        output = model(unit, state, unit.speech_codes)
        state = output.state
        total = total + compute_losses(output, unit)["total"]

    assert output is not None
    max_tokens = smoke_config.model.kv_units * smoke_config.model.perceiver_slots
    assert all(cache.key.shape[2] == max_tokens for cache in state.layer_kv)
    assert state.semantic_memory.shape == (
        1,
        smoke_config.model.semantic_memory_slots,
        smoke_config.model.model_dim,
    )
    total.backward()
    assert model.audio_encoder.conv.weight.grad is not None
    assert model.vision_encoder.encoder[0].weight.grad is not None
    assert model.semantic_slot_identity.grad is not None
    assert model.slow_memories[0].write_gate.weight.grad is not None
    assert model.speech_head.depth_embeddings[0].weight.grad is not None
    assert model.action_head.kind_output.weight.grad is not None
    assert model.speech_head.mode.weight.grad is not None
    assert output.perceiver_slots.shape == (1, 16, smoke_config.model.model_dim)
    assert output.jepa_prediction.shape == output.perceiver_slots.shape
    assert output.value.shape == (1,)


def test_training_value_head_receives_policy_value_gradient(smoke_config: ProjectConfig) -> None:
    model = StreamingLatentLoop(smoke_config.model)
    unit = SyntheticEpisodeDataset(smoke_config.data, smoke_config.model).make_episode(0).units[0]
    output = model(unit, model.initial_state(1, "cpu"), unit.speech_codes)

    output.value.square().mean().backward()

    assert model.value_head.network[-1].weight.grad is not None


def test_semantic_slots_are_backbone_state_but_not_recent_kv(smoke_config: ProjectConfig) -> None:
    model = StreamingLatentLoop(smoke_config.model).eval()
    unit = SyntheticEpisodeDataset(smoke_config.data, smoke_config.model).make_episode(0).units[0]
    initial = model.initial_state(1, "cpu")
    altered = initial.detach()
    altered.semantic_memory = altered.semantic_memory + 1.0

    with torch.no_grad():
        baseline = model(unit, initial, unit.speech_codes)
        changed = model(unit, altered, unit.speech_codes)

    assert not torch.equal(baseline.hidden, changed.hidden)
    expected_cached = smoke_config.model.perceiver_slots
    assert all(cache.key.shape[2] == expected_cached for cache in baseline.state.layer_kv)
    assert torch.equal(initial.semantic_memory[0], model.semantic_slot_identity)


def test_detach_breaks_tbptt_graph(smoke_config: ProjectConfig) -> None:
    model = StreamingLatentLoop(smoke_config.model)
    unit = SyntheticEpisodeDataset(smoke_config.data, smoke_config.model).make_episode(0).units[0]
    state = model(unit, model.initial_state(1, "cpu")).state.detach()
    assert state.semantic_memory.grad_fn is None
    assert all(memory.matrix.grad_fn is None for memory in state.slow_memory)
    assert all(cache.key.grad_fn is None for cache in state.layer_kv)


def test_activation_checkpointing_supports_backward(smoke_config: ProjectConfig) -> None:
    smoke_config.model.activation_checkpointing = True
    model = StreamingLatentLoop(smoke_config.model)
    unit = SyntheticEpisodeDataset(smoke_config.data, smoke_config.model).make_episode(0).units[0]
    output = model(unit, model.initial_state(1, "cpu"), unit.speech_codes)

    compute_losses(output, unit)["total"].backward()

    assert model.layers[0].self_attention.qkv.weight.grad is not None


def test_speech_loss_averages_over_valid_codec_tokens(smoke_config: ProjectConfig) -> None:
    model = StreamingLatentLoop(smoke_config.model)
    unit = SyntheticEpisodeDataset(smoke_config.data, smoke_config.model).make_episode(0).units[0]
    output = model(unit, model.initial_state(1, "cpu"), unit.speech_codes)
    output.speech_codec_logits = torch.zeros_like(output.speech_codec_logits)
    output.speech_mode_logits = torch.zeros_like(output.speech_mode_logits)

    speech_loss = compute_losses(output, unit)["speech"]

    expected = math.log(smoke_config.model.speech_codebook_size) + math.log(2)
    assert torch.isclose(speech_loss, torch.tensor(expected), atol=1e-4)


def test_speech_loss_follows_model_dtype(smoke_config: ProjectConfig) -> None:
    model = StreamingLatentLoop(smoke_config.model).half()
    unit = SyntheticEpisodeDataset(smoke_config.data, smoke_config.model).make_episode(0).units[0]
    unit = unit.to("cpu", dtype=torch.float16)
    output = model(unit, model.initial_state(1, "cpu"), unit.speech_codes)

    losses = compute_losses(output, unit)

    assert torch.isfinite(losses["speech"])


def test_stream_unit_casts_continuous_action_targets_with_model_dtype(
    smoke_config: ProjectConfig,
) -> None:
    model = StreamingLatentLoop(smoke_config.model).half()
    unit = SyntheticEpisodeDataset(smoke_config.data, smoke_config.model).make_episode(0).units[0]

    unit = unit.to("cpu", dtype=torch.float16)
    output = model(
        unit,
        model.initial_state(1, "cpu"),
        unit.speech_codes,
        speech_teacher_mode=unit.speech_mode,
        action_teacher_frame=unit.action,
        action_teacher_mask=unit.action_supervision_mask,
    )

    assert unit.action.coordinate_residual.dtype == torch.float16
    assert unit.action.scroll_delta.dtype == torch.float16
    assert unit.action.kind.dtype == torch.long
    assert torch.isfinite(output.action.kind_logits).all()
