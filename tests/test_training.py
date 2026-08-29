from __future__ import annotations

import pytest
import torch
from data import SyntheticEpisodeDataset, write_episode_shards
from model import StreamingLatentLoop, compute_jepa_loss
from model.losses import compute_losses
from runtime.config import ProjectConfig
from training import train
from training.training import (
    _accumulate_metric_numerator,
    _rematerialized_segment,
    _require_finite_training_losses,
)


class _RecordingTracker:
    def __init__(self, *args, **kwargs) -> None:
        self.logs: list[tuple[dict[str, float], int]] = []

    def log(self, metrics: dict[str, float], step: int) -> None:
        self.logs.append((metrics, step))

    def record_checkpoint(self, path, digest: str, step: int) -> None:
        pass

    def finish(self) -> None:
        pass

    @property
    def effective_mode(self) -> None:
        return None

    @property
    def run_url(self) -> None:
        return None


def test_fp16_metric_accumulation_is_finite_and_detached() -> None:
    accumulated = None
    value = torch.tensor(55.0, dtype=torch.float16, requires_grad=True)
    for _ in range(375):
        accumulated = _accumulate_metric_numerator(accumulated, value, 8.0)

    assert accumulated is not None
    assert accumulated.dtype == torch.float32
    assert torch.isfinite(accumulated)
    assert not accumulated.requires_grad


def test_non_finite_training_loss_is_rejected() -> None:
    with pytest.raises(RuntimeError, match="loss_speech_codec"):
        _require_finite_training_losses(
            {"train/loss_total": 1.0, "train/loss_speech_codec": float("inf")}
        )


def test_recurrent_rematerialization_matches_direct_unroll(
    smoke_config: ProjectConfig,
) -> None:
    torch.manual_seed(11)
    direct = StreamingLatentLoop(smoke_config.model)
    rematerialized = StreamingLatentLoop(smoke_config.model)
    rematerialized.load_state_dict(direct.state_dict())
    units = SyntheticEpisodeDataset(smoke_config.data, smoke_config.model).make_episode(0).units[:4]

    direct_state = direct.initial_state(1, "cpu")
    direct_outputs = []
    for unit in units:
        output = direct(
            unit,
            direct_state,
            unit.speech_codes,
            speech_teacher_mode=unit.speech_mode,
            action_teacher_frame=unit.action,
            action_teacher_mask=unit.action_supervision_mask,
        )
        direct_outputs.append(output)
        direct_state = output.state
    direct_behavior = sum(
        compute_losses(output, unit)["total"]
        for output, unit in zip(direct_outputs, units, strict=True)
    )
    direct_jepa = compute_jepa_loss(
        torch.stack([output.jepa_prediction for output in direct_outputs[:-1]]),
        torch.stack([output.perceiver_slots for output in direct_outputs[:-1]]),
        torch.stack([output.perceiver_slots for output in direct_outputs[1:]]),
    )["total"]
    direct_loss = direct_behavior + direct_jepa
    direct_loss.backward()

    first_outputs, boundary_state = _rematerialized_segment(
        rematerialized,
        units[:2],
        rematerialized.initial_state(1, "cpu"),
        [unit.speech_codes for unit in units[:2]],
    )
    second_outputs, rematerialized_state = _rematerialized_segment(
        rematerialized,
        units[2:],
        boundary_state,
        [unit.speech_codes for unit in units[2:]],
    )
    rematerialized_outputs = first_outputs + second_outputs
    assert all(not hasattr(output, "state") for output in rematerialized_outputs)
    assert boundary_state.layer_kv[0].key.shape[2] == (
        len(units[:2]) * smoke_config.model.perceiver_slots
    )
    for direct_output, rematerialized_output in zip(
        direct_outputs, rematerialized_outputs, strict=True
    ):
        assert torch.equal(
            direct_output.speech_mode_logits, rematerialized_output.speech_mode_logits
        )
        assert torch.equal(
            direct_output.speech_codec_logits, rematerialized_output.speech_codec_logits
        )
        assert torch.equal(
            direct_output.action.kind_logits,
            rematerialized_output.action.kind_logits,
        )
        assert torch.equal(direct_output.jepa_prediction, rematerialized_output.jepa_prediction)
        assert torch.equal(direct_output.value, rematerialized_output.value)
    rematerialized_behavior = sum(
        compute_losses(output, unit)["total"]
        for output, unit in zip(rematerialized_outputs, units, strict=True)
    )
    rematerialized_jepa = compute_jepa_loss(
        torch.stack([output.jepa_prediction for output in rematerialized_outputs[:-1]]),
        torch.stack([output.perceiver_slots for output in rematerialized_outputs[:-1]]),
        torch.stack([output.perceiver_slots for output in rematerialized_outputs[1:]]),
    )["total"]
    rematerialized_loss = rematerialized_behavior + rematerialized_jepa
    rematerialized_loss.backward()

    assert torch.equal(direct_state.hidden, rematerialized_state.hidden)
    assert torch.equal(direct_state.semantic_memory, rematerialized_state.semantic_memory)
    assert all(
        torch.equal(left.matrix, right.matrix)
        for left, right in zip(
            direct_state.slow_memory, rematerialized_state.slow_memory, strict=True
        )
    )
    for (name, direct_parameter), rematerialized_parameter in zip(
        direct.named_parameters(), rematerialized.parameters(), strict=True
    ):
        if direct_parameter.grad is None or rematerialized_parameter.grad is None:
            assert direct_parameter.grad is rematerialized_parameter.grad, name
            continue
        assert torch.allclose(
            direct_parameter.grad, rematerialized_parameter.grad, atol=1e-6, rtol=1e-5
        ), name


def test_smoke_training_and_atomic_checkpoint(smoke_config: ProjectConfig) -> None:
    result = train(smoke_config)
    checkpoint_dir = smoke_config.runtime.root_path() / "checkpoints"
    checkpoints = sorted(checkpoint_dir.glob("step-*.pt"))
    assert result["train_state"]["update"] == smoke_config.training.max_updates
    assert "train/loss_total" in result["metrics"]
    assert "train/loss_jepa" in result["metrics"]
    assert result["metrics"]["runtime/elapsed_seconds"] > 0
    assert result["metrics"]["runtime/units_per_second"] > 0
    assert result["tracking"]["requested_mode"] == smoke_config.tracking.mode
    assert result["tracking"]["effective_mode"] is None
    assert result["tracking"]["run_url"] is None
    assert checkpoints
    assert not list(checkpoint_dir.glob("*.tmp.pt"))


def test_training_enables_deterministic_cuda_algorithms(
    smoke_config: ProjectConfig,
) -> None:
    train(smoke_config, stop_after_updates=1)
    if torch.cuda.is_available():
        assert torch.are_deterministic_algorithms_enabled()


def test_short_training_logs_final_metrics(smoke_config: ProjectConfig, monkeypatch) -> None:
    smoke_config.training.max_updates = 2
    smoke_config.training.log_every = 10
    tracker = _RecordingTracker()
    monkeypatch.setattr("training.training.Tracker", lambda *args, **kwargs: tracker)

    train(smoke_config)

    assert tracker.logs
    final_metrics, final_step = tracker.logs[-1]
    assert final_step == 2
    assert "train/loss_total" in final_metrics
    assert "speech/codec_accuracy_q0" in final_metrics
    assert "runtime/elapsed_seconds" in final_metrics


def test_interrupted_training_matches_continuous_training(
    smoke_config: ProjectConfig,
) -> None:
    continuous = train(smoke_config)
    continuous_model = continuous["model"]
    continuous_state = {
        name: value.detach().clone() for name, value in continuous_model.state_dict().items()
    }

    interrupted = train(smoke_config, stop_after_updates=2)
    assert interrupted["train_state"]["update"] == 2
    checkpoint = smoke_config.runtime.root_path() / "checkpoints" / "step-00000002.pt"
    resumed = train(smoke_config, resume=str(checkpoint))
    assert resumed["train_state"]["update"] == smoke_config.training.max_updates
    for name, value in resumed["model"].state_dict().items():
        assert value.equal(continuous_state[name]), name


def test_gradient_accumulation_tracks_all_consumed_units(
    smoke_config: ProjectConfig,
) -> None:
    smoke_config.training.gradient_accumulation_steps = 2
    smoke_config.training.max_updates = 2
    smoke_config.training.checkpoint_every = 1

    result = train(smoke_config)

    assert result["train_state"]["update"] == 2
    assert result["train_state"]["consumed_units"] == (
        2
        * smoke_config.training.gradient_accumulation_steps
        * smoke_config.training.memory_horizon_units
    )


def test_gradient_accumulation_resume_matches_continuous_training(
    smoke_config: ProjectConfig,
) -> None:
    smoke_config.training.gradient_accumulation_steps = 2
    smoke_config.training.max_updates = 2
    smoke_config.training.checkpoint_every = 1
    continuous = train(smoke_config)
    continuous_state = {
        name: value.detach().clone() for name, value in continuous["model"].state_dict().items()
    }

    interrupted = train(smoke_config, stop_after_updates=1)
    assert interrupted["train_state"]["consumed_units"] == 16
    checkpoint = smoke_config.runtime.root_path() / "checkpoints" / "step-00000001.pt"
    resumed = train(smoke_config, resume=str(checkpoint))

    assert resumed["train_state"]["consumed_units"] == 32
    for name, value in resumed["model"].state_dict().items():
        assert value.equal(continuous_state[name]), name


def test_webdataset_resume_across_shards_matches_continuous_training(
    smoke_config: ProjectConfig,
) -> None:
    smoke_config.data.episode_units = 4
    smoke_config.data.train_episodes = 2
    episodes = list(SyntheticEpisodeDataset(smoke_config.data, smoke_config.model))
    data_dir = smoke_config.runtime.data_path() / "generated"
    write_episode_shards(episodes, data_dir / "train-%06d.tar", max_size=1)
    smoke_config.data.source = "webdataset"
    smoke_config.data.shards = str(data_dir / "train-*.tar")
    smoke_config.data.manifest = str(data_dir / "train-manifest.jsonl")
    smoke_config.training.max_updates = 4
    smoke_config.training.checkpoint_every = 2

    continuous = train(smoke_config)
    continuous_state = {
        name: value.detach().clone() for name, value in continuous["model"].state_dict().items()
    }
    interrupted = train(smoke_config, stop_after_updates=2)
    cursor = interrupted["data_cursor"]
    assert (cursor.ordered_shard_index, cursor.sample_index_in_shard) == (1, 1)

    checkpoint = smoke_config.runtime.root_path() / "checkpoints" / "step-00000002.pt"
    resumed = train(smoke_config, resume=str(checkpoint))

    assert resumed["train_state"]["update"] == 4
    for name, value in resumed["model"].state_dict().items():
        assert value.equal(continuous_state[name]), name
