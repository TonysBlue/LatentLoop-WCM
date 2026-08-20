from __future__ import annotations

import torch
from data import SyntheticEpisodeDataset, write_episode_shards
from runtime.config import ProjectConfig
from training import train


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
