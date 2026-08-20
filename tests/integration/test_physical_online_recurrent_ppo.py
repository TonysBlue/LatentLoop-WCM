from __future__ import annotations

import json
import threading
from pathlib import Path

import torch
from contracts import ActuationSignal
from harness.transport.control import HarnessControlServer
from model import StreamingLatentLoop
from runtime.codec import CodecIdentity
from runtime.config import load_config
from training.checkpoint import CheckpointManager, CheckpointMetadata, DataCursor
from training.training import train_online_ppo

from tests.support.physical_workers import CodecWorker, PhysicalBackend, RewardWorker


def test_formal_physical_online_recurrent_ppo_runs_one_update(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    config = load_config(
        "configs/smoke.yaml",
        [
            f"runtime.data_root={tmp_path / 'datasets'}",
            f"runtime.experiment_root={tmp_path / 'experiment'}",
            "data.dataset=synthetic",
            "data.audio_sample_rate=24000",
            "data.unit_audio_samples=1920",
            "training.stage=rl",
            "training.max_updates=1",
            "training.rl.ppo_window_units=8",
            "training.rl.candidate_max_reference_kl=1000000000",
            "training.rl.candidate_max_eval_loss_ratio=1000000000",
            "training.rl.judge_revision=test-revision",
            "training.rl.rubric_sha256=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "tracking.enabled=false",
            "tracking.mode=disabled",
        ],
    )
    session_manifest = tmp_path / "sessions.jsonl"
    session_manifest.write_text(
        '{"session_id":"life","initial_snapshot_id":"snapshot","seed":17}\n',
        encoding="utf-8",
    )
    config.training.rl.session_manifest = str(session_manifest)
    config.training.rl.timeline_root = str(tmp_path / "timeline")
    config.training.rl.environment_socket = str(tmp_path / "harness.sock")
    config.training.rl.codec_socket = str(tmp_path / "codec.sock")
    config.training.rl.reward_socket = str(tmp_path / "reward.sock")
    config.training.rl.environment_id = "test-physical"
    config.training.rl.environment_version = "1"
    config.training.rl.environment_protocol_version = "realtime-v2"
    identity = CodecIdentity(
        "synthetic-codec-v1", "synthetic", "synthetic", sample_rate=24000,
        frame_samples=1920, codebooks=2, codebook_size=32
    )
    codec = CodecWorker(Path(config.training.rl.codec_socket), identity)
    reward = RewardWorker(
        Path(config.training.rl.reward_socket), config, delayed=True
    )
    backend = PhysicalBackend()
    server = HarnessControlServer(
        lambda: backend,
        config.training.rl.environment_socket,
        expected_environment_id="test-physical",
        expected_environment_version="1",
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    for _ in range(100):
        if Path(config.training.rl.environment_socket).exists():
            break
    init = StreamingLatentLoop(config.model)
    init_path = tmp_path / "sft.pt"
    manager = CheckpointManager(tmp_path / "init")
    sft_config = load_config(
        "configs/smoke.yaml",
        [
            "data.audio_sample_rate=24000",
            "data.unit_audio_samples=1920",
            "training.stage=sft",
            "training.jepa_loss_weight=0.5",
        ],
    )
    manager.save(
        "sft",
        model=init,
        optimizer=torch.optim.AdamW(init.parameters(), lr=1e-3),
        scheduler=None,
        scaler=None,
        recurrent_state=None,
        train_state={},
        data_cursor=DataCursor(),
        metadata=CheckpointMetadata(
            "synthetic", identity.codec_id, identity.weight_sha256, "test",
            codec_revision=identity.revision, stage="sft"
        ),
        config=sft_config.as_dict(),
    )
    init_path = tmp_path / "init" / "sft.pt"
    try:
        result = train_online_ppo(
            config,
            init_from=str(init_path),
            model=StreamingLatentLoop(config.model),
            stop_after_updates=1,
        )
    finally:
        server.close()
        codec.close()
        reward.close()
    assert result["train_state"]["update"] == 1
    assert backend.seen
    assert all(isinstance(output, ActuationSignal) for output in backend.seen)
    assert all(not hasattr(output, "action_tokens") for output in backend.seen)
    assert result["train_state"]["consumed_units"] > config.training.rl.ppo_window_units
    assert result["train_state"]["dropped_windows"] >= 1
    assert result["metrics"]["rl/reward_mean"] > 0
    assert torch.isfinite(torch.tensor(result["metrics"]["train/loss_jepa_on_policy"]))
    assert torch.isfinite(torch.tensor(result["metrics"]["train/loss_jepa_replay"]))
    assert result["metrics"]["rl/finalization_lag_units"] >= 0
    assert result["metrics"]["runtime/elapsed_seconds"] > 0
    assert result["metrics"]["runtime/units_per_second"] > 0
    assert result["metrics"]["runtime/peak_memory_allocated_bytes"] >= 0
    trainable_windows = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in (tmp_path / "experiment" / "rollout-windows").glob("window-*.json")
    ]
    assert len(trainable_windows) == 1
    window = trainable_windows[0]
    assert window["lookahead_unit_index"] == window["end_unit"] + 1
    assert len(window["lookahead_payload_sha256"]) == 64


def test_online_recurrent_ppo_resumes_the_same_lifetime_session(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    config = load_config(
        "configs/smoke.yaml",
        [
            f"runtime.data_root={tmp_path / 'datasets'}",
            f"runtime.experiment_root={tmp_path / 'experiment'}",
            "data.dataset=synthetic",
            "data.audio_sample_rate=24000",
            "data.unit_audio_samples=1920",
            "training.stage=rl",
            "training.max_updates=2",
            "training.rl.ppo_window_units=8",
            "training.rl.candidate_max_reference_kl=1000000000",
            "training.rl.candidate_max_eval_loss_ratio=1000000000",
            "training.rl.judge_revision=test-revision",
            "training.rl.rubric_sha256=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "tracking.enabled=false",
            "tracking.mode=disabled",
        ],
    )
    session_manifest = tmp_path / "sessions.jsonl"
    session_manifest.write_text(
        '{"session_id":"life","initial_snapshot_id":"snapshot","seed":17}\n',
        encoding="utf-8",
    )
    config.training.rl.session_manifest = str(session_manifest)
    config.training.rl.timeline_root = str(tmp_path / "timeline")
    config.training.rl.environment_socket = str(tmp_path / "harness.sock")
    config.training.rl.codec_socket = str(tmp_path / "codec.sock")
    config.training.rl.reward_socket = str(tmp_path / "reward.sock")
    config.training.rl.environment_id = "test-physical"
    config.training.rl.environment_version = "1"
    config.training.rl.environment_protocol_version = "realtime-v2"
    identity = CodecIdentity(
        "synthetic-codec-v1",
        "synthetic",
        "synthetic",
        sample_rate=24000,
        frame_samples=1920,
        codebooks=2,
        codebook_size=32,
    )
    codec = CodecWorker(Path(config.training.rl.codec_socket), identity)
    reward = RewardWorker(Path(config.training.rl.reward_socket), config)
    backend = PhysicalBackend()
    server = HarnessControlServer(
        lambda: backend,
        config.training.rl.environment_socket,
        expected_environment_id="test-physical",
        expected_environment_version="1",
    )
    threading.Thread(target=server.serve_forever, daemon=True).start()
    for _ in range(100):
        if Path(config.training.rl.environment_socket).exists():
            break
    init = StreamingLatentLoop(config.model)
    init_manager = CheckpointManager(tmp_path / "init")
    sft_config = load_config(
        "configs/smoke.yaml",
        [
            "data.audio_sample_rate=24000",
            "data.unit_audio_samples=1920",
            "training.stage=sft",
            "training.jepa_loss_weight=0.5",
        ],
    )
    init_manager.save(
        "sft",
        model=init,
        optimizer=torch.optim.AdamW(init.parameters(), lr=1e-3),
        scheduler=None,
        scaler=None,
        recurrent_state=None,
        train_state={},
        data_cursor=DataCursor(),
        metadata=CheckpointMetadata(
            "synthetic",
            identity.codec_id,
            identity.weight_sha256,
            "test",
            codec_revision=identity.revision,
            stage="sft",
        ),
        config=sft_config.as_dict(),
    )
    sft_path = tmp_path / "init" / "sft.pt"
    try:
        first = train_online_ppo(
            config,
            init_from=str(sft_path),
            model=StreamingLatentLoop(config.model),
            stop_after_updates=1,
        )
        checkpoint = tmp_path / "experiment" / "checkpoints" / "step-00000001.pt"
        first_units = first["train_state"]["consumed_units"]
        resumed = train_online_ppo(
            config,
            resume=str(checkpoint),
            model=StreamingLatentLoop(config.model),
            stop_after_updates=2,
        )
    finally:
        server.close()
        codec.close()
        reward.close()
    assert resumed["train_state"]["update"] == 2
    assert resumed["train_state"]["consumed_units"] > first_units
    assert backend.start_count == 1
