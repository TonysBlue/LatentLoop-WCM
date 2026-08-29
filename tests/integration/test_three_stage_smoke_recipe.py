from __future__ import annotations

import json
import os
import subprocess
import threading
from pathlib import Path

import torch
from harness.transport.control import HarnessControlServer
from runtime.codec import CodecIdentity
from runtime.config import load_config
from training.checkpoint import file_sha256

from tests.support.physical_workers import CodecWorker, PhysicalBackend, RewardWorker


def test_public_smoke_recipe_runs_all_three_stages(tmp_path: Path) -> None:
    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir()
    experiment_root = tmp_path / "experiments"
    session_manifest = runtime_root / "sessions.jsonl"
    session_manifest.write_text(
        '{"session_id":"smoke-life","initial_snapshot_id":"snapshot","seed":17}\n',
        encoding="utf-8",
    )
    harness_socket = runtime_root / "harness.sock"
    codec_socket = runtime_root / "codec.sock"
    reward_socket = runtime_root / "reward.sock"
    timeline_root = runtime_root / "timeline"
    overrides = [
        f"runtime.data_root={tmp_path / 'datasets'}",
        f"training.rl.environment_socket={harness_socket}",
        f"training.rl.codec_socket={codec_socket}",
        f"training.rl.reward_socket={reward_socket}",
        f"training.rl.timeline_root={timeline_root}",
        f"training.rl.session_manifest={session_manifest}",
        "tracking.enabled=false",
        "tracking.mode=disabled",
        "model.rematerialization_segment_units=2",
    ]
    rl_config = load_config("configs/stages/smoke-rl.yaml", overrides)
    identity = CodecIdentity(
        "synthetic-codec-v1",
        "synthetic",
        "synthetic",
        sample_rate=24000,
        frame_samples=1920,
        codebooks=2,
        codebook_size=32,
    )
    codec = CodecWorker(codec_socket, identity)
    reward = RewardWorker(reward_socket, rl_config)
    backend = PhysicalBackend()
    server = HarnessControlServer(
        lambda: backend,
        str(harness_socket),
        expected_environment_id="test-physical",
        expected_environment_version="1",
    )
    threading.Thread(target=server.serve_forever, daemon=True).start()
    command = [
        "scripts/run-training.sh",
        "--recipe",
        "configs/recipes/smoke.yaml",
        "--run-id",
        "three-stage",
    ]
    for override in overrides:
        command.extend(("--set", override))
    environment = os.environ.copy()
    environment["LATENTLOOP_EXPERIMENT_ROOT"] = str(experiment_root)
    try:
        completed = subprocess.run(
            command,
            cwd=Path(__file__).parents[2],
            env=environment,
            text=True,
            capture_output=True,
            timeout=180,
            check=False,
        )
    finally:
        server.close()
        codec.close()
        reward.close()
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert backend.seen

    run_root = experiment_root / "smoke" / "three-stage"
    checkpoint_paths = {
        stage: run_root / stage / "checkpoints" / "step-00000001.pt"
        for stage in ("pretrain", "sft", "rl")
    }
    assert all(path.is_file() for path in checkpoint_paths.values())
    payloads = {
        stage: torch.load(path, map_location="cpu", weights_only=False)
        for stage, path in checkpoint_paths.items()
    }
    pretrain_sha256 = file_sha256(checkpoint_paths["pretrain"])
    sft_sha256 = file_sha256(checkpoint_paths["sft"])
    assert payloads["pretrain"]["metadata"]["stage"] == "pretrain"
    assert payloads["sft"]["metadata"]["stage"] == "sft"
    assert payloads["sft"]["metadata"]["parent_sha256"] == pretrain_sha256
    assert payloads["rl"]["metadata"]["stage"] == "rl"
    assert payloads["rl"]["metadata"]["algorithm"] == "online_recurrent_ppo"
    assert payloads["rl"]["metadata"]["parent_sha256"] == sft_sha256
    assert payloads["rl"]["metadata"]["reference_checkpoint_sha256"] == sft_sha256

    for stage in ("pretrain", "sft", "rl"):
        validation = json.loads(
            (run_root / stage / "reports" / "validation.json").read_text(encoding="utf-8")
        )
        assert validation["episodes"] > 0
    final_test = json.loads(
        (run_root / "rl" / "reports" / "test.json").read_text(encoding="utf-8")
    )
    assert final_test["episodes"] > 0
    report = json.loads(
        (run_root / "rl" / "recipe-report.json").read_text(encoding="utf-8")
    )
    assert [stage["stage"] for stage in report["stages"]] == ["pretrain", "sft", "rl"]
    assert report["stages"][-1]["train"]["metrics"]["rl/reward_mean"] > 0
    assert report["stages"][-1]["train"]["metrics"]["rl/finalization_lag_units"] >= 0
    assert report["stages"][-1]["train"]["metrics"]["runtime/elapsed_seconds"] > 0
    assert report["stages"][-1]["train"]["metrics"]["runtime/units_per_second"] > 0
    assert rl_config.model.rematerialization_segment_units == 2
    assert rl_config.training.rl.ppo_window_units > 2
