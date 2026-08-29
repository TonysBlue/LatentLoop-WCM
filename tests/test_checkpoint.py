from __future__ import annotations

from pathlib import Path

import pytest
import torch
from data import SyntheticEpisodeDataset
from model import StreamingLatentLoop
from model.losses import compute_losses
from runtime.config import ProjectConfig
from training.checkpoint import (
    CheckpointManager,
    CheckpointMetadata,
    DataCursor,
    file_sha256,
)
from training.training import initialize_compatible_weights, initialize_exact_weights


def test_checkpoint_restores_full_recurrent_step(
    tmp_path: Path, smoke_config: ProjectConfig
) -> None:
    model = StreamingLatentLoop(smoke_config.model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    episode = SyntheticEpisodeDataset(smoke_config.data, smoke_config.model).make_episode(0)
    first = model(episode.units[0], model.initial_state(1, "cpu"))
    loss = compute_losses(first, episode.units[0])["total"]
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    manager = CheckpointManager(tmp_path)
    path, digest = manager.save(
        "state",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
        recurrent_state=first.state.detach(),
        train_state={"update": 1, "episode": 1, "unit": 1},
        data_cursor=DataCursor(epoch=0, episode=0, unit=1),
        metadata=CheckpointMetadata(
            data_identity="test-data",
            codec_id=smoke_config.data.codec_id,
            codec_weight_hash=smoke_config.data.codec_weight_hash,
            git_commit="test",
            codec_revision=smoke_config.data.codec_revision,
        ),
        config=smoke_config.as_dict(),
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert "format_version" not in payload
    assert "world_state_update_version" not in payload["metadata"]
    assert "semantic_memory" in payload["recurrent_state"]
    assert "slow_memory" in payload["recurrent_state"]
    assert "delta_time_encoder_version" not in payload["metadata"]
    expected = model(episode.units[1], first.state.detach()).speech_codec_logits.detach()

    restored_model = StreamingLatentLoop(smoke_config.model)
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=1e-3)
    restored_scheduler = torch.optim.lr_scheduler.LambdaLR(restored_optimizer, lambda _: 1.0)
    train_state, cursor, recurrent, metadata = manager.load(
        path,
        model=restored_model,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
        scaler=None,
        device=torch.device("cpu"),
        config=smoke_config.as_dict(),
        expected_metadata=CheckpointMetadata(
            data_identity="test-data",
            codec_id=smoke_config.data.codec_id,
            codec_weight_hash=smoke_config.data.codec_weight_hash,
            git_commit="different-commit-is-allowed",
            codec_revision=smoke_config.data.codec_revision,
        ),
    )
    assert recurrent is not None
    actual = restored_model(episode.units[1], recurrent).speech_codec_logits.detach()
    assert train_state["update"] == 1
    assert cursor.unit == 1
    assert metadata.codec_id == smoke_config.data.codec_id
    assert digest == file_sha256(path)
    assert torch.equal(actual, expected)
    assert (tmp_path / "manifest.json").exists()


def test_checkpoint_restores_reference_recurrent_state(
    tmp_path: Path, smoke_config: ProjectConfig
) -> None:
    model = StreamingLatentLoop(smoke_config.model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    state = model.initial_state(1, "cpu")
    state.unit_index.fill_(9)
    manager = CheckpointManager(tmp_path)
    path, _ = manager.save(
        "reference-state",
        model=model,
        optimizer=optimizer,
        scheduler=None,
        scaler=None,
        recurrent_state=state,
        reference_recurrent_state=state,
        train_state={"update": 0},
        data_cursor=DataCursor(),
        metadata=CheckpointMetadata("data", "codec", "hash", "test"),
        config=smoke_config.as_dict(),
    )
    from training.checkpoint import load_reference_recurrent_state

    restored = load_reference_recurrent_state(path, torch.device("cpu"))
    assert restored is not None
    assert int(restored.unit_index.item()) == 9


def test_checkpoint_rejects_codec_mismatch(tmp_path: Path, smoke_config: ProjectConfig) -> None:
    model = StreamingLatentLoop(smoke_config.model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    manager = CheckpointManager(tmp_path)
    path, _ = manager.save(
        "codec",
        model=model,
        optimizer=optimizer,
        scheduler=None,
        scaler=None,
        recurrent_state=None,
        train_state={"update": 0},
        data_cursor=DataCursor(),
        metadata=CheckpointMetadata("data", "codec-a", "hash-a", "test", "revision"),
        config=smoke_config.as_dict(),
    )
    try:
        manager.load(
            path,
            model=model,
            optimizer=optimizer,
            scheduler=None,
            scaler=None,
            device=torch.device("cpu"),
            config=smoke_config.as_dict(),
            expected_metadata=CheckpointMetadata("data", "codec-b", "hash-a", "test", "revision"),
        )
    except ValueError as error:
        assert "codec_id" in str(error)
    else:
        raise AssertionError("codec mismatch must be rejected")


def test_checkpoint_rejects_rl_algorithm_mismatch(
    tmp_path: Path, smoke_config: ProjectConfig
) -> None:
    model = StreamingLatentLoop(smoke_config.model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    manager = CheckpointManager(tmp_path)
    metadata = CheckpointMetadata(
        "data", "codec", "hash", "test", stage="rl", algorithm="online_recurrent_ppo"
    )
    path, _ = manager.save(
        "rl",
        model=model,
        optimizer=optimizer,
        scheduler=None,
        scaler=None,
        recurrent_state=None,
        train_state={"update": 0},
        data_cursor=DataCursor(),
        metadata=metadata,
        config=smoke_config.as_dict(),
    )

    with pytest.raises(ValueError, match="algorithm"):
        manager.load(
            path,
            model=model,
            optimizer=optimizer,
            scheduler=None,
            scaler=None,
            device=torch.device("cpu"),
            config=smoke_config.as_dict(),
            expected_metadata=CheckpointMetadata(
                "data", "codec", "hash", "test", stage="rl", algorithm="other"
            ),
        )


def test_checkpoint_rejects_incomplete_current_payload(
    tmp_path: Path, smoke_config: ProjectConfig
) -> None:
    model = StreamingLatentLoop(smoke_config.model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    manager = CheckpointManager(tmp_path)
    path, _ = manager.save(
        "incomplete",
        model=model,
        optimizer=optimizer,
        scheduler=None,
        scaler=None,
        recurrent_state=None,
        train_state={"update": 0},
        data_cursor=DataCursor(),
        metadata=CheckpointMetadata("data", "codec", "hash", "test"),
        config=smoke_config.as_dict(),
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    del payload["metadata"]
    incomplete_path = tmp_path / "incomplete.pt"
    torch.save(payload, incomplete_path)

    with pytest.raises(ValueError, match="checkpoint is incomplete"):
        manager.load(
            incomplete_path,
            model=model,
            optimizer=optimizer,
            scheduler=None,
            scaler=None,
            device=torch.device("cpu"),
            config=smoke_config.as_dict(),
            expected_metadata=CheckpointMetadata("data", "codec", "hash", "test"),
        )


def test_checkpoint_rejects_removed_objective_metadata(
    tmp_path: Path, smoke_config: ProjectConfig
) -> None:
    model = StreamingLatentLoop(smoke_config.model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    manager = CheckpointManager(tmp_path)
    path, _ = manager.save(
        "obsolete-objective",
        model=model,
        optimizer=optimizer,
        scheduler=None,
        scaler=None,
        recurrent_state=None,
        train_state={"update": 0},
        data_cursor=DataCursor(),
        metadata=CheckpointMetadata("data", "codec", "hash", "test"),
        config=smoke_config.as_dict(),
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["metadata"]["objective"] = "supervised"
    obsolete = tmp_path / "obsolete.pt"
    torch.save(payload, obsolete)

    with pytest.raises(ValueError, match="objective is removed"):
        manager.load(
            obsolete,
            model=model,
            optimizer=optimizer,
            scheduler=None,
            scaler=None,
            device=torch.device("cpu"),
            config=smoke_config.as_dict(),
            expected_metadata=CheckpointMetadata("data", "codec", "hash", "test"),
        )


def test_ppo_exact_initialization_rejects_missing_value_head(
    tmp_path: Path, smoke_config: ProjectConfig
) -> None:
    model = StreamingLatentLoop(smoke_config.model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    path, _ = CheckpointManager(tmp_path).save(
        "sft",
        model=model,
        optimizer=optimizer,
        scheduler=None,
        scaler=None,
        recurrent_state=None,
        train_state={},
        data_cursor=DataCursor(),
        metadata=CheckpointMetadata("data", "codec", "hash", "test", stage="sft"),
        config=smoke_config.as_dict(),
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    del payload["model"]["value_head.network.3.bias"]
    incomplete = tmp_path / "incomplete-sft.pt"
    torch.save(payload, incomplete)

    with pytest.raises(ValueError, match="complete current model"):
        initialize_exact_weights(StreamingLatentLoop(smoke_config.model), incomplete)


def test_weight_initialization_rejects_obsolete_architecture(
    tmp_path: Path, smoke_config: ProjectConfig
) -> None:
    model = StreamingLatentLoop(smoke_config.model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    path, _ = CheckpointManager(tmp_path).save(
        "obsolete-architecture",
        model=model,
        optimizer=optimizer,
        scheduler=None,
        scaler=None,
        recurrent_state=None,
        train_state={},
        data_cursor=DataCursor(),
        metadata=CheckpointMetadata("data", "codec", "hash", "test", stage="sft"),
        config=smoke_config.as_dict(),
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["metadata"]["architecture_id"] = "latentloop-legacy-v0"
    obsolete = tmp_path / "obsolete.pt"
    torch.save(payload, obsolete)

    with pytest.raises(ValueError, match="architecture is incompatible"):
        initialize_compatible_weights(model, obsolete)
    with pytest.raises(ValueError, match="architecture is incompatible"):
        initialize_exact_weights(model, obsolete)


def test_checkpoint_manifest_rejects_obsolete_format_field(
    tmp_path: Path, smoke_config: ProjectConfig
) -> None:
    model = StreamingLatentLoop(smoke_config.model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    manager = CheckpointManager(tmp_path)
    manager.save(
        "current",
        model=model,
        optimizer=optimizer,
        scheduler=None,
        scaler=None,
        recurrent_state=None,
        train_state={"update": 0},
        data_cursor=DataCursor(),
        metadata=CheckpointMetadata("data", "codec", "hash", "test"),
        config=smoke_config.as_dict(),
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        manifest_path.read_text(encoding="utf-8").replace(
            '{\n  "checkpoints":', '{\n  "format_version": 1,\n  "checkpoints":'
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="obsolete format_version"):
        manager.save(
            "next",
            model=model,
            optimizer=optimizer,
            scheduler=None,
            scaler=None,
            recurrent_state=None,
            train_state={"update": 1},
            data_cursor=DataCursor(),
            metadata=CheckpointMetadata("data", "codec", "hash", "test"),
            config=smoke_config.as_dict(),
        )
