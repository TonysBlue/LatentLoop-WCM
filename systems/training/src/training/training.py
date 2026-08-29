from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import subprocess
import time
from collections.abc import Iterable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from accelerate import Accelerator
from contracts import ACTION_SCHEMA_ID, ObservationSignal
from contracts.protocol import observation_to_payload
from data import EpisodeShardReader, SyntheticEpisodeDataset
from data.curation.readiness import check_readiness
from model import StreamingLatentLoop, action_frame_log_prob, compute_jepa_loss, compute_losses
from model.types import (
    ActionFrame,
    Episode,
    RecurrentState,
    SpeechSamplingConfig,
    StreamUnit,
)
from runtime.config import ProjectConfig
from torch.utils import _pytree
from torch.utils.checkpoint import checkpoint

from training.checkpoint import (
    CheckpointManager,
    CheckpointMetadata,
    DataCursor,
    config_hash,
    file_sha256,
    inspect_checkpoint,
    load_reference_recurrent_state,
)
from training.physical_rollout import PhysicalRolloutClient, observation_to_stream_unit
from training.ppo import (
    generalized_advantage_estimate,
    recurrent_ppo_loss,
    sampled_reference_kl,
    time_discounts,
)
from training.tracking import Tracker


def _loss_denominators(units: Sequence[StreamUnit]) -> dict[str, float]:
    return {
        "total": float(len(units)),
        "speech": float(max(sum(int(u.speech_mode_mask.sum()) for u in units), 1)),
        "speech_mode": float(max(sum(int(u.speech_mode_mask.sum()) for u in units), 1)),
        "speech_codec": float(
            max(
                sum(
                    int((u.speech_codec_mask & u.speech_mode.eq(1)[:, None]).sum())
                    * u.speech_codes.shape[-1]
                    for u in units
                ),
                1,
            )
        ),
        "action": float(max(sum(int(u.action_supervision_mask.sum()) for u in units), 1)),
    }


def _sequence_jepa_loss(
    model: StreamingLatentLoop,
    outputs: Sequence[Any],
    lookahead: StreamUnit | None,
) -> tuple[dict[str, torch.Tensor], int]:
    """Pair consecutive outputs and optionally encode one target-only successor."""
    if not outputs:
        raise ValueError("JEPA sequence requires at least one source output")
    predicted = [output.jepa_prediction for output in outputs[:-1]]
    sources = [output.perceiver_slots for output in outputs[:-1]]
    targets = [output.perceiver_slots for output in outputs[1:]]
    if lookahead is not None:
        target, _ = model.encode_observation(
            lookahead,
            outputs[-1].state.audio_cache.clone(),
        )
        predicted.append(outputs[-1].jepa_prediction)
        sources.append(outputs[-1].perceiver_slots)
        targets.append(target)
    if not predicted:
        zero = outputs[0].jepa_prediction.sum() * 0.0
        return {"total": zero, "prediction": zero, "variance": zero}, 0
    losses = compute_jepa_loss(
        torch.stack(predicted),
        torch.stack(sources),
        torch.stack(targets),
    )
    pairs = sum(value.shape[0] for value in predicted)
    return losses, pairs


def _rematerialized_segment(
    model: StreamingLatentLoop,
    units: Sequence[StreamUnit],
    state: RecurrentState,
    teacher_codes: Sequence[torch.Tensor | None],
    *,
    action_masks: Sequence[torch.Tensor] | None = None,
    action_frames: Sequence[ActionFrame] | None = None,
) -> tuple[tuple[Any, ...], RecurrentState]:
    """Rematerialize each recurrent transition without cutting the segment graph."""
    if not units:
        raise ValueError("rematerialization segment must contain at least one unit")
    masks = action_masks or [unit.action_supervision_mask for unit in units]
    frames = action_frames or [unit.action for unit in units]
    current = state
    outputs: list[Any] = []
    for unit, codes, action_mask, action_frame in zip(
        units, teacher_codes, masks, frames, strict=True
    ):
        state_flat, state_spec = _pytree.tree_flatten(current)
        output_spec: list[Any] = []

        def make_run(
            step_unit: StreamUnit,
            step_codes: torch.Tensor | None,
            step_action_mask: torch.Tensor,
            step_action_frame: ActionFrame,
            step_state_spec: Any,
            step_output_spec: list[Any],
        ) -> Any:
            def run(*initial_tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
                initial = _pytree.tree_unflatten(list(initial_tensors), step_state_spec)
                output = model(
                    step_unit,
                    initial,
                    step_codes,
                    speech_teacher_mode=step_unit.speech_mode,
                    action_teacher_frame=step_action_frame,
                    action_teacher_mask=step_action_mask,
                )
                flat, spec = _pytree.tree_flatten(output)
                if not step_output_spec:
                    unique: list[torch.Tensor] = []
                    positions: dict[int, int] = {}
                    aliases: list[int] = []
                    for tensor in flat:
                        identity = id(tensor)
                        if identity not in positions:
                            positions[identity] = len(unique)
                            unique.append(tensor)
                        aliases.append(positions[identity])
                    step_output_spec.extend((spec, aliases))
                    return tuple(unique)
                aliases = step_output_spec[1]
                return tuple(flat[index] for index in _first_alias_positions(aliases))

            return run

        flat_output = checkpoint(
            make_run(unit, codes, action_mask, action_frame, state_spec, output_spec),
            *state_flat,
            use_reentrant=False,
            preserve_rng_state=True,
        )
        output = _pytree.tree_unflatten(
            [flat_output[index] for index in output_spec[1]], output_spec[0]
        )
        outputs.append(output)
        current = output.state
    return tuple(outputs), current


def _first_alias_positions(aliases: Sequence[int]) -> tuple[int, ...]:
    """Map unique tensor indices back to their first position in a flattened tree."""
    first: dict[int, int] = {}
    for position, unique_index in enumerate(aliases):
        first.setdefault(unique_index, position)
    return tuple(first[index] for index in range(len(first)))


def _aggregate_update_metrics(
    records: list[dict[str, Any]], config: ProjectConfig
) -> dict[str, float]:
    """Aggregate losses and codec accuracy over chunks in one optimizer update."""
    names = (
        "total",
        "speech",
        "speech_mode",
        "speech_codec",
        "action",
        "jepa",
        "jepa_prediction",
        "jepa_variance",
    )
    metrics: dict[str, float] = {}
    for name in names:
        numerator = sum(item["losses"][name] for item in records)
        denominator = sum(item["denoms"][name] for item in records)
        metrics[f"train/loss_{name}"] = numerator / denominator if denominator else float("nan")
    speech_valid = sum(item["speech_valid"] for item in records)
    mode_correct = sum(item["mode_correct"] for item in records)
    action_correct = sum(item["action_correct"] for item in records)
    for codebook, correct in enumerate(
        sum(
            (item["speech_correct"] for item in records),
            torch.zeros(config.model.speech_codebooks, dtype=torch.long),
        )
    ):
        metrics[f"speech/codec_accuracy_q{codebook}"] = (
            float(correct) / speech_valid if speech_valid else float("nan")
        )
    target_units = sum(item["target_units"] for item in records)
    metrics.update(
        {
            "speech/mode_accuracy": float(
                mode_correct / max(sum(item["mode_valid"] for item in records), 1)
            ),
            "action/kind_accuracy": float(
                action_correct / max(sum(item["action_valid"] for item in records), 1)
            ),
            "speech/valid_frames": float(speech_valid),
            "speech/active_unit_fraction": float(speech_valid / max(target_units, 1)),
            "speech/codec_accuracy_mean": (
                float(
                    sum(
                        metrics[f"speech/codec_accuracy_q{i}"]
                        for i in range(config.model.speech_codebooks)
                    )
                )
                / config.model.speech_codebooks
                if speech_valid
                else float("nan")
            ),
            "data/update_units": float(target_units),
            "data/update_chunks": float(len(records)),
        }
    )
    return metrics


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)


def build_episodes(config: ProjectConfig) -> Iterable[Episode]:
    if config.data.source == "synthetic":
        return SyntheticEpisodeDataset(config.data, config.model)
    assert config.data.shards is not None
    return EpisodeShardReader(config.data.shards, config.data, config.model)


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "nogit"


def _checkpoint_metadata(
    config: ProjectConfig,
    parent_sha256: str | None = None,
    reference_checkpoint_sha256: str | None = None,
    *,
    lineage_id: str | None = None,
    policy_version: str | None = None,
    observation_chain_sha256: str | None = None,
    policy_sample_chain_sha256: str | None = None,
    sft_checkpoint_path: str | None = None,
) -> CheckpointMetadata:
    return CheckpointMetadata(
        data_identity=_data_identity(config),
        codec_id=config.data.codec_id,
        codec_weight_hash=config.data.codec_weight_hash,
        git_commit=_git_commit(),
        codec_revision=config.data.codec_revision,
        parent_sha256=parent_sha256,
        reference_checkpoint_sha256=reference_checkpoint_sha256,
        stage=config.training.stage,
        algorithm=(
            config.training.rl.algorithm if config.training.stage == "rl" else None
        ),
        action_schema_id=ACTION_SCHEMA_ID,
        environment_id=config.training.rl.environment_id or None,
        session_manifest_sha256=(
            file_sha256(Path(config.training.rl.session_manifest).expanduser())
            if config.training.rl.session_manifest
            and Path(config.training.rl.session_manifest).expanduser().is_file()
            else None
        ),
        reward_spec_id=config.training.rl.reward_spec_id if config.training.stage == "rl" else None,
        judge_model_id=(
            config.training.rl.judge_model_id if config.training.stage == "rl" else None
        ),
        judge_revision=(
            config.training.rl.judge_revision if config.training.stage == "rl" else None
        ),
        rubric_sha256=(config.training.rl.rubric_sha256 if config.training.stage == "rl" else None),
        lineage_id=lineage_id,
        policy_version=policy_version,
        observation_chain_sha256=observation_chain_sha256,
        policy_sample_chain_sha256=policy_sample_chain_sha256,
        sft_checkpoint_path=sft_checkpoint_path,
        sft_replay_manifest_sha256=(
            file_sha256(Path(config.training.rl.sft_replay_manifest).expanduser())
            if config.training.stage == "rl"
            and config.training.rl.sft_replay_manifest
            and Path(config.training.rl.sft_replay_manifest).expanduser().is_file()
            else None
        ),
        sft_preservation_manifest_sha256=(
            file_sha256(
                Path(config.training.rl.sft_preservation_manifest).expanduser()
            )
            if config.training.stage == "rl"
            and config.training.rl.sft_preservation_manifest
            and Path(config.training.rl.sft_preservation_manifest).expanduser().is_file()
            else None
        ),
    )


def _data_identity(config: ProjectConfig) -> str:
    if config.data.manifest:
        manifest_path = Path(config.data.manifest).expanduser()
        if not manifest_path.is_absolute():
            manifest_path = config.runtime.data_path() / manifest_path
        return file_sha256(manifest_path)
    return config_hash(config.as_dict()["data"])


def train(
    config: ProjectConfig,
    *,
    resume: str | None = None,
    init_from: str | None = None,
    model: StreamingLatentLoop | None = None,
    stop_after_updates: int | None = None,
) -> dict[str, Any]:
    if resume and init_from:
        raise ValueError("resume and init_from are mutually exclusive")
    if config.data.dataset == "canary" and not torch.cuda.is_available():
        raise RuntimeError(
            "formal real-data training requires a CUDA GPU; "
            "torch.cuda.is_available() is false"
        )
    if config.training.stage == "rl":
        return train_online_ppo(
            config,
            resume=resume,
            init_from=init_from,
            model=model,
            stop_after_updates=stop_after_updates,
        )
    if config.data.dataset != "synthetic":
        require_initial = (
            config.training.backbone_train_mode in {"frozen", "selective"}
            and config.data.dataset != "direct-speech-overfit"
        )
        check_readiness(
            config.runtime.data_path(),
            config=config,
            require_checkpoint=init_from if require_initial else None,
        )
        if require_initial and not (init_from or resume):
            raise ValueError(
                "frozen/selective real-data training requires --init-from with a "
                "compatible checkpoint"
            )
    seed_everything(config.data.seed)
    accelerator = Accelerator(
        mixed_precision=config.training.mixed_precision,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
    )
    model = model or StreamingLatentLoop(config.model)
    if init_from:
        initialize_exact_weights(model, init_from)
    configure_trainable_parameters(model, config)
    head_parameters = []
    backbone_parameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith(("speech_head.", "action_head.")):
            head_parameters.append(parameter)
        else:
            backbone_parameters.append(parameter)
    parameter_groups: list[dict[str, Any]] = []
    if head_parameters:
        parameter_groups.append(
            {"params": head_parameters, "lr": config.training.head_learning_rate}
        )
    if backbone_parameters:
        parameter_groups.append(
            {"params": backbone_parameters, "lr": config.training.backbone_learning_rate}
        )
    optimizer = torch.optim.AdamW(
        parameter_groups,
        weight_decay=config.training.weight_decay,
    )
    warmup_updates = int(config.training.max_updates * config.training.warmup_ratio)

    def learning_rate_scale(step: int) -> float:
        if warmup_updates and step < warmup_updates:
            return (step + 1) / warmup_updates
        progress = (step - warmup_updates) / max(config.training.max_updates - warmup_updates, 1)
        cosine = 0.5 * (1 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
        return (
            config.training.min_learning_rate_ratio
            + (1.0 - config.training.min_learning_rate_ratio) * cosine
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_scale)
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    root = config.runtime.root_path()
    checkpoint_manager = CheckpointManager(root / "checkpoints")
    unwrapped_model = accelerator.unwrap_model(model)
    tracker = Tracker(
        config,
        model_name=f"{unwrapped_model.parameter_count()}p",
        parameter_count=unwrapped_model.parameter_count(),
        data_identity=_data_identity(config),
        parent_checkpoint_sha256=(
            file_sha256(resume or init_from) if (resume or init_from) else None
        ),
    )
    train_state: dict[str, Any] = {"update": 0, "epoch": 0, "episode": 0, "unit": 0}
    cursor = DataCursor()
    recurrent: RecurrentState | None = None
    parent_sha256: str | None = file_sha256(init_from) if init_from else None
    if resume:
        parent_sha256 = file_sha256(resume)
        train_state, cursor, recurrent, _ = checkpoint_manager.load(
            resume,
            model=accelerator.unwrap_model(model),
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=accelerator.scaler,
            device=accelerator.device,
            config=config.as_dict(),
            expected_metadata=_checkpoint_metadata(config),
        )

    optimizer.zero_grad(set_to_none=True)
    if accelerator.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(accelerator.device)
    training_started = time.perf_counter()
    last_metrics: dict[str, float] = {}
    last_logged_update = 0
    pending_update_metrics: list[dict[str, Any]] = []
    target_updates = min(
        config.training.max_updates,
        stop_after_updates if stop_after_updates is not None else config.training.max_updates,
    )
    last_checkpoint_update = 0
    tracking: dict[str, str | None] = {}
    try:
        epoch = cursor.epoch
        while train_state["update"] < target_updates:
            yielded = False
            for episode_index, episode in enumerate(build_episodes(config)):
                yielded = True
                episode_location = (
                    episode.ordered_shard_index,
                    episode.sample_index_in_shard,
                )
                cursor_location = (
                    cursor.ordered_shard_index,
                    cursor.sample_index_in_shard,
                )
                if epoch == cursor.epoch and episode_location < cursor_location:
                    continue
                is_resumed_episode = epoch == cursor.epoch and episode_location == cursor_location
                unit_start = cursor.unit if is_resumed_episode else 0
                if not is_resumed_episode or unit_start == 0:
                    recurrent = None
                # A memory horizon is one autograd segment: detaching at a
                # shorter sampling window would sever future action/speech
                # The full memory horizon preserves supervision through C_t and SlowMemory.
                chunk_size = config.training.memory_horizon_units
                for chunk_start in range(unit_start, len(episode.units), chunk_size):
                    if train_state["update"] >= target_updates:
                        break
                    chunk = episode.units[chunk_start : chunk_start + chunk_size]
                    moved = [unit.to(accelerator.device) for unit in chunk]
                    if recurrent is None:
                        recurrent = accelerator.unwrap_model(model).initial_state(
                            moved[0].batch_size, accelerator.device
                        )
                    with accelerator.accumulate(model):
                        chunk_losses: dict[str, torch.Tensor] = {}
                        chunk_numerators: dict[str, torch.Tensor] = {}
                        chunk_denoms: dict[str, float] = {
                            name: 0.0
                            for name in (
                                "total",
                                "speech",
                                "speech_mode",
                                "speech_codec",
                                "action",
                                "jepa",
                                "jepa_prediction",
                                "jepa_variance",
                            )
                        }
                        speech_correct = torch.zeros(
                            config.model.speech_codebooks,
                            device=accelerator.device,
                            dtype=torch.long,
                        )
                        speech_valid = torch.zeros((), device=accelerator.device, dtype=torch.long)
                        output = None
                        outputs: list[Any] = []
                        segment_units = config.model.rematerialization_segment_units
                        sampling_probability = scheduled_sampling_probability(
                            train_state["update"], config
                        )
                        teacher_codes = [
                            unit.speech_codes
                            if sampling_probability <= 0
                            or random.random() >= sampling_probability
                            else None
                            for unit in moved
                        ]
                        for segment_start in range(0, len(moved), segment_units):
                            segment = moved[segment_start : segment_start + segment_units]
                            segment_teachers = teacher_codes[
                                segment_start : segment_start + segment_units
                            ]
                            segment_outputs, recurrent = _rematerialized_segment(
                                model, segment, recurrent, segment_teachers
                            )
                            outputs.extend(segment_outputs)
                        for unit, output in zip(moved, outputs, strict=True):
                            unit_losses = compute_losses(
                                output,
                                unit,
                                config.training.speech_loss_weight,
                                config.training.action_loss_weight,
                            )
                            for name, value in unit_losses.items():
                                accumulated = chunk_losses.get(name, torch.zeros_like(value))
                                chunk_losses[name] = accumulated + value
                                unit_denoms = _loss_denominators([unit])
                                chunk_numerators[name] = (
                                    chunk_numerators.get(name, torch.zeros_like(value))
                                    + value * unit_denoms[name]
                                )
                                chunk_denoms[name] += unit_denoms[name]
                            predictions = output.speech_codec_logits.detach().argmax(dim=-1)
                            valid = (unit.speech_codec_mask & unit.speech_mode.eq(1)[:, None])[
                                :, :, None
                            ]
                            speech_correct += (predictions.eq(unit.speech_codes) & valid).sum(
                                dim=(0, 1)
                            )
                            speech_valid += valid[:, :, 0].sum()
                        next_index = chunk_start + len(chunk)
                        lookahead = (
                            episode.units[next_index].to(accelerator.device)
                            if next_index < len(episode.units)
                            else None
                        )
                        jepa, jepa_pairs = _sequence_jepa_loss(
                            accelerator.unwrap_model(model), outputs, lookahead
                        )
                        losses = {name: value / len(moved) for name, value in chunk_losses.items()}
                        losses["jepa"] = jepa["total"]
                        losses["jepa_prediction"] = jepa["prediction"]
                        losses["jepa_variance"] = jepa["variance"]
                        losses["total"] = (
                            losses["total"]
                            + config.training.jepa_loss_weight * losses["jepa"]
                        )
                        for name, value in (
                            ("jepa", jepa["total"]),
                            ("jepa_prediction", jepa["prediction"]),
                            ("jepa_variance", jepa["variance"]),
                        ):
                            chunk_numerators[name] = value * jepa_pairs
                            chunk_denoms[name] = float(jepa_pairs)
                        chunk_numerators["total"] = (
                            chunk_numerators["total"]
                            + config.training.jepa_loss_weight * jepa["total"] * len(moved)
                        )
                        accelerator.backward(losses["total"])
                        if accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(
                                model.parameters(), config.training.max_grad_norm
                            )
                            optimizer.step()
                            scheduler.step()
                            optimizer.zero_grad(set_to_none=True)
                    assert output is not None
                    recurrent = recurrent.detach()
                    next_unit = chunk_start + len(chunk)
                    if next_unit >= len(episode.units):
                        cursor = DataCursor(
                            epoch=epoch,
                            episode=episode_index + 1,
                            unit=0,
                            ordered_shard_index=episode.ordered_shard_index,
                            sample_index_in_shard=episode.sample_index_in_shard + 1,
                        )
                        checkpoint_recurrent = None
                    else:
                        cursor = DataCursor(
                            epoch=epoch,
                            episode=episode_index,
                            unit=next_unit,
                            ordered_shard_index=episode.ordered_shard_index,
                            sample_index_in_shard=episode.sample_index_in_shard,
                        )
                        checkpoint_recurrent = recurrent
                    train_state.update(
                        {
                            "epoch": epoch,
                            "episode": episode_index,
                            "unit": next_unit,
                            "consumed_units": train_state.get("consumed_units", 0) + len(chunk),
                        }
                    )
                    if not accelerator.sync_gradients:
                        pending_update_metrics.append(
                            {
                                "losses": {
                                    name: float(value.detach().float().item())
                                    for name, value in chunk_numerators.items()
                                },
                                "denoms": chunk_denoms,
                                "speech_correct": speech_correct.detach().cpu(),
                                "speech_valid": int(speech_valid.item()),
                                "target_units": len(moved),
                                "mode_correct": int(
                                    (output.speech_mode_logits.argmax(-1) == unit.speech_mode)
                                    .sum()
                                    .item()
                                ),
                                "mode_valid": int(unit.speech_mode_mask.sum().item()),
                                "action_correct": int(
                                    (output.action.kind_logits.argmax(-1) == unit.action.kind)
                                    .masked_select(unit.action_supervision_mask)
                                    .sum()
                                    .item()
                                ),
                                "action_valid": int(unit.action_supervision_mask.sum().item()),
                            }
                        )
                        continue

                    train_state["update"] += 1
                    pending_update_metrics.append(
                        {
                            "losses": {
                                name: float(value.detach().float().item())
                                for name, value in chunk_numerators.items()
                            },
                            "denoms": chunk_denoms,
                            "speech_correct": speech_correct.detach().cpu(),
                            "speech_valid": int(speech_valid.item()),
                            "target_units": len(moved),
                            "mode_correct": int(
                                (output.speech_mode_logits.argmax(-1) == moved[-1].speech_mode)
                                .sum()
                                .item()
                            ),
                            "mode_valid": int(moved[-1].speech_mode_mask.sum().item()),
                            "action_correct": int(
                                (output.action.kind_logits.argmax(-1) == moved[-1].action.kind)
                                .masked_select(moved[-1].action_supervision_mask)
                                .sum()
                                .item()
                            ),
                            "action_valid": int(moved[-1].action_supervision_mask.sum().item()),
                        }
                    )
                    last_metrics = _aggregate_update_metrics(pending_update_metrics, config)
                    pending_update_metrics = []
                    last_metrics.update(
                        {
                            "stream/kv_tokens": float(output.state.layer_kv[0].key.shape[2]),
                            "data/episode": float(episode_index),
                            "data/unit": float(next_unit),
                            "train/learning_rate": float(scheduler.get_last_lr()[0]),
                            "speech/scheduled_sampling": sampling_probability,
                        }
                    )
                    should_log = (
                        accelerator.is_main_process
                        and train_state["update"] % config.training.log_every == 0
                    )
                    if should_log:
                        tracker.log(last_metrics, train_state["update"])
                        last_logged_update = train_state["update"]
                    should_checkpoint = (
                        accelerator.is_main_process
                        and train_state["update"] % config.training.checkpoint_every == 0
                    )
                    if should_checkpoint:
                        path, digest = checkpoint_manager.save(
                            f"step-{train_state['update']:08d}",
                            model=accelerator.unwrap_model(model),
                            optimizer=optimizer,
                            scheduler=scheduler,
                            scaler=accelerator.scaler,
                            recurrent_state=checkpoint_recurrent,
                            train_state=train_state,
                            data_cursor=cursor,
                            metadata=_checkpoint_metadata(config, parent_sha256),
                            config=config.as_dict(),
                        )
                        parent_sha256 = digest
                        last_checkpoint_update = train_state["update"]
                        tracker.record_checkpoint(path, digest, train_state["update"])
                if train_state["update"] >= target_updates:
                    break
            if not yielded:
                raise RuntimeError("training source produced no episodes")
            if train_state["update"] < target_updates:
                epoch += 1
                cursor = DataCursor(epoch=epoch, episode=0, unit=0)
                recurrent = None
    finally:
        if (
            accelerator.is_main_process
            and train_state["update"]
            and train_state["update"] != last_checkpoint_update
        ):
            path, digest = checkpoint_manager.save(
                f"step-{train_state['update']:08d}",
                model=accelerator.unwrap_model(model),
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=accelerator.scaler,
                recurrent_state=recurrent,
                train_state=train_state,
                data_cursor=cursor,
                metadata=_checkpoint_metadata(config, parent_sha256),
                config=config.as_dict(),
            )
            tracker.record_checkpoint(path, digest, train_state["update"])
        elapsed_seconds = time.perf_counter() - training_started
        runtime_metrics = {
            "runtime/elapsed_seconds": elapsed_seconds,
            "runtime/units_per_second": train_state.get("consumed_units", 0)
            / max(elapsed_seconds, 1e-9),
        }
        if accelerator.device.type == "cuda":
            runtime_metrics.update(
                {
                    "runtime/peak_memory_allocated_bytes": float(
                        torch.cuda.max_memory_allocated(accelerator.device)
                    ),
                    "runtime/peak_memory_reserved_bytes": float(
                        torch.cuda.max_memory_reserved(accelerator.device)
                    ),
                }
            )
        last_metrics.update(runtime_metrics)
        if accelerator.is_main_process and train_state["update"]:
            if last_logged_update == train_state["update"]:
                # The final training metrics were already logged at this step;
                # append runtime metrics without duplicating the training point.
                tracker.log(runtime_metrics, train_state["update"])
            else:
                # Always make short smoke runs visible in W&B, even when the
                # update count is smaller than the configured log interval.
                tracker.log(last_metrics, train_state["update"])
        tracking = {
            "requested_mode": config.tracking.mode,
            "effective_mode": tracker.effective_mode,
            "run_url": tracker.run_url,
        }
        tracker.finish()
    return {
        "train_state": train_state,
        "data_cursor": cursor,
        "metrics": last_metrics,
        "tracking": tracking,
        "model": model,
    }


def initialize_exact_weights(model: StreamingLatentLoop, path: str | Path) -> None:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("initial checkpoint metadata is incomplete")
    if metadata.get("action_schema_id") != ACTION_SCHEMA_ID:
        raise ValueError("initial checkpoint action schema is incompatible")
    source = payload.get("model")
    if not isinstance(source, dict):
        raise ValueError("initial checkpoint does not contain a model state")
    current = model.state_dict()
    if source.keys() != current.keys() or any(
        source[name].shape != current[name].shape for name in current
    ):
        raise ValueError("SFT checkpoint does not contain the complete current model")
    model.load_state_dict(source, strict=True)


def scheduled_sampling_probability(update: int, config: ProjectConfig) -> float:
    target = config.training.codec_scheduled_sampling
    start = config.training.codec_scheduled_sampling_start
    progress = update / max(config.training.max_updates, 1)
    if progress <= start:
        return 0.0
    return target * min((progress - start) / (1 - start), 1.0)


def configure_trainable_parameters(model: StreamingLatentLoop, config: ProjectConfig) -> None:
    mode = config.training.backbone_train_mode
    if mode == "all":
        return
    head_prefixes = ("speech_head.", "action_head.")
    selective_prefixes = (
        "audio_encoder.",
        "semantic_slot_identity",
        "slow_memories.",
        "final_norm.",
    )
    first_top_layer = max(0, config.model.num_layers * 3 // 4)
    for name, parameter in model.named_parameters():
        trainable = name.startswith(head_prefixes)
        if mode == "selective":
            trainable = trainable or name.startswith(selective_prefixes)
            if name.startswith("layers."):
                layer_index = int(name.split(".", 2)[1])
                trainable = trainable or layer_index >= first_top_layer
        parameter.requires_grad = trainable


def _sample_logprob(
    output: Any,
    mode: torch.Tensor,
    codes: torch.Tensor,
    actions: ActionFrame,
    sampling: SpeechSamplingConfig,
) -> torch.Tensor:
    temperature = sampling.temperature
    mode_lp = (
        torch.log_softmax(output.speech_mode_logits / temperature, dim=-1)
        .gather(-1, mode[:, None])
        .squeeze(-1)
    )
    codec_lp = (
        torch.log_softmax(output.speech_codec_logits / temperature, dim=-1)
        .gather(-1, codes[..., None])
        .squeeze(-1)
    )
    codec_lp = torch.where(
        mode.eq(1)[:, None, None], codec_lp, torch.zeros_like(codec_lp)
    )
    action_lp = action_frame_log_prob(output.action, actions, temperature)
    return torch.cat(
        (
            mode_lp.reshape(mode.shape[0], -1),
            codec_lp.reshape(mode.shape[0], -1),
            action_lp[:, None],
        ),
        dim=-1,
    )


def _policy_logprobs(
    output: Any,
    mode: torch.Tensor,
    codes: torch.Tensor,
    actions: ActionFrame,
    sampling: SpeechSamplingConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    values = _sample_logprob(output, mode, codes, actions, sampling)
    return values[:, :-1].sum(dim=-1), values[:, -1]


def _session_specs(path: str) -> list[tuple[str, str, int]]:
    values: list[tuple[str, str, int]] = []
    with Path(path).expanduser().open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError("RL session manifest entries must be objects")
            values.append(
                (
                    str(item["session_id"]),
                    str(item["initial_snapshot_id"]),
                    int(item.get("seed", 0)),
                )
            )
    if not values:
        raise ValueError("RL session manifest contains no lifetime sessions")
    return values


@dataclass(slots=True)
class _PPOUnit:
    observation: ObservationSignal
    mode: torch.Tensor
    codes: torch.Tensor
    action: ActionFrame
    old_speech_logprob: torch.Tensor
    old_action_logprob: torch.Tensor
    old_value: torch.Tensor
    reference_speech_logprob: torch.Tensor
    reference_action_logprob: torch.Tensor


@dataclass(slots=True)
class _CandidateResult:
    accepted: bool
    reason: str | None
    model_state: dict[str, torch.Tensor]
    optimizer_state: dict[str, Any]
    metrics: dict[str, float]


def _action_sample(action: ActionFrame) -> dict[str, Any]:
    return {
        name: value.detach().cpu().tolist()[0]
        for name, value in action.items()
    }


def _trace_sample(
    mode: torch.Tensor, codes: torch.Tensor, action: ActionFrame
) -> dict[str, Any]:
    return {
        "speech_mode": int(mode.detach().cpu().reshape(-1)[0]),
        "speech_codes": codes.detach().cpu().tolist()[0],
        "action": _action_sample(action),
    }


def _state_pending_utf8(state: RecurrentState) -> bytes:
    length = int(state.action_local.pending_utf8_length.reshape(-1)[0].item())
    return bytes(
        int(value)
        for value in state.action_local.pending_utf8_bytes[0, :length].tolist()
    )


def _supervised_episode_objectives(
    model: StreamingLatentLoop,
    episode: Episode,
    config: ProjectConfig,
    device: torch.device,
    *,
    include_jepa: bool = True,
) -> dict[str, torch.Tensor]:
    state = model.initial_state(1, device)
    losses: list[torch.Tensor] = []
    outputs: list[Any] = []
    selected_units = episode.units[: config.training.memory_horizon_units]
    for raw_unit in selected_units:
        unit = raw_unit.to(device)
        output = model.forward_step(
            unit,
            state,
            speech_teacher_codes=unit.speech_codes,
            speech_teacher_mode=unit.speech_mode,
            action_teacher_frame=unit.action,
            action_teacher_mask=unit.action_supervision_mask,
        )
        state = output.state
        outputs.append(output)
        losses.append(
            compute_losses(
                output,
                unit,
                config.training.speech_loss_weight,
                config.training.action_loss_weight,
            )["total"]
        )
    if not losses:
        raise RuntimeError("SFT preservation episode contains no units")
    if include_jepa:
        lookahead = (
            episode.units[len(selected_units)].to(device)
            if len(selected_units) < len(episode.units)
            else None
        )
        jepa, _ = _sequence_jepa_loss(model, outputs, lookahead)
    else:
        zero = losses[0].sum() * 0.0
        jepa = {"total": zero}
    return {"behavior": torch.stack(losses).mean(), "jepa": jepa["total"]}


def _supervised_episode_loss(
    model: StreamingLatentLoop,
    episode: Episode,
    config: ProjectConfig,
    device: torch.device,
) -> torch.Tensor:
    """Behavior-only objective used by the preservation acceptance gate."""
    return _supervised_episode_objectives(
        model, episode, config, device, include_jepa=False
    )["behavior"]


def _ppo_guard_episodes(config: ProjectConfig) -> tuple[Episode, Episode]:
    rl = config.training.rl
    if config.data.source == "synthetic":
        episodes = SyntheticEpisodeDataset(config.data, config.model)
        return episodes.make_episode(0), episodes.make_episode(1)
    paths = (
        (rl.sft_replay_shards, rl.sft_replay_manifest),
        (rl.sft_preservation_shards, rl.sft_preservation_manifest),
    )
    guard_episodes: list[Episode] = []
    for shards, manifest in paths:
        if not shards or not manifest:
            raise ValueError("PPO requires explicit SFT replay and preservation datasets")
        data = replace(config.data, shards=shards, manifest=manifest)
        episode = next(iter(EpisodeShardReader(shards, data, config.model)), None)
        if episode is None:
            raise RuntimeError("PPO SFT guard dataset contains no episodes")
        guard_episodes.append(episode)
    return guard_episodes[0], guard_episodes[1]


def _train_ppo_candidate(
    serving: StreamingLatentLoop,
    optimizer_state: dict[str, Any],
    units: tuple[_PPOUnit, ...],
    lookahead_observation: ObservationSignal,
    window_start_state: RecurrentState,
    rewards: tuple[float, ...],
    bootstrap_masks: tuple[float, ...],
    bootstrap_value: torch.Tensor,
    config: ProjectConfig,
    device: torch.device,
    sampling: SpeechSamplingConfig,
) -> _CandidateResult:
    candidate = serving.to(device).eval()
    candidate_optimizer = torch.optim.AdamW(
        [parameter for parameter in candidate.parameters() if parameter.requires_grad],
        lr=config.training.backbone_learning_rate,
        weight_decay=config.training.weight_decay,
    )
    candidate_optimizer.load_state_dict(optimizer_state)
    old_value = torch.cat([unit.old_value for unit in units]).to(device)
    reward_tensor = torch.tensor(rewards, device=device)
    discounts = time_discounts(
        torch.tensor([unit.observation.delta_ms for unit in units], device=device),
        config.training.rl.discount_time_constant_ms,
    )
    with torch.no_grad():
        advantages, value_targets = generalized_advantage_estimate(
            reward_tensor,
            old_value,
            bootstrap_value.to(device),
            discounts,
            torch.tensor(bootstrap_masks, device=device),
            config.training.rl.gae_lambda,
        )
        advantages = (advantages - advantages.mean()) / advantages.std(
            unbiased=False
        ).clamp_min(config.training.rl.advantage_epsilon)

    replay_episode, evaluation_episode = _ppo_guard_episodes(config)
    with torch.no_grad():
        baseline_eval_loss = _supervised_episode_loss(
            candidate, evaluation_episode, config, device
        )
    old_speech = torch.cat([unit.old_speech_logprob for unit in units]).to(device)
    old_action = torch.cat([unit.old_action_logprob for unit in units]).to(device)
    reference_speech = torch.cat(
        [unit.reference_speech_logprob for unit in units]
    ).to(device)
    reference_action = torch.cat(
        [unit.reference_action_logprob for unit in units]
    ).to(device)
    last_loss: torch.Tensor | None = None
    last_components: dict[str, torch.Tensor] = {}
    last_sft_replay_loss: torch.Tensor | None = None
    last_on_policy_jepa: torch.Tensor | None = None
    last_replay_jepa: torch.Tensor | None = None
    for _ in range(config.training.rl.ppo_epochs):
        state = window_start_state
        current_speech: list[torch.Tensor] = []
        current_action: list[torch.Tensor] = []
        current_values: list[torch.Tensor] = []
        outputs: list[Any] = []
        stream_units = [
            observation_to_stream_unit(item.observation, config).to(device) for item in units
        ]
        segment_units = config.model.rematerialization_segment_units
        for segment_start in range(0, len(units), segment_units):
            segment_items = units[segment_start : segment_start + segment_units]
            segment = stream_units[segment_start : segment_start + segment_units]
            segment_outputs, state = _rematerialized_segment(
                candidate,
                segment,
                state,
                [item.codes.to(device) for item in segment_items],
                action_masks=[
                    torch.ones_like(item.mode, dtype=torch.bool, device=device)
                    for item in segment_items
                ],
                action_frames=[item.action.to(device) for item in segment_items],
            )
            outputs.extend(segment_outputs)
        for item, output in zip(units, outputs, strict=True):
            speech_logprob, action_logprob = _policy_logprobs(
                output,
                item.mode.to(device),
                item.codes.to(device),
                item.action.to(device),
                sampling,
            )
            current_speech.append(speech_logprob)
            current_action.append(action_logprob)
            current_values.append(output.value)
        ppo_inputs = {
            "current_speech": torch.cat(current_speech),
            "old_speech": old_speech,
            "current_action": torch.cat(current_action),
            "old_action": old_action,
            "advantage": advantages,
            "current_value": torch.cat(current_values),
            "old_value": old_value,
            "value_target": value_targets,
            "reference_speech": reference_speech,
            "reference_action": reference_action,
        }
        invalid_inputs = [
            name
            for name, value in ppo_inputs.items()
            if not bool(torch.isfinite(value).all())
        ]
        if invalid_inputs:
            return _CandidateResult(
                False,
                "non-finite PPO inputs: " + ", ".join(invalid_inputs),
                {},
                {},
                {},
            )
        ppo_loss, components = recurrent_ppo_loss(
            ppo_inputs["current_speech"],
            ppo_inputs["old_speech"],
            ppo_inputs["current_action"],
            ppo_inputs["old_action"],
            ppo_inputs["advantage"],
            ppo_inputs["current_value"],
            ppo_inputs["old_value"],
            ppo_inputs["value_target"],
            ppo_inputs["reference_speech"],
            ppo_inputs["reference_action"],
            clip_epsilon=config.training.rl.clip_epsilon,
            value_coef=config.training.rl.value_coef,
            entropy_coef=config.training.rl.entropy_coef,
            reference_kl_beta=config.training.rl.reference_kl_beta,
            entropy=-0.5
            * (ppo_inputs["current_speech"] + ppo_inputs["current_action"]),
        )
        on_policy_jepa, _ = _sequence_jepa_loss(
            candidate,
            outputs,
            observation_to_stream_unit(lookahead_observation, config).to(device),
        )
        replay_objectives = _supervised_episode_objectives(
            candidate, replay_episode, config, device
        )
        sft_replay_loss = replay_objectives["behavior"]
        replay_jepa = replay_objectives["jepa"]
        loss = (
            ppo_loss
            + config.training.rl.sft_replay_coef * sft_replay_loss
            + config.training.rl.on_policy_jepa_coef * on_policy_jepa["total"]
            + config.training.rl.replay_jepa_coef * replay_jepa
        )
        diagnostic_tensors = {
            "total": loss,
            "sft_replay": sft_replay_loss,
            "jepa_on_policy": on_policy_jepa["total"],
            "jepa_replay": replay_jepa,
            **components,
        }
        invalid = [
            name
            for name, value in diagnostic_tensors.items()
            if not bool(torch.isfinite(value).all())
        ]
        if invalid:
            return _CandidateResult(
                False, "non-finite candidate loss: " + ", ".join(invalid), {}, {}, {}
            )
        candidate_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradients_finite = all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for parameter in candidate.parameters()
        )
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            candidate.parameters(), config.training.max_grad_norm
        )
        if not gradients_finite or not bool(torch.isfinite(gradient_norm)):
            return _CandidateResult(False, "non-finite candidate gradients", {}, {}, {})
        candidate_optimizer.step()
        if not all(
            bool(torch.isfinite(parameter).all()) for parameter in candidate.parameters()
        ):
            return _CandidateResult(False, "non-finite candidate weights", {}, {}, {})
        last_loss = loss
        last_components = components
        last_sft_replay_loss = sft_replay_loss
        last_on_policy_jepa = on_policy_jepa["total"]
        last_replay_jepa = replay_jepa

    assert (
        last_loss is not None
        and last_sft_replay_loss is not None
        and last_on_policy_jepa is not None
        and last_replay_jepa is not None
    )

    with torch.no_grad():
        candidate_eval_loss = _supervised_episode_loss(
            candidate, evaluation_episode, config, device
        )
        post_state = window_start_state
        post_speech: list[torch.Tensor] = []
        post_action: list[torch.Tensor] = []
        for item in units:
            stream_unit = observation_to_stream_unit(item.observation, config).to(device)
            output = candidate.forward_step(
                stream_unit,
                post_state,
                speech_teacher_codes=item.codes.to(device),
                speech_teacher_mode=item.mode.to(device),
                action_teacher_frame=item.action.to(device),
                action_teacher_mask=torch.ones_like(
                    item.mode, dtype=torch.bool, device=device
                ),
            )
            post_state = output.state
            speech_logprob, action_logprob = _policy_logprobs(
                output,
                item.mode.to(device),
                item.codes.to(device),
                item.action.to(device),
                sampling,
            )
            post_speech.append(speech_logprob)
            post_action.append(action_logprob)
        post_reference_kl = 0.5 * (
            sampled_reference_kl(
                torch.cat(post_speech), ppo_inputs["reference_speech"]
            )
            + sampled_reference_kl(
                torch.cat(post_action), ppo_inputs["reference_action"]
            )
        )
    eval_ratio = candidate_eval_loss / baseline_eval_loss.clamp_min(1e-8)
    metrics = {
        "train/loss_ppo": float(last_loss.detach().cpu()),
        "train/loss_actor": float(last_components["actor"].detach().cpu()),
        "train/loss_value": float(last_components["value"].detach().cpu()),
        "train/loss_sft_replay": float(last_sft_replay_loss.detach().cpu()),
        "train/loss_jepa_on_policy": float(last_on_policy_jepa.detach().cpu()),
        "train/loss_jepa_replay": float(last_replay_jepa.detach().cpu()),
        "rl/candidate_reference_kl": float(post_reference_kl.cpu()),
        "rl/candidate_eval_loss_ratio": float(eval_ratio.cpu()),
        "rl/reward_mean": float(reward_tensor.mean().cpu()),
    }
    if float(post_reference_kl) > config.training.rl.candidate_max_reference_kl:
        return _CandidateResult(
            False,
            "candidate reference KL gate failed: "
            f"{float(post_reference_kl):.6g} > "
            f"{config.training.rl.candidate_max_reference_kl:.6g}",
            {},
            {},
            metrics,
        )
    if float(eval_ratio) > config.training.rl.candidate_max_eval_loss_ratio:
        return _CandidateResult(
            False,
            "candidate SFT preservation gate failed: "
            f"{float(eval_ratio):.6g} > "
            f"{config.training.rl.candidate_max_eval_loss_ratio:.6g}",
            {},
            {},
            metrics,
        )
    return _CandidateResult(
        True,
        None,
        {name: value.detach().cpu() for name, value in candidate.state_dict().items()},
        candidate_optimizer.state_dict(),
        metrics,
    )


def _apply_candidate_result(
    policy: StreamingLatentLoop,
    optimizer: torch.optim.Optimizer,
    result: _CandidateResult,
) -> bool:
    if not result.accepted:
        return False
    policy.load_state_dict(result.model_state)
    optimizer.load_state_dict(result.optimizer_state)
    return True


def train_online_ppo(
    config: ProjectConfig,
    *,
    resume: str | None = None,
    init_from: str | None = None,
    model: StreamingLatentLoop | None = None,
    stop_after_updates: int | None = None,
) -> dict[str, Any]:
    if not resume and not init_from:
        raise ValueError("Online Recurrent PPO requires the final SFT checkpoint as --init-from")
    if config.data.dataset == "canary" and not torch.cuda.is_available():
        raise RuntimeError(
            "formal Online Recurrent PPO requires a CUDA GPU; "
            "torch.cuda.is_available() is false"
        )
    if config.data.dataset != "synthetic":
        check_readiness(config.runtime.data_path(), config=config)
    rl = config.training.rl
    if not rl.session_manifest or not rl.timeline_root or not rl.reward_socket:
        raise ValueError(
            "Online Recurrent PPO requires session_manifest, timeline_root and reward_socket"
        )
    sessions = _session_specs(rl.session_manifest)
    if len(sessions) != 1:
        raise ValueError("one Online Recurrent PPO worker requires exactly one lifetime session")
    seed_everything(config.data.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    policy = model or StreamingLatentLoop(config.model)
    session_id, snapshot_id, seed = sessions[0]
    resume_state: dict[str, Any] | None = None
    resume_metadata: CheckpointMetadata | None = None
    if resume:
        resume_state, resume_metadata = inspect_checkpoint(resume)
        if not resume_metadata.sft_checkpoint_path:
            raise ValueError("PPO checkpoint is missing the frozen SFT checkpoint path")
        sft_checkpoint = Path(resume_metadata.sft_checkpoint_path).expanduser().resolve()
        if (
            not sft_checkpoint.is_file()
            or file_sha256(sft_checkpoint)
            != resume_metadata.reference_checkpoint_sha256
        ):
            raise ValueError("PPO frozen SFT reference checkpoint is unavailable or changed")
    else:
        assert init_from is not None
        sft_checkpoint = Path(init_from).expanduser().resolve()
    _, sft_metadata = inspect_checkpoint(sft_checkpoint)
    if sft_metadata.stage != "sft" or sft_metadata.algorithm is not None:
        raise ValueError("Online Recurrent PPO requires a final supervised SFT checkpoint")
    sft_sha256 = file_sha256(sft_checkpoint)
    initialize_exact_weights(policy, sft_checkpoint)
    policy = policy.to(device).eval()
    configure_trainable_parameters(policy, config)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in policy.parameters() if parameter.requires_grad],
        lr=config.training.backbone_learning_rate,
        weight_decay=config.training.weight_decay,
    )
    reference = StreamingLatentLoop(config.model)
    initialize_exact_weights(reference, sft_checkpoint)
    reference = reference.to(device).eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    lineage_id = (
        resume_metadata.lineage_id
        if resume_metadata is not None
        else f"{session_id}-{sft_sha256[:12]}"
    )
    if not lineage_id:
        raise ValueError("PPO checkpoint is missing lifetime lineage identity")
    if resume_state is not None and resume_state.get("session_id") != session_id:
        raise ValueError("PPO checkpoint session does not match the session manifest")
    tracker = Tracker(
        config,
        model_name=f"{policy.parameter_count()}p",
        parameter_count=policy.parameter_count(),
        data_identity=_data_identity(config),
        parent_checkpoint_sha256=file_sha256(resume) if resume else sft_sha256,
    )
    manager = CheckpointManager(config.runtime.root_path() / "checkpoints")
    train_state: dict[str, Any] = {
        "update": 0,
        "candidate_attempts": 0,
        "candidate_rejections": 0,
        "consecutive_rejections": 0,
        "consumed_units": 0,
        "dropped_windows": 0,
        "session_id": session_id,
        "lineage_id": lineage_id,
        "next_unit": 0,
        "finalized_through": -1,
        "goal_tracker": {"active_goal_id": None, "finalized": []},
    }
    sampling = SpeechSamplingConfig(
        temperature=config.training.rl.sampling_temperature,
        top_k=config.training.rl.sampling_top_k,
        greedy=False,
    )
    target_updates = min(
        config.training.max_updates,
        stop_after_updates if stop_after_updates is not None else config.training.max_updates,
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    training_started = time.perf_counter()
    from data import ObservationTimeline, PolicySampleTrace
    from harness.reward import PerceptualRewardClient, SingleActiveGoalTracker

    from training.window import RolloutWindowStore, SealedRolloutWindow

    policy_version = resume_metadata.policy_version if resume_metadata else sft_sha256
    if not policy_version:
        raise ValueError("PPO checkpoint is missing serving policy version")
    timeline = ObservationTimeline(
        lineage_id, session_id, Path(rl.timeline_root).expanduser() / lineage_id
    )
    policy_trace = PolicySampleTrace(
        lineage_id, session_id, Path(rl.timeline_root).expanduser() / lineage_id
    )
    window_store = RolloutWindowStore(config.runtime.root_path() / "rollout-windows")
    if not resume and (timeline.records or policy_trace.records):
        raise ValueError("Online Recurrent PPO fresh run requires an empty lifetime timeline")
    judge = PerceptualRewardClient(
        rl.reward_socket,
        spec_id=rl.reward_spec_id,
        judge_model_id=rl.judge_model_id,
        judge_revision=rl.judge_revision,
        rubric_sha256=rl.rubric_sha256,
    )
    judge_identity = judge.identity()
    expected_judge = {
        "spec_id": rl.reward_spec_id,
        "judge_model_id": rl.judge_model_id,
        "judge_revision": rl.judge_revision,
        "rubric_sha256": rl.rubric_sha256,
    }
    if any(judge_identity.get(key) != value for key, value in expected_judge.items()):
        raise ValueError("Reward Judge identity does not match configuration")
    goal_tracker = SingleActiveGoalTracker()
    env = PhysicalRolloutClient(config)
    identity = env.identity()
    if (
        identity.get("environment_id") != rl.environment_id
        or identity.get("environment_version") != rl.environment_version
        or identity.get("protocol_version") != rl.environment_protocol_version
        or identity.get("action_schema_id") != ACTION_SCHEMA_ID
    ):
        raise ValueError("Harness environment identity does not match configuration")
    if resume:
        assert resume_metadata is not None
        expected_metadata = _checkpoint_metadata(
            config,
            reference_checkpoint_sha256=sft_sha256,
            lineage_id=lineage_id,
            policy_version=policy_version,
            observation_chain_sha256=resume_metadata.observation_chain_sha256,
            policy_sample_chain_sha256=resume_metadata.policy_sample_chain_sha256,
            sft_checkpoint_path=str(sft_checkpoint),
        )
        train_state, _, policy_state, _ = manager.load(
            resume,
            model=policy,
            optimizer=optimizer,
            scheduler=None,
            scaler=None,
            device=device,
            config=config.as_dict(),
            expected_metadata=expected_metadata,
        )
        reference_state = load_reference_recurrent_state(resume, device)
        if policy_state is None or reference_state is None:
            raise ValueError("PPO checkpoint is missing recurrent serving/reference state")
        next_unit = int(train_state.get("next_unit", -1))
        if (
            next_unit != len(timeline.records)
            or next_unit != len(policy_trace.records)
            or timeline.chain_sha256 != resume_metadata.observation_chain_sha256
            or policy_trace.chain_sha256 != resume_metadata.policy_sample_chain_sha256
            or int(policy_state.unit_index.item()) != next_unit
            or int(reference_state.unit_index.item()) != next_unit
        ):
            raise ValueError("PPO checkpoint and lifetime timeline cursors do not match")
        goal_tracker.load_state_dict(train_state.get("goal_tracker", {}))
        speech_history = []
        for item in policy_trace.records:
            sample = item.sample
            if int(sample["speech_mode"]) == 1:
                codes = torch.tensor([sample["speech_codes"]], dtype=torch.long)
                speech_history.append(codes.transpose(1, 2))
        observation = env.resume_lifetime_session(
            session_id,
            next_unit,
            pending_utf8=_state_pending_utf8(policy_state),
            speech_history=speech_history,
        )
    else:
        observation = env.start_lifetime_session(snapshot_id, seed, session_id)
        policy_state = policy.initial_state(1, device)
        reference_state = reference.initial_state(1, device)
    last_metrics: dict[str, float] = {}
    finalized_through = int(train_state.get("finalized_through", -1))
    reward_by_unit: dict[int, float] = {}
    terminal_units: set[int] = set()
    parent_checkpoint_sha256 = file_sha256(resume) if resume else sft_sha256
    last_checkpoint_update = -1
    checkpoint_path: Path | None = None
    checkpoint_digest: str | None = None

    def rollout_one() -> tuple[_PPOUnit, tuple[str, ...]]:
        nonlocal observation, policy_state, reference_state, finalized_through
        record = timeline.append(observation, policy_version)
        result = judge.observe(
            lineage_id=lineage_id,
            session_id=session_id,
            unit_index=observation.unit_index,
            observation_payload=record.payload,
            observation_chain_sha256=record.chain_sha256,
        )
        previous_finalized_through = finalized_through
        if result.finalized_through_unit < previous_finalized_through:
            raise RuntimeError("Reward Judge finalization watermark regressed")
        finalized_through = result.finalized_through_unit
        reward_event_ids: list[str] = []
        for event in result.events:
            is_new = goal_tracker.accept(event)
            if (
                event.evidence_end_unit >= len(timeline.records)
                or event.observation_chain_end_sha256
                != timeline.records[event.evidence_end_unit].chain_sha256
            ):
                raise RuntimeError("Reward Judge event observation chain does not match")
            if is_new and event.status.value == "finalized":
                reward_event_ids.append(event.event_id)
            if is_new and event.trainable:
                if event.outcome_unit <= previous_finalized_through:
                    raise RuntimeError(
                        "Reward Judge emitted an event behind its finalization watermark"
                    )
                reward_by_unit[event.outcome_unit] = (
                    reward_by_unit.get(event.outcome_unit, 0.0) + event.reward.total
                )
                terminal_units.add(event.outcome_unit)
        unit = observation_to_stream_unit(observation, config).to(device)
        with torch.no_grad():
            generated = policy.generate_step(unit, policy_state, sampling)
            policy_state = generated.output.state
            mode, codes, action = (
                generated.speech_mode,
                generated.speech_codes,
                generated.action_frame,
            )
            speech_logprob, action_logprob = _policy_logprobs(
                generated.output, mode, codes, action, sampling
            )
            reference_output = reference.forward_step(
                unit,
                reference_state,
                speech_teacher_codes=codes,
                speech_teacher_mode=mode,
                action_teacher_frame=action,
                action_teacher_mask=torch.ones_like(mode, dtype=torch.bool),
            )
            reference_state = reference_output.state
            reference_speech, reference_action = _policy_logprobs(
                reference_output, mode, codes, action, sampling
            )
        next_observation, receipt, _ = env.step(observation, mode, codes, action)
        if receipt.infrastructure_failure:
            raise RuntimeError(
                f"physical rollout infrastructure failure: {receipt.infrastructure_failure}"
            )
        if not receipt.accepted:
            raise RuntimeError(
                f"physical rollout action rejected: {receipt.safety_violation or 'unknown'}"
            )
        policy_trace.append(
            unit_index=observation.unit_index,
            policy_version=policy_version,
            observation_payload_sha256=record.payload_sha256,
            sample=_trace_sample(mode, codes, action),
        )
        item = _PPOUnit(
            observation,
            mode.cpu(),
            codes.cpu(),
            action.cpu(),
            speech_logprob.cpu(),
            action_logprob.cpu(),
            generated.output.value.cpu(),
            reference_speech.cpu(),
            reference_action.cpu(),
        )
        train_state["consumed_units"] += 1
        observation = next_observation
        train_state["next_unit"] = observation.unit_index
        train_state["finalized_through"] = finalized_through
        train_state["goal_tracker"] = goal_tracker.state_dict()
        return item, tuple(reward_event_ids)

    def seal_stale(
        units: list[_PPOUnit], disposition: str, reward_event_ids: Iterable[str] = ()
    ) -> None:
        if not units:
            return
        attempt = int(train_state["candidate_attempts"])
        lookahead_hash = hashlib.sha256(observation_to_payload(observation)).hexdigest()
        window_store.seal(
            SealedRolloutWindow(
                window_id=(
                    f"stale-{attempt:08d}-{units[0].observation.unit_index:012d}"
                ),
                lineage_id=lineage_id,
                session_id=session_id,
                policy_version=timeline.records[
                    units[0].observation.unit_index
                ].policy_version,
                start_unit=units[0].observation.unit_index,
                end_unit=units[-1].observation.unit_index,
                observation_chain_sha256=timeline.records[
                    units[-1].observation.unit_index
                ].chain_sha256,
                finalized_through_unit=finalized_through,
                reward_event_ids=tuple(dict.fromkeys(reward_event_ids)),
                consumed_units=len(units),
                lookahead_unit_index=observation.unit_index,
                lookahead_payload_sha256=lookahead_hash,
                eligible_for_update=False,
                disposition=disposition,
            )
        )
        stale_indices = {item.observation.unit_index for item in units}
        for unit_index in stale_indices:
            reward_by_unit.pop(unit_index, None)
        terminal_units.difference_update(stale_indices)
        train_state["dropped_windows"] += 1

    def save_checkpoint() -> tuple[Path, str]:
        nonlocal parent_checkpoint_sha256, last_checkpoint_update
        train_state["next_unit"] = observation.unit_index
        train_state["finalized_through"] = finalized_through
        train_state["goal_tracker"] = goal_tracker.state_dict()
        path, digest = manager.save(
            f"step-{train_state['update']:08d}",
            model=policy,
            optimizer=optimizer,
            scheduler=None,
            scaler=None,
            recurrent_state=policy_state.detach(),
            reference_recurrent_state=reference_state.detach(),
            train_state=train_state,
            data_cursor=DataCursor(unit=observation.unit_index),
            metadata=_checkpoint_metadata(
                config,
                parent_checkpoint_sha256,
                sft_sha256,
                lineage_id=lineage_id,
                policy_version=policy_version,
                observation_chain_sha256=timeline.chain_sha256,
                policy_sample_chain_sha256=policy_trace.chain_sha256,
                sft_checkpoint_path=str(sft_checkpoint),
            ),
            config=config.as_dict(),
        )
        parent_checkpoint_sha256 = digest
        last_checkpoint_update = int(train_state["update"])
        tracker.record_checkpoint(path, digest, train_state["update"])
        return path, digest

    completed_normally = False
    try:
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="ppo-candidate") as pool:
            while train_state["update"] < target_updates:
                window_start_state = policy_state.detach()
                units: list[_PPOUnit] = []
                reward_event_ids: list[str] = []
                for _ in range(rl.ppo_window_units):
                    item, event_ids = rollout_one()
                    units.append(item)
                    reward_event_ids.extend(event_ids)
                window_end = units[-1].observation.unit_index
                lookahead_observation = observation
                if lookahead_observation.unit_index != window_end + 1:
                    raise RuntimeError("PPO lookahead observation is not the immediate successor")
                lookahead_payload_sha256 = hashlib.sha256(
                    observation_to_payload(lookahead_observation)
                ).hexdigest()
                with torch.no_grad():
                    bootstrap_value = policy.forward_step(
                        observation_to_stream_unit(observation, config).to(device),
                        policy_state,
                    ).value.cpu()
                pending_units: list[_PPOUnit] = []
                pending_reward_event_ids: list[str] = []
                while finalized_through < window_end:
                    if observation.unit_index - window_end > rl.max_pending_reward_units:
                        raise RuntimeError(
                            "PPO reward finalization exceeded the configured horizon"
                        )
                    item, event_ids = rollout_one()
                    pending_units.append(item)
                    pending_reward_event_ids.extend(event_ids)
                    reward_event_ids.extend(event_ids)
                seal_stale(
                    pending_units, "reward_pending", pending_reward_event_ids
                )
                train_state["candidate_attempts"] += 1
                attempt = int(train_state["candidate_attempts"])
                window_store.seal(
                    SealedRolloutWindow(
                        window_id=f"window-{attempt:08d}",
                        lineage_id=lineage_id,
                        session_id=session_id,
                        policy_version=policy_version,
                        start_unit=units[0].observation.unit_index,
                        end_unit=window_end,
                        observation_chain_sha256=timeline.records[
                            window_end
                        ].chain_sha256,
                        finalized_through_unit=finalized_through,
                        reward_event_ids=tuple(dict.fromkeys(reward_event_ids)),
                        consumed_units=len(units),
                        lookahead_unit_index=lookahead_observation.unit_index,
                        lookahead_payload_sha256=lookahead_payload_sha256,
                    )
                )
                rewards = tuple(
                    reward_by_unit.pop(item.observation.unit_index, 0.0)
                    for item in units
                )
                bootstrap_masks = tuple(
                    0.0 if item.observation.unit_index in terminal_units else 1.0
                    for item in units
                )
                terminal_units.difference_update(
                    item.observation.unit_index for item in units
                )
                candidate_snapshot = copy.deepcopy(policy)
                future: Future[_CandidateResult] = pool.submit(
                    _train_ppo_candidate,
                    candidate_snapshot,
                    copy.deepcopy(optimizer.state_dict()),
                    tuple(units),
                    lookahead_observation,
                    window_start_state,
                    rewards,
                    bootstrap_masks,
                    bootstrap_value,
                    config,
                    device,
                    sampling,
                )
                stale_units: list[_PPOUnit] = []
                stale_reward_event_ids: list[str] = []
                stale_item, event_ids = rollout_one()
                stale_units.append(stale_item)
                stale_reward_event_ids.extend(event_ids)
                while not future.done():
                    stale_item, event_ids = rollout_one()
                    stale_units.append(stale_item)
                    stale_reward_event_ids.extend(event_ids)
                result = future.result()
                seal_stale(
                    stale_units, "candidate_training", stale_reward_event_ids
                )
                last_metrics = {
                    **result.metrics,
                    "rl/finalization_lag_units": float(
                        max(observation.unit_index - 1 - finalized_through, 0)
                    ),
                    "data/consumed_units": float(train_state["consumed_units"]),
                    "rl/sealed_windows": float(train_state["candidate_attempts"]),
                    "rl/dropped_windows": float(train_state["dropped_windows"]),
                }
                if _apply_candidate_result(policy, optimizer, result):
                    train_state["update"] += 1
                    train_state["consecutive_rejections"] = 0
                    policy_version = f"candidate-{train_state['update']:08d}"
                    policy_state = policy_state.detach()
                    last_metrics["rl/candidate_accepted"] = 1.0
                else:
                    train_state["candidate_rejections"] += 1
                    train_state["consecutive_rejections"] += 1
                    last_metrics["rl/candidate_accepted"] = 0.0
                    if (
                        train_state["consecutive_rejections"]
                        >= rl.candidate_max_rejections
                    ):
                        raise RuntimeError(
                            "PPO candidate acceptance failed repeatedly: "
                            + str(result.reason)
                        )
                if train_state["candidate_attempts"] % config.training.log_every == 0:
                    tracker.log(last_metrics, train_state["update"])
                if (
                    train_state["update"] > 0
                    and train_state["update"] % config.training.checkpoint_every == 0
                    and train_state["update"] != last_checkpoint_update
                ):
                    checkpoint_path, checkpoint_digest = save_checkpoint()
        if train_state["update"] != last_checkpoint_update:
            checkpoint_path, checkpoint_digest = save_checkpoint()
        completed_normally = True
    finally:
        if completed_normally and train_state["update"] < config.training.max_updates:
            env.detach()
        else:
            env.close()
    assert checkpoint_path is not None and checkpoint_digest is not None
    tracker.finish()
    elapsed = time.perf_counter() - training_started
    last_metrics.update(
        {
            "runtime/elapsed_seconds": elapsed,
            "runtime/units_per_second": train_state["consumed_units"] / max(elapsed, 1e-9),
            "runtime/peak_memory_allocated_bytes": (
                float(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0.0
            ),
            "runtime/peak_memory_reserved_bytes": (
                float(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0.0
            ),
        }
    )
    last_metrics["checkpoint/sha256"] = checkpoint_digest
    return {
        "train_state": train_state,
        "data_cursor": DataCursor(),
        "metrics": last_metrics,
        "tracking": {
            "requested_mode": config.tracking.mode,
            "effective_mode": tracker.effective_mode,
            "run_url": tracker.run_url,
        },
        "model": policy,
    }
