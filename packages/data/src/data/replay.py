"""Model-free Harness-in-the-loop episode replay."""

from __future__ import annotations

import base64
import json
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from contracts import ActuationSignal, SpeechSignal, decode_action_frame
from contracts.protocol import message_to_actuation

from data.capture import CaptureLedger, action_frame_from_dict, action_frame_to_dict
from data.webdataset import EpisodeShardReader


@dataclass(frozen=True, slots=True)
class ReplayEvent:
    unit_index: int
    receipt_accepted: bool
    execution_latency_ms: float
    observation_unit_index: int
    elapsed_ms: float


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def replay_capture(
    harness: Any,
    transitions: Iterable[dict[str, Any]],
    *,
    session_id: str,
    snapshot_id: str,
    seed: int,
    realtime: bool = True,
    step: bool = False,
    on_started: Callable[[], None] | None = None,
    on_event: Callable[[ReplayEvent], None] | None = None,
) -> list[ReplayEvent]:
    """Apply recorded actions/speech through Harness without loading a model."""

    harness.start_lifetime_session(snapshot_id, seed, session_id)
    if on_started is not None:
        on_started()
    events: list[ReplayEvent] = []
    started = time.monotonic()
    pending = b""
    for index, transition in enumerate(transitions):
        if int(transition.get("unit_index", index)) != index:
            raise ValueError("replay transition units must be contiguous")
        frame = action_frame_from_dict(dict(transition["action_frame"]))
        speech_b64 = str(transition.get("speech_pcm_b64", ""))
        try:
            pcm = base64.b64decode(speech_b64, validate=True)
        except (ValueError, TypeError) as error:
            raise ValueError(f"replay unit {index} speech PCM is invalid") from error
        speech = SpeechSignal(pcm, silent=bool(transition.get("speech_silent", False)))
        decoded = decode_action_frame(
            frame, event_id=f"{session_id}-replay-{index}", pending_utf8=pending
        )
        pending = decoded.pending_utf8
        output = ActuationSignal(session_id, index, speech, controls=decoded.controls)
        observation, receipt = harness.apply(output)
        if observation.unit_index != index + 1 or receipt.unit_index != index:
            raise RuntimeError(f"replay unit {index} returned a non-contiguous result")
        if not receipt.accepted:
            failure = receipt.safety_violation or receipt.infrastructure_failure or "rejected"
            raise RuntimeError(f"Harness rejected replay unit {index}: {failure}")
        elapsed = (time.monotonic() - started) * 1000
        event = ReplayEvent(
            index,
            receipt.accepted,
            receipt.execution_latency_ms,
            observation.unit_index,
            elapsed,
        )
        events.append(event)
        if on_event is not None:
            on_event(event)
        if realtime:
            target = (index + 1) * 80.0
            delay = target - elapsed
            if delay > 0:
                time.sleep(delay / 1000.0)
        if step:
            input(f"replay unit {index} complete; press Enter to continue")
    if pending:
        raise ValueError("replay ended with an incomplete UTF-8 TYPE frame")
    return events


def replay_manifest(
    path: str | Path,
    *,
    harness: Any,
    snapshot_id: str,
    seed: int,
    session_id: str,
    realtime: bool,
    step: bool,
    on_event: Callable[[ReplayEvent], None] | None = None,
) -> list[ReplayEvent]:
    records = _read_jsonl(Path(path).expanduser())
    return replay_capture(
        harness,
        records,
        session_id=session_id,
        snapshot_id=snapshot_id,
        seed=seed,
        realtime=realtime,
        step=step,
        on_event=on_event,
    )


def replay_episode(
    shards: str,
    *,
    data_config: Any,
    model_config: Any,
    harness: Any,
    snapshot_id: str,
    seed: int,
    session_id: str,
    episode_id: str | None = None,
    realtime: bool = True,
    step: bool = False,
    on_started: Callable[[], None] | None = None,
    on_event: Callable[[ReplayEvent], None] | None = None,
) -> list[ReplayEvent]:
    """Replay target speech and ActionFrames from one current WebDataset episode."""

    selected = None
    for episode in EpisodeShardReader(shards, data_config, model_config):
        if episode_id is None or episode.episode_id == episode_id:
            selected = episode
            break
    if selected is None:
        raise ValueError(f"replay episode is absent: {episode_id}")
    if selected.target_speech is None:
        raise ValueError("replay episode has no target speech waveform")
    identity = harness.identity()
    expected_identity = {
        name: str(selected.metadata.get(name, ""))
        for name in (
            "environment_id",
            "environment_version",
            "protocol_version",
            "action_schema_id",
        )
    }
    if not all(expected_identity.values()) or any(
        identity.get(name) != value for name, value in expected_identity.items()
    ):
        raise ValueError("replay Harness identity does not match the episode")
    frame_samples = data_config.unit_audio_samples
    if frame_samples != 1_920:
        raise ValueError("Harness replay requires the physical 1920-sample 80 ms audio clock")
    prepared: list[dict[str, Any]] = []
    for index, unit in enumerate(selected.units):
        start = index * frame_samples
        pcm = selected.target_speech[start : start + frame_samples]
        if pcm.numel() < frame_samples:
            pcm = torch.nn.functional.pad(pcm, (0, frame_samples - pcm.numel()))
        prepared.append(
            {
                "unit_index": index,
                "action_frame": action_frame_to_dict(unit.action.as_contract()),
                "speech_pcm_b64": base64.b64encode(
                    pcm.detach().cpu().float().numpy().tobytes()
                ).decode(),
                "speech_silent": not bool(unit.speech_codec_mask.any()),
            }
        )
    try:
        return replay_capture(
            harness,
            prepared,
            session_id=session_id,
            snapshot_id=snapshot_id,
            seed=seed,
            realtime=realtime,
            step=step,
            on_started=on_started,
            on_event=on_event,
        )
    finally:
        harness.close(session_id)


def replay_ledger(
    root: str | Path,
    *,
    harness: Any,
    snapshot_id: str,
    seed: int,
    session_id: str,
    realtime: bool = True,
    step: bool = False,
    on_started: Callable[[], None] | None = None,
    on_event: Callable[[ReplayEvent], None] | None = None,
) -> list[ReplayEvent]:
    """Replay a CaptureLedger directory without model/service connections."""

    ledger = CaptureLedger.load_sealed(root)
    identity = harness.identity()
    expected_identity = {
        name: str(ledger.seal_metadata.get(name, ""))
        for name in (
            "environment_id",
            "environment_version",
            "protocol_version",
            "action_schema_id",
        )
    }
    if not all(expected_identity.values()) or any(
        identity.get(name) != value for name, value in expected_identity.items()
    ):
        raise ValueError("replay Harness identity does not match the capture ledger")
    ledger_root = ledger.root
    transitions = [record.action_frame for record in ledger.records]
    prepared: list[dict[str, Any]] = []
    for index, action_frame in enumerate(transitions):
        actuation = message_to_actuation(
            (ledger_root / f"actuation-{index:012d}.pb").read_bytes()
        )
        prepared.append(
            {
                "unit_index": index,
                "action_frame": action_frame,
                "speech_pcm_b64": base64.b64encode(actuation.speech.pcm).decode(),
                "speech_silent": actuation.speech.silent,
            }
        )
    try:
        return replay_capture(
            harness,
            prepared,
            session_id=session_id,
            snapshot_id=snapshot_id,
            seed=seed,
            realtime=realtime,
            step=step,
            on_started=on_started,
            on_event=on_event,
        )
    finally:
        harness.close(session_id)


__all__ = [
    "ReplayEvent",
    "replay_capture",
    "replay_episode",
    "replay_ledger",
    "replay_manifest",
]
