"""Harness-backed expert capture and append-only session ledgers.

The collector deliberately keeps the physical boundary in Harness.  It accepts
an expert ``ActionFrame`` and one 80 ms speech PCM frame, converts the frame to
validated controls, and records the observation/actuation/receipt transition.
No model or privileged environment state is involved.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from contracts import (
    ACTION_SCHEMA_ID,
    ActionKind,
    ActuationSignal,
    EnvironmentReceipt,
    ObservationSignal,
    PointerButton,
    PointerButtonPhase,
    SpeechSignal,
    decode_action_frame,
)
from contracts import (
    ActionFrame as ContractActionFrame,
)
from contracts.protocol import (
    actuation_to_payload,
    message_to_actuation,
    message_to_observation,
    observation_to_payload,
    payload_to_receipt,
    receipt_to_payload,
)
from model.types import ActionFrame, Episode, SpeechMode, StreamUnit


def action_frame_from_dict(value: dict[str, Any]) -> ContractActionFrame:
    """Parse the control-console JSON representation of one action frame."""

    if not isinstance(value, dict):
        raise ValueError("action frame must be an object")
    raw_kind = value.get("kind", "NO_ACTION")
    try:
        kind = (
            ActionKind[str(raw_kind).upper()]
            if isinstance(raw_kind, str)
            else ActionKind(int(raw_kind))
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid action kind: {raw_kind!r}") from error
    coordinate = value.get("coordinate_residual", value.get("coordinate", (0.0, 0.0)))
    if isinstance(coordinate, list):
        coordinate = tuple(float(item) for item in coordinate)
    residual = (
        tuple(coordinate)
        if isinstance(coordinate, tuple) and len(coordinate) == 2
        else (0.0, 0.0)
    )
    scroll = value.get("scroll_delta", (0.0, 0.0))
    if isinstance(scroll, list):
        scroll = tuple(float(item) for item in scroll)
    scroll_delta = tuple(scroll) if isinstance(scroll, tuple) and len(scroll) == 2 else (0.0, 0.0)
    button = value.get("button", 0)
    phase = value.get("button_phase", "CLICK")
    button = (
        PointerButton[str(button).upper()]
        if isinstance(button, str)
        else PointerButton(int(button))
    )
    phase = (
        PointerButtonPhase[str(phase).upper()]
        if isinstance(phase, str)
        else PointerButtonPhase(int(phase))
    )
    text = value.get("text_bytes", value.get("text", b""))
    if isinstance(text, str):
        text = text.encode("utf-8")
    elif isinstance(text, list):
        text = bytes(int(item) for item in text)
    keys = value.get("hotkey_keys", value.get("keys", ()))
    return ContractActionFrame(
        kind=kind,
        coordinate_cell=int(value.get("coordinate_cell", 0)),
        coordinate_residual=residual,
        button=button,
        button_phase=phase,
        scroll_delta=scroll_delta,
        text_bytes=bytes(text),
        hotkey_keys=tuple(int(item) for item in keys),
    )


def action_frame_to_dict(frame: ContractActionFrame) -> dict[str, Any]:
    return {
        "kind": frame.kind.name,
        "coordinate_cell": frame.coordinate_cell,
        "coordinate_residual": list(frame.coordinate_residual),
        "button": frame.button.name,
        "button_phase": frame.button_phase.name,
        "scroll_delta": list(frame.scroll_delta),
        "text_bytes": list(frame.text_bytes),
        "hotkey_keys": list(frame.hotkey_keys),
    }


def speech_signal_from_base64(value: str, *, silent: bool = False) -> SpeechSignal:
    try:
        pcm = base64.b64decode(value, validate=True)
    except (ValueError, TypeError) as error:
        raise ValueError("speech PCM base64 is invalid") from error
    return SpeechSignal(pcm, silent=silent)


def _payload_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class CaptureTransition:
    session_id: str
    unit_index: int
    observation_sha256: str
    next_observation_sha256: str
    actuation_sha256: str
    receipt_sha256: str
    previous_chain_sha256: str
    chain_sha256: str
    action_frame: dict[str, Any]
    speech_silent: bool
    receipt_accepted: bool


class CaptureLedger:
    """Crash-safe transition ledger for one expert lifetime session."""

    def __init__(self, root: str | Path, *, session_id: str, lineage_id: str) -> None:
        if not session_id or not lineage_id:
            raise ValueError("capture ledger identity is required")
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.session_id = session_id
        self.lineage_id = lineage_id
        self._manifest = self.root / "transitions.jsonl"
        self._initial: ObservationSignal | None = None
        self._current: ObservationSignal | None = None
        self._records: list[CaptureTransition] = []
        self._chain = "0" * 64
        self.seal_metadata: dict[str, Any] = {}
        if self._manifest.exists():
            raise ValueError("capture ledger resume requires an explicit recovery implementation")

    @classmethod
    def load_sealed(cls, root: str | Path) -> CaptureLedger:
        """Load and fully verify a sealed ledger for export or replay."""

        resolved = Path(root).expanduser().resolve()
        marker = resolved / "sealed.json"
        manifest = resolved / "transitions.jsonl"
        if not marker.is_file() or not manifest.is_file():
            raise ValueError("capture ledger is not sealed")
        seal = json.loads(marker.read_text(encoding="utf-8"))
        instance = cls.__new__(cls)
        instance.root = resolved
        instance.session_id = str(seal["session_id"])
        instance.lineage_id = str(seal["lineage_id"])
        instance._manifest = manifest
        instance._records = []
        instance._chain = "0" * 64
        instance._initial = None
        instance._current = None
        instance.seal_metadata = dict(seal.get("metadata", {}))
        instance._load_verified()
        if (
            int(seal["unit_count"]) != len(instance._records)
            or str(seal["chain_sha256"]) != instance._chain
        ):
            raise ValueError("capture seal does not match the verified ledger")
        return instance

    def _load_verified(self) -> None:
        initial_path = self.root / "observation-000000000000.pb"
        if not initial_path.is_file():
            raise ValueError("capture ledger initial observation is missing")
        initial = message_to_observation(initial_path.read_bytes())
        if initial.session_id != self.session_id or initial.unit_index != 0:
            raise ValueError("capture ledger initial observation identity is invalid")
        self._initial = initial
        self._current = initial
        for expected, line in enumerate(
            self._manifest.read_text(encoding="utf-8").splitlines()
        ):
            item = json.loads(line)
            if (
                str(item["lineage_id"]) != self.lineage_id
                or str(item["session_id"]) != self.session_id
                or int(item["unit_index"]) != expected
            ):
                raise ValueError("capture ledger transition identity is invalid")
            observation_payload = (
                self.root / f"observation-{expected:012d}.pb"
            ).read_bytes()
            next_observation_payload = (
                self.root / f"observation-{expected + 1:012d}.pb"
            ).read_bytes()
            actuation_payload = (self.root / f"actuation-{expected:012d}.pb").read_bytes()
            receipt_payload = (self.root / f"receipt-{expected:012d}.pb").read_bytes()
            hashes = {
                "observation_sha256": _payload_sha256(observation_payload),
                "next_observation_sha256": _payload_sha256(next_observation_payload),
                "actuation_sha256": _payload_sha256(actuation_payload),
                "receipt_sha256": _payload_sha256(receipt_payload),
            }
            if any(str(item[name]) != digest for name, digest in hashes.items()):
                raise ValueError("capture ledger payload hash is invalid")
            identity = {
                "lineage_id": self.lineage_id,
                "session_id": self.session_id,
                "unit_index": expected,
                **hashes,
                "action_frame": item["action_frame"],
                "speech_silent": bool(item["speech_silent"]),
                "receipt_accepted": bool(item["receipt_accepted"]),
            }
            encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
            chain = hashlib.sha256(bytes.fromhex(self._chain) + encoded).hexdigest()
            if (
                str(item["previous_chain_sha256"]) != self._chain
                or str(item["chain_sha256"]) != chain
            ):
                raise ValueError("capture ledger hash chain is invalid")
            actuation = message_to_actuation(actuation_payload)
            receipt = payload_to_receipt(receipt_payload)
            next_observation = message_to_observation(next_observation_payload)
            if (
                actuation.session_id != self.session_id
                or actuation.unit_index != expected
                or receipt.session_id != self.session_id
                or receipt.unit_index != expected
                or next_observation.session_id != self.session_id
                or next_observation.unit_index != expected + 1
                or receipt.accepted != identity["receipt_accepted"]
            ):
                raise ValueError("capture ledger physical identity is invalid")
            record = CaptureTransition(
                session_id=self.session_id,
                unit_index=expected,
                observation_sha256=hashes["observation_sha256"],
                next_observation_sha256=hashes["next_observation_sha256"],
                actuation_sha256=hashes["actuation_sha256"],
                receipt_sha256=hashes["receipt_sha256"],
                previous_chain_sha256=self._chain,
                chain_sha256=chain,
                action_frame=dict(item["action_frame"]),
                speech_silent=bool(item["speech_silent"]),
                receipt_accepted=receipt.accepted,
            )
            self._records.append(record)
            self._chain = chain
            self._current = next_observation

    @property
    def records(self) -> tuple[CaptureTransition, ...]:
        return tuple(self._records)

    @property
    def chain_sha256(self) -> str:
        return self._chain

    @property
    def initial_observation(self) -> ObservationSignal:
        if self._initial is None:
            raise RuntimeError("capture session has not started")
        return self._initial

    @property
    def current_observation(self) -> ObservationSignal:
        if self._current is None:
            raise RuntimeError("capture session has not started")
        return self._current

    def start(self, observation: ObservationSignal) -> None:
        if self._initial is not None:
            raise ValueError("capture session is already started")
        if observation.session_id != self.session_id or observation.unit_index != 0:
            raise ValueError("initial observation identity is invalid")
        if observation.delta_ms != 80:
            raise ValueError("capture observation must use the 80 ms physical clock")
        self._initial = observation
        self._current = observation
        self._write_payload("observation-000000000000.pb", observation_to_payload(observation))

    def append(
        self,
        actuation: ActuationSignal,
        next_observation: ObservationSignal,
        receipt: EnvironmentReceipt,
        action_frame: ContractActionFrame,
    ) -> CaptureTransition:
        if self._current is None:
            raise RuntimeError("capture session has not started")
        expected = len(self._records)
        if actuation.session_id != self.session_id or actuation.unit_index != expected:
            raise ValueError("actuation identity is not contiguous")
        if (
            next_observation.session_id != self.session_id
            or next_observation.unit_index != expected + 1
        ):
            raise ValueError("next observation identity is not contiguous")
        if (
            next_observation.delta_ms != 80
            or next_observation.timestamp_ms <= self._current.timestamp_ms
        ):
            raise ValueError("next observation physical clock is invalid")
        if receipt.session_id != self.session_id or receipt.unit_index != expected:
            raise ValueError("receipt identity is invalid")
        observation_payload = observation_to_payload(self._current)
        actuation_payload = actuation_to_payload(actuation)
        receipt_payload = receipt_to_payload(receipt)
        next_observation_payload = observation_to_payload(next_observation)
        identity = {
            "lineage_id": self.lineage_id,
            "session_id": self.session_id,
            "unit_index": expected,
            "observation_sha256": _payload_sha256(observation_payload),
            "next_observation_sha256": _payload_sha256(next_observation_payload),
            "actuation_sha256": _payload_sha256(actuation_payload),
            "receipt_sha256": _payload_sha256(receipt_payload),
            "action_frame": action_frame_to_dict(action_frame),
            "speech_silent": actuation.speech.silent,
            "receipt_accepted": receipt.accepted,
        }
        encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        chain = hashlib.sha256(bytes.fromhex(self._chain) + encoded).hexdigest()
        record = CaptureTransition(
            self.session_id,
            expected,
            identity["observation_sha256"],
            identity["next_observation_sha256"],
            identity["actuation_sha256"],
            identity["receipt_sha256"],
            self._chain,
            chain,
            identity["action_frame"],
            actuation.speech.silent,
            receipt.accepted,
        )
        self._write_payload(
            f"observation-{expected + 1:012d}.pb", next_observation_payload
        )
        self._write_payload(f"actuation-{expected:012d}.pb", actuation_payload)
        self._write_payload(f"receipt-{expected:012d}.pb", receipt_payload)
        self._append_json({**identity, "previous_chain_sha256": self._chain, "chain_sha256": chain})
        self._records.append(record)
        self._chain = chain
        self._current = next_observation
        return record

    def seal(self, metadata: dict[str, Any] | None = None) -> Path:
        if self._initial is None or not self._records:
            raise ValueError("cannot seal an empty capture session")
        if any(not record.receipt_accepted for record in self._records):
            raise ValueError("cannot seal a capture session containing a rejected receipt")
        marker = self.root / "sealed.json"
        if marker.exists():
            raise ValueError("capture session is already sealed")
        self.seal_metadata = dict(metadata or {})
        payload = {
            "lineage_id": self.lineage_id,
            "session_id": self.session_id,
            "unit_count": len(self._records),
            "chain_sha256": self._chain,
            "metadata": self.seal_metadata,
        }
        self._atomic_json(marker, payload)
        return marker

    def _write_payload(self, name: str, payload: bytes) -> None:
        path = self.root / name
        if path.exists():
            raise ValueError(f"capture payload already exists: {name}")
        with path.open("xb") as target:
            target.write(payload)
            target.flush()
            os.fsync(target.fileno())

    def _append_json(self, value: dict[str, Any]) -> None:
        with self._manifest.open("a", encoding="utf-8") as target:
            target.write(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
            target.flush()
            os.fsync(target.fileno())

    @staticmethod
    def _atomic_json(path: Path, value: dict[str, Any]) -> None:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, mode="w", encoding="utf-8", delete=False
        ) as target:
            temporary = Path(target.name)
            json.dump(value, target, sort_keys=True, separators=(",", ":"))
            target.write("\n")
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)


class ExpertCaptureSession:
    """Drive one Harness session from expert-provided frames and speech."""

    def __init__(self, harness: Any, ledger: CaptureLedger, *, seed: int, snapshot_id: str) -> None:
        self.harness = harness
        self.ledger = ledger
        self.seed = seed
        self.snapshot_id = snapshot_id
        self._pending_utf8 = b""
        self._started = False

    def start(self) -> ObservationSignal:
        observation = self.harness.start_lifetime_session(
            self.snapshot_id, self.seed, self.ledger.session_id
        )
        self.ledger.start(observation)
        self._started = True
        return observation

    def step(
        self, frame: ContractActionFrame, speech: SpeechSignal
    ) -> tuple[ObservationSignal, EnvironmentReceipt]:
        if not self._started:
            raise RuntimeError("capture session has not started")
        unit_index = len(self.ledger.records)
        decoded = decode_action_frame(
            frame,
            event_id=f"{self.ledger.session_id}-unit-{unit_index}",
            pending_utf8=self._pending_utf8,
        )
        self._pending_utf8 = decoded.pending_utf8
        output = ActuationSignal(
            self.ledger.session_id,
            unit_index,
            speech,
            controls=decoded.controls,
        )
        observation, receipt = self.harness.apply(output)
        self.ledger.append(output, observation, receipt, frame)
        if not receipt.accepted:
            failure = receipt.safety_violation or receipt.infrastructure_failure or "rejected"
            raise RuntimeError(f"Harness rejected capture unit {unit_index}: {failure}")
        return observation, receipt

    def finish(self, metadata: dict[str, Any] | None = None) -> Episode:
        if self._pending_utf8:
            raise ValueError("capture ended with an incomplete UTF-8 TYPE frame")
        episode = episode_from_capture(self.ledger, metadata=metadata)
        episode.validate(
            audio_samples=1_920,
            speech_frames=1,
            speech_codebooks=8,
            speech_codebook_size=2_048,
        )
        self.ledger.seal(metadata)
        return episode

    def close(self) -> None:
        self.harness.close(self.ledger.session_id)


def _speech_waveform(signal: SpeechSignal) -> torch.Tensor:
    if signal.silent:
        return torch.zeros(1_920)
    if signal.encoding == "pcm_f32le":
        return torch.from_numpy(np.frombuffer(signal.pcm, dtype="<f4").copy()).float()
    values = np.frombuffer(signal.pcm, dtype="<i2").astype(np.float32) / 32768.0
    return torch.from_numpy(values)


def _mic_waveform(observation: ObservationSignal) -> torch.Tensor:
    if observation.mic.encoding == "pcm_f32le":
        values = np.frombuffer(observation.mic.samples, dtype="<f4").copy()
    else:
        values = np.frombuffer(observation.mic.samples, dtype="<i2").astype(np.float32)
        values /= 32768.0
    if not np.isfinite(values).all() or np.max(np.abs(values)) > 1.0:
        raise ValueError("capture microphone PCM is not normalized and finite")
    return torch.from_numpy(values).float().reshape(1, -1)


def episode_from_capture(
    ledger: CaptureLedger, *, metadata: dict[str, Any] | None = None
) -> Episode:
    initial = ledger.initial_observation
    observations = [initial]
    actions: list[ContractActionFrame] = []
    speech: list[torch.Tensor] = []
    speech_active: list[bool] = []
    for index, record in enumerate(ledger.records):
        observation_path = ledger.root / f"observation-{index + 1:012d}.pb"
        next_observation = message_to_observation(observation_path.read_bytes())
        actuation = message_to_actuation((ledger.root / f"actuation-{index:012d}.pb").read_bytes())
        observations.append(next_observation)
        actions.append(action_frame_from_dict(record.action_frame))
        speech.append(_speech_waveform(actuation.speech))
        speech_active.append(not actuation.speech.silent)
    units: list[StreamUnit] = []
    for _index, (observation, frame, _target, speaking) in enumerate(
        zip(observations[:-1], actions, speech, speech_active, strict=True)
    ):
        image = np.frombuffer(observation.screen.image, dtype=np.uint8)
        channels = 4 if observation.screen.pixel_format == "rgba32" else 3
        screen = torch.from_numpy(
            image.copy()
            .reshape(observation.screen.height, observation.screen.width, channels)[..., :3]
        )
        screen = screen.permute(2, 0, 1).float()[None]
        screen = screen.div(255)
        model_action = ActionFrame(
            kind=torch.tensor([int(frame.kind)]),
            coordinate_cell=torch.tensor([frame.coordinate_cell]),
            coordinate_residual=torch.tensor([frame.coordinate_residual], dtype=torch.float32),
            button=torch.tensor([int(frame.button)]),
            button_phase=torch.tensor([int(frame.button_phase)]),
            scroll_delta=torch.tensor([frame.scroll_delta], dtype=torch.float32),
            text_bytes=torch.tensor([list(frame.text_bytes) + [0] * (16 - len(frame.text_bytes))]),
            text_length=torch.tensor([len(frame.text_bytes)]),
            hotkey_keys=torch.tensor(
                [list(frame.hotkey_keys) + [0] * (8 - len(frame.hotkey_keys))]
            ),
            hotkey_length=torch.tensor([len(frame.hotkey_keys)]),
        )
        units.append(
            StreamUnit(
                timestamp_ms=torch.tensor([observation.timestamp_ms]),
                delta_ms=torch.tensor([observation.delta_ms]),
                mic_audio=_mic_waveform(observation),
                screen=screen,
                speech_mode=torch.tensor(
                    [int(SpeechMode.SPEECH if speaking else SpeechMode.SILENCE)]
                ),
                speech_mode_mask=torch.tensor([True]),
                speech_codes=torch.zeros(1, 1, 8, dtype=torch.long),
                speech_codec_mask=torch.tensor([[speaking]]),
                action=model_action,
                action_supervision_mask=torch.tensor([True]),
            )
        )
    target_speech = torch.cat(speech) if speech else torch.empty(0)
    supplied = metadata or {}
    codec_id = str(supplied.get("codec_id", ""))
    codec_weight_hash = str(supplied.get("codec_weight_hash", ""))
    codec_revision = str(supplied.get("codec_revision", ""))
    if not all((codec_id, codec_weight_hash, codec_revision)):
        raise ValueError("capture export requires complete codec identity metadata")
    environment_id = str(supplied.get("environment_id", ""))
    environment_version = str(supplied.get("environment_version", ""))
    protocol_version = str(supplied.get("protocol_version", ""))
    action_schema_id = str(supplied.get("action_schema_id", ""))
    if not all(
        (environment_id, environment_version, protocol_version, action_schema_id)
    ):
        raise ValueError("capture export requires complete Harness identity metadata")
    if action_schema_id != ACTION_SCHEMA_ID:
        raise ValueError("capture export action schema identity is incompatible")
    base = {
        "stage": "sft",
        "dataset_scale": "canary",
        "sample_kind": "expert_capture",
        "supervision_kind": "speech_action",
        "action_source": "expert",
        "task_id": ledger.session_id,
        "environment_id": environment_id,
        "environment_version": environment_version,
        "protocol_version": protocol_version,
        "action_schema_id": action_schema_id,
        "runtime_identity": {
            "protocol_version": protocol_version,
            "environment_id": environment_id,
            "environment_version": environment_version,
            "action_schema_id": action_schema_id,
        },
        "sample_rate": 24_000,
        "unit_ms": 80,
        "codec_frame_rate": 12.5,
        "codec_id": codec_id,
        "codec_weight_hash": codec_weight_hash,
        "codec_revision": codec_revision,
        "speech_codes_encoded": False,
        "capture_lineage_id": ledger.lineage_id,
        "capture_chain_sha256": ledger.chain_sha256,
    }
    base.update(supplied)
    return Episode(ledger.session_id, units, base, target_speech=target_speech)


__all__ = [
    "CaptureLedger",
    "CaptureTransition",
    "ExpertCaptureSession",
    "action_frame_from_dict",
    "action_frame_to_dict",
    "episode_from_capture",
    "speech_signal_from_base64",
]
