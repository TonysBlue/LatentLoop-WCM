from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import torch
from contracts import (
    HOTKEY_KEYS_PER_UNIT,
    KEY_VOCAB_SIZE,
    TYPE_BYTES_PER_UNIT,
    ActionKind,
    PointerButton,
    PointerButtonPhase,
    decode_action_frame,
)
from contracts import ActionFrame as ContractActionFrame
from torch import Tensor
from torch.utils import _pytree


class SpeechMode(IntEnum):
    SILENCE = 0
    SPEECH = 1


@dataclass(slots=True)
class ActionFrame:
    """Batched tensor representation of one structured action per 80 ms unit."""

    kind: Tensor
    coordinate_cell: Tensor
    coordinate_residual: Tensor
    button: Tensor
    button_phase: Tensor
    scroll_delta: Tensor
    text_bytes: Tensor
    text_length: Tensor
    hotkey_keys: Tensor
    hotkey_length: Tensor

    @property
    def batch_size(self) -> int:
        return self.kind.shape[0]

    @classmethod
    def no_action(cls, batch: int, device: torch.device | str = "cpu") -> ActionFrame:
        return cls(
            kind=torch.full((batch,), int(ActionKind.NO_ACTION), dtype=torch.long, device=device),
            coordinate_cell=torch.zeros(batch, dtype=torch.long, device=device),
            coordinate_residual=torch.zeros(batch, 2, device=device),
            button=torch.zeros(batch, dtype=torch.long, device=device),
            button_phase=torch.zeros(batch, dtype=torch.long, device=device),
            scroll_delta=torch.zeros(batch, 2, device=device),
            text_bytes=torch.zeros(
                batch, TYPE_BYTES_PER_UNIT, dtype=torch.long, device=device
            ),
            text_length=torch.zeros(batch, dtype=torch.long, device=device),
            hotkey_keys=torch.zeros(
                batch, HOTKEY_KEYS_PER_UNIT, dtype=torch.long, device=device
            ),
            hotkey_length=torch.zeros(batch, dtype=torch.long, device=device),
        )

    def to(
        self,
        device: torch.device | str,
        dtype: torch.dtype | None = None,
    ) -> ActionFrame:
        return ActionFrame(
            **{
                name: value.to(
                    device=device,
                    dtype=dtype if dtype is not None and value.is_floating_point() else value.dtype,
                )
                for name, value in self.items()
            }
        )

    def detach(self) -> ActionFrame:
        return ActionFrame(**{name: value.detach() for name, value in self.items()})

    def cpu(self) -> ActionFrame:
        return self.to("cpu")

    def items(self) -> tuple[tuple[str, Tensor], ...]:
        return (
            ("kind", self.kind),
            ("coordinate_cell", self.coordinate_cell),
            ("coordinate_residual", self.coordinate_residual),
            ("button", self.button),
            ("button_phase", self.button_phase),
            ("scroll_delta", self.scroll_delta),
            ("text_bytes", self.text_bytes),
            ("text_length", self.text_length),
            ("hotkey_keys", self.hotkey_keys),
            ("hotkey_length", self.hotkey_length),
        )

    def validate(self) -> None:
        batch = self.batch_size
        expected = {
            "kind": (batch,),
            "coordinate_cell": (batch,),
            "coordinate_residual": (batch, 2),
            "button": (batch,),
            "button_phase": (batch,),
            "scroll_delta": (batch, 2),
            "text_bytes": (batch, TYPE_BYTES_PER_UNIT),
            "text_length": (batch,),
            "hotkey_keys": (batch, HOTKEY_KEYS_PER_UNIT),
            "hotkey_length": (batch,),
        }
        for name, value in self.items():
            if value.shape != expected[name]:
                actual_shape = tuple(value.shape)
                raise ValueError(
                    f"action {name} has shape {actual_shape}, expected {expected[name]}"
                )
        if torch.any((self.kind < 0) | (self.kind >= len(ActionKind))):
            raise ValueError("action kind is outside the structured schema")
        if torch.any((self.coordinate_cell < 0) | (self.coordinate_cell >= 32**2)):
            raise ValueError("action coordinate cell is outside the 32x32 grid")
        if torch.any((self.coordinate_residual < 0) | (self.coordinate_residual > 1)):
            raise ValueError("action coordinate residual must be in [0, 1]")
        if torch.any((self.button < 0) | (self.button >= 3)):
            raise ValueError("action button is outside the schema")
        if torch.any((self.button_phase < 0) | (self.button_phase >= 3)):
            raise ValueError("action button phase is outside the schema")
        if torch.any((self.scroll_delta < -1) | (self.scroll_delta > 1)):
            raise ValueError("action scroll delta must be in [-1, 1]")
        if torch.any((self.text_length < 0) | (self.text_length > TYPE_BYTES_PER_UNIT)):
            raise ValueError("action text length is outside the per-unit limit")
        if torch.any((self.text_bytes < 0) | (self.text_bytes > 255)):
            raise ValueError("action text bytes must be uint8 values")
        if torch.any((self.hotkey_length < 0) | (self.hotkey_length > HOTKEY_KEYS_PER_UNIT)):
            raise ValueError("action hotkey length is outside the per-unit limit")
        if torch.any((self.hotkey_keys < 0) | (self.hotkey_keys >= KEY_VOCAB_SIZE)):
            raise ValueError("action hotkey key is outside the configured table")
        is_type = self.kind.eq(int(ActionKind.TYPE))
        is_hotkey = self.kind.eq(int(ActionKind.HOTKEY))
        if torch.any(is_type & self.text_length.eq(0)):
            raise ValueError("TYPE frame requires at least one byte")
        if torch.any(is_hotkey & self.hotkey_length.eq(0)):
            raise ValueError("HOTKEY frame requires at least one key")
        positions = torch.arange(HOTKEY_KEYS_PER_UNIT, device=self.kind.device)[None]
        active_keys = positions < self.hotkey_length[:, None]
        for index in range(batch):
            keys = self.hotkey_keys[index][active_keys[index]].tolist()
            if len(set(keys)) != len(keys):
                raise ValueError("HOTKEY keys must be unique")
        move = self.kind.eq(int(ActionKind.POINTER_MOVE))
        pointer_button = self.kind.eq(int(ActionKind.POINTER_BUTTON))
        scroll = self.kind.eq(int(ActionKind.SCROLL))
        if torch.any(~move & self.coordinate_cell.ne(0)) or torch.any(
            ~move[:, None] & self.coordinate_residual.ne(0)
        ):
            raise ValueError("only POINTER_MOVE accepts coordinate parameters")
        if torch.any(~pointer_button & self.button.ne(0)) or torch.any(
            ~pointer_button & self.button_phase.ne(0)
        ):
            raise ValueError("only POINTER_BUTTON accepts button parameters")
        if torch.any(~scroll[:, None] & self.scroll_delta.ne(0)):
            raise ValueError("only SCROLL accepts scroll parameters")
        if torch.any(~is_type & self.text_length.ne(0)):
            raise ValueError("only TYPE accepts text bytes")
        if torch.any(~is_hotkey & self.hotkey_length.ne(0)):
            raise ValueError("only HOTKEY accepts keys")

    def as_contract(self, index: int = 0) -> ContractActionFrame:
        kind = ActionKind(int(self.kind[index].item()))
        text_length = int(self.text_length[index].item()) if kind is ActionKind.TYPE else 0
        key_length = int(self.hotkey_length[index].item()) if kind is ActionKind.HOTKEY else 0
        return ContractActionFrame(
            kind=kind,
            coordinate_cell=(
                int(self.coordinate_cell[index].item())
                if kind is ActionKind.POINTER_MOVE
                else 0
            ),
            coordinate_residual=(
                tuple(
                    float(value)
                    for value in self.coordinate_residual[index].detach().cpu().tolist()
                )
                if kind is ActionKind.POINTER_MOVE
                else (0.0, 0.0)
            ),
            button=(
                PointerButton(int(self.button[index].item()))
                if kind is ActionKind.POINTER_BUTTON
                else PointerButton.LEFT
            ),
            button_phase=(
                PointerButtonPhase(int(self.button_phase[index].item()))
                if kind is ActionKind.POINTER_BUTTON
                else PointerButtonPhase.CLICK
            ),
            scroll_delta=(
                tuple(
                    float(value)
                    for value in self.scroll_delta[index].detach().cpu().tolist()
                )
                if kind is ActionKind.SCROLL
                else (0.0, 0.0)
            ),
            text_bytes=bytes(
                int(value)
                for value in self.text_bytes[index, :text_length].detach().cpu().tolist()
            ),
            hotkey_keys=tuple(
                int(value)
                for value in self.hotkey_keys[index, :key_length].detach().cpu().tolist()
            ),
        )


@dataclass(slots=True)
class StreamUnit:
    """One batched, time-aligned multimodal training unit."""

    timestamp_ms: Tensor
    delta_ms: Tensor
    mic_audio: Tensor
    screen: Tensor
    speech_mode: Tensor
    speech_mode_mask: Tensor
    speech_codes: Tensor
    speech_codec_mask: Tensor
    action: ActionFrame
    action_supervision_mask: Tensor

    @property
    def batch_size(self) -> int:
        return self.mic_audio.shape[0]

    def to(
        self,
        device: torch.device | str,
        dtype: torch.dtype | None = None,
    ) -> StreamUnit:
        return StreamUnit(
            timestamp_ms=self.timestamp_ms.to(device),
            delta_ms=self.delta_ms.to(device),
            mic_audio=self.mic_audio.to(device=device, dtype=dtype),
            screen=self.screen.to(device=device, dtype=dtype),
            speech_mode=self.speech_mode.to(device),
            speech_mode_mask=self.speech_mode_mask.to(device),
            speech_codes=self.speech_codes.to(device),
            speech_codec_mask=self.speech_codec_mask.to(device),
            action=self.action.to(device, dtype=dtype),
            action_supervision_mask=self.action_supervision_mask.to(device),
        )

    def validate(
        self,
        *,
        audio_samples: int,
        speech_frames: int,
        speech_codebooks: int,
        speech_codebook_size: int = 2**16,
        **_: int,
    ) -> None:
        batch = self.batch_size
        if self.mic_audio.shape != (batch, audio_samples):
            raise ValueError(f"mic_audio must have shape [B, {audio_samples}]")
        if self.screen.shape != (batch, 3, 224, 224):
            raise ValueError("screen must have shape [B, 3, 224, 224]")
        if self.speech_mode.shape != (batch,) or self.speech_mode_mask.shape != (batch,):
            raise ValueError("speech mode tensors must have shape [B]")
        if torch.any((self.speech_mode < 0) | (self.speech_mode > 1)):
            raise ValueError("speech mode must be SILENCE or SPEECH")
        if self.speech_codes.shape != (batch, speech_frames, speech_codebooks):
            raise ValueError("speech_codes has an incompatible shape")
        if self.speech_codec_mask.shape != (batch, speech_frames):
            raise ValueError("speech_codec_mask has an incompatible shape")
        valid_codes = self.speech_codes[self.speech_codec_mask]
        if valid_codes.numel() and (
            valid_codes.min() < 0 or valid_codes.max() >= speech_codebook_size
        ):
            raise ValueError("speech code is outside the configured codebook")
        self.action.validate()
        if self.action.batch_size != batch or self.action_supervision_mask.shape != (batch,):
            raise ValueError("action frame and supervision mask must have batch shape [B]")
        if torch.any(self.delta_ms <= 0):
            raise ValueError("delta_ms must be positive")


@dataclass(slots=True)
class Episode:
    episode_id: str
    units: list[StreamUnit]
    metadata: dict[str, Any]
    target_speech: Tensor | None = None
    ordered_shard_index: int = 0
    sample_index_in_shard: int = 0

    def validate(self, **unit_shape: int) -> None:
        if not self.units:
            raise ValueError("episode must contain at least one stream unit")
        previous = -1
        pending_utf8 = [b""] * self.units[0].batch_size
        held_buttons = [set() for _ in range(self.units[0].batch_size)]
        for unit in self.units:
            unit.validate(**unit_shape)
            timestamp = int(unit.timestamp_ms[0].item())
            if timestamp <= previous:
                raise ValueError("episode timestamps must be strictly increasing")
            previous = timestamp
            for index in range(unit.batch_size):
                if not bool(unit.action_supervision_mask[index]):
                    if pending_utf8[index]:
                        raise ValueError(
                            "TYPE continuation cannot cross missing action supervision"
                        )
                    if held_buttons[index]:
                        raise ValueError(
                            "held pointer button cannot cross missing action supervision"
                        )
                    continue
                frame = unit.action.as_contract(index)
                result = decode_action_frame(
                    frame,
                    event_id=f"{self.episode_id}-{timestamp}-{index}",
                    pending_utf8=pending_utf8[index],
                )
                pending_utf8[index] = result.pending_utf8
                if frame.kind is ActionKind.POINTER_BUTTON:
                    button = int(frame.button)
                    if frame.button_phase is PointerButtonPhase.DOWN:
                        if button in held_buttons[index]:
                            raise ValueError("pointer button is already held")
                        held_buttons[index].add(button)
                    elif frame.button_phase is PointerButtonPhase.UP:
                        if button not in held_buttons[index]:
                            raise ValueError("pointer button is not held")
                        held_buttons[index].remove(button)
                    elif button in held_buttons[index]:
                        raise ValueError("held pointer button requires an UP frame")
        if any(pending_utf8):
            raise ValueError("episode ends with incomplete UTF-8 action bytes")
        if self.target_speech is not None:
            expected = len(self.units) * unit_shape["audio_samples"]
            if self.target_speech.numel() != expected:
                raise ValueError("target_speech must exactly cover the episode timeline")


@dataclass(slots=True)
class LayerKV:
    key: Tensor
    value: Tensor

    def detach(self) -> LayerKV:
        return LayerKV(self.key.detach(), self.value.detach())


@dataclass(slots=True)
class SlowMemoryState:
    """Fixed-capacity per-layer associative memory."""

    matrix: Tensor
    normalizer: Tensor

    def detach(self) -> SlowMemoryState:
        return SlowMemoryState(self.matrix.detach(), self.normalizer.detach())


@dataclass(slots=True)
class SpeechLocalState:
    temporal: Tensor
    previous_codes: Tensor

    def detach(self) -> SpeechLocalState:
        return SpeechLocalState(self.temporal.detach(), self.previous_codes.detach())


@dataclass(slots=True)
class ActionLocalState:
    previous_frame_embedding: Tensor
    type_decoder_state: Tensor
    pending_utf8_bytes: Tensor
    pending_utf8_length: Tensor
    type_active: Tensor
    held_buttons: Tensor
    held_keys: Tensor

    def detach(self) -> ActionLocalState:
        return ActionLocalState(
            previous_frame_embedding=self.previous_frame_embedding.detach(),
            type_decoder_state=self.type_decoder_state.detach(),
            pending_utf8_bytes=self.pending_utf8_bytes.detach(),
            pending_utf8_length=self.pending_utf8_length.detach(),
            type_active=self.type_active.detach(),
            held_buttons=self.held_buttons.detach(),
            held_keys=self.held_keys.detach(),
        )


@dataclass(slots=True)
class RecurrentState:
    layer_kv: tuple[LayerKV, ...]
    semantic_memory: Tensor
    slow_memory: tuple[SlowMemoryState, ...]
    audio_cache: Tensor
    hidden: Tensor
    speech_local: SpeechLocalState
    action_local: ActionLocalState
    unit_index: Tensor

    def detach(self) -> RecurrentState:
        return RecurrentState(
            layer_kv=tuple(cache.detach() for cache in self.layer_kv),
            semantic_memory=self.semantic_memory.detach(),
            slow_memory=tuple(memory.detach() for memory in self.slow_memory),
            audio_cache=self.audio_cache.detach(),
            hidden=self.hidden.detach(),
            speech_local=self.speech_local.detach(),
            action_local=self.action_local.detach(),
            unit_index=self.unit_index.detach(),
        )


@dataclass(slots=True)
class ActionHeadOutput:
    frame: ActionFrame
    kind_logits: Tensor
    coordinate_cell_logits: Tensor
    coordinate_residual_alpha: Tensor
    coordinate_residual_beta: Tensor
    button_logits: Tensor
    button_phase_logits: Tensor
    scroll_alpha: Tensor
    scroll_beta: Tensor
    text_length_logits: Tensor
    text_byte_logits: Tensor
    hotkey_length_logits: Tensor
    hotkey_key_logits: Tensor


@dataclass(slots=True)
class StepOutput:
    state: RecurrentState
    speech_mode_logits: Tensor
    speech_codec_logits: Tensor
    action: ActionHeadOutput
    hidden: Tensor
    perceiver_slots: Tensor
    jepa_prediction: Tensor
    value: Tensor
    selected_speech_mode: Tensor

    def sampled_logprob(
        self, speech_mode: Tensor, speech_codes: Tensor, action_frame: ActionFrame
    ) -> Tensor:
        from model.action import action_frame_log_prob

        mode = torch.log_softmax(self.speech_mode_logits, dim=-1).gather(
            -1, speech_mode.unsqueeze(-1)
        ).squeeze(-1)
        codec = torch.log_softmax(self.speech_codec_logits, dim=-1).gather(
            -1, speech_codes.unsqueeze(-1)
        ).squeeze(-1)
        action = action_frame_log_prob(self.action, action_frame)
        return torch.cat(
            (mode.reshape(mode.shape[0], -1), codec.reshape(codec.shape[0], -1), action[:, None]),
            dim=-1,
        )


@dataclass(slots=True)
class TrainingStepOutput:
    """Per-unit differentiable outputs without a retained recurrent state."""

    speech_mode_logits: Tensor
    speech_codec_logits: Tensor
    action: ActionHeadOutput
    perceiver_slots: Tensor
    jepa_prediction: Tensor
    value: Tensor


def training_output(output: StepOutput) -> TrainingStepOutput:
    return TrainingStepOutput(
        speech_mode_logits=output.speech_mode_logits,
        speech_codec_logits=output.speech_codec_logits,
        action=output.action,
        perceiver_slots=output.perceiver_slots,
        jepa_prediction=output.jepa_prediction,
        value=output.value,
    )


@dataclass(frozen=True, slots=True)
class SpeechSamplingConfig:
    temperature: float = 0.8
    top_k: int = 250
    greedy: bool = False


@dataclass(slots=True)
class GenerationOutput:
    output: StepOutput
    speech_mode: Tensor
    speech_codes: Tensor
    action_frame: ActionFrame


# Activation checkpointing must see every tensor nested in recurrent and step
# dataclasses; otherwise recomputation silently drops parts of the autograd graph.
for _checkpoint_dataclass in (
    ActionFrame,
    LayerKV,
    SlowMemoryState,
    SpeechLocalState,
    ActionLocalState,
    RecurrentState,
    ActionHeadOutput,
    StepOutput,
    TrainingStepOutput,
):
    _pytree.register_dataclass(_checkpoint_dataclass)
