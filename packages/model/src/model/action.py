from __future__ import annotations

import math

import torch
from contracts import (
    HOTKEY_KEYS_PER_UNIT,
    KEY_VOCAB_SIZE,
    TYPE_BYTES_PER_UNIT,
    ActionKind,
    PointerButtonPhase,
)
from runtime.config import ModelConfig
from torch import Tensor, nn
from torch.distributions import Beta
from torch.nn import functional as F

from model.types import ActionFrame, ActionHeadOutput, ActionLocalState


def _categorical(logits: Tensor, temperature: float | None) -> Tensor:
    if temperature is None or temperature <= 0:
        return logits.argmax(dim=-1)
    probabilities = torch.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probabilities, 1).squeeze(-1)


def _beta_parameters(raw: Tensor) -> tuple[Tensor, Tensor]:
    raw = torch.nan_to_num(raw, nan=0.0, posinf=20.0, neginf=-20.0)
    alpha, beta = raw.unbind(dim=-1)
    return F.softplus(alpha) + 1.0, F.softplus(beta) + 1.0


def _beta_value(alpha: Tensor, beta: Tensor, sample: bool) -> Tensor:
    if sample:
        return Beta(alpha, beta).sample()
    return alpha / (alpha + beta)


def _advance_utf8(pending: bytes, value: int) -> bytes | None:
    candidate = pending + bytes((value,))
    try:
        candidate.decode("utf-8")
        return b""
    except UnicodeDecodeError as error:
        if error.reason == "unexpected end of data" and len(candidate) - error.start <= 3:
            return candidate[error.start:]
        return None


def _pending_from_state(state: ActionLocalState, index: int) -> bytes:
    length = int(state.pending_utf8_length[index].item())
    return bytes(int(value) for value in state.pending_utf8_bytes[index, :length].tolist())


class ActionHead(nn.Module):
    """One kind-conditioned structured action distribution per stream unit."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        dim = config.model_dim
        self.kind_count = len(ActionKind)
        self.context = nn.Sequential(nn.Linear(dim * 2, dim), nn.GELU(), nn.Linear(dim, dim))
        self.state_query = nn.Parameter(torch.zeros(1, dim))
        self.spatial_queries = nn.Parameter(torch.zeros(16, dim))
        self.query_attention = nn.MultiheadAttention(
            dim, config.num_heads, dropout=config.dropout, batch_first=True
        )
        self.query_norm = nn.LayerNorm(dim)
        self.visual_grid_size = 4
        self.local_coordinate_grid_size = 8
        self.kind_output = nn.Linear(dim, self.kind_count)
        self.coordinate_cell_output = nn.Linear(
            dim * 2, self.local_coordinate_grid_size**2
        )
        self.coordinate_residual_output = nn.Linear(dim * 2, 2 * 2)
        self.button_output = nn.Linear(dim, 3)
        self.button_phase_output = nn.Linear(dim, 3)
        self.scroll_output = nn.Linear(dim, 2 * 2)
        self.text_length_output = nn.Linear(dim, config.action_type_bytes_per_unit + 1)
        self.hotkey_length_output = nn.Linear(dim, config.action_hotkey_keys_per_unit + 1)

        self.kind_embedding = nn.Embedding(self.kind_count, dim)
        self.cell_embedding = nn.Embedding(config.action_coordinate_grid_size**2, dim)
        self.button_embedding = nn.Embedding(3, dim)
        self.button_phase_embedding = nn.Embedding(3, dim)
        self.byte_embedding = nn.Embedding(256, dim)
        self.key_embedding = nn.Embedding(KEY_VOCAB_SIZE, dim)
        self.continuous_embedding = nn.Linear(4, dim)
        self.frame_norm = nn.LayerNorm(dim)

        self.type_start = nn.Linear(dim, dim)
        self.type_decoder = nn.GRUCell(dim, dim)
        self.type_output = nn.Linear(dim, 256)
        self.hotkey_start = nn.Linear(dim, dim)
        self.hotkey_decoder = nn.GRUCell(dim, dim)
        self.hotkey_output = nn.Linear(dim, KEY_VOCAB_SIZE)
        self.type_bytes_per_unit = config.action_type_bytes_per_unit
        self.hotkey_keys_per_unit = config.action_hotkey_keys_per_unit
        nn.init.normal_(self.state_query, std=0.02)
        nn.init.normal_(self.spatial_queries, std=0.02)

    def initial_state(
        self, batch: int, device: torch.device, dtype: torch.dtype
    ) -> ActionLocalState:
        dim = self.kind_embedding.embedding_dim
        return ActionLocalState(
            previous_frame_embedding=torch.zeros(batch, dim, device=device, dtype=dtype),
            type_decoder_state=torch.zeros(batch, dim, device=device, dtype=dtype),
            pending_utf8_bytes=torch.zeros(batch, 3, device=device, dtype=torch.long),
            pending_utf8_length=torch.zeros(batch, device=device, dtype=torch.long),
            type_active=torch.zeros(batch, device=device, dtype=torch.bool),
            held_buttons=torch.zeros(batch, 3, device=device, dtype=torch.bool),
            held_keys=torch.zeros(batch, KEY_VOCAB_SIZE, device=device, dtype=torch.bool),
        )

    def _decode_text(
        self,
        context: Tensor,
        state: ActionLocalState,
        teacher: ActionFrame | None,
        teacher_mask: Tensor | None,
        temperature: float | None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        batch = context.shape[0]
        current = torch.where(
            state.type_active[:, None], state.type_decoder_state, self.type_start(context)
        )
        previous_embedding = torch.zeros_like(context)
        logits: list[Tensor] = []
        values: list[Tensor] = []
        states: list[Tensor] = []
        pending = [_pending_from_state(state, index) for index in range(batch)]
        for index in range(self.type_bytes_per_unit):
            current = self.type_decoder(previous_embedding + context, current)
            current_logits = torch.nan_to_num(
                self.type_output(current), nan=0.0, posinf=30.0, neginf=-30.0
            )
            constrained = current_logits.clone()
            for batch_index in range(batch):
                allowed = [
                    value
                    for value in range(256)
                    if _advance_utf8(pending[batch_index], value) is not None
                ]
                mask = torch.ones(256, dtype=torch.bool, device=context.device)
                mask[allowed] = False
                constrained[batch_index].masked_fill_(mask, -torch.inf)
            logits.append(constrained)
            sampled = _categorical(constrained, temperature)
            use_teacher = (
                teacher_mask & teacher.text_length.gt(index)
                if teacher is not None and teacher_mask is not None
                else None
            )
            value = (
                torch.where(use_teacher, teacher.text_bytes[:, index], sampled)
                if teacher is not None and teacher_mask is not None
                else sampled
            )
            for batch_index, byte_value in enumerate(value.tolist()):
                next_pending = _advance_utf8(pending[batch_index], int(byte_value))
                if next_pending is None:
                    raise ValueError("TYPE target contains invalid UTF-8")
                pending[batch_index] = next_pending
            values.append(value)
            states.append(current)
            previous_embedding = self.byte_embedding(value)
        return (
            torch.stack(logits, dim=1),
            torch.stack(values, dim=1),
            torch.stack(states, dim=1),
            context,
        )

    def _decode_hotkey(
        self,
        context: Tensor,
        teacher: ActionFrame | None,
        teacher_mask: Tensor | None,
        temperature: float | None,
    ) -> tuple[Tensor, Tensor]:
        current = self.hotkey_start(context)
        previous_embedding = torch.zeros_like(context)
        logits: list[Tensor] = []
        values: list[Tensor] = []
        selected = torch.zeros(
            context.shape[0], KEY_VOCAB_SIZE, dtype=torch.bool, device=context.device
        )
        for index in range(self.hotkey_keys_per_unit):
            current = self.hotkey_decoder(previous_embedding + context, current)
            current_logits = torch.nan_to_num(
                self.hotkey_output(current), nan=0.0, posinf=30.0, neginf=-30.0
            )
            constrained = current_logits.masked_fill(selected.clone(), -torch.inf)
            logits.append(constrained)
            sampled = _categorical(constrained, temperature)
            use_teacher = (
                teacher_mask & teacher.hotkey_length.gt(index)
                if teacher is not None and teacher_mask is not None
                else None
            )
            value = (
                torch.where(use_teacher, teacher.hotkey_keys[:, index], sampled)
                if teacher is not None and teacher_mask is not None
                else sampled
            )
            if torch.any(selected.gather(1, value[:, None]).squeeze(1)):
                raise ValueError("HOTKEY target contains duplicate keys")
            selected.scatter_(1, value[:, None], True)
            values.append(value)
            previous_embedding = self.key_embedding(value)
        return torch.stack(logits, dim=1), torch.stack(values, dim=1)

    def _next_pending(
        self, frame: ActionFrame, state: ActionLocalState
    ) -> tuple[Tensor, Tensor]:
        next_bytes = torch.zeros_like(state.pending_utf8_bytes)
        next_lengths = torch.zeros_like(state.pending_utf8_length)
        for index in range(frame.batch_size):
            pending = _pending_from_state(state, index)
            if int(frame.kind[index].item()) != int(ActionKind.TYPE):
                if pending:
                    raise ValueError("cannot switch from TYPE with incomplete UTF-8 bytes")
                continue
            length = int(frame.text_length[index].item())
            for value in frame.text_bytes[index, :length].tolist():
                advanced = _advance_utf8(pending, int(value))
                if advanced is None:
                    raise ValueError("TYPE target contains invalid UTF-8")
                pending = advanced
            next_lengths[index] = len(pending)
            if pending:
                next_bytes[index, : len(pending)] = torch.tensor(
                    tuple(pending), dtype=torch.long, device=frame.kind.device
                )
        return next_bytes, next_lengths

    def _frame_embedding(self, frame: ActionFrame) -> Tensor:
        batch = frame.batch_size
        embedding = self.kind_embedding(frame.kind)
        move = frame.kind.eq(int(ActionKind.POINTER_MOVE))[:, None]
        pointer = self.cell_embedding(frame.coordinate_cell) + self.continuous_embedding(
            torch.cat((frame.coordinate_residual, torch.zeros_like(frame.coordinate_residual)), -1)
        )
        embedding = embedding + move * pointer
        button = frame.kind.eq(int(ActionKind.POINTER_BUTTON))[:, None]
        button_value = self.button_embedding(frame.button) + self.button_phase_embedding(
            frame.button_phase
        )
        embedding = embedding + button * button_value
        scroll = frame.kind.eq(int(ActionKind.SCROLL))[:, None]
        scroll_value = self.continuous_embedding(
            torch.cat((torch.zeros_like(frame.scroll_delta), frame.scroll_delta), -1)
        )
        embedding = embedding + scroll * scroll_value
        positions = torch.arange(TYPE_BYTES_PER_UNIT, device=frame.kind.device)[None]
        text_mask = positions < frame.text_length[:, None]
        text_value = (self.byte_embedding(frame.text_bytes) * text_mask[:, :, None]).sum(dim=1)
        text_value = text_value / frame.text_length.clamp_min(1)[:, None]
        embedding = embedding + frame.kind.eq(int(ActionKind.TYPE))[:, None] * text_value
        key_positions = torch.arange(HOTKEY_KEYS_PER_UNIT, device=frame.kind.device)[None]
        key_mask = key_positions < frame.hotkey_length[:, None]
        key_value = (self.key_embedding(frame.hotkey_keys) * key_mask[:, :, None]).sum(dim=1)
        key_value = key_value / frame.hotkey_length.clamp_min(1)[:, None]
        embedding = embedding + frame.kind.eq(int(ActionKind.HOTKEY))[:, None] * key_value
        return self.frame_norm(embedding.reshape(batch, -1))

    def forward(
        self,
        hidden: Tensor,
        state: ActionLocalState,
        teacher_frame: ActionFrame | None = None,
        teacher_mask: Tensor | None = None,
        sampling_temperature: float | None = None,
    ) -> tuple[ActionHeadOutput, ActionLocalState]:
        queries = torch.cat((self.state_query, self.spatial_queries), dim=0)
        queries = queries[None].expand(hidden.shape[0], -1, -1)
        queried, _ = self.query_attention(queries, hidden, hidden, need_weights=False)
        queried = self.query_norm(queried)
        state_query = queried[:, 0]
        spatial_hidden = queried[:, 1:]
        recurrent_context = torch.cat(
            (state_query, state.previous_frame_embedding), dim=-1
        )
        context = torch.nan_to_num(
            self.context(recurrent_context),
            nan=0.0,
            posinf=30.0,
            neginf=-30.0,
        )
        raw_kind_logits = torch.nan_to_num(
            self.kind_output(context), nan=0.0, posinf=30.0, neginf=-30.0
        )
        has_pending = state.pending_utf8_length.gt(0)
        kind_logits = raw_kind_logits.clone()
        kind_logits[has_pending] = -torch.inf
        kind_logits[has_pending, int(ActionKind.TYPE)] = 0.0
        if teacher_frame is not None:
            if teacher_mask is None:
                teacher_mask = torch.ones_like(has_pending)
            teacher_mask = teacher_mask.to(dtype=torch.bool, device=hidden.device)
            if teacher_mask.shape != has_pending.shape:
                raise ValueError("action teacher mask must have shape [B]")
            if torch.any(
                teacher_mask & has_pending & teacher_frame.kind.ne(int(ActionKind.TYPE))
            ):
                raise ValueError("incomplete UTF-8 requires a following TYPE frame")
            sampled_kind = _categorical(kind_logits, sampling_temperature)
            frame_kind = torch.where(teacher_mask, teacher_frame.kind, sampled_kind)
        else:
            frame_kind = _categorical(kind_logits, sampling_temperature)

        position_context = context[:, None].expand(-1, self.visual_grid_size**2, -1)
        local_cell_logits = self.coordinate_cell_output(
            torch.cat((position_context, spatial_hidden), dim=-1)
        )
        cell_logits = (
            local_cell_logits.view(
                -1,
                self.visual_grid_size,
                self.visual_grid_size,
                self.local_coordinate_grid_size,
                self.local_coordinate_grid_size,
            )
            .permute(0, 1, 3, 2, 4)
            .reshape(-1, (self.visual_grid_size * self.local_coordinate_grid_size) ** 2)
        )
        cell_logits = torch.nan_to_num(cell_logits, nan=0.0, posinf=30.0, neginf=-30.0)
        sampled_cell = _categorical(cell_logits, sampling_temperature)
        selected_cell = (
            torch.where(teacher_mask, teacher_frame.coordinate_cell, sampled_cell)
            if teacher_frame is not None and teacher_mask is not None
            else sampled_cell
        )
        cell_x = selected_cell % (self.visual_grid_size * self.local_coordinate_grid_size)
        cell_y = selected_cell // (self.visual_grid_size * self.local_coordinate_grid_size)
        visual_index = (cell_y // self.local_coordinate_grid_size) * self.visual_grid_size + (
            cell_x // self.local_coordinate_grid_size
        )
        selected_spatial = spatial_hidden.gather(
            1, visual_index[:, None, None].expand(-1, 1, spatial_hidden.shape[-1])
        ).squeeze(1)
        coordinate_raw = self.coordinate_residual_output(
            torch.cat((context, selected_spatial), dim=-1)
        ).view(-1, 2, 2)
        coordinate_alpha, coordinate_beta = _beta_parameters(coordinate_raw)
        button_logits = torch.nan_to_num(
            self.button_output(context), nan=0.0, posinf=30.0, neginf=-30.0
        )
        sampled_button = _categorical(button_logits, sampling_temperature)
        selected_button = (
            torch.where(teacher_mask, teacher_frame.button, sampled_button)
            if teacher_frame is not None and teacher_mask is not None
            else sampled_button
        )
        phase_logits = torch.nan_to_num(
            self.button_phase_output(context), nan=0.0, posinf=30.0, neginf=-30.0
        )
        selected_held = state.held_buttons.gather(1, selected_button[:, None]).squeeze(1)
        phase_logits = phase_logits.clone()
        phase_logits[selected_held, : int(PointerButtonPhase.UP)] = -torch.inf
        phase_logits[~selected_held, int(PointerButtonPhase.UP)] = -torch.inf
        selected_phase = _categorical(phase_logits, sampling_temperature)
        scroll_raw = self.scroll_output(context).view(-1, 2, 2)
        scroll_alpha, scroll_beta = _beta_parameters(scroll_raw)
        text_length_logits = torch.nan_to_num(
            self.text_length_output(context), nan=0.0, posinf=30.0, neginf=-30.0
        )
        text_length_logits = text_length_logits.clone()
        text_length_logits[:, 0] = -torch.inf
        hotkey_length_logits = torch.nan_to_num(
            self.hotkey_length_output(context), nan=0.0, posinf=30.0, neginf=-30.0
        )
        hotkey_length_logits = hotkey_length_logits.clone()
        hotkey_length_logits[:, 0] = -torch.inf
        text_logits, text_values, type_states, _ = self._decode_text(
            context, state, teacher_frame, teacher_mask, sampling_temperature
        )
        hotkey_logits, hotkey_values = self._decode_hotkey(
            context, teacher_frame, teacher_mask, sampling_temperature
        )

        sample_continuous = sampling_temperature is not None and sampling_temperature > 0
        sampled_frame = ActionFrame(
            kind=frame_kind,
            coordinate_cell=sampled_cell,
            coordinate_residual=_beta_value(
                coordinate_alpha, coordinate_beta, sample_continuous
            ),
            button=selected_button,
            button_phase=selected_phase,
            scroll_delta=(
                _beta_value(scroll_alpha, scroll_beta, sample_continuous) * 2.0 - 1.0
            ),
            text_bytes=text_values,
            text_length=_categorical(text_length_logits, sampling_temperature),
            hotkey_keys=hotkey_values,
            hotkey_length=_categorical(hotkey_length_logits, sampling_temperature),
        )
        if teacher_frame is not None and teacher_mask is not None:
            fields: dict[str, Tensor] = {}
            for name, sampled_value in sampled_frame.items():
                teacher_value = dict(teacher_frame.items())[name]
                expanded_mask = teacher_mask.reshape(
                    teacher_mask.shape + (1,) * (sampled_value.ndim - 1)
                )
                fields[name] = torch.where(expanded_mask, teacher_value, sampled_value)
            frame = ActionFrame(**fields)
        else:
            frame = sampled_frame
        move = frame.kind.eq(int(ActionKind.POINTER_MOVE))
        pointer = frame.kind.eq(int(ActionKind.POINTER_BUTTON))
        scroll_kind = frame.kind.eq(int(ActionKind.SCROLL))
        type_kind = frame.kind.eq(int(ActionKind.TYPE))
        hotkey_kind = frame.kind.eq(int(ActionKind.HOTKEY))
        frame = ActionFrame(
            kind=frame.kind,
            coordinate_cell=torch.where(
                move, frame.coordinate_cell, torch.zeros_like(frame.coordinate_cell)
            ),
            coordinate_residual=torch.where(
                move[:, None],
                frame.coordinate_residual,
                torch.zeros_like(frame.coordinate_residual),
            ),
            button=torch.where(pointer, frame.button, torch.zeros_like(frame.button)),
            button_phase=torch.where(
                pointer, frame.button_phase, torch.zeros_like(frame.button_phase)
            ),
            scroll_delta=torch.where(
                scroll_kind[:, None], frame.scroll_delta, torch.zeros_like(frame.scroll_delta)
            ),
            text_bytes=torch.where(
                type_kind[:, None], frame.text_bytes, torch.zeros_like(frame.text_bytes)
            ),
            text_length=torch.where(
                type_kind, frame.text_length, torch.zeros_like(frame.text_length)
            ),
            hotkey_keys=torch.where(
                hotkey_kind[:, None], frame.hotkey_keys, torch.zeros_like(frame.hotkey_keys)
            ),
            hotkey_length=torch.where(
                hotkey_kind, frame.hotkey_length, torch.zeros_like(frame.hotkey_length)
            ),
        )
        frame.validate()
        pending_bytes, pending_length = self._next_pending(frame, state)
        is_type = frame.kind.eq(int(ActionKind.TYPE))
        state_index = frame.text_length.clamp_min(1).sub(1)
        selected_type_state = type_states.gather(
            1,
            state_index[:, None, None].expand(-1, 1, type_states.shape[-1]),
        ).squeeze(1)
        next_type_state = torch.where(
            is_type[:, None], selected_type_state, torch.zeros_like(selected_type_state)
        )
        held_buttons = state.held_buttons.clone()
        pointer = frame.kind.eq(int(ActionKind.POINTER_BUTTON))
        for index in range(frame.batch_size):
            if not bool(pointer[index]):
                continue
            button_index = int(frame.button[index].item())
            phase = int(frame.button_phase[index].item())
            was_held = bool(held_buttons[index, button_index])
            if was_held and phase != int(PointerButtonPhase.UP):
                if teacher_frame is not None and bool(teacher_mask[index]):
                    raise ValueError("held pointer button requires an UP frame")
                continue
            if not was_held and phase == int(PointerButtonPhase.UP):
                if teacher_frame is not None and bool(teacher_mask[index]):
                    raise ValueError("pointer button is not held")
                continue
            if phase == int(PointerButtonPhase.DOWN):
                held_buttons[index, button_index] = True
            elif phase == int(PointerButtonPhase.UP):
                held_buttons[index, button_index] = False
        next_state = ActionLocalState(
            previous_frame_embedding=self._frame_embedding(frame),
            type_decoder_state=next_type_state,
            pending_utf8_bytes=pending_bytes,
            pending_utf8_length=pending_length,
            type_active=is_type,
            held_buttons=held_buttons,
            held_keys=torch.zeros_like(state.held_keys),
        )
        output = ActionHeadOutput(
            frame=frame,
            kind_logits=kind_logits,
            coordinate_cell_logits=cell_logits,
            coordinate_residual_alpha=coordinate_alpha,
            coordinate_residual_beta=coordinate_beta,
            button_logits=button_logits,
            button_phase_logits=phase_logits,
            scroll_alpha=scroll_alpha,
            scroll_beta=scroll_beta,
            text_length_logits=text_length_logits,
            text_byte_logits=text_logits,
            hotkey_length_logits=hotkey_length_logits,
            hotkey_key_logits=hotkey_logits,
        )
        return output, next_state


def action_log_prob_components(
    output: ActionHeadOutput,
    frame: ActionFrame,
    sampling_temperature: float | None = None,
) -> dict[str, Tensor]:
    def categorical_logprob(logits: Tensor, target: Tensor) -> Tensor:
        scaled = (
            logits / sampling_temperature
            if sampling_temperature is not None and sampling_temperature > 0
            else logits
        )
        return F.log_softmax(scaled, dim=-1).gather(
            -1, target[..., None]
        ).squeeze(-1)

    eps = torch.finfo(output.coordinate_residual_alpha.dtype).eps
    kind = categorical_logprob(output.kind_logits, frame.kind)
    cell = categorical_logprob(output.coordinate_cell_logits, frame.coordinate_cell)
    residual = Beta(
        output.coordinate_residual_alpha, output.coordinate_residual_beta
    ).log_prob(frame.coordinate_residual.clamp(eps, 1.0 - eps)).sum(dim=-1)
    button = categorical_logprob(output.button_logits, frame.button)
    phase = categorical_logprob(output.button_phase_logits, frame.button_phase)
    scroll_unit = ((frame.scroll_delta + 1.0) * 0.5).clamp(eps, 1.0 - eps)
    scroll = (
        Beta(output.scroll_alpha, output.scroll_beta).log_prob(scroll_unit) - math.log(2.0)
    ).sum(dim=-1)
    text_length = categorical_logprob(output.text_length_logits, frame.text_length)
    text_bytes = categorical_logprob(output.text_byte_logits, frame.text_bytes)
    text_positions = torch.arange(TYPE_BYTES_PER_UNIT, device=frame.kind.device)[None]
    text_mask = text_positions < frame.text_length[:, None]
    text = text_length + torch.where(
        text_mask, text_bytes, torch.zeros_like(text_bytes)
    ).sum(dim=-1)
    hotkey_length = categorical_logprob(output.hotkey_length_logits, frame.hotkey_length)
    keys = categorical_logprob(output.hotkey_key_logits, frame.hotkey_keys)
    key_positions = torch.arange(HOTKEY_KEYS_PER_UNIT, device=frame.kind.device)[None]
    key_mask = key_positions < frame.hotkey_length[:, None]
    hotkey = hotkey_length + torch.where(
        key_mask, keys, torch.zeros_like(keys)
    ).sum(dim=-1)
    return {
        "kind": kind,
        "pointer_move": cell + residual,
        "pointer_button": button + phase,
        "scroll": scroll,
        "type": text,
        "hotkey": hotkey,
    }


def action_frame_log_prob(
    output: ActionHeadOutput,
    frame: ActionFrame,
    sampling_temperature: float | None = None,
) -> Tensor:
    components = action_log_prob_components(output, frame, sampling_temperature)
    result = components["kind"]
    conditions = (
        (ActionKind.POINTER_MOVE, "pointer_move"),
        (ActionKind.POINTER_BUTTON, "pointer_button"),
        (ActionKind.SCROLL, "scroll"),
        (ActionKind.TYPE, "type"),
        (ActionKind.HOTKEY, "hotkey"),
    )
    for kind, name in conditions:
        # Irrelevant conditional distributions may contain -inf after grammar
        # masking. Multiplication would turn 0 * -inf into NaN, so select the
        # active branch explicitly.
        result = result + torch.where(
            frame.kind.eq(int(kind)), components[name], torch.zeros_like(result)
        )
    if not torch.isfinite(result).all():
        raise RuntimeError("structured action log-probability is not finite")
    return result
