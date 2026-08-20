"""Model output loss helpers shared by supervised Training stages."""

from __future__ import annotations

import torch
from contracts import ActionKind
from torch import Tensor
from torch.nn import functional as F

from model.action import action_log_prob_components
from model.types import SpeechMode, StepOutput, StreamUnit


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    weights = mask.to(values.dtype)
    selected = torch.where(mask, values, torch.zeros_like(values))
    return selected.sum() / weights.sum().clamp_min(1)


def compute_jepa_loss(
    predicted_next_slots: Tensor,
    source_slots: Tensor,
    target_slots: Tensor,
    valid_mask: Tensor | None = None,
) -> dict[str, Tensor]:
    """Compute slot-aligned next-observation prediction and variance-floor losses."""
    if not (
        predicted_next_slots.shape == source_slots.shape == target_slots.shape
        and predicted_next_slots.ndim >= 3
    ):
        raise ValueError("JEPA tensors must have matching [..., slots, dim] shapes")
    sample_shape = predicted_next_slots.shape[:-2]
    if valid_mask is None:
        valid_mask = torch.ones(sample_shape, dtype=torch.bool, device=source_slots.device)
    if valid_mask.shape != sample_shape:
        raise ValueError("JEPA valid_mask must match the batch/time sample dimensions")
    flat_mask = valid_mask.reshape(-1)
    predicted = predicted_next_slots.reshape(-1, *predicted_next_slots.shape[-2:])[flat_mask]
    source = source_slots.reshape(-1, *source_slots.shape[-2:])[flat_mask]
    target = target_slots.reshape(-1, *target_slots.shape[-2:])[flat_mask]
    if predicted.shape[0] == 0:
        zero = predicted_next_slots.sum() * 0.0
        return {"total": zero, "prediction": zero, "variance": zero}

    predicted_normalized = F.normalize(predicted, dim=-1, eps=1e-4)
    target_normalized = F.normalize(target.detach(), dim=-1, eps=1e-4)
    prediction = (predicted_normalized - target_normalized).square().mean()
    if source.shape[0] < 2:
        variance = source.sum() * 0.0
    else:
        standard_deviation = torch.sqrt(source.var(dim=0, unbiased=False) + 1e-4)
        variance = torch.relu(1.0 - standard_deviation).mean()
    return {"total": prediction + variance, "prediction": prediction, "variance": variance}


def structured_action_loss(output: StepOutput, target: StreamUnit) -> Tensor:
    components = action_log_prob_components(output.action, target.action)
    supervised = target.action_supervision_mask
    losses = [_masked_mean(-components["kind"], supervised)]
    conditions = (
        (ActionKind.POINTER_MOVE, "pointer_move"),
        (ActionKind.POINTER_BUTTON, "pointer_button"),
        (ActionKind.SCROLL, "scroll"),
        (ActionKind.TYPE, "type"),
        (ActionKind.HOTKEY, "hotkey"),
    )
    for kind, name in conditions:
        mask = supervised & target.action.kind.eq(int(kind))
        if bool(mask.any()):
            losses.append(_masked_mean(-components[name], mask))
    return sum(losses) / len(losses)


def compute_losses(
    output: StepOutput,
    target: StreamUnit,
    speech_loss_weight: float = 1.0,
    action_loss_weight: float = 1.0,
) -> dict[str, Tensor]:
    mode_values = F.cross_entropy(output.speech_mode_logits, target.speech_mode, reduction="none")
    mode_loss = _masked_mean(mode_values, target.speech_mode_mask)
    batch, frames, codebooks, vocab = output.speech_codec_logits.shape
    codec_values = F.cross_entropy(
        output.speech_codec_logits.reshape(batch * frames * codebooks, vocab),
        target.speech_codes.reshape(-1),
        reduction="none",
    ).view(batch, frames, codebooks)
    codec_mask = (
        target.speech_codec_mask & target.speech_mode.eq(int(SpeechMode.SPEECH))[:, None]
    )[:, :, None].expand_as(codec_values)
    codec_loss = _masked_mean(codec_values, codec_mask)
    action_loss = structured_action_loss(output, target)
    total = speech_loss_weight * (mode_loss + codec_loss) + action_loss_weight * action_loss
    return {
        "total": total,
        "speech": mode_loss + codec_loss,
        "speech_mode": mode_loss,
        "speech_codec": codec_loss,
        "action": action_loss,
    }
