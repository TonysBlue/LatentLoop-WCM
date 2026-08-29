from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path

import torch
from data import EpisodeShardReader, SyntheticEpisodeDataset
from model import StreamingLatentLoop
from model.types import SpeechMode
from runtime.config import ProjectConfig

from training.checkpoint import file_sha256


@dataclass(frozen=True, slots=True)
class Evaluation:
    split: str
    episodes: int
    speech_mode_accuracy: float
    speech_codec_accuracy: list[float]
    action_kind_accuracy: float
    speech_silence_precision: float
    speech_silence_recall: float
    passed: bool | None = None


def build_evaluation_report(
    config: ProjectConfig, checkpoint: str | Path, split: str, result: Evaluation
) -> dict[str, object]:
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    payload: dict[str, object] = {
        "dataset": config.data.dataset,
        "split": split,
        "evaluation_kind": "streaming",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "data_identity": None,
        **(asdict(result) if is_dataclass(result) else dict(result.__dict__)),
    }
    if config.data.manifest:
        manifest = Path(config.data.manifest).expanduser()
        if not manifest.is_absolute():
            manifest = config.runtime.data_path() / manifest
        if manifest.is_file():
            payload["data_identity"] = file_sha256(manifest)
    return payload


def load_evaluation_model(
    config: ProjectConfig,
    checkpoint: str | Path,
    device: torch.device,
    *,
    require_data_identity: bool = True,
) -> StreamingLatentLoop:
    payload = torch.load(Path(checkpoint).expanduser(), map_location="cpu", weights_only=False)
    if not isinstance(payload.get("model"), dict) or not isinstance(
        payload.get("metadata"), dict
    ):
        raise ValueError("evaluation requires a complete current checkpoint")
    metadata = payload.get("metadata", {})
    for field, expected in (
        ("codec_id", config.data.codec_id),
        ("codec_weight_hash", config.data.codec_weight_hash),
        ("codec_revision", config.data.codec_revision),
    ):
        if metadata.get(field) != expected:
            raise ValueError(f"checkpoint {field} does not match the evaluation config")
    if require_data_identity and config.data.manifest:
        manifest = Path(config.data.manifest).expanduser()
        if not manifest.is_absolute():
            manifest = config.runtime.data_path() / manifest
        if manifest.is_file() and metadata.get("data_identity") != file_sha256(manifest):
            raise ValueError("checkpoint data identity does not match the evaluation manifest")
    model = StreamingLatentLoop(config.model)
    model.load_state_dict(payload["model"])
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    return model.to(device=device, dtype=dtype).eval()


def evaluate_checkpoint(
    config: ProjectConfig,
    checkpoint: str | Path,
    *,
    split: str = "validation",
    device: str | torch.device | None = None,
    codec_threshold: float = 0.9,
    control_f1_threshold: float | None = None,
) -> Evaluation:
    del control_f1_threshold
    selected_device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    model = load_evaluation_model(config, checkpoint, selected_device, require_data_identity=False)
    if config.data.source == "synthetic":
        episodes_source = SyntheticEpisodeDataset(config.data, config.model)
    else:
        if not config.data.shards:
            raise ValueError("evaluation requires WebDataset shards")
        episodes_source = EpisodeShardReader(config.data.shards, config.data, config.model)
    mode_correct = mode_total = action_correct = action_total = 0
    codec_correct = torch.zeros(config.model.speech_codebooks, dtype=torch.long)
    codec_total = 0
    silence_tp = silence_pred = silence_target = 0
    episodes = 0
    with torch.inference_mode():
        for episode in episodes_source:
            episodes += 1
            state = model.initial_state(1, selected_device)
            for raw_unit in episode.units:
                unit = raw_unit.to(selected_device, dtype=next(model.parameters()).dtype)
                output = model(
                    unit,
                    state,
                    unit.speech_codes,
                    speech_teacher_mode=unit.speech_mode,
                    action_teacher_frame=unit.action,
                    action_teacher_mask=unit.action_supervision_mask,
                )
                state = output.state
                mode = output.speech_mode_logits.argmax(-1)
                mode_mask = unit.speech_mode_mask
                mode_correct += int(((mode == unit.speech_mode) & mode_mask).sum())
                mode_total += int(mode_mask.sum())
                target_codec = (
                    unit.speech_codec_mask & unit.speech_mode.eq(int(SpeechMode.SPEECH))[:, None]
                )
                predicted_codec = output.speech_codec_logits.argmax(-1)
                codec_correct += (
                    ((predicted_codec == unit.speech_codes) & target_codec[:, :, None])
                    .sum(dim=(0, 1))
                    .cpu()
                )
                codec_total += int(target_codec.sum())
                action_mask = unit.action_supervision_mask
                action_correct += int(
                    ((output.action.kind_logits.argmax(-1) == unit.action.kind) & action_mask).sum()
                )
                action_total += int(action_mask.sum())
                silence_target += int((unit.speech_mode == int(SpeechMode.SILENCE)).sum())
                silence_pred += int((mode == int(SpeechMode.SILENCE)).sum())
                silence_tp += int(
                    (
                        (mode == int(SpeechMode.SILENCE))
                        & (unit.speech_mode == int(SpeechMode.SILENCE))
                    ).sum()
                )
    if episodes == 0:
        raise ValueError("evaluation dataset contains no episodes")
    silence_precision = silence_tp / max(silence_pred, 1)
    silence_recall = silence_tp / max(silence_target, 1)
    codec_accuracy = (codec_correct.float() / max(codec_total, 1)).tolist()
    passed = min(codec_accuracy, default=0.0) >= codec_threshold
    return Evaluation(
        split,
        episodes,
        mode_correct / max(mode_total, 1),
        codec_accuracy,
        action_correct / max(action_total, 1),
        silence_precision,
        silence_recall,
        passed,
    )


def evaluate_overfit_checkpoint(
    config: ProjectConfig,
    checkpoint: str | Path,
    *,
    device: str | torch.device | None = None,
    codec_threshold: float = 0.9,
    control_f1_threshold: float | None = None,
) -> Evaluation:
    """Evaluate the direct-speech gate through the same two-head protocol."""
    return evaluate_checkpoint(
        config,
        checkpoint,
        split="train",
        device=device,
        codec_threshold=codec_threshold,
        control_f1_threshold=control_f1_threshold,
    )


def evaluate_canary_checkpoint(
    config: ProjectConfig,
    checkpoint: str | Path,
    *,
    split: str = "validation",
    device: str | torch.device | None = None,
) -> Evaluation:
    return evaluate_checkpoint(config, checkpoint, split=split, device=device)
