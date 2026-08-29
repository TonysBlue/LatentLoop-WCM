from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


@dataclass(slots=True)
class ModelConfig:
    model_dim: int = 256
    num_layers: int = 4
    num_heads: int = 8
    ffn_dim: int = 1024
    audio_tokens: int = 4
    audio_kernel: int = 400
    audio_stride: int = 160
    vision_tokens: int = 16
    perceiver_slots: int = 16
    perceiver_layers: int = 2
    jepa_layers: int = 2
    semantic_memory_slots: int = 8
    slow_memory_type: str = "gated_delta"
    rematerialization_segment_units: int = 32
    delta_time_fourier_bands: int = 8
    delta_time_base_period_ms: int = 80
    # The formal Canary profile retains the most recent 60 seconds.
    kv_units: int = 750
    kv_window_ms: int = 60_000
    speech_frames_per_unit: int = 1
    speech_codebooks: int = 8
    speech_codebook_size: int = 2048
    speech_depth_layers: int = 2
    speech_depth_heads: int = 4
    speech_depth_ffn_dim: int = 1024
    action_schema_id: str = "structured-action-v1"
    action_coordinate_grid_size: int = 32
    action_type_bytes_per_unit: int = 16
    action_hotkey_keys_per_unit: int = 8
    dropout: float = 0.1
    activation_checkpointing: bool = False

    @property
    def tokens_per_unit(self) -> int:
        return self.perceiver_slots


@dataclass(slots=True)
class DataConfig:
    # Identifies the dataset contract used by readiness, training and eval.
    # ``synthetic`` is an explicit local-development exemption.
    dataset: str = "synthetic"
    source: str = "synthetic"
    shards: str | None = None
    manifest: str | None = None
    audio_sample_rate: int = 24_000
    codec_frame_rate: float = 12.5
    codec_id: str = "mimi-24khz-8x2048"
    codec_weight_hash: str = "09b782f0629851a271227fb9d36db65c041790365f11bbe5d3d59369cf863f50"
    codec_revision: str = "a49141e28b3d9c947cf9aa5314431e1b11cbd2f5"
    codec_codebooks: int = 8
    codec_codebook_size: int = 2048
    unit_ms: int = 80
    unit_audio_samples: int = 1_920
    screen_height: int = 224
    screen_width: int = 224
    episode_units: int = 32
    train_episodes: int = 1_000
    seed: int = 17


@dataclass(slots=True)
class TrainingConfig:
    stage: str = "pretrain"
    max_updates: int = 10_000
    weight_decay: float = 0.1
    gradient_accumulation_steps: int = 16
    tbptt_units: int = 750
    mixed_precision: str = "fp16"
    max_grad_norm: float = 1.0
    checkpoint_every: int = 500
    log_every: int = 10
    head_learning_rate: float = 1e-4
    backbone_learning_rate: float = 3e-5
    warmup_ratio: float = 0.03
    codec_scheduled_sampling: float = 0.0
    codec_scheduled_sampling_start: float = 0.7
    backbone_train_mode: str = "all"
    speech_loss_weight: float = 1.0
    action_loss_weight: float = 1.0
    jepa_loss_weight: float = 1.0
    memory_horizon_units: int = 750
    min_learning_rate_ratio: float = 0.1
    rl: RLConfig = field(default_factory=lambda: RLConfig())


@dataclass(slots=True)
class RLConfig:
    algorithm: str = "online_recurrent_ppo"
    ppo_window_units: int = 750
    ppo_epochs: int = 4
    gae_lambda: float = 0.95
    discount_time_constant_ms: float = 10_000.0
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    sft_replay_coef: float = 0.1
    on_policy_jepa_coef: float = 0.1
    replay_jepa_coef: float = 0.1
    max_pending_reward_units: int = 750
    candidate_max_reference_kl: float = 1.0
    candidate_max_eval_loss_ratio: float = 1.25
    candidate_max_rejections: int = 3
    clip_epsilon: float = 0.2
    reference_kl_beta: float = 0.02
    environment_socket: str | None = None
    codec_socket: str | None = None
    reward_socket: str | None = None
    environment_id: str = ""
    environment_version: str = "1"
    environment_protocol_version: str = "realtime-v2"
    reward_spec_id: str = "realtime-v2"
    judge_model_id: str = "external-frozen-multimodal"
    judge_revision: str = "unconfigured"
    rubric_sha256: str = "unconfigured"
    timeline_root: str | None = None
    session_manifest: str | None = None
    sft_replay_shards: str | None = None
    sft_replay_manifest: str | None = None
    sft_preservation_shards: str | None = None
    sft_preservation_manifest: str | None = None
    sampling_temperature: float = 0.8
    sampling_top_k: int = 0
    advantage_epsilon: float = 1e-6


@dataclass(slots=True)
class TrackingConfig:
    enabled: bool = True
    project: str = "latentloop"
    mode: str = "online"
    base_url: str = "http://127.0.0.1:8080"
    media_samples_per_run: int = 4
    media_warning_gb: float = 40.0
    media_stop_gb: float = 50.0


@dataclass(slots=True)
class RuntimeConfig:
    data_root: str = "~/latentloop-data/datasets"
    experiment_root: str = "~/latentloop-data/experiments/local"
    run_name: str = "train"
    recipe_name: str | None = None
    stage_name: str | None = None

    def root_path(self) -> Path:
        return Path(self.experiment_root).expanduser().resolve()

    def data_path(self) -> Path:
        return Path(self.data_root).expanduser().resolve()

    def tracking_path(self) -> Path:
        """Return the shared W&B SDK directory, separate from experiment artifacts."""
        value = os.environ.get("LATENTLOOP_TRACKING_ROOT")
        if value:
            return Path(value).expanduser().resolve()
        return self.data_path().parent / "tracking"


@dataclass(slots=True)
class ProjectConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        if self.model.model_dim % self.model.num_heads:
            raise ValueError("model_dim must be divisible by num_heads")
        if self.model.audio_kernel < self.model.audio_stride:
            raise ValueError("audio_kernel must be >= audio_stride")
        if self.model.kv_units < 1 or self.model.semantic_memory_slots < 1:
            raise ValueError("kv_units and semantic_memory_slots must be positive")
        expected_kv_units = -(-self.model.kv_window_ms // self.data.unit_ms)
        if self.model.kv_units != expected_kv_units:
            raise ValueError("KV must exactly cover kv_window_ms at the configured unit_ms")
        if self.model.vision_tokens != 16:
            raise ValueError("vision_tokens must be 16")
        if (
            self.model.perceiver_slots != 16
            or self.model.perceiver_layers != 2
            or self.model.jepa_layers != 2
        ):
            raise ValueError("Perceiver/JEPA topology must be 16 slots and 2/2 layers")
        if self.model.slow_memory_type != "gated_delta":
            raise ValueError("slow_memory_type must be gated_delta")
        if self.model.semantic_memory_slots < 1:
            raise ValueError("semantic_memory_slots must be positive")
        if self.model.rematerialization_segment_units < 1:
            raise ValueError("rematerialization segment must be positive")
        if self.model.delta_time_fourier_bands < 1:
            raise ValueError("delta time Fourier bands must be positive")
        if self.model.delta_time_base_period_ms < 1:
            raise ValueError("delta time base period must be positive")
        if self.model.action_schema_id != "structured-action-v1":
            raise ValueError("model.action_schema_id must be structured-action-v1")
        if self.model.action_coordinate_grid_size != 32:
            raise ValueError("structured action requires a 32x32 coordinate grid")
        if self.model.action_type_bytes_per_unit != 16:
            raise ValueError("structured action requires 16 TYPE bytes per unit")
        if self.model.action_hotkey_keys_per_unit != 8:
            raise ValueError("structured action requires at most 8 HOTKEY keys per unit")
        if self.data.dataset not in {
            "synthetic",
            "canary",
            "direct-speech-overfit",
        }:
            raise ValueError(
                "data.dataset must be synthetic, canary, or direct-speech-overfit"
            )
        if self.data.source not in {"synthetic", "webdataset"}:
            raise ValueError("data.source must be synthetic or webdataset")
        if self.data.dataset != "synthetic" and self.data.source != "webdataset":
            raise ValueError("real and gate datasets must use webdataset data.source")
        if self.data.source == "webdataset" and not self.data.shards:
            raise ValueError("data.shards is required for webdataset")
        if not self.data.codec_id or not self.data.codec_weight_hash:
            raise ValueError("codec identity and weight hash are required")
        if self.data.unit_audio_samples * 1_000 != self.data.audio_sample_rate * self.data.unit_ms:
            raise ValueError("unit_audio_samples must exactly match the audio clock")
        audio_frames = self.data.unit_audio_samples // self.model.audio_stride
        if (
            self.data.unit_audio_samples % self.model.audio_stride
            or audio_frames % self.model.audio_tokens
        ):
            raise ValueError("audio convolution frames must divide into model audio tokens")
        if self.data.screen_height != 224 or self.data.screen_width != 224:
            raise ValueError("screen input must be exactly 224x224")
        if self.data.codec_codebooks != self.model.speech_codebooks:
            raise ValueError("data and model codec codebook counts must match")
        if self.data.codec_codebook_size != self.model.speech_codebook_size:
            raise ValueError("data and model codec vocabularies must match")
        if self.data.unit_ms * self.data.codec_frame_rate != 1_000:
            raise ValueError("direct speech requires exactly one codec frame per stream unit")
        if self.model.speech_frames_per_unit != 1:
            raise ValueError("direct speech requires exactly one speech frame per stream unit")
        if (
            self.model.speech_depth_heads < 1
            or self.model.model_dim % self.model.speech_depth_heads
        ):
            raise ValueError("model_dim must be divisible by speech_depth_heads")
        if self.training.tbptt_units < 1:
            raise ValueError("tbptt_units must be positive")
        if not 0 <= self.training.codec_scheduled_sampling <= 1:
            raise ValueError("codec_scheduled_sampling must be in [0, 1]")
        if not 0 <= self.training.codec_scheduled_sampling_start < 1:
            raise ValueError("codec_scheduled_sampling_start must be in [0, 1)")
        if self.training.backbone_train_mode not in {"frozen", "selective", "all"}:
            raise ValueError("backbone_train_mode must be frozen, selective, or all")
        if self.training.stage not in {"pretrain", "sft", "rl"}:
            raise ValueError("training.stage must be pretrain, sft, or rl")
        if self.data.dataset == "canary":
            if self.training.backbone_train_mode != "all":
                raise ValueError(
                    "formal stages must train the full model with backbone_train_mode=all"
                )
        rl = self.training.rl
        if rl.algorithm != "online_recurrent_ppo":
            raise ValueError("RL algorithm must be online_recurrent_ppo")
        if rl.ppo_window_units < 1 or rl.ppo_epochs < 2 or rl.max_pending_reward_units < 0:
            raise ValueError("PPO window and pending reward horizon must be non-negative")
        if self.training.stage == "rl" and rl.ppo_window_units < self.training.memory_horizon_units:
            raise ValueError("PPO window must cover the recurrent memory horizon")
        if not 0 < rl.gae_lambda <= 1 or rl.discount_time_constant_ms <= 0:
            raise ValueError("GAE lambda and discount time constant are invalid")
        if not 0 < rl.clip_epsilon < 1 or rl.reference_kl_beta < 0:
            raise ValueError("RL clip_epsilon must be in (0,1) and KL beta non-negative")
        if (
            rl.candidate_max_reference_kl <= 0
            or rl.candidate_max_eval_loss_ratio < 1
            or rl.candidate_max_rejections < 1
        ):
            raise ValueError("PPO candidate acceptance gates are invalid")
        if rl.sampling_temperature <= 0 or rl.sampling_top_k < 0:
            raise ValueError("RL temperature and top_k are invalid")
        if self.training.stage == "rl" and rl.sampling_top_k != 0:
            raise ValueError("PPO requires full-support sampling_top_k=0")
        if self.training.stage == "rl" and (
            rl.judge_revision == "unconfigured"
            or len(rl.rubric_sha256) != 64
            or any(character not in "0123456789abcdef" for character in rl.rubric_sha256)
        ):
            raise ValueError("PPO requires a locked Judge revision and rubric SHA-256")
        if self.training.stage == "rl" and self.data.dataset == "canary":
            if (
                not rl.environment_socket
                or not rl.codec_socket
                or not rl.reward_socket
                or not rl.environment_id
                or not rl.environment_version
                or not rl.environment_protocol_version
                or not rl.session_manifest
                or not rl.sft_replay_shards
                or not rl.sft_replay_manifest
                or not rl.sft_preservation_shards
                or not rl.sft_preservation_manifest
            ):
                raise ValueError(
                    "formal RL requires environment, codec and reward sockets, "
                    "environment identity, session_manifest and SFT guard datasets"
                )
            if (
                Path(rl.sft_replay_shards).expanduser()
                == Path(rl.sft_preservation_shards).expanduser()
                or Path(rl.sft_replay_manifest).expanduser()
                == Path(rl.sft_preservation_manifest).expanduser()
            ):
                raise ValueError("PPO replay and preservation datasets must be independent")
        if self.training.speech_loss_weight <= 0 or self.training.action_loss_weight <= 0:
            raise ValueError("speech_loss_weight and action_loss_weight must be positive")
        expected_jepa_weight = 0.5 if self.training.stage == "sft" else 1.0
        if self.training.stage != "rl" and self.training.jepa_loss_weight != expected_jepa_weight:
            raise ValueError(
                f"{self.training.stage} requires jepa_loss_weight={expected_jepa_weight}"
            )
        if (
            rl.sft_replay_coef != 0.1
            or rl.on_policy_jepa_coef != 0.1
            or rl.replay_jepa_coef != 0.1
        ):
            raise ValueError("Online RL SFT replay and both JEPA coefficients must be 0.1")
        if self.training.memory_horizon_units < 1:
            raise ValueError("memory_horizon_units must be positive")
        if self.training.tbptt_units != self.training.memory_horizon_units:
            raise ValueError("tbptt_units must equal memory_horizon_units")
        if (
            self.data.dataset == "canary" and self.training.memory_horizon_units != 750
        ):
            raise ValueError("formal Canary memory horizon must be exactly 750 units")
        if (
            self.data.dataset == "canary"
            and self.model.rematerialization_segment_units
            >= self.training.memory_horizon_units
        ):
            raise ValueError(
                "formal Canary rematerialization segment must be shorter than 750 units"
            )
        if not 0 <= self.training.min_learning_rate_ratio <= 1:
            raise ValueError("min_learning_rate_ratio must be in [0, 1]")
        if not 0 <= self.training.warmup_ratio < 1:
            raise ValueError("warmup_ratio must be in [0, 1)")
        if self.training.mixed_precision not in {"no", "fp16"}:
            raise ValueError("RTX 2080 SUPER profiles support mixed_precision=no or fp16")


def load_config(path: str | Path, overrides: list[str] | None = None) -> ProjectConfig:
    for override in overrides or []:
        if override.split("=", 1)[0] == "training.objective":
            raise ValueError(
                "training.objective is removed; use training.stage and training.rl.algorithm"
            )
        if "speech_control_class_weights" in override:
            raise ValueError("speech_control_class_weights is removed; use speech_loss_weight")
        if "speech_control_loss_weight" in override:
            raise ValueError("speech_control_loss_weight is removed; use speech_loss_weight")
    schema = OmegaConf.structured(ProjectConfig)
    config_path = Path(path).expanduser().resolve()
    loaded = OmegaConf.load(config_path)
    defaults = loaded.pop("defaults", []) if "defaults" in loaded else []
    base_configs = []
    for default in defaults:
        if default in {"_self_", None}:
            continue
        default_path = Path(str(default))
        if default_path.suffix != ".yaml":
            default_path = default_path.with_suffix(".yaml")
        base_configs.append(OmegaConf.load((config_path.parent / default_path).resolve()))
    loaded = OmegaConf.merge(*base_configs, loaded)
    merged = OmegaConf.merge(schema, loaded, OmegaConf.from_dotlist(overrides or []))
    raw = OmegaConf.to_container(merged, resolve=True)
    assert isinstance(raw, dict)
    config = ProjectConfig(
        model=ModelConfig(**raw["model"]),
        data=DataConfig(**raw["data"]),
        training=TrainingConfig(
            **{key: value for key, value in raw["training"].items() if key != "rl"},
            rl=RLConfig(**raw["training"].get("rl", {})),
        ),
        tracking=TrackingConfig(**raw["tracking"]),
        runtime=RuntimeConfig(**raw["runtime"]),
    )
    config.validate()
    return config
