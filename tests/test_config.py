from __future__ import annotations

import pytest
from runtime.config import load_config


def test_load_config_with_override() -> None:
    config = load_config("configs/smoke.yaml", ["model.semantic_memory_slots=6"])
    assert config.model.semantic_memory_slots == 6
    assert config.model.tokens_per_unit == config.model.perceiver_slots == 16
    assert config.model.perceiver_layers == config.model.jepa_layers == 2
    assert config.data.unit_ms == 80
    assert config.data.codec_frame_rate == 12.5
    assert config.model.delta_time_fourier_bands == 8
    assert config.model.delta_time_base_period_ms == 80


def test_world_state_and_delta_time_configuration_is_bounded() -> None:
    config = load_config("configs/smoke.yaml")
    assert config.model.slow_memory_type == "gated_delta"
    with pytest.raises(ValueError, match="Fourier bands"):
        load_config("configs/smoke.yaml", ["model.delta_time_fourier_bands=0"])
    with pytest.raises(ValueError, match="base period"):
        load_config("configs/smoke.yaml", ["model.delta_time_base_period_ms=0"])
    with pytest.raises(ValueError, match="16 slots"):
        load_config("configs/smoke.yaml", ["model.perceiver_slots=8"])


def test_stage_jepa_weights_are_fixed() -> None:
    pretrain = load_config("configs/stages/smoke-pretrain.yaml")
    sft = load_config("configs/stages/smoke-sft.yaml")
    rl = load_config("configs/stages/smoke-rl.yaml")

    assert pretrain.training.jepa_loss_weight == 1.0
    assert sft.training.jepa_loss_weight == 0.5
    assert rl.training.rl.on_policy_jepa_coef == 0.1
    assert rl.training.rl.replay_jepa_coef == 0.1
    with pytest.raises(ValueError, match="jepa_loss_weight=1.0"):
        load_config("configs/smoke.yaml", ["training.jepa_loss_weight=0.5"])
    with pytest.raises(ValueError, match="both JEPA coefficients"):
        load_config(
            "configs/stages/smoke-rl.yaml",
            ["training.rl.on_policy_jepa_coef=0.2"],
        )


def test_local_dev_and_canary_profiles_are_explicit() -> None:
    local = load_config("configs/local-dev.yaml")
    canary = load_config("configs/canary.yaml")
    assert local.data.dataset == "synthetic"
    assert canary.data.dataset == "canary"
    assert canary.model.action_schema_id == "structured-action-v1"


@pytest.mark.parametrize("removed_scale", ["pilot", "production"])
def test_removed_scales_are_rejected(removed_scale: str) -> None:
    with pytest.raises(ValueError, match="synthetic, canary, or direct-speech-overfit"):
        load_config("configs/smoke.yaml", [f"data.dataset={removed_scale}"])


def test_training_stage_is_the_only_top_level_dispatch_key() -> None:
    with pytest.raises(ValueError, match="pretrain, sft, or rl"):
        load_config("configs/smoke.yaml", ["training.stage=obsolete"])
    with pytest.raises(ValueError, match="training.objective is removed"):
        load_config("configs/smoke.yaml", ["training.objective=ppo"])


def test_formal_rl_requires_real_environment_configuration() -> None:
    with pytest.raises(ValueError, match="environment, codec and reward sockets"):
        load_config(
            "configs/canary.yaml",
            ["training.stage=rl", "training.rl.environment_socket=null"],
        )
    with pytest.raises(ValueError, match="SFT guard datasets"):
        load_config(
            "configs/canary.yaml",
            ["training.stage=rl", "training.rl.sft_replay_shards=null"],
        )


def test_formal_rl_environment_identity_matches_harness_service() -> None:
    training = load_config("configs/canary.yaml")
    assert training.training.rl.environment_id == "isolated-qemu-v1"


def test_ppo_candidate_acceptance_gates_are_validated() -> None:
    with pytest.raises(ValueError, match="candidate acceptance gates"):
        load_config(
            "configs/smoke.yaml",
            ["training.rl.candidate_max_eval_loss_ratio=0.9"],
        )
    with pytest.raises(ValueError, match="PPO window"):
        load_config("configs/smoke.yaml", ["training.rl.ppo_epochs=1"])
    with pytest.raises(ValueError, match="full-support"):
        load_config(
            "configs/smoke.yaml",
            ["training.stage=rl", "training.rl.sampling_top_k=4"],
        )
    with pytest.raises(ValueError, match="locked Judge revision"):
        load_config(
            "configs/smoke.yaml",
            ["training.stage=rl"],
        )


def test_online_rl_uses_the_online_recurrent_ppo_algorithm() -> None:
    config = load_config("configs/canary.yaml")
    assert config.training.rl.algorithm == "online_recurrent_ppo"
    with pytest.raises(ValueError, match="online_recurrent_ppo"):
        load_config("configs/smoke.yaml", ["training.rl.algorithm=recurrent_ppo"])


def test_formal_canary_training_requires_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch
    from training.training import train

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="requires a CUDA GPU"):
        train(load_config("configs/stages/canary-pretrain.yaml"))


def test_formal_canary_requires_segmented_recurrent_rematerialization() -> None:
    config = load_config("configs/canary.yaml")
    assert config.model.rematerialization_segment_units < config.training.memory_horizon_units
    with pytest.raises(ValueError, match="segment must be shorter"):
        load_config(
            "configs/canary.yaml",
            ["model.rematerialization_segment_units=750"],
        )


def test_config_rejects_incompatible_attention_width() -> None:
    with pytest.raises(ValueError, match="divisible"):
        load_config("configs/smoke.yaml", ["model.model_dim=63"])


def test_direct_speech_requires_one_codec_frame_per_tick() -> None:
    with pytest.raises(ValueError, match="exactly one speech frame"):
        load_config("configs/smoke.yaml", ["model.speech_frames_per_unit=2"])


def test_audio_convolution_frames_must_divide_into_tokens() -> None:
    with pytest.raises(ValueError, match="audio convolution frames"):
        load_config("configs/smoke.yaml", ["model.audio_tokens=3"])


def test_config_rejects_removed_speech_control_weights() -> None:
    with pytest.raises(ValueError, match="removed"):
        load_config(
            "configs/smoke.yaml",
            ["training.speech_control_class_weights=[1,1,1]"],
        )


def test_config_rejects_removed_speech_control_loss_weight() -> None:
    with pytest.raises(ValueError, match="removed"):
        load_config(
            "configs/smoke.yaml",
            ["training.speech_control_loss_weight=0"],
        )
