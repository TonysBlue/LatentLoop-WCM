from __future__ import annotations

import torch
from runtime.config import ModelConfig
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from model.action import ActionHead
from model.attention import StreamingTransformerLayer
from model.encoders import DeltaTimeEncoder, StreamingAudioEncoder, VisionEncoder
from model.perceiver import Perceiver, PredictionAdapter, Predictor
from model.speech import FactorizedSpeechHead
from model.types import (
    ActionFrame,
    GenerationOutput,
    LayerKV,
    RecurrentState,
    SpeechLocalState,
    SpeechMode,
    SpeechSamplingConfig,
    StepOutput,
    StreamUnit,
)
from model.value import ValueHead


class WorldStateUpdate(nn.Module):
    def __init__(self, model_dim: int, latent_dim: int, heads: int, slots: int) -> None:
        super().__init__()
        self.latent_to_model = nn.Linear(latent_dim, model_dim)
        self.slot_identity = nn.Parameter(torch.zeros(slots, model_dim))
        self.slot_identity_latent = nn.Parameter(torch.zeros(slots, latent_dim))
        self.read = nn.MultiheadAttention(model_dim, heads, batch_first=True)
        self.model_to_latent = nn.Linear(model_dim, latent_dim)
        self.candidate = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim * 2),
            nn.GELU(),
            nn.Linear(latent_dim * 2, latent_dim),
        )
        self.gate = nn.Linear(latent_dim * 2, latent_dim)
        self.gate_bias = nn.Parameter(torch.tensor(-2.0))
        self.residual_scale = 0.1
        self.norm = nn.LayerNorm(latent_dim)
        nn.init.normal_(self.slot_identity, std=0.02)
        nn.init.normal_(self.slot_identity_latent, std=0.02)

    def _context(self, latent: Tensor, previous_hidden: Tensor) -> Tensor:
        query = self.latent_to_model(latent) + self.slot_identity[None]
        context, _ = self.read(query, previous_hidden, previous_hidden, need_weights=False)
        return self.model_to_latent(context)

    def forward(self, latent: Tensor, previous_hidden: Tensor) -> Tensor:
        context = self._context(latent, previous_hidden)
        combined = torch.cat((latent, context), dim=-1)
        # Slot identity must affect the first write even when Z_0 and H_0 are
        # both zero; otherwise every slot remains exactly symmetric.
        candidate = self.candidate(combined) + self.slot_identity_latent[None]
        gate = torch.sigmoid(self.gate(combined) + self.gate_bias)
        return self.norm(latent + self.residual_scale * gate * candidate)


class StreamingLatentLoop(nn.Module):
    """Final target: bounded KV, recurrent Z/H state, independent speech/action heads."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        dim = config.model_dim
        self.audio_encoder = StreamingAudioEncoder(
            dim, config.audio_tokens, config.audio_kernel, config.audio_stride
        )
        self.vision_encoder = VisionEncoder(dim)
        self.delta_time_encoder = DeltaTimeEncoder(
            dim,
            bands=config.delta_time_fourier_bands,
            base_period_ms=config.delta_time_base_period_ms,
        )
        self.type_embedding = nn.Embedding(3, dim)
        self.perceiver = Perceiver(
            dim,
            config.num_heads,
            config.ffn_dim,
            config.perceiver_slots,
            config.perceiver_layers,
            config.dropout,
        )
        self.predictor = Predictor(
            dim,
            config.latent_dim,
            config.num_heads,
            config.ffn_dim,
            config.perceiver_slots,
            config.predictor_layers,
            config.dropout,
        )
        self.prediction_adapter = PredictionAdapter(dim)
        self.future_embedding = nn.Parameter(torch.zeros(config.perceiver_slots, dim))
        self.latent_reader = nn.Linear(config.latent_dim, dim)
        self.layers = nn.ModuleList(
            StreamingTransformerLayer(
                dim=dim,
                heads=config.num_heads,
                ffn_dim=config.ffn_dim,
                dropout=config.dropout,
                cross_world=(index + 1) % config.cross_attention_every == 0,
            )
            for index in range(config.num_layers)
        )
        self.final_norm = nn.LayerNorm(dim)
        self.world_state_update = WorldStateUpdate(
            dim, config.latent_dim, config.num_heads, config.latent_slots
        )
        self.speech_head = FactorizedSpeechHead(config)
        self.action_head = ActionHead(config)
        self.value_head = ValueHead(config.model_dim, config.latent_dim, config.num_heads)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.future_embedding, std=0.02)

    def initial_state(self, batch_size: int, device: torch.device | str) -> RecurrentState:
        dtype = next(self.parameters()).dtype
        device = torch.device(device)
        head_dim = self.config.model_dim // self.config.num_heads
        empty_kv = tuple(
            LayerKV(
                key=torch.empty(
                    batch_size, self.config.num_heads, 0, head_dim, device=device, dtype=dtype
                ),
                value=torch.empty(
                    batch_size, self.config.num_heads, 0, head_dim, device=device, dtype=dtype
                ),
            )
            for _ in self.layers
        )
        return RecurrentState(
            layer_kv=empty_kv,
            latent=torch.zeros(
                batch_size,
                self.config.latent_slots,
                self.config.latent_dim,
                device=device,
                dtype=dtype,
            ),
            audio_cache=torch.zeros(
                batch_size, self.audio_encoder.cache_samples, device=device, dtype=dtype
            ),
            hidden=torch.zeros(
                batch_size,
                self.config.perceiver_slots,
                self.config.model_dim,
                device=device,
                dtype=dtype,
            ),
            speech_local=SpeechLocalState(
                temporal=torch.zeros(batch_size, self.config.model_dim, device=device, dtype=dtype),
                previous_codes=torch.zeros(
                    batch_size, self.config.speech_codebooks, device=device, dtype=torch.long
                ),
            ),
            action_local=self.action_head.initial_state(batch_size, device, dtype),
            unit_index=torch.zeros(batch_size, device=device, dtype=torch.long),
        )

    def encode_observation(self, unit: StreamUnit, audio_cache: Tensor) -> tuple[Tensor, Tensor]:
        """Encode an observation without advancing any cognitive recurrent state."""
        audio, next_audio_cache = self.audio_encoder(unit.mic_audio, audio_cache)
        vision = self.vision_encoder(unit.screen)
        delta_time = self.delta_time_encoder(unit.delta_ms)
        inputs = torch.cat(
            (
                delta_time + self.type_embedding.weight[0],
                audio + self.type_embedding.weight[1],
                vision + self.type_embedding.weight[2],
            ),
            dim=1,
        )
        return self.perceiver(inputs), next_audio_cache

    def forward_step(
        self,
        unit: StreamUnit,
        state: RecurrentState,
        *,
        speech_teacher_codes: Tensor | None = None,
        speech_teacher_mode: Tensor | None = None,
        action_teacher_frame: ActionFrame | None = None,
        action_teacher_mask: Tensor | None = None,
        sampling: SpeechSamplingConfig | None = None,
    ) -> StepOutput:
        perceived, audio_cache = self.encode_observation(unit, state.audio_cache)
        updated_latent = self.world_state_update(state.latent, state.hidden)
        predicted_next = self.predictor(perceived, updated_latent)
        future = self.prediction_adapter(predicted_next.detach()) + self.future_embedding[None]
        world = self.latent_reader(updated_latent)
        new_caches: list[LayerKV] = []
        gates: list[Tensor] = []
        max_kv_tokens = self.config.kv_units * self.config.perceiver_slots
        hidden = perceived
        for layer, cache in zip(self.layers, state.layer_kv, strict=True):
            if self.training and self.config.activation_checkpointing:

                def layer_forward(
                    current: Tensor,
                    current_world: Tensor,
                    current_future: Tensor,
                    key: Tensor,
                    value: Tensor,
                    layer: StreamingTransformerLayer = layer,
                ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
                    output, updated, gate = layer(
                        current,
                        current_world,
                        current_future,
                        LayerKV(key, value),
                        max_kv_tokens,
                    )
                    return output, updated.key, updated.value, gate

                hidden, key, value, gate = checkpoint(
                    layer_forward,
                    hidden,
                    world,
                    future,
                    cache.key,
                    cache.value,
                    use_reentrant=False,
                )
                new_cache = LayerKV(key, value)
            else:
                hidden, new_cache, gate = layer(
                    hidden,
                    world,
                    future,
                    cache,
                    max_kv_tokens,
                )
            new_caches.append(new_cache)
            gates.append(gate)
        hidden = self.final_norm(hidden)
        speech_context = self.speech_head.context(hidden)
        speech_temporal = self.speech_head.update_temporal(speech_context, state.speech_local)
        speech_mode_logits = self.speech_head.mode_logits(speech_context)
        if speech_teacher_mode is not None:
            mode = speech_teacher_mode
        elif sampling is not None and not sampling.greedy and sampling.temperature > 0:
            mode = torch.multinomial(
                torch.softmax(speech_mode_logits / sampling.temperature, dim=-1), 1
            ).squeeze(-1)
        else:
            mode = speech_mode_logits.argmax(dim=-1)
        if speech_teacher_codes is not None:
            speech_codec_logits = self.speech_head.teacher_logits(
                speech_temporal, speech_teacher_codes
            )
            next_codes = speech_teacher_codes[:, 0]
        else:
            speech_codec_logits, generated_codes = self.speech_head.generate(
                speech_temporal, sampling or SpeechSamplingConfig(greedy=True)
            )
            next_codes = generated_codes[:, 0]
        next_codes = torch.where(
            mode[:, None] == int(SpeechMode.SPEECH), next_codes, torch.zeros_like(next_codes)
        )
        action, action_local = self.action_head(
            hidden,
            state.action_local,
            action_teacher_frame,
            action_teacher_mask,
            sampling_temperature=(sampling.temperature if sampling is not None else None),
        )
        next_state = RecurrentState(
            layer_kv=tuple(new_caches),
            latent=updated_latent,
            audio_cache=audio_cache,
            hidden=hidden,
            speech_local=SpeechLocalState(temporal=speech_temporal, previous_codes=next_codes),
            action_local=action_local,
            unit_index=state.unit_index + 1,
        )
        return StepOutput(
            state=next_state,
            speech_mode_logits=speech_mode_logits,
            speech_codec_logits=speech_codec_logits,
            action=action,
            hidden=hidden,
            perceiver_slots=perceived,
            predicted_next_slots=predicted_next,
            future_gate_mean=torch.stack([gate.mean() for gate in gates]).mean(),
            future_gate_max=torch.stack([gate.max() for gate in gates]).max(),
            value=self.value_head(hidden, updated_latent),
            selected_speech_mode=mode,
        )

    def forward(
        self,
        unit: StreamUnit,
        state: RecurrentState,
        speech_teacher_codes: Tensor | None = None,
        *,
        speech_teacher_mode: Tensor | None = None,
        action_teacher_frame: ActionFrame | None = None,
        action_teacher_mask: Tensor | None = None,
    ) -> StepOutput:
        return self.forward_step(
            unit,
            state,
            speech_teacher_codes=speech_teacher_codes,
            speech_teacher_mode=speech_teacher_mode,
            action_teacher_frame=action_teacher_frame,
            action_teacher_mask=action_teacher_mask,
        )

    @torch.no_grad()
    def generate_step(
        self, unit: StreamUnit, state: RecurrentState, sampling: SpeechSamplingConfig | None = None
    ) -> GenerationOutput:
        output = self.forward_step(unit, state, sampling=sampling)
        return GenerationOutput(
            output=output,
            speech_mode=output.selected_speech_mode,
            speech_codes=output.state.speech_local.previous_codes[:, None],
            action_frame=output.action.frame,
        )

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
