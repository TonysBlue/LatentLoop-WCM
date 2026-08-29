from __future__ import annotations

import torch
from runtime.config import ModelConfig
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from model.action import ActionHead
from model.attention import SlowMemory, StreamingTransformerLayer
from model.encoders import DeltaTimeEncoder, StreamingAudioEncoder, VisionEncoder
from model.perceiver import JEPAHead, Perceiver
from model.speech import FactorizedSpeechHead
from model.types import (
    ActionFrame,
    GenerationOutput,
    LayerKV,
    RecurrentState,
    SlowMemoryState,
    SpeechLocalState,
    SpeechMode,
    SpeechSamplingConfig,
    StepOutput,
    StreamUnit,
)
from model.value import ValueHead


class StreamingLatentLoop(nn.Module):
    """Streaming multimodal model with shared semantic slots and long memory."""

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
        self.jepa_head = JEPAHead(
            dim,
            config.model_dim,
            config.num_heads,
            config.ffn_dim,
            config.perceiver_slots,
            config.predictor_layers,
            config.dropout,
        )
        self.semantic_slot_identity = nn.Parameter(torch.zeros(config.semantic_memory_slots, dim))
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
        self.semantic_attention = nn.MultiheadAttention(dim, config.num_heads, batch_first=True)
        self.semantic_gate = nn.Linear(dim * 2, 1)
        self.slow_memory = SlowMemory(config.num_heads, dim // config.num_heads)
        self.predictor = self.jepa_head
        self.speech_head = FactorizedSpeechHead(config)
        self.action_head = ActionHead(config)
        self.value_head = ValueHead(config.model_dim, config.model_dim, config.num_heads)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.semantic_slot_identity, std=0.02)
        nn.init.constant_(self.semantic_gate.bias, -2.0)

    def semantic_memory_update(self, previous: Tensor, hidden: Tensor) -> Tensor:
        query = previous + self.semantic_slot_identity[None]
        context, _ = self.semantic_attention(query, hidden, hidden, need_weights=False)
        gate = torch.sigmoid(self.semantic_gate(torch.cat((query, context), dim=-1)))
        return previous + gate * (context - previous)

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
            semantic_memory=torch.zeros(
                batch_size,
                self.config.semantic_memory_slots,
                self.config.model_dim,
                device=device,
                dtype=dtype,
            ),
            slow_memory=tuple(
                SlowMemoryState(
                    matrix=torch.zeros(
                        batch_size, self.config.num_heads, head_dim, head_dim,
                        device=device, dtype=dtype,
                    ),
                    normalizer=torch.zeros(
                        batch_size, self.config.num_heads, head_dim,
                        device=device, dtype=dtype,
                    ),
                ) for _ in self.layers
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
        semantic = state.semantic_memory + self.semantic_slot_identity[None]
        new_caches: list[LayerKV] = []
        new_memories: list[SlowMemoryState] = []
        gates: list[Tensor] = []
        max_kv_tokens = self.config.kv_units * self.config.perceiver_slots
        hidden = perceived + state.hidden
        for layer, cache in zip(self.layers, state.layer_kv, strict=True):
            if self.training and self.config.activation_checkpointing:

                def layer_forward(
                    current: Tensor,
                    key: Tensor,
                    value: Tensor,
                    matrix: Tensor,
                    normalizer: Tensor,
                    layer: StreamingTransformerLayer = layer,
                ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
                    output, updated, memory, gate = layer(
                        current, cache=LayerKV(key, value), max_tokens=max_kv_tokens,
                        slow_memory=self.slow_memory,
                        slow_state=SlowMemoryState(matrix, normalizer),
                        semantic_memory=semantic,
                    )
                    return (
                        output, updated.key, updated.value,
                        memory.matrix, memory.normalizer, gate,
                    )

                hidden, key, value, matrix, normalizer, gate = checkpoint(
                    layer_forward, hidden, cache.key, cache.value,
                    state.slow_memory[len(new_caches)].matrix,
                    state.slow_memory[len(new_caches)].normalizer,
                    use_reentrant=False,
                )
                new_cache = LayerKV(key, value)
                new_memory = SlowMemoryState(matrix, normalizer)
            else:
                hidden, new_cache, new_memory, gate = layer(
                    hidden, cache=cache, max_tokens=max_kv_tokens,
                    slow_memory=self.slow_memory,
                    slow_state=state.slow_memory[len(new_caches)],
                    semantic_memory=semantic,
                )
            new_caches.append(new_cache)
            new_memories.append(new_memory)
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
        semantic = self.semantic_memory_update(semantic, hidden)
        predicted_next = self.jepa_head(hidden, semantic)
        next_state = RecurrentState(
            layer_kv=tuple(new_caches),
            semantic_memory=semantic,
            slow_memory=tuple(new_memories),
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
            value=self.value_head(hidden, semantic),
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
