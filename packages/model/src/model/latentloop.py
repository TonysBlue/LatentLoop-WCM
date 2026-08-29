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
            config.num_heads,
            config.ffn_dim,
            config.perceiver_slots,
            config.jepa_layers,
            config.dropout,
        )
        self.semantic_slot_identity = nn.Parameter(torch.zeros(config.semantic_memory_slots, dim))
        self.layers = nn.ModuleList(
            StreamingTransformerLayer(
                dim=dim,
                heads=config.num_heads,
                ffn_dim=config.ffn_dim,
                dropout=config.dropout,
            )
            for index in range(config.num_layers)
        )
        self.final_norm = nn.LayerNorm(dim)
        self.slow_memories = nn.ModuleList(
            SlowMemory(config.num_heads, dim // config.num_heads) for _ in self.layers
        )
        self.speech_head = FactorizedSpeechHead(config)
        self.action_head = ActionHead(config)
        self.value_head = ValueHead(config.model_dim, config.num_heads)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.semantic_slot_identity, std=0.02)

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
            semantic_memory=self.semantic_slot_identity.to(device=device, dtype=dtype)
            .unsqueeze(0)
            .expand(batch_size, -1, -1),
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
        semantic = state.semantic_memory
        new_caches: list[LayerKV] = []
        new_memories: list[SlowMemoryState] = []
        max_kv_tokens = self.config.kv_units * self.config.perceiver_slots
        hidden_tokens = perceived + state.hidden
        sequence = torch.cat((hidden_tokens, semantic), dim=1)
        hidden_count = hidden_tokens.shape[1]
        for layer_index, (layer, cache, slow_memory) in enumerate(
            zip(self.layers, state.layer_kv, self.slow_memories, strict=True)
        ):
            if self.training and self.config.activation_checkpointing:

                def layer_forward(
                    current: Tensor,
                    key: Tensor,
                    value: Tensor,
                    matrix: Tensor,
                    normalizer: Tensor,
                    layer: StreamingTransformerLayer = layer,
                    memory_module: SlowMemory = slow_memory,
                ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
                    output, updated, memory = layer(
                        current, cache=LayerKV(key, value), max_tokens=max_kv_tokens,
                        cache_tokens=hidden_count,
                        slow_memory=memory_module,
                        slow_state=SlowMemoryState(matrix, normalizer),
                    )
                    return (
                        output, updated.key, updated.value,
                        memory.matrix, memory.normalizer,
                    )

                sequence, key, value, matrix, normalizer = checkpoint(
                    layer_forward, sequence, cache.key, cache.value,
                    state.slow_memory[layer_index].matrix,
                    state.slow_memory[layer_index].normalizer,
                    use_reentrant=False,
                )
                new_cache = LayerKV(key, value)
                new_memory = SlowMemoryState(matrix, normalizer)
            else:
                sequence, new_cache, new_memory = layer(
                    sequence, cache=cache, max_tokens=max_kv_tokens,
                    cache_tokens=hidden_count,
                    slow_memory=slow_memory,
                    slow_state=state.slow_memory[layer_index],
                )
            new_caches.append(new_cache)
            new_memories.append(new_memory)
        sequence = self.final_norm(sequence)
        hidden = sequence[:, :hidden_count]
        semantic = sequence[:, hidden_count:]
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
        predicted_next = self.jepa_head(hidden)
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
            jepa_prediction=predicted_next,
            value=self.value_head(hidden),
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
