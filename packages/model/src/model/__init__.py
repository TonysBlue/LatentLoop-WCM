"""Model Core public surface.

Only tensor code is exported here.  Serving and training systems depend on
this package boundary, so they cannot accidentally acquire device or reward
dependencies.
"""

from model.action import ActionHead, action_frame_log_prob, action_log_prob_components
from model.attention import SlowMemory
from model.core import FactorizedSpeechHead, StreamingLatentLoop
from model.encoders import DeltaTimeEncoder
from model.losses import compute_jepa_loss, compute_losses
from model.perceiver import JEPAHead, Perceiver
from model.types import (
    ActionFrame,
    ActionHeadOutput,
    ActionLocalState,
    Episode,
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

__all__ = [
    "ActionFrame", "ActionHead", "ActionHeadOutput", "ActionLocalState", "Episode",
    "FactorizedSpeechHead",
    "Perceiver", "JEPAHead", "SlowMemory",
    "DeltaTimeEncoder",
    "GenerationOutput", "LayerKV", "RecurrentState", "SpeechLocalState", "SpeechMode",
    "SpeechSamplingConfig", "StepOutput", "StreamUnit", "StreamingLatentLoop",
    "ValueHead",
    "action_frame_log_prob", "action_log_prob_components", "compute_jepa_loss", "compute_losses",
]
