"""voice-kit: a reusable realtime voice-agent stack.

The top-level package is framework-free (no LiveKit import). Optional subpackages:

* :mod:`voice_kit.livekit` - LiveKit Agents worker/session helpers (``livekit`` extra)
* :mod:`voice_kit.tokens` - access tokens and agent dispatch (``tokens`` extra)
* :mod:`voice_kit.retell` - Retell custom-LLM websocket adapter (``retell`` extra)
* :mod:`voice_kit.keepalive` - self-ping middleware (``server`` extra)
"""

from .context import ContextStore
from .env import env_bool, env_float, env_int, env_list, env_str
from .llm import LLMModelSpec, init_chat_model_from_spec, llm_model_init_args, parse_llm_model_spec
from .presets import PRESETS, VoicePreset, get_preset, preset_from_env, register_preset
from .streaming import (
    FILLER,
    OpenerDropper,
    SayStreamer,
    ToolMarkupFilter,
    WordGate,
    chunk_text,
    is_farewell,
    message_text,
    strip_leading_filler,
    strip_opener,
)
from .turns import (
    TurnDecision,
    TurnGate,
    UserTurn,
    always_respond,
    chain,
    decide,
    is_question,
    keyword_gate,
    llm_gate,
    min_words_gate,
    question_gate,
    reference_text_gate,
    text_overlap,
)

__version__ = "0.1.0"

__all__ = [
    "FILLER",
    "PRESETS",
    "ContextStore",
    "LLMModelSpec",
    "OpenerDropper",
    "SayStreamer",
    "ToolMarkupFilter",
    "TurnDecision",
    "TurnGate",
    "UserTurn",
    "VoicePreset",
    "WordGate",
    "always_respond",
    "chain",
    "chunk_text",
    "decide",
    "env_bool",
    "env_float",
    "env_int",
    "env_list",
    "env_str",
    "get_preset",
    "init_chat_model_from_spec",
    "is_farewell",
    "is_question",
    "keyword_gate",
    "llm_gate",
    "llm_model_init_args",
    "message_text",
    "min_words_gate",
    "parse_llm_model_spec",
    "preset_from_env",
    "question_gate",
    "reference_text_gate",
    "register_preset",
    "strip_leading_filler",
    "strip_opener",
    "text_overlap",
]
