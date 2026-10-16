"""Build LiveKit ``AgentSession`` pieces from a :class:`~voice_kit.presets.VoicePreset`."""

from __future__ import annotations

import logging
from typing import Any

from livekit.agents import AgentSession, TurnHandlingOptions, inference
from livekit.agents.voice import room_io

from ..presets import VoicePreset, get_preset

log = logging.getLogger(__name__)

__all__ = [
    "build_session",
    "build_stt",
    "build_tts",
    "build_turn_handling",
    "graph_llm",
    "noise_canceller",
    "room_options",
]


def build_stt(preset: str | VoicePreset = "default") -> Any:
    p = get_preset(preset)
    return inference.STT(model=p.stt_model, language=p.language)


def build_tts(preset: str | VoicePreset = "default") -> Any:
    p = get_preset(preset)
    kwargs: dict[str, Any] = {"model": p.tts_model}
    if p.tts_voice:
        kwargs["voice"] = p.tts_voice
    return inference.TTS(**kwargs)


def build_turn_handling(preset: str | VoicePreset = "default") -> TurnHandlingOptions:
    p = get_preset(preset)
    opts: dict[str, Any] = p.turn_handling()
    if p.turn_detection == "model":
        opts["turn_detection"] = inference.TurnDetector()
    return TurnHandlingOptions(**opts)


def noise_canceller(preset: str | VoicePreset = "default") -> Any | None:
    """Noise-cancellation filter for the preset, or ``None``.

    Uses ``livekit-plugins-noise-cancellation`` (LiveKit Cloud only). If the plugin is missing a
    warning is logged and audio is left unfiltered.
    """
    p = get_preset(preset)
    if p.noise_cancellation == "none":
        return None
    try:
        from livekit.plugins import noise_cancellation
    except ImportError:
        log.warning(
            "preset %r wants %s noise cancellation but livekit-plugins-noise-cancellation is not installed",
            p.name,
            p.noise_cancellation,
        )
        return None
    factory = {
        "nc": noise_cancellation.NC,
        "bvc": noise_cancellation.BVC,
        "bvc_telephony": noise_cancellation.BVCTelephony,
    }[p.noise_cancellation]
    return factory()


def room_options(preset: str | VoicePreset = "default", **kwargs: Any) -> room_io.RoomOptions:
    """``RoomOptions`` with the preset's noise cancellation; ``kwargs`` pass through to RoomOptions."""
    nc = noise_canceller(preset)
    if nc is not None and "audio_input" not in kwargs:
        kwargs["audio_input"] = room_io.AudioInputOptions(noise_cancellation=nc)
    return room_io.RoomOptions(**kwargs)


def graph_llm(
    graph: Any,
    *,
    stream_mode: Any = "custom",
    config: Any = None,
    context: Any = None,
) -> Any:
    """Wrap a LangGraph graph (or any runnable with LangGraph's ``astream``) as a LiveKit LLM.

    ``stream_mode="custom"`` speaks whatever the graph emits through ``get_stream_writer()`` (text
    you have already filtered); ``"messages"`` speaks the model's raw message chunks.
    """
    from livekit.plugins import langchain

    kwargs: dict[str, Any] = {"graph": graph, "stream_mode": stream_mode}
    if config is not None:
        kwargs["config"] = config
    if context is not None:
        kwargs["context"] = context
    return langchain.LLMAdapter(**kwargs)


def build_session(
    preset: str | VoicePreset = "default",
    *,
    llm: Any = None,
    graph: Any = None,
    stream_mode: Any = "custom",
    stt: Any = None,
    tts: Any = None,
    vad: Any = None,
    turn_handling: TurnHandlingOptions | None = None,
    **kwargs: Any,
) -> AgentSession:
    """``AgentSession`` for a preset. Pass ``graph`` (wrapped with :func:`graph_llm`) or ``llm``.

    Explicit ``stt``/``tts``/``vad``/``turn_handling`` replace the preset's; other ``kwargs`` go
    straight to ``AgentSession`` (e.g. ``userdata``).
    """
    p = get_preset(preset)
    if llm is None and graph is not None:
        llm = graph_llm(graph, stream_mode=stream_mode)
    session_kwargs: dict[str, Any] = {
        "stt": stt if stt is not None else build_stt(p),
        "tts": tts if tts is not None else build_tts(p),
        "turn_handling": turn_handling if turn_handling is not None else build_turn_handling(p),
        "user_away_timeout": p.user_away_timeout,
    }
    if llm is not None:
        session_kwargs["llm"] = llm
    if vad is not None:
        session_kwargs["vad"] = vad
    session_kwargs.update(kwargs)
    return AgentSession(**session_kwargs)
