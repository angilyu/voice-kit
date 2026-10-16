"""Named speech/turn-taking presets.

A :class:`VoicePreset` is plain data (no LiveKit import) describing STT/TTS models, turn
detection, endpointing, interruption, noise cancellation and preemptive generation. The LiveKit
helpers in :mod:`voice_kit.livekit` turn it into ``AgentSession`` arguments; ``None`` fields leave
the LiveKit default in place.

Presets:

``default``
    Muse's current LiveKit session: AssemblyAI STT, Fish Audio TTS, LiveKit turn-detector model,
    LiveKit defaults for everything else.
``telephony``
    ``default`` plus telephony-tuned noise cancellation (BVC telephony) for SIP calls.
``patient``
    For slow, quiet or hesitant speakers such as young children reading aloud: long end-of-turn
    delay, interruption only on deliberate speech, background-voice noise cancellation and no
    preemptive generation (nothing is generated until the speaker has really finished).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Literal

from .env import env_float, env_str

__all__ = [
    "NoiseCancellation",
    "PRESETS",
    "TurnDetection",
    "VoicePreset",
    "get_preset",
    "preset_from_env",
    "register_preset",
]

NoiseCancellation = Literal["none", "nc", "bvc", "bvc_telephony"]
"""``nc`` = standard noise cancellation, ``bvc`` = background voice + noise cancellation (removes
other people talking in the room), ``bvc_telephony`` = BVC tuned for phone audio."""

TurnDetection = Literal["model", "vad", "stt", "manual", "auto"]
"""``model`` = LiveKit's end-of-turn model (``inference.TurnDetector``); ``auto`` lets the session
choose."""


@dataclass(frozen=True)
class VoicePreset:
    name: str = "default"
    stt_model: str = "assemblyai/universal-3-5-pro"
    tts_model: str = "fishaudio/s2.1-pro"
    tts_voice: str | None = "fa4c9eb3dccc4806b382b40d61c6b10a"
    language: str = "en"
    turn_detection: TurnDetection = "model"
    # Seconds of silence before the turn may end / must end. None keeps LiveKit's default (0.5/3.0).
    min_endpointing_delay: float | None = None
    max_endpointing_delay: float | None = None
    allow_interruptions: bool | None = None
    # Speech needed before the agent stops talking. None keeps LiveKit's default (0.5 s / 0 words).
    min_interruption_duration: float | None = None
    min_interruption_words: int | None = None
    resume_false_interruption: bool | None = None
    false_interruption_timeout: float | None = None
    # None keeps LiveKit's default (enabled).
    preemptive_generation: bool | None = None
    noise_cancellation: NoiseCancellation = "none"
    # Seconds of mutual silence before the user is marked "away". None disables.
    user_away_timeout: float | None = 15.0

    def with_overrides(self, **changes: Any) -> VoicePreset:
        return dataclasses.replace(self, **changes)

    def turn_handling(self) -> dict[str, Any]:
        """``TurnHandlingOptions``-shaped dict, without the turn-detector object.

        ``turn_detection`` is only set for the string modes; :func:`voice_kit.livekit.build_turn_handling`
        adds the model detector for ``"model"``.
        """
        out: dict[str, Any] = {}
        if self.turn_detection in ("vad", "stt", "manual"):
            out["turn_detection"] = self.turn_detection
        endpointing: dict[str, Any] = {}
        if self.min_endpointing_delay is not None:
            endpointing["min_delay"] = self.min_endpointing_delay
        if self.max_endpointing_delay is not None:
            endpointing["max_delay"] = self.max_endpointing_delay
        if endpointing:
            out["endpointing"] = endpointing
        interruption: dict[str, Any] = {}
        if self.allow_interruptions is not None:
            interruption["enabled"] = self.allow_interruptions
        if self.min_interruption_duration is not None:
            interruption["min_duration"] = self.min_interruption_duration
        if self.min_interruption_words is not None:
            interruption["min_words"] = self.min_interruption_words
        if self.resume_false_interruption is not None:
            interruption["resume_false_interruption"] = self.resume_false_interruption
        if self.false_interruption_timeout is not None:
            interruption["false_interruption_timeout"] = self.false_interruption_timeout
        if interruption:
            out["interruption"] = interruption
        if self.preemptive_generation is not None:
            out["preemptive_generation"] = {"enabled": self.preemptive_generation}
        return out


DEFAULT = VoicePreset()

TELEPHONY = DEFAULT.with_overrides(name="telephony", noise_cancellation="bvc_telephony")

PATIENT = DEFAULT.with_overrides(
    name="patient",
    # Children pause mid-sentence while sounding out words; wait well past a normal pause.
    min_endpointing_delay=1.6,
    max_endpointing_delay=6.0,
    # Coughs, "um", page turns and short background words should not cut the agent off.
    allow_interruptions=True,
    min_interruption_duration=0.8,
    min_interruption_words=2,
    resume_false_interruption=True,
    false_interruption_timeout=1.5,
    # Don't start answering while the speaker may still be mid-thought.
    preemptive_generation=False,
    # Remove other voices (TV, siblings, adults talking) and background noise.
    noise_cancellation="bvc",
    user_away_timeout=60.0,
)

PRESETS: dict[str, VoicePreset] = {p.name: p for p in (DEFAULT, TELEPHONY, PATIENT)}


def register_preset(preset: VoicePreset) -> VoicePreset:
    """Add (or replace) a named preset so ``get_preset(name)`` finds it."""
    PRESETS[preset.name] = preset
    return preset


def get_preset(name: str | VoicePreset = "default", **overrides: Any) -> VoicePreset:
    """Look up a preset by name (case-insensitive) and apply field overrides."""
    if isinstance(name, VoicePreset):
        preset = name
    else:
        key = (name or "default").strip().lower()
        if key not in PRESETS:
            raise ValueError(f"Unknown voice preset {name!r}; choose one of {sorted(PRESETS)}")
        preset = PRESETS[key]
    return preset.with_overrides(**overrides) if overrides else preset


def preset_from_env(default: str = "default", *, prefix: str = "") -> VoicePreset:
    """Preset named by ``{prefix}VOICE_PRESET`` with optional env overrides.

    Recognised variables (all optional, all prefixed): ``VOICE_PRESET``, ``STT_MODEL``,
    ``STT_LANGUAGE``, ``TTS_MODEL``, ``TTS_VOICE``, ``MIN_ENDPOINTING_DELAY``,
    ``MAX_ENDPOINTING_DELAY``, ``NOISE_CANCELLATION``.
    """
    preset = get_preset(env_str(f"{prefix}VOICE_PRESET", default))
    changes: dict[str, Any] = {}
    for env_name, field_name in (
        ("STT_MODEL", "stt_model"),
        ("STT_LANGUAGE", "language"),
        ("TTS_MODEL", "tts_model"),
        ("TTS_VOICE", "tts_voice"),
    ):
        value = env_str(f"{prefix}{env_name}")
        if value:
            changes[field_name] = value
    for env_name, field_name in (
        ("MIN_ENDPOINTING_DELAY", "min_endpointing_delay"),
        ("MAX_ENDPOINTING_DELAY", "max_endpointing_delay"),
    ):
        if env_str(f"{prefix}{env_name}"):
            changes[field_name] = env_float(f"{prefix}{env_name}", 0.0)
    nc = env_str(f"{prefix}NOISE_CANCELLATION").lower()
    if nc:
        if nc not in ("none", "nc", "bvc", "bvc_telephony"):
            raise ValueError(f"Unknown NOISE_CANCELLATION {nc!r}")
        changes["noise_cancellation"] = nc
    return preset.with_overrides(**changes) if changes else preset
