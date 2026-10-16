import pytest

from voice_kit.presets import (
    DEFAULT,
    PATIENT,
    VoicePreset,
    get_preset,
    preset_from_env,
    register_preset,
)


def test_default_preset_keeps_livekit_defaults():
    assert DEFAULT.turn_handling() == {}
    assert DEFAULT.noise_cancellation == "none"


def test_patient_preset_waits_longer_and_cancels_background_voices():
    th = PATIENT.turn_handling()
    assert th["endpointing"]["min_delay"] >= 1.5
    assert th["endpointing"]["max_delay"] >= 5
    assert th["interruption"]["min_words"] >= 2
    assert th["preemptive_generation"] == {"enabled": False}
    assert PATIENT.noise_cancellation == "bvc"
    assert "turn_detection" not in th  # model detector is added by voice_kit.livekit


def test_get_preset_by_name_with_overrides():
    p = get_preset("Patient", tts_voice="v1", min_endpointing_delay=2.0)
    assert p.name == "patient" and p.tts_voice == "v1"
    assert p.turn_handling()["endpointing"]["min_delay"] == 2.0
    assert get_preset(p) is p
    assert get_preset("telephony").noise_cancellation == "bvc_telephony"
    with pytest.raises(ValueError):
        get_preset("nope")


def test_register_custom_preset():
    register_preset(VoicePreset(name="vk-test", turn_detection="vad"))
    assert get_preset("vk-test").turn_handling() == {"turn_detection": "vad"}


def test_preset_from_env(monkeypatch):
    monkeypatch.setenv("APP_VOICE_PRESET", "patient")
    monkeypatch.setenv("APP_TTS_VOICE", "kid-voice")
    monkeypatch.setenv("APP_MIN_ENDPOINTING_DELAY", "2.5")
    monkeypatch.setenv("APP_NOISE_CANCELLATION", "nc")
    p = preset_from_env(prefix="APP_")
    assert p.name == "patient" and p.tts_voice == "kid-voice"
    assert p.min_endpointing_delay == 2.5 and p.noise_cancellation == "nc"
    assert preset_from_env("telephony", prefix="NONE_").name == "telephony"
