import json

import pytest

pytest.importorskip("livekit.agents")

from livekit.agents import StopResponse, llm  # noqa: E402

from voice_kit.context import ContextStore  # noqa: E402
from voice_kit.livekit import (  # noqa: E402
    GraphAgent,
    attach_room_context,
    build_turn_handling,
    default_context_message,
    room_options,
)
from voice_kit.presets import get_preset  # noqa: E402
from voice_kit.turns import TurnDecision  # noqa: E402


def test_build_turn_handling_from_patient_preset():
    th = build_turn_handling("patient")
    assert th["endpointing"]["min_delay"] == 1.6
    assert th["turn_detection"] is not None
    assert build_turn_handling(get_preset("default", turn_detection="vad"))["turn_detection"] == "vad"


def test_room_options_without_noise_cancellation():
    opts = room_options("default", close_on_disconnect=False)
    assert opts.close_on_disconnect is False


def test_default_context_message():
    assert default_context_message({}) is None
    msg = default_context_message({"page": 3, "page_text": "The bear."})
    assert "page" in msg and "The bear." in msg


def _message(text):
    return llm.ChatMessage(role="user", content=[text])


async def _run(agent, text):
    turn_ctx = llm.ChatContext()
    updates = []

    async def fake_update(ctx):
        updates.append(ctx)

    agent.update_chat_ctx = fake_update
    try:
        await agent.on_user_turn_completed(turn_ctx, _message(text))
    except StopResponse:
        return "stopped", turn_ctx, updates
    return "respond", turn_ctx, updates


async def test_graph_agent_applies_gate():
    decisions = []
    store = ContextStore({"page": 2})

    def gate(turn):
        assert turn.context == {"page": 2}
        return {"read": TurnDecision.LISTEN, "noise": TurnDecision.IGNORE}.get(turn.text, TurnDecision.RESPOND)

    agent = GraphAgent(gate=gate, context=store, on_decision=lambda turn, d: decisions.append(d))

    outcome, _, updates = await _run(agent, "noise")
    assert outcome == "stopped" and updates == []

    outcome, _, updates = await _run(agent, "read")
    assert outcome == "stopped" and len(updates) == 1
    assert updates[0].items[-1].text_content == "read"

    outcome, turn_ctx, updates = await _run(agent, "why?")
    assert outcome == "respond" and updates == []
    assert any("page" in (item.text_content or "") for item in turn_ctx.items)
    assert decisions == [TurnDecision.IGNORE, TurnDecision.LISTEN, TurnDecision.RESPOND]


class FakeParticipant:
    def __init__(self, identity, attributes=None):
        self.identity = identity
        self.attributes = attributes or {}


class FakePacket:
    def __init__(self, data, topic, participant):
        self.data, self.topic, self.participant = data, topic, participant


class FakeRoom:
    def __init__(self):
        self.handlers = {}
        self.text_handlers = {}
        self.local_participant = FakeParticipant("agent")
        self.remote_participants = {"kid": FakeParticipant("kid", {"page": "1", "lk.agent.state": "x"})}

    def on(self, event, cb=None):
        self.handlers.setdefault(event, []).append(cb)
        return cb

    def off(self, event, cb):
        self.handlers[event].remove(cb)

    def emit(self, event, *args):
        for cb in list(self.handlers.get(event, [])):
            cb(*args)

    def register_text_stream_handler(self, topic, handler):
        self.text_handlers[topic] = handler

    def unregister_text_stream_handler(self, topic):
        self.text_handlers.pop(topic, None)


def test_attach_room_context_attributes_and_data():
    store = ContextStore()
    room = FakeRoom()
    detach = attach_room_context(store, room)
    assert store.snapshot() == {"page": "1"}

    kid = room.remote_participants["kid"]
    room.emit("participant_attributes_changed", {"page": "2", "title": ""}, kid)
    assert store["page"] == "2"
    room.emit("participant_attributes_changed", {"page": "99"}, room.local_participant)
    assert store["page"] == "2"
    room.emit("data_received", FakePacket(json.dumps({"page_text": "Hi"}).encode(), "context", kid))
    room.emit("data_received", FakePacket(b'{"page": "x"}', "other", kid))
    assert store["page_text"] == "Hi" and store["page"] == "2"
    assert "context" in room.text_handlers

    detach()
    assert "context" not in room.text_handlers
    room.emit("participant_attributes_changed", {"page": "3"}, kid)
    assert store["page"] == "2"
