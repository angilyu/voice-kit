import asyncio
import json

import httpx
import pytest

pytest.importorskip("starlette")

from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402
from starlette.applications import Starlette  # noqa: E402
from starlette.responses import PlainTextResponse  # noqa: E402
from starlette.routing import Route  # noqa: E402
from starlette.testclient import TestClient  # noqa: E402
from starlette.websockets import WebSocketDisconnect  # noqa: E402

from voice_kit.retell import (  # noqa: E402
    GraphRetellHandler,
    RetellCallHandler,
    RetellChunk,
    RetellClient,
    RetellError,
    RetellWebsocketRouter,
    retell_llm_websocket_url,
    sync_agent_websocket_url,
    transcript_to_messages,
)

SECRET = "s3cret"


def _collect(ws, response_id):
    text, chunks = "", []
    while True:
        msg = ws.receive_json()
        assert msg["response_type"] == "response" and msg["response_id"] == response_id
        chunks.append(msg)
        text += msg["content"]
        if msg["content_complete"]:
            return text, msg, chunks


class EchoHandler(RetellCallHandler):
    def __init__(self, call_id, delay=0.0):
        self.call_id = call_id
        self.delay = delay
        self.transcripts = []
        self.after = []
        self.closed = False

    async def on_call_details(self, call):
        return call.get("metadata", {}).get("ok", True)

    async def on_transcript(self, transcript):
        self.transcripts.append(transcript)

    async def respond(self, turn):
        last = turn.messages[-1].content if turn.messages else ""
        if self.delay:
            await asyncio.sleep(self.delay)
        if last == "boom":
            raise RuntimeError("brain failed")
        if last == "bye":
            turn.end_call = True
            yield RetellChunk("Goodbye!", no_interruption_allowed=True)
            return
        if last == "press":
            turn.digit_to_press = "1"
        yield "Hmm, you said "
        yield last

    async def after_turn(self, turn):
        self.after.append(turn.response_id)
        if turn.end_call:
            await self.session.update_agent({"reminder_max_count": 0})

    async def on_close(self):
        self.closed = True


def _app(handlers, **session_kwargs):
    async def home(request):
        return PlainTextResponse("ok")

    def factory(call_id):
        handlers[call_id] = EchoHandler(call_id, **handlers.pop("_kwargs", {}))
        return handlers[call_id]

    inner = Starlette(routes=[Route("/", home)])
    return RetellWebsocketRouter(inner, secret=SECRET, handler_factory=factory, session_kwargs=session_kwargs)


def _user(text):
    return [{"role": "user", "content": text}]


def test_websocket_url_helper():
    assert retell_llm_websocket_url("https://x.example/", SECRET) == f"wss://x.example/retell/llm/{SECRET}"
    assert retell_llm_websocket_url("http://localhost:8000", "k", "/ws/") == "ws://localhost:8000/ws/k"
    with pytest.raises(ValueError):
        retell_llm_websocket_url("", SECRET)


def test_transcript_to_messages():
    msgs = transcript_to_messages(
        [{"role": "agent", "content": "Hi"}, {"role": "user", "content": " "}, {"role": "user", "content": "Yo"}]
    )
    assert [type(m) for m in msgs] == [AIMessage, HumanMessage]
    assert transcript_to_messages(None) == []


def test_conversation_protocol():
    handlers = {}
    client = TestClient(_app(handlers, begin_message="Hello!"))
    assert client.get("/").text == "ok"
    with client.websocket_connect(f"/retell/llm/{SECRET}/rc_1") as ws:
        assert ws.receive_json() == {"response_type": "config", "config": {"auto_reconnect": True, "call_details": True}}
        assert ws.receive_json() == {"response_type": "response", "response_id": 0, "content": "Hello!", "content_complete": True}
        ws.send_json({"interaction_type": "ping_pong", "timestamp": 5})
        assert ws.receive_json() == {"response_type": "ping_pong", "timestamp": 5}
        ws.send_json({"interaction_type": "call_details", "call": {"metadata": {}}})
        ws.send_json({"interaction_type": "update_only", "transcript": _user("hel")})
        ws.send_json({"interaction_type": "response_required", "response_id": 1, "transcript": _user("hello")})
        text, last, _ = _collect(ws, 1)
        assert text == "Hmm, you said hello" and last["end_call"] is False and "digit_to_press" not in last

        ws.send_json({"interaction_type": "response_required", "response_id": 2, "transcript": _user("press")})
        _, last, _ = _collect(ws, 2)
        assert last["digit_to_press"] == "1"

        ws.send_json({"interaction_type": "response_required", "response_id": 3, "transcript": _user("boom")})
        text, last, _ = _collect(ws, 3)
        assert text == "Sorry, could you say that again?"

        ws.send_json({"interaction_type": "response_required", "response_id": 4, "transcript": _user("bye")})
        text, last, chunks = _collect(ws, 4)
        assert text == "Goodbye!" and last["end_call"] is True
        assert chunks[0]["no_interruption_allowed"] is True
        assert ws.receive_json() == {"response_type": "update_agent", "agent_config": {"reminder_max_count": 0}}
    h = handlers["rc_1"]
    assert h.closed
    assert h.after == [1, 2, 4]
    assert h.transcripts[0] == _user("hel")


def test_filler_is_spoken_while_slow_and_not_repeated():
    handlers = {"_kwargs": {"delay": 0.3}}
    client = TestClient(_app(handlers, filler_seconds=0.05))
    with client.websocket_connect(f"/retell/llm/{SECRET}/rc_2") as ws:
        ws.receive_json(), ws.receive_json()
        ws.send_json({"interaction_type": "call_details", "call": {}})
        ws.send_json({"interaction_type": "response_required", "response_id": 1, "transcript": _user("hi")})
        text, _, chunks = _collect(ws, 1)
        assert chunks[0]["content"] == "Hmm, "
        assert text == "Hmm, you said hi"


def test_rejected_call_and_bad_secret_close():
    handlers = {}
    client = TestClient(_app(handlers))
    with pytest.raises(WebSocketDisconnect) as exc:
        with client.websocket_connect(f"/retell/llm/wrong/rc_3") as ws:
            ws.receive_json()
    assert exc.value.code == 1008
    with client.websocket_connect(f"/retell/llm/{SECRET}/rc_4") as ws:
        ws.receive_json(), ws.receive_json()
        ws.send_json({"interaction_type": "call_details", "call": {"metadata": {"ok": False}}})
        with pytest.raises(WebSocketDisconnect) as exc:
            ws.receive_json()
        assert exc.value.code == 1008


class FakeGraph:
    def __init__(self, items):
        self.items = items
        self.inputs = []

    async def astream(self, inputs, config=None, stream_mode="custom"):
        self.inputs.append((inputs, stream_mode))
        for item in self.items:
            yield item


async def test_graph_handler_normalizes_stream_items():
    graph = FakeGraph(["a", {"content": "b"}, (AIMessage("c"), {}), RetellChunk("d"), {"other": 1}])
    handler = GraphRetellHandler(graph)
    from voice_kit.retell import RetellTurn

    out = [x async for x in handler.respond(RetellTurn("response_required", [HumanMessage("q")]))]
    assert [x.content if isinstance(x, RetellChunk) else x for x in out] == ["a", "b", "c", "d"]
    assert graph.inputs[0][0]["messages"][0].content == "q"


async def test_interrupt_speaks_first():
    sent = []

    class FakeWS:
        async def send_text(self, text):
            sent.append(json.loads(text))

    class Hello(RetellCallHandler):
        async def respond(self, turn):
            yield "Are you there?"

    from voice_kit.retell import RetellLLMSession

    session = RetellLLMSession(FakeWS(), "c", Hello())
    await session.interrupt([])
    assert sent[0] == {"response_type": "agent_interrupt", "interrupt_id": 1, "content": "Are you there?", "content_complete": False}
    assert sent[-1]["content_complete"] and sent[-1]["interrupt_id"] == 1
    assert not session.turn_active


async def test_client_and_sync_agent_url():
    calls = []

    def handler(request):
        calls.append((request.method, request.url.path, request.headers["authorization"], request.content))
        if request.url.path == "/get-agent/ag":
            return httpx.Response(200, json={"response_engine": {"type": "custom-llm", "llm_websocket_url": "wss://old"}})
        if request.url.path == "/get-call/bad":
            return httpx.Response(404, text="nope")
        return httpx.Response(200, json={"ok": True})

    async with RetellClient("key", transport=httpx.MockTransport(handler)) as client:
        assert await sync_agent_websocket_url(client, "ag", "wss://new") == "wss://new"
        with pytest.raises(RetellError):
            await client._req("GET", "/get-call/bad")
    assert calls[0][2] == "Bearer key"
    assert calls[1][0] == "PATCH" and json.loads(calls[1][3]) == {
        "response_engine": {"type": "custom-llm", "llm_websocket_url": "wss://new"}
    }
