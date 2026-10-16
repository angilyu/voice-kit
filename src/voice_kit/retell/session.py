"""Retell custom-LLM websocket protocol (https://docs.retellai.com/api-references/llm-websocket).

:class:`RetellLLMSession` speaks the protocol (config, ping-pong, call details, streamed responses,
agent interrupts, filler while the brain is slow). What to say is up to a
:class:`RetellCallHandler`; :class:`GraphRetellHandler` streams a LangGraph graph.
"""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import json
import logging
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from starlette.types import ASGIApp, Receive, Scope, Send
from starlette.websockets import WebSocket, WebSocketDisconnect

from ..streaming import FILLER, message_text, strip_leading_filler
from .client import WS_PATH_PREFIX

log = logging.getLogger(__name__)

__all__ = [
    "ERROR_REPLY",
    "GraphRetellHandler",
    "RetellCallHandler",
    "RetellChunk",
    "RetellLLMSession",
    "RetellTurn",
    "RetellWebsocketRouter",
    "transcript_to_messages",
]

ERROR_REPLY = "Sorry, could you say that again?"


def transcript_to_messages(transcript: list[dict[str, Any]] | None) -> list[BaseMessage]:
    """Retell transcript (``role`` agent/user, ``content``) to LangChain messages; blanks dropped."""
    messages: list[BaseMessage] = []
    for utt in transcript or []:
        content = (utt.get("content") or "").strip()
        if not content:
            continue
        if utt.get("role") == "agent":
            messages.append(AIMessage(content=content))
        else:
            messages.append(HumanMessage(content=content))
    return messages


@dataclass
class RetellTurn:
    """One reply the session must produce. Handlers may set ``end_call``/``digit_to_press``."""

    kind: str
    """``response_required``, ``reminder_required`` or ``interrupt`` (agent speaks first)."""
    messages: list[BaseMessage]
    transcript: list[dict[str, Any]] = field(default_factory=list)
    response_id: int | None = None
    interrupt_id: int | None = None
    end_call: bool = False
    digit_to_press: str | None = None


@dataclass
class RetellChunk:
    content: str
    no_interruption_allowed: bool = False


class RetellCallHandler:
    """Override :meth:`respond` (required) and any hooks. ``self.session`` is set before use."""

    session: RetellLLMSession

    async def on_call_details(self, call: dict[str, Any]) -> bool:
        """Called with Retell's call object. Return False to reject the call (close 1008)."""
        return True

    async def on_transcript(self, transcript: list[dict[str, Any]]) -> None:
        """Called whenever Retell sends a changed transcript."""

    def respond(self, turn: RetellTurn) -> AsyncIterator[str | RetellChunk]:
        """Yield the reply text in chunks. May set ``turn.end_call``/``turn.digit_to_press``."""
        raise NotImplementedError

    def filler_wait(self, turn: RetellTurn) -> float | None:
        """Seconds to wait for the first chunk before saying a filler; None for no filler.

        Default: ``session.filler_seconds`` for replies to something the user said.
        """
        if turn.kind != "response_required" or not self.session.filler_seconds:
            return None
        last = turn.messages[-1] if turn.messages else None
        return self.session.filler_seconds if isinstance(last, HumanMessage) else None

    async def after_turn(self, turn: RetellTurn) -> None:
        """Called after the final message of a turn was sent."""

    async def on_close(self) -> None:
        """Called once when the websocket ends."""


class GraphRetellHandler(RetellCallHandler):
    """Streams ``graph.astream({"messages": ...})``; ``custom`` mode yields what the graph writes."""

    def __init__(self, graph: Any, *, stream_mode: str = "custom", config: Any = None) -> None:
        self.graph = graph
        self.stream_mode = stream_mode
        self.config = config

    def graph_input(self, turn: RetellTurn) -> dict[str, Any]:
        return {"messages": turn.messages}

    async def respond(self, turn: RetellTurn) -> AsyncIterator[str | RetellChunk]:
        async for item in self.graph.astream(
            self.graph_input(turn), self.config, stream_mode=self.stream_mode
        ):
            if isinstance(item, tuple):  # "messages" mode: (chunk, metadata)
                item = item[0]
            if isinstance(item, RetellChunk):
                yield item
            elif isinstance(item, str):
                yield item
            elif isinstance(item, dict):
                text = item.get("content")
                if isinstance(text, str):
                    yield text
            else:
                yield message_text(item)


class RetellLLMSession:
    """Handles one Retell custom-LLM websocket connection (one call)."""

    def __init__(
        self,
        ws: WebSocket,
        call_id: str,
        handler: RetellCallHandler,
        *,
        begin_message: str = "",
        filler_text: str = FILLER,
        filler_seconds: float | None = None,
        error_reply: str = ERROR_REPLY,
        bind_timeout: float = 10.0,
        auto_reconnect: bool = True,
    ) -> None:
        self.ws = ws
        self.call_id = call_id
        self.handler = handler
        handler.session = self
        # Spoken right after connect; "" lets the other side talk first.
        self.begin_message = begin_message
        self.filler_text = filler_text
        self.filler_seconds = filler_seconds
        self.error_reply = error_reply
        self.bind_timeout = bind_timeout
        self.auto_reconnect = auto_reconnect
        self.call: dict[str, Any] | None = None
        self.last_transcript: list[dict[str, Any]] | None = None
        self.heard_response_request = False
        self._bound = asyncio.Event()
        self._turn: asyncio.Task[None] | None = None
        self._interrupt_id = 0
        self._send_lock = asyncio.Lock()
        self._closed = False

    @property
    def turn_active(self) -> bool:
        return self._turn is not None and not self._turn.done()

    @property
    def closed(self) -> bool:
        return self._closed

    async def close(self, code: int = 1000) -> None:
        if not self._closed:
            self._closed = True
            with contextlib.suppress(Exception):
                await self.ws.close(code=code)

    async def send(self, payload: dict[str, Any]) -> None:
        async with self._send_lock:
            await self.ws.send_text(json.dumps(payload))

    async def update_agent(self, agent_config: dict[str, Any]) -> None:
        """Change Retell agent settings for this call (e.g. ``reminder_trigger_ms``)."""
        await self.send({"response_type": "update_agent", "agent_config": agent_config})

    def interrupt(self, messages: list[BaseMessage] | None = None) -> asyncio.Task[None]:
        """Speak without being asked (``agent_interrupt``), cancelling any running turn."""
        self._interrupt_id += 1
        turn = RetellTurn(
            kind="interrupt",
            messages=messages if messages is not None else transcript_to_messages(self.last_transcript),
            transcript=list(self.last_transcript or []),
            interrupt_id=self._interrupt_id,
        )
        return self._start(self._run_turn(turn))

    def _start(self, coro: Any) -> asyncio.Task[None]:
        if self._turn and not self._turn.done():
            self._turn.cancel()  # Retell discards responses to older ids anyway
        self._turn = asyncio.create_task(coro)
        return self._turn

    async def run(self) -> None:
        await self.ws.accept()
        await self.send(
            {
                "response_type": "config",
                "config": {"auto_reconnect": self.auto_reconnect, "call_details": True},
            }
        )
        await self.send(
            {
                "response_type": "response",
                "response_id": 0,
                "content": self.begin_message,
                "content_complete": True,
            }
        )
        try:
            while not self._closed:
                msg = json.loads(await self.ws.receive_text())
                await self._handle(msg)
        except (WebSocketDisconnect, RuntimeError):
            log.info("retell websocket closed (call %s)", self.call_id)
        finally:
            if self._turn and not self._turn.done():
                self._turn.cancel()
            with contextlib.suppress(Exception):
                await self.handler.on_close()

    async def _handle(self, msg: dict[str, Any]) -> None:
        kind = msg.get("interaction_type")
        if kind == "ping_pong":
            await self.send({"response_type": "ping_pong", "timestamp": msg.get("timestamp")})
        elif kind == "call_details":
            await self._bind(msg.get("call") or {})
        elif kind == "update_only":
            await self._save_transcript(msg.get("transcript"))
        elif kind in ("response_required", "reminder_required"):
            await self._save_transcript(msg.get("transcript"))
            if kind == "response_required":
                self.heard_response_request = True
            self._start(self._respond(msg))

    async def _bind(self, call: dict[str, Any]) -> None:
        self.call = call
        try:
            ok = await self.handler.on_call_details(call)
        except Exception:
            log.exception("on_call_details failed (call %s)", self.call_id)
            ok = False
        if ok is False:
            log.warning("rejecting retell call %s", self.call_id)
            await self.close(1008)
            return
        self._bound.set()

    async def _save_transcript(self, transcript: list[dict[str, Any]] | None) -> None:
        if transcript is None or transcript == self.last_transcript:
            return
        self.last_transcript = transcript
        try:
            await self.handler.on_transcript(transcript)
        except Exception:
            log.exception("on_transcript failed (call %s)", self.call_id)

    async def _respond(self, msg: dict[str, Any]) -> None:
        try:
            await asyncio.wait_for(self._bound.wait(), self.bind_timeout)
        except TimeoutError:
            log.error("no call_details for retell call %s; closing", self.call_id)
            await self.close(1011)
            return
        transcript = msg.get("transcript") or []
        turn = RetellTurn(
            kind=msg.get("interaction_type", "response_required"),
            messages=transcript_to_messages(transcript),
            transcript=transcript,
            response_id=msg.get("response_id"),
        )
        await self._run_turn(turn)

    async def _run_turn(self, turn: RetellTurn) -> None:
        if turn.interrupt_id is None:
            base: dict[str, Any] = {"response_type": "response", "response_id": turn.response_id}
        else:
            base = {"response_type": "agent_interrupt", "interrupt_id": turn.interrupt_id}
        filler_wait = self.handler.filler_wait(turn)
        chunks = self.handler.respond(turn).__aiter__()
        pending: asyncio.Future[Any] | None = asyncio.ensure_future(chunks.__anext__())
        filler_sent = False
        try:
            if filler_wait is not None and self.filler_text:
                done, _ = await asyncio.wait({pending}, timeout=filler_wait)
                if not done:
                    filler_sent = True
                    await self.send({**base, "content": self.filler_text, "content_complete": False})
            first = True
            while pending is not None:
                try:
                    item = await pending
                except StopAsyncIteration:
                    pending = None
                    break
                pending = asyncio.ensure_future(chunks.__anext__())
                chunk = item if isinstance(item, RetellChunk) else RetellChunk(str(item or ""))
                if filler_sent and first:
                    chunk.content = strip_leading_filler(chunk.content)  # no "Hmm, hmm, ..."
                first = False
                if not chunk.content:
                    continue
                event = {**base, "content": chunk.content, "content_complete": False}
                if chunk.no_interruption_allowed:
                    event["no_interruption_allowed"] = True
                await self.send(event)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("turn failed (retell call %s)", self.call_id)
            await self.send({**base, "content": self.error_reply, "content_complete": True})
            return
        finally:
            if pending is not None and not pending.done():
                pending.cancel()
                with contextlib.suppress(BaseException):
                    await pending
            aclose = getattr(chunks, "aclose", None)
            if aclose is not None:
                with contextlib.suppress(Exception):
                    await aclose()
        final: dict[str, Any] = {
            **base,
            "content": "",
            "content_complete": True,
            "end_call": turn.end_call,
        }
        if turn.digit_to_press:
            final["digit_to_press"] = turn.digit_to_press
        await self.send(final)
        try:
            await self.handler.after_turn(turn)
        except Exception:
            log.exception("after_turn failed (retell call %s)", self.call_id)


class RetellWebsocketRouter:
    """ASGI wrapper: serves Retell's custom-LLM websocket at ``{path_prefix}{secret}/{call_id}``
    and passes every other request to ``app``.

    Retell can't send a bearer token, so the secret path segment authenticates it; handlers can
    additionally reject unknown calls in :meth:`RetellCallHandler.on_call_details`.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        secret: str,
        handler_factory: Callable[[str], RetellCallHandler],
        path_prefix: str = WS_PATH_PREFIX,
        session_kwargs: dict[str, Any] | Callable[[], dict[str, Any]] | None = None,
    ) -> None:
        self.app = app
        self.secret = secret
        self.handler_factory = handler_factory
        self.path_prefix = "/" + path_prefix.strip("/") + "/"
        self.session_kwargs = session_kwargs

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "websocket":
            await self.app(scope, receive, send)
            return
        path: str = scope["path"]
        if not path.startswith(self.path_prefix):
            await self.app(scope, receive, send)
            return
        ws = WebSocket(scope, receive, send)
        supplied, _, call_id = path[len(self.path_prefix) :].partition("/")
        if not self.secret or not call_id or not hmac.compare_digest(supplied.encode(), self.secret.encode()):
            await ws.close(code=1008)
            return
        kwargs = self.session_kwargs() if callable(self.session_kwargs) else (self.session_kwargs or {})
        await RetellLLMSession(ws, call_id, self.handler_factory(call_id), **kwargs).run()
