"""Run any LangGraph graph as a LiveKit voice agent with a few lines of code."""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from livekit.agents import Agent, AgentServer, AgentSession, JobContext, cli

from ..context import ContextStore
from ..presets import VoicePreset, get_preset
from ..turns import TurnGate
from .agent import ContextFormatter, DecisionCallback, GraphAgent, default_context_message
from .room import attach_room_context
from .session import build_session, graph_llm, room_options

log = logging.getLogger(__name__)

__all__ = ["GraphSession", "create_server", "job_metadata", "run_app", "start_graph_session"]


@dataclass
class GraphSession:
    session: AgentSession
    agent: Agent
    context: ContextStore
    detach_context: Callable[[], None]


def job_metadata(ctx: JobContext) -> dict[str, Any]:
    """Dispatch metadata as a dict (JSON object), or ``{}``; non-JSON metadata is under ``"raw"``."""
    raw = ctx.job.metadata or ""
    if not raw.strip():
        return {}
    try:
        value = json.loads(raw)
    except ValueError:
        return {"raw": raw}
    return value if isinstance(value, dict) else {"raw": value}


async def start_graph_session(
    ctx: JobContext,
    graph: Any,
    *,
    preset: str | VoicePreset = "default",
    gate: TurnGate | None = None,
    instructions: str = "",
    context: ContextStore | None = None,
    context_topic: str | None = "context",
    context_participant: str | None = None,
    context_message: ContextFormatter | None = default_context_message,
    on_decision: DecisionCallback | None = None,
    greeting: str | None = None,
    stream_mode: Any = "custom",
    graph_config: Any = None,
    graph_context: Any = None,
    agent: Agent | None = None,
    room_kwargs: dict[str, Any] | None = None,
    **session_kwargs: Any,
) -> GraphSession:
    """Connect, wire client context, and start an ``AgentSession`` driven by ``graph``.

    The graph receives ``{"messages": [...]}`` built from the conversation and should stream the
    spoken reply (``get_stream_writer()`` with ``stream_mode="custom"``, the default). ``agent``
    replaces the default :class:`GraphAgent` (then ``gate``/``instructions``/``context_message``/
    ``on_decision`` are ignored). Extra kwargs go to :func:`~voice_kit.livekit.build_session`.
    """
    p = get_preset(preset)
    await ctx.connect()
    store = context if context is not None else ContextStore()
    detach = attach_room_context(
        store, ctx.room, topic=context_topic, participant_identity=context_participant
    )
    session = build_session(
        p,
        llm=graph_llm(graph, stream_mode=stream_mode, config=graph_config, context=graph_context),
        **session_kwargs,
    )
    if agent is None:
        agent = GraphAgent(
            instructions=instructions,
            gate=gate,
            context=store,
            context_message=context_message,
            on_decision=on_decision,
        )
    await session.start(agent=agent, room=ctx.room, room_options=room_options(p, **(room_kwargs or {})))
    if greeting:
        session.say(greeting)
    return GraphSession(session=session, agent=agent, context=store, detach_context=detach)


def create_server(
    entrypoint: Callable[[JobContext], Awaitable[None]],
    *,
    agent_name: str = "",
    **kwargs: Any,
) -> AgentServer:
    """``AgentServer`` with ``entrypoint`` registered. A non-empty ``agent_name`` enables explicit
    dispatch (via token room config or :func:`voice_kit.tokens.dispatch_agent`)."""
    server = AgentServer()
    server.rtc_session(entrypoint, agent_name=agent_name, **kwargs)
    return server


def run_app(server_or_entrypoint: AgentServer | Callable[[JobContext], Awaitable[None]], *, agent_name: str = "") -> None:
    """CLI entry (``dev``, ``start``, ``console``...) for a server or a bare entrypoint."""
    server = (
        server_or_entrypoint
        if isinstance(server_or_entrypoint, AgentServer)
        else create_server(server_or_entrypoint, agent_name=agent_name)
    )
    cli.run_app(server)
