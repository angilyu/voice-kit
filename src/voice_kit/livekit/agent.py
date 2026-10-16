"""A LiveKit ``Agent`` that gates turns and feeds client context to a LangGraph brain."""

from __future__ import annotations

import inspect
import json
import logging
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from livekit.agents import Agent, llm
from livekit.agents.llm import StopResponse

from ..context import ContextStore
from ..turns import TurnDecision, TurnGate, UserTurn, decide

log = logging.getLogger(__name__)

__all__ = ["ContextFormatter", "DecisionCallback", "GraphAgent", "default_context_message"]

ContextFormatter = Callable[[Mapping[str, Any]], str | None]
DecisionCallback = Callable[[UserTurn, TurnDecision], "None | Awaitable[None]"]


def default_context_message(context: Mapping[str, Any]) -> str | None:
    """System note with the latest client context, or ``None`` when there is none."""
    if not context:
        return None
    lines = ["Current app context (from the user's device):"]
    for key, value in context.items():
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        lines.append(f"- {key}: {text}")
    return "\n".join(lines)


def _history(chat_ctx: llm.ChatContext, exclude_id: str | None = None) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for item in chat_ctx.items:
        if getattr(item, "type", None) != "message" or getattr(item, "id", None) == exclude_id:
            continue
        role = getattr(item, "role", None)
        text = getattr(item, "text_content", None)
        if role in ("user", "assistant") and text:
            out.append((role, text))
    return out


class GraphAgent(Agent):
    """``Agent`` whose replies come from the session LLM (e.g. a LangGraph graph), with:

    * ``gate``: a :data:`~voice_kit.turns.TurnGate` deciding RESPOND / LISTEN / IGNORE per turn.
      LISTEN keeps the utterance in the chat history without replying; IGNORE drops it.
    * ``context``: a :class:`~voice_kit.context.ContextStore` (fed by
      :func:`~voice_kit.livekit.attach_room_context`). Its snapshot is passed to the gate and, via
      ``context_message``, injected as a system message for the reply being generated (not kept in
      history). Pass ``context_message=None`` to skip injection, e.g. when the graph reads the store
      itself.
    * ``on_decision``: called with every turn and decision (sync or async), e.g. for logging.
    """

    def __init__(
        self,
        *,
        instructions: str = "",
        gate: TurnGate | None = None,
        context: ContextStore | None = None,
        context_message: ContextFormatter | None = default_context_message,
        on_decision: DecisionCallback | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(instructions=instructions, **kwargs)
        self.gate = gate
        self.context = context if context is not None else ContextStore()
        self.context_message = context_message
        self.on_decision = on_decision

    async def on_user_turn_completed(
        self, turn_ctx: llm.ChatContext, new_message: llm.ChatMessage
    ) -> None:
        snapshot = self.context.snapshot()
        turn = UserTurn(
            text=new_message.text_content or "",
            context=snapshot,
            history=_history(turn_ctx, exclude_id=new_message.id),
        )
        decision = await decide(self.gate, turn)
        if self.on_decision is not None:
            try:
                result = self.on_decision(turn, decision)
                if inspect.isawaitable(result):
                    await result
            except Exception:
                log.exception("on_decision callback failed")
        if decision is TurnDecision.IGNORE:
            raise StopResponse()
        if decision is TurnDecision.LISTEN:
            # StopResponse skips adding the user message; keep it so later answers see it.
            ctx = self.chat_ctx.copy()
            ctx.items.append(new_message)
            await self.update_chat_ctx(ctx)
            raise StopResponse()
        if self.context_message is not None:
            note = self.context_message(snapshot)
            if note:
                turn_ctx.add_message(role="system", content=note)
