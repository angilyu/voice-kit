"""Turn gates: decide whether the agent should answer a finished user turn at all.

A gate receives a :class:`UserTurn` and returns ``True``/``False`` or a :class:`TurnDecision`.
Gates can be sync or async. ``True`` means respond, ``False`` means ignore.

* ``RESPOND``: run the agent brain and speak.
* ``LISTEN``: keep the utterance in the conversation history but stay silent (for example a child
  reading the page aloud).
* ``IGNORE``: drop the utterance entirely (background chatter, TV, off-topic talk).

Gates are framework-free; :class:`voice_kit.livekit.GraphAgent` applies them inside LiveKit's
``on_user_turn_completed`` hook.
"""

from __future__ import annotations

import inspect
import logging
import re
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

log = logging.getLogger(__name__)

__all__ = [
    "TurnDecision",
    "TurnGate",
    "UserTurn",
    "always_respond",
    "chain",
    "decide",
    "is_question",
    "keyword_gate",
    "llm_gate",
    "min_words_gate",
    "question_gate",
    "reference_text_gate",
    "text_overlap",
]


class TurnDecision(StrEnum):
    RESPOND = "respond"
    LISTEN = "listen"
    IGNORE = "ignore"


@dataclass(frozen=True)
class UserTurn:
    text: str
    """Final transcript of the user's turn."""
    context: Mapping[str, Any] = field(default_factory=dict)
    """Client-pushed context (e.g. ``{"page": 3, "page_text": "..."}``); see :class:`ContextStore`."""
    history: Sequence[tuple[str, str]] = ()
    """Prior conversation as ``(role, text)`` pairs, oldest first; role is ``user`` or ``assistant``."""


TurnGate = Callable[[UserTurn], "bool | TurnDecision | Awaitable[bool | TurnDecision]"]


def _coerce(result: Any) -> TurnDecision:
    if isinstance(result, TurnDecision):
        return result
    if isinstance(result, str):
        return TurnDecision(result.strip().lower())
    return TurnDecision.RESPOND if result else TurnDecision.IGNORE


async def decide(gate: TurnGate | None, turn: UserTurn) -> TurnDecision:
    """Run a gate (sync or async). ``None`` always responds."""
    if gate is None:
        return TurnDecision.RESPOND
    result = gate(turn)
    if inspect.isawaitable(result):
        result = await result
    return _coerce(result)


def always_respond(turn: UserTurn) -> TurnDecision:
    return TurnDecision.RESPOND


def chain(*gates: TurnGate) -> TurnGate:
    """Run gates in order; the first non-RESPOND decision wins."""

    async def gate(turn: UserTurn) -> TurnDecision:
        for g in gates:
            decision = await decide(g, turn)
            if decision is not TurnDecision.RESPOND:
                return decision
        return TurnDecision.RESPOND

    return gate


_WORD = re.compile(r"[\w']+")


def _words(text: str) -> list[str]:
    return [w.lower() for w in _WORD.findall(text or "")]


def min_words_gate(n: int, *, otherwise: TurnDecision = TurnDecision.IGNORE) -> TurnGate:
    """Respond only when the turn has at least ``n`` words (drops coughs, "um", single stray words)."""

    def gate(turn: UserTurn) -> TurnDecision:
        return TurnDecision.RESPOND if len(_words(turn.text)) >= n else otherwise

    return gate


def keyword_gate(
    keywords: Iterable[str], *, otherwise: TurnDecision = TurnDecision.IGNORE
) -> TurnGate:
    """Respond only when the turn contains one of ``keywords`` (whole words/phrases, any case).

    Useful for wake words ("hey owl") or required topic words.
    """
    patterns = [re.compile(r"\b" + re.escape(k.strip().lower()) + r"\b") for k in keywords if k.strip()]

    def gate(turn: UserTurn) -> TurnDecision:
        text = (turn.text or "").lower()
        return TurnDecision.RESPOND if any(p.search(text) for p in patterns) else otherwise

    return gate


_QUESTION_START = re.compile(
    r"^\s*(?:(?:um+|uh+|hey|so|and|but|ok(?:ay)?|wait)[,\s]+)*"
    r"(?:what|what's|whats|why|how|who|who's|where|where's|when|which|whose|"
    r"is|are|am|was|were|do|does|did|can|could|will|would|should|shall|have|has|may|might|"
    r"tell me|explain|i wonder|i don't (?:know|understand|get)|what does)\b",
    re.I,
)


def is_question(text: str) -> bool:
    """Heuristic: ends with ``?`` or starts like a question/request for explanation.

    Speech-to-text often drops the question mark, so the opening words matter.
    """
    value = (text or "").strip()
    return bool(value) and (value.endswith("?") or bool(_QUESTION_START.match(value)))


def question_gate(*, otherwise: TurnDecision = TurnDecision.LISTEN) -> TurnGate:
    """Respond to questions; ``otherwise`` (default LISTEN) for statements."""

    def gate(turn: UserTurn) -> TurnDecision:
        return TurnDecision.RESPOND if is_question(turn.text) else otherwise

    return gate


def text_overlap(spoken: str, reference: str) -> float:
    """Fraction of the spoken words that appear in ``reference`` (0.0 to 1.0)."""
    spoken_words = _words(spoken)
    if not spoken_words:
        return 0.0
    ref = set(_words(reference))
    return sum(1 for w in spoken_words if w in ref) / len(spoken_words)


def reference_text_gate(
    context_key: str = "page_text",
    *,
    threshold: float = 0.6,
    min_words: int = 3,
    decision: TurnDecision = TurnDecision.LISTEN,
) -> TurnGate:
    """Return ``decision`` (default LISTEN) when the user is reading reference text aloud.

    The reference text comes from ``turn.context[context_key]``, e.g. the sentence on the page the
    app is showing. A turn counts as reading when at least ``min_words`` words were spoken and at
    least ``threshold`` of them appear in the reference. A question counts as reading only if its
    opening word is in the reference too (the page itself says "Where is the bear?").
    """

    def gate(turn: UserTurn) -> TurnDecision:
        reference = turn.context.get(context_key)
        spoken = _words(turn.text)
        if not reference or len(spoken) < min_words:
            return TurnDecision.RESPOND
        ref_words = set(_words(str(reference)))
        if is_question(turn.text) and spoken[0] not in ref_words:
            return TurnDecision.RESPOND
        if text_overlap(turn.text, str(reference)) >= threshold:
            return decision
        return TurnDecision.RESPOND

    return gate


_LLM_GATE_DEFAULT = """\
You decide whether a voice assistant should answer what its microphone just heard.
Reply with exactly one word:
RESPOND - the user is talking to the assistant and expects an answer.
LISTEN - the user is talking but no answer is wanted yet (for example reading aloud or thinking).
IGNORE - background noise, other people talking, or speech unrelated to the assistant's purpose."""


def llm_gate(
    model: Any,
    instructions: str = "",
    *,
    on_error: TurnDecision = TurnDecision.RESPOND,
    max_history: int = 6,
    context_keys: Sequence[str] | None = None,
) -> TurnGate:
    """Ask a (small, fast) LangChain chat model to classify the turn.

    ``instructions`` describe the assistant's purpose (e.g. "Only answer questions about the
    picture book on screen"). The model must answer RESPOND, LISTEN or IGNORE; anything else, or an
    error, yields ``on_error``. ``context_keys`` limits which context entries are shown (default all).
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    system = _LLM_GATE_DEFAULT + (f"\n\nAssistant purpose:\n{instructions.strip()}" if instructions else "")

    async def gate(turn: UserTurn) -> TurnDecision:
        lines: list[str] = []
        ctx = {k: v for k, v in turn.context.items() if context_keys is None or k in context_keys}
        if ctx:
            lines.append("Context:")
            lines.extend(f"- {k}: {v}" for k, v in ctx.items())
        recent = list(turn.history)[-max_history:] if max_history else []
        if recent:
            lines.append("Recent conversation:")
            lines.extend(f"{role}: {text}" for role, text in recent)
        lines.append(f"Just heard: {turn.text}")
        try:
            reply = await model.ainvoke([SystemMessage(system), HumanMessage("\n".join(lines))])
        except Exception:
            log.exception("llm_gate failed; using %s", on_error)
            return on_error
        content = getattr(reply, "content", reply)
        if isinstance(content, list):
            content = " ".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
        match = re.search(r"\b(RESPOND|LISTEN|IGNORE)\b", str(content), re.I)
        return TurnDecision(match.group(1).lower()) if match else on_error

    return gate
