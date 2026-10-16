"""Streaming text helpers for spoken LLM output.

All helpers are incremental: feed them chunks as they arrive and ``flush()`` at the end.
"""

from __future__ import annotations

import json
import re
from typing import Any

__all__ = [
    "FILLER",
    "OpenerDropper",
    "SayStreamer",
    "ToolMarkupFilter",
    "WordGate",
    "chunk_text",
    "is_farewell",
    "message_text",
    "strip_leading_filler",
    "strip_opener",
]

FILLER = "Hmm, "
"""Short sound spoken while a slow reply is still being generated."""

_LEADING_FILLER = re.compile(r"^\s*(?:hmm+|um+|uh+)\b[,.!]?\s*", re.I)

_TOOL_MARKER = "<function="
_TOOL_MARKUP = re.compile(r"<function=.*?/>", re.S)


def strip_leading_filler(text: str) -> str:
    """Drop a leading "Hmm,"/"Um,"/"Uh," so it isn't repeated after a spoken filler."""
    return _LEADING_FILLER.sub("", text, count=1)


def chunk_text(chunk: Any) -> str:
    """Text content of a LangChain message/chunk (string or content-part list)."""
    return message_text(chunk)


def message_text(msg: Any) -> str:
    content = getattr(msg, "content", msg)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part) for part in content
        )
    return "" if content is None else str(content)


class ToolMarkupFilter:
    """Removes ``<function=...>...</>``-style inline tool markup some models emit as text.

    Text before a possible marker is released immediately; a partial marker at the end of a chunk
    is held back until the next chunk shows whether it is markup.
    """

    def __init__(self) -> None:
        self._buf = ""

    def feed(self, text: str) -> str:
        self._buf += text
        out: list[str] = []
        while self._buf:
            start = self._buf.find(_TOOL_MARKER)
            if start < 0:
                keep = self._partial_marker_len(self._buf)
                out.append(self._buf[: len(self._buf) - keep])
                self._buf = self._buf[len(self._buf) - keep :]
                break
            out.append(self._buf[:start])
            end = self._buf.find("/>", start + len(_TOOL_MARKER))
            if end < 0:
                self._buf = self._buf[start:]
                break
            self._buf = self._buf[end + 2 :]
        return "".join(out)

    def flush(self) -> str:
        buf, self._buf = self._buf, ""
        if buf.startswith(_TOOL_MARKER):
            return ""
        return _TOOL_MARKUP.sub("", buf)

    @staticmethod
    def _partial_marker_len(text: str) -> int:
        for n in range(min(len(text), len(_TOOL_MARKER) - 1), 0, -1):
            if _TOOL_MARKER.startswith(text[-n:]):
                return n
        return 0


_PARTIAL_FIELD_CACHE: dict[str, re.Pattern[str]] = {}


def _partial_field_re(field: str) -> re.Pattern[str]:
    pattern = _PARTIAL_FIELD_CACHE.get(field)
    if pattern is None:
        pattern = re.compile(r'"' + re.escape(field) + r'"\s*:\s*"((?:[^"\\]|\\.)*)(")?')
        _PARTIAL_FIELD_CACHE[field] = pattern
    return pattern


class SayStreamer:
    """Streams one string argument of a tool call while its JSON arguments are still arriving.

    Lets a model speak a final line through a tool (e.g. ``record_outcome(say="Bye!")``) without
    waiting for the whole call. Feed every ``AIMessageChunk``; each call returns newly decoded text.
    """

    def __init__(self, tool_name: str = "record_outcome", field: str = "say") -> None:
        self.tool_name = tool_name
        self.field = field
        self._pattern = _partial_field_re(field)
        self._args: dict[int, str] = {}
        self._names: dict[int, str] = {}
        self.spoken = ""
        self.done = False

    def feed(self, chunk: Any) -> str:
        if self.done:
            return ""
        for tc in getattr(chunk, "tool_call_chunks", None) or []:
            idx = tc.get("index") or 0
            if tc.get("name"):
                self._names[idx] = tc["name"]
            if tc.get("args"):
                self._args[idx] = self._args.get(idx, "") + tc["args"]
        for idx, name in self._names.items():
            if name != self.tool_name:
                continue
            match = self._pattern.search(self._args.get(idx, ""))
            if not match:
                continue
            raw = match.group(1)
            if match.group(2):
                self.done = True
            else:
                # Drop an escape sequence that was cut mid-chunk.
                raw = re.sub(r"\\(?:u[0-9a-fA-F]{0,3})?$", "", raw)
            try:
                text = json.loads(f'"{raw}"')
            except ValueError:
                return ""
            new = text[len(self.spoken) :] if text.startswith(self.spoken) else ""
            self.spoken = text if text.startswith(self.spoken) else self.spoken
            return new
        return ""


class OpenerDropper:
    """Drops a fixed opener (e.g. a disclosure already spoken) if the model repeats it."""

    def __init__(self, opener: str) -> None:
        self.opener = opener.strip().lower()
        self.buf = ""
        self.done = not self.opener

    def feed(self, text: str) -> str:
        if self.done:
            return text
        self.buf += text
        head = self.buf.lstrip().lower()
        if len(head) < len(self.opener) and self.opener.startswith(head):
            return ""
        self.done = True
        out, self.buf = self.buf, ""
        if head.startswith(self.opener):
            return out.lstrip()[len(self.opener) :].lstrip()
        return out

    def flush(self) -> str:
        self.done = True
        out, self.buf = self.buf, ""
        return out


def strip_opener(text: str, opener: str) -> str:
    """Remove ``opener`` from the start of ``text`` (case-insensitive)."""
    stripped = text.lstrip()
    if opener and stripped.lower().startswith(opener.strip().lower()):
        return stripped[len(opener.strip()) :].lstrip()
    return text


class WordGate:
    """Holds streamed text until it contains a word character.

    Prevents speaking stray punctuation/whitespace (or committing to a reply) before the model has
    produced any real content.
    """

    _WORD = re.compile(r"\w")

    def __init__(self) -> None:
        self._buf = ""
        self.open = False

    def feed(self, text: str) -> str:
        if self.open:
            return text
        self._buf += text
        if self._WORD.search(self._buf):
            self.open = True
            out, self._buf = self._buf, ""
            return out
        return ""

    def flush(self) -> str:
        out, self._buf = self._buf, ""
        return out if self.open else ""


_FAREWELL = re.compile(
    r"^(?:(?:ok(?:ay)?|alright|all right|great|perfect|awesome|sounds (?:good|great)|cool|got it|"
    r"will do|you'?re all set|yep|yeah|yes|sure|thanks?|thank you(?: so much| very much)?|"
    r"you too|same to you|no problem|bye(?:[- ]bye)?|goodbye|"
    r"have a (?:good|great|nice) (?:one|day|night|evening)|take care|see you|see ya|"
    r"you'?re welcome)[\s,.!]*)+$",
    re.I,
)
_FAREWELL_WORD = re.compile(r"\bbye\b|goodbye|take care|have a (?:good|great|nice)|see (?:you|ya)", re.I)


def is_farewell(text: str) -> bool:
    """True when ``text`` is only pleasantries and contains an actual goodbye."""
    value = (text or "").strip()
    return bool(value and _FAREWELL.match(value) and _FAREWELL_WORD.search(value))
