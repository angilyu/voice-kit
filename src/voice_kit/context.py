"""Client-pushed context (e.g. which page of a book is on screen) shared with the agent."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping
from typing import Any

log = logging.getLogger(__name__)

__all__ = ["ContextStore"]

Listener = Callable[[dict[str, Any]], None]


class ContextStore:
    """A small dict of client context with change listeners.

    Clients push context through participant attributes or data messages (see
    :func:`voice_kit.livekit.attach_room_context`). ``update`` merges a mapping; a ``None`` value
    deletes the key.
    """

    def __init__(self, initial: Mapping[str, Any] | None = None) -> None:
        self._data: dict[str, Any] = {}
        self._listeners: list[Listener] = []
        if initial:
            self.update(initial)

    def update(self, values: Mapping[str, Any]) -> dict[str, Any]:
        """Merge ``values`` and return the keys that actually changed."""
        changed: dict[str, Any] = {}
        for key, value in values.items():
            if value is None:
                if key in self._data:
                    del self._data[key]
                    changed[key] = None
            elif self._data.get(key, _MISSING) != value:
                self._data[key] = value
                changed[key] = value
        if changed:
            for listener in list(self._listeners):
                try:
                    listener(changed)
                except Exception:
                    log.exception("context listener failed")
        return changed

    def handle_data(self, payload: bytes | str, *, key: str = "message") -> dict[str, Any]:
        """Apply a data message: a JSON object is merged, anything else is stored under ``key``."""
        text = payload.decode("utf-8", errors="replace") if isinstance(payload, bytes) else payload
        try:
            value = json.loads(text)
        except ValueError:
            value = None
        if isinstance(value, dict):
            return self.update(value)
        return self.update({key: text})

    def on_change(self, listener: Listener) -> Callable[[], None]:
        """Register ``listener(changed)``; returns an unsubscribe function."""
        self._listeners.append(listener)

        def unsubscribe() -> None:
            if listener in self._listeners:
                self._listeners.remove(listener)

        return unsubscribe

    def snapshot(self) -> dict[str, Any]:
        return dict(self._data)

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __contains__(self, key: object) -> bool:
        return key in self._data

    def __len__(self) -> int:
        return len(self._data)


_MISSING = object()
