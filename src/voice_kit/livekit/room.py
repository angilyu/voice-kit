"""Feed client context from a LiveKit room into a :class:`~voice_kit.context.ContextStore`."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import Any

from livekit import rtc

from ..context import ContextStore

log = logging.getLogger(__name__)

__all__ = ["attach_room_context"]


def attach_room_context(
    store: ContextStore,
    room: rtc.Room,
    *,
    topic: str | None = "context",
    attributes: bool = True,
    participant_identity: str | None = None,
) -> Callable[[], None]:
    """Keep ``store`` in sync with what client participants push. Returns a ``detach`` function.

    Sources (all optional):

    * Participant attributes (``attributes=True``): every non-``lk.`` attribute; an empty value
      deletes the key. Set them from the client SDK or in the access token
      (:func:`voice_kit.tokens.mint_token` ``attributes=``).
    * Data packets on ``topic``: a JSON object is merged; any other payload is stored under
      ``"message"``.
    * Text streams on ``topic`` (``sendText(..., topic=...)``), same format as data packets.

    ``participant_identity`` restricts updates to one participant (default: any remote one).
    """

    def accepted(participant: Any) -> bool:
        if participant_identity is None:
            return True
        return participant is not None and getattr(participant, "identity", None) == participant_identity

    def apply_attributes(values: dict[str, str]) -> None:
        update = {k: (v if v != "" else None) for k, v in values.items() if not k.startswith("lk.")}
        if update:
            store.update(update)

    handlers: list[tuple[str, Callable[..., Any]]] = []
    tasks: set[asyncio.Task[Any]] = set()

    if attributes:
        for participant in room.remote_participants.values():
            if accepted(participant) and participant.attributes:
                apply_attributes(dict(participant.attributes))

        def on_connected(participant: rtc.RemoteParticipant) -> None:
            if accepted(participant) and participant.attributes:
                apply_attributes(dict(participant.attributes))

        def on_attributes(changed: dict[str, str], participant: rtc.Participant) -> None:
            if accepted(participant) and participant.identity != room.local_participant.identity:
                apply_attributes(dict(changed))

        handlers.append(("participant_connected", on_connected))
        handlers.append(("participant_attributes_changed", on_attributes))

    stream_registered = False
    if topic is not None:

        def on_data(packet: rtc.DataPacket) -> None:
            if packet.topic == topic and accepted(packet.participant):
                store.handle_data(packet.data)

        handlers.append(("data_received", on_data))

        async def read_stream(reader: Any, identity: str) -> None:
            try:
                text = await reader.read_all()
            except Exception:
                log.exception("failed to read context text stream from %s", identity)
                return
            if participant_identity is None or identity == participant_identity:
                store.handle_data(text)

        def on_stream(reader: Any, identity: str) -> None:
            task = asyncio.create_task(read_stream(reader, identity))
            tasks.add(task)
            task.add_done_callback(tasks.discard)

        try:
            room.register_text_stream_handler(topic, on_stream)
            stream_registered = True
        except ValueError:
            log.warning("a text stream handler for topic %r is already registered", topic)

    for event, handler in handlers:
        room.on(event, handler)

    def detach() -> None:
        for event, handler in handlers:
            room.off(event, handler)
        if stream_registered:
            room.unregister_text_stream_handler(topic)
        for task in list(tasks):
            task.cancel()

    return detach
