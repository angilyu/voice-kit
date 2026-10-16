"""Mint LiveKit access tokens and dispatch agents from an app backend.

Client apps (iOS, web) call a small backend endpoint that uses these helpers, so the LiveKit API
secret never ships in the app. Requires the ``tokens`` extra (``livekit-api``).
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from datetime import timedelta
from typing import Any

__all__ = ["dispatch_agent", "mint_token"]


def _creds(api_key: str | None, api_secret: str | None) -> tuple[str, str]:
    key = api_key or os.getenv("LIVEKIT_API_KEY", "")
    secret = api_secret or os.getenv("LIVEKIT_API_SECRET", "")
    if not key or not secret:
        raise ValueError("LiveKit API key/secret missing; pass them or set LIVEKIT_API_KEY/LIVEKIT_API_SECRET")
    return key, secret


def _metadata(value: str | Mapping[str, Any] | None) -> str:
    if value is None:
        return ""
    return value if isinstance(value, str) else json.dumps(dict(value))


def mint_token(
    *,
    room: str,
    identity: str,
    name: str | None = None,
    metadata: str | Mapping[str, Any] | None = None,
    attributes: Mapping[str, str] | None = None,
    agent_name: str | None = None,
    agent_metadata: str | Mapping[str, Any] | None = None,
    ttl: timedelta = timedelta(hours=1),
    can_publish: bool = True,
    can_subscribe: bool = True,
    can_publish_data: bool = True,
    api_key: str | None = None,
    api_secret: str | None = None,
) -> str:
    """Return a JWT that lets ``identity`` join ``room``.

    With ``agent_name`` the token also carries a room configuration that dispatches that agent when
    the room is created by this participant joining, so no separate dispatch call is needed.
    ``attributes`` become the participant's initial attributes (a cheap way to send context like the
    current page before the agent starts).
    """
    from livekit import api

    key, secret = _creds(api_key, api_secret)
    token = (
        api.AccessToken(key, secret)
        .with_identity(identity)
        .with_ttl(ttl)
        .with_grants(
            api.VideoGrants(
                room_join=True,
                room=room,
                can_publish=can_publish,
                can_subscribe=can_subscribe,
                can_publish_data=can_publish_data,
            )
        )
    )
    if name:
        token = token.with_name(name)
    if metadata is not None:
        token = token.with_metadata(_metadata(metadata))
    if attributes:
        token = token.with_attributes(dict(attributes))
    if agent_name:
        token = token.with_room_config(
            api.RoomConfiguration(
                agents=[api.RoomAgentDispatch(agent_name=agent_name, metadata=_metadata(agent_metadata))]
            )
        )
    return token.to_jwt()


async def dispatch_agent(
    *,
    room: str,
    agent_name: str,
    metadata: str | Mapping[str, Any] | None = None,
    url: str | None = None,
    api_key: str | None = None,
    api_secret: str | None = None,
) -> Any:
    """Explicitly dispatch ``agent_name`` into ``room`` (creating the room if needed).

    Returns LiveKit's ``AgentDispatch``. ``url`` defaults to ``LIVEKIT_URL``.
    """
    from livekit import api

    key, secret = _creds(api_key, api_secret)
    lk_url = url or os.getenv("LIVEKIT_URL", "")
    if not lk_url:
        raise ValueError("LiveKit URL missing; pass url or set LIVEKIT_URL")
    lkapi = api.LiveKitAPI(lk_url, key, secret)
    try:
        return await lkapi.agent_dispatch.create_dispatch(
            api.CreateAgentDispatchRequest(agent_name=agent_name, room=room, metadata=_metadata(metadata))
        )
    finally:
        await lkapi.aclose()
