"""Keep a free-tier web host awake by pinging its own URL. Requires the ``server`` extra."""

from __future__ import annotations

import asyncio
import logging

import httpx

log = logging.getLogger(__name__)

__all__ = ["KeepAliveMiddleware", "keep_awake"]


async def keep_awake(url: str, interval: float, transport: httpx.AsyncBaseTransport | None = None) -> None:
    """GET ``url`` every ``interval`` seconds forever; failures are logged, never raised."""
    async with httpx.AsyncClient(timeout=30, transport=transport) as client:
        while True:
            await asyncio.sleep(interval)
            try:
                resp = await client.get(url)
                if resp.status_code != 200:
                    log.warning("keepalive %s -> %s", url, resp.status_code)
            except httpx.HTTPError as exc:
                log.warning("keepalive %s failed: %s", url, exc)


class KeepAliveMiddleware:
    """ASGI middleware that starts :func:`keep_awake` on the first lifespan event."""

    def __init__(
        self,
        app,
        url: str,
        interval: float,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.app = app
        self.url = url
        self.interval = interval
        self.transport = transport
        self.task: asyncio.Task | None = None

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] == "lifespan" and self.task is None:
            self.task = asyncio.create_task(keep_awake(self.url, self.interval, self.transport))
        await self.app(scope, receive, send)
