import asyncio

import httpx

from voice_kit import keepalive
from voice_kit.keepalive import KeepAliveMiddleware, keep_awake


async def test_keep_awake_pings_and_survives_errors(monkeypatch):
    hits = []

    def handler(request):
        hits.append(str(request.url))
        if len(hits) == 1:
            raise httpx.ConnectError("down")
        return httpx.Response(200)

    task = asyncio.create_task(keep_awake("https://app.example/health", 0.01, httpx.MockTransport(handler)))
    while len(hits) < 3:
        await asyncio.sleep(0.01)
    task.cancel()
    assert hits[0] == "https://app.example/health"


async def test_middleware_starts_once_on_lifespan(monkeypatch):
    started = []

    async def fake_keep_awake(url, interval, transport=None):
        started.append((url, interval))

    monkeypatch.setattr(keepalive, "keep_awake", fake_keep_awake)
    calls = []

    async def app(scope, receive, send):
        calls.append(scope["type"])

    mw = KeepAliveMiddleware(app, "https://x/health", 60)
    await mw({"type": "http"}, None, None)
    assert mw.task is None
    await mw({"type": "lifespan"}, None, None)
    await mw({"type": "lifespan"}, None, None)
    await asyncio.sleep(0)
    assert started == [("https://x/health", 60)]
    assert calls == ["http", "lifespan", "lifespan"]
