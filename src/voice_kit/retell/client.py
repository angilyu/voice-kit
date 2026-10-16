"""Minimal async client for the Retell REST endpoints, plus URL helpers."""

from __future__ import annotations

from typing import Any

import httpx

__all__ = [
    "API_BASE",
    "WS_PATH_PREFIX",
    "RetellClient",
    "RetellError",
    "retell_llm_websocket_url",
    "sync_agent_websocket_url",
]

API_BASE = "https://api.retellai.com"
WS_PATH_PREFIX = "/retell/llm/"


class RetellError(Exception):
    pass


class RetellClient:
    def __init__(
        self,
        api_key: str,
        base_url: str = API_BASE,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._http = httpx.AsyncClient(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=20,
            transport=transport,
        )

    async def __aenter__(self) -> RetellClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._http.aclose()

    async def _req(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> Any:
        resp = await self._http.request(method, path, json=body, params=params)
        if resp.status_code >= 400:
            raise RetellError(f"{method} {path} -> {resp.status_code}: {resp.text[:500]}")
        return resp.json() if resp.content else None

    async def create_phone_call(
        self,
        *,
        from_number: str,
        to_number: str,
        agent_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"from_number": from_number, "to_number": to_number}
        if agent_id:
            body["override_agent_id"] = agent_id
        if metadata:
            body["metadata"] = metadata
        return await self._req("POST", "/v2/create-phone-call", body)

    async def get_call(self, retell_call_id: str) -> dict[str, Any]:
        return await self._req("GET", f"/v2/get-call/{retell_call_id}")

    async def list_calls(self, *, limit: int = 20, agent_id: str | None = None) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"limit": limit}
        if agent_id:
            params["agent_id"] = agent_id
        try:
            data = await self._req("GET", "/v2/list-calls", params=params)
        except RetellError:
            body: dict[str, Any] = {"limit": limit}
            if agent_id:
                body["filter_criteria"] = {"agent_id": [agent_id]}
            data = await self._req("POST", "/v2/list-calls", body)
        if isinstance(data, dict):
            for key in ("calls", "data", "results"):
                if isinstance(data.get(key), list):
                    return data[key]
        return data if isinstance(data, list) else []

    async def list_agents(self) -> list[dict[str, Any]]:
        return await self._req("GET", "/list-agents")

    async def get_agent(self, agent_id: str) -> dict[str, Any]:
        return await self._req("GET", f"/get-agent/{agent_id}")

    async def create_agent(self, body: dict[str, Any]) -> dict[str, Any]:
        return await self._req("POST", "/create-agent", body)

    async def update_agent(self, agent_id: str, body: dict[str, Any]) -> dict[str, Any]:
        return await self._req("PATCH", f"/update-agent/{agent_id}", body)

    async def list_phone_numbers(self) -> list[dict[str, Any]]:
        return await self._req("GET", "/list-phone-numbers")

    async def import_phone_number(self, body: dict[str, Any]) -> dict[str, Any]:
        return await self._req("POST", "/import-phone-number", body)

    async def update_phone_number(self, number: str, body: dict[str, Any]) -> dict[str, Any]:
        return await self._req("PATCH", f"/update-phone-number/{number}", body)


def retell_llm_websocket_url(public_base_url: str, secret: str, path_prefix: str = WS_PATH_PREFIX) -> str:
    """The custom-LLM websocket URL to give Retell; Retell appends ``/{call_id}``."""
    base = (public_base_url or "").strip().rstrip("/")
    if not base or not secret:
        raise ValueError("public base URL and websocket secret are required")
    if base.startswith("https://"):
        base = "wss://" + base[len("https://") :]
    elif base.startswith("http://"):
        base = "ws://" + base[len("http://") :]
    return f"{base}/{path_prefix.strip('/')}/{secret}"


async def sync_agent_websocket_url(client: RetellClient, agent_id: str, url: str) -> str:
    """Point a Retell agent at ``url`` (e.g. after a tunnel restart); no-op if already set."""
    agent = await client.get_agent(agent_id)
    engine = agent.get("response_engine") or {}
    if engine.get("type") == "custom-llm" and engine.get("llm_websocket_url") == url:
        return url
    await client.update_agent(agent_id, {"response_engine": {"type": "custom-llm", "llm_websocket_url": url}})
    return url
