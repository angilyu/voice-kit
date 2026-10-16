import json

import pytest

api = pytest.importorskip("livekit.api")

from voice_kit.tokens import dispatch_agent, mint_token  # noqa: E402

KEY, SECRET = "devkey", "devsecret-devsecret-devsecret-0123456789"


def test_mint_token_with_agent_dispatch_and_attributes():
    jwt = mint_token(
        room="book-123",
        identity="kid-ipad",
        name="Reader",
        metadata={"book": "bears"},
        attributes={"page": "3"},
        agent_name="story-buddy",
        agent_metadata={"book": "bears"},
        api_key=KEY,
        api_secret=SECRET,
    )
    claims = api.TokenVerifier(KEY, SECRET).verify(jwt)
    assert claims.identity == "kid-ipad"
    assert claims.name == "Reader"
    assert claims.video.room == "book-123" and claims.video.room_join
    assert json.loads(claims.metadata) == {"book": "bears"}
    assert claims.attributes == {"page": "3"}
    dispatch = claims.room_config.agents[0]
    assert dispatch.agent_name == "story-buddy"
    assert json.loads(dispatch.metadata) == {"book": "bears"}


def test_mint_token_reads_env_credentials(monkeypatch):
    monkeypatch.setenv("LIVEKIT_API_KEY", KEY)
    monkeypatch.setenv("LIVEKIT_API_SECRET", SECRET)
    claims = api.TokenVerifier(KEY, SECRET).verify(mint_token(room="r", identity="u"))
    assert claims.identity == "u" and not claims.room_config


def test_missing_credentials(monkeypatch):
    monkeypatch.delenv("LIVEKIT_API_KEY", raising=False)
    monkeypatch.delenv("LIVEKIT_API_SECRET", raising=False)
    with pytest.raises(ValueError):
        mint_token(room="r", identity="u")


async def test_dispatch_agent_requires_url(monkeypatch):
    monkeypatch.delenv("LIVEKIT_URL", raising=False)
    with pytest.raises(ValueError):
        await dispatch_agent(room="r", agent_name="a", api_key=KEY, api_secret=SECRET)
