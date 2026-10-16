# voice-kit

A reusable realtime voice-agent stack, extracted from MuseVoiceAgent so several apps can share it.

- **LiveKit worker helpers**: run any LangGraph graph (or LangChain runnable) as the agent brain.
- **Turn-taking presets**, including `patient` for slow, quiet speakers such as young children
  (long end-of-turn delay, tolerant interruptions, background-voice cancellation).
- **Turn gates**: per-turn RESPOND / LISTEN / IGNORE decisions, so an agent can listen quietly
  while someone reads aloud and ignore background talk.
- **Client context**: apps push context such as "current page" through participant attributes or
  data messages, and the agent sees it.
- **Token minting and agent dispatch** for app backends, so iOS and web clients never hold LiveKit
  secrets.
- **Streaming text helpers**: tool-markup filtering, streaming a tool's `say` argument, opener
  de-duplication, filler, and farewell detection.
- **Retell custom-LLM adapter**: the websocket protocol, the REST client, and a LangGraph handler.

Domain logic (prompts, tools, call outcomes, phone numbers) stays in each app.

## Install

Requires Python 3.12+.

```bash
# Voice agent worker (LiveKit + OpenAI models)
uv add "voice-kit[livekit,openai] @ git+https://github.com/angilyu/voice-kit"

# App backend that only mints tokens / dispatches agents
uv add "voice-kit[tokens] @ git+https://github.com/angilyu/voice-kit"

# Retell phone agent (websocket adapter + keepalive middleware)
uv add "voice-kit[retell,server,openai] @ git+https://github.com/angilyu/voice-kit"
```

Pin a commit or tag with `@ git+https://github.com/angilyu/voice-kit@<sha-or-tag>`.

| Extra | Enables | Pulls in |
|---|---|---|
| (core) | `voice_kit` (presets, turns, context, streaming, llm, env) | langchain, langchain-core, pydantic |
| `livekit` | `voice_kit.livekit` | livekit-agents[langchain] 1.8, livekit-plugins-noise-cancellation |
| `tokens` | `voice_kit.tokens` | livekit-api |
| `retell` | `voice_kit.retell` | httpx, starlette |
| `server` | `voice_kit.keepalive` | httpx, starlette |
| `openai` | `openai:` model specs | langchain[openai] |

## Quickstart: a LiveKit agent driven by a LangGraph graph

```python
# agent.py — run with: uv run python agent.py dev
from langgraph.config import get_stream_writer
from langgraph.graph import START, MessagesState, StateGraph

from voice_kit import init_chat_model_from_spec
from voice_kit.livekit import run_app, start_graph_session

model = init_chat_model_from_spec("openai:gpt-5.4-mini")


async def reply(state: MessagesState):
    writer = get_stream_writer()
    text = ""
    async for chunk in model.astream(state["messages"]):
        writer(chunk.content)  # stream_mode="custom": whatever you write is spoken
        text += chunk.content
    return {"messages": [{"role": "assistant", "content": text}]}


graph = StateGraph(MessagesState).add_node("reply", reply).add_edge(START, "reply").compile()


async def entrypoint(ctx):
    await start_graph_session(ctx, graph, preset="default", greeting="Hi! How can I help?")


if __name__ == "__main__":
    run_app(entrypoint, agent_name="my-agent")
```

The worker needs `LIVEKIT_URL`, `LIVEKIT_API_KEY`, and `LIVEKIT_API_SECRET`, plus your model
provider's key (for example `OPENAI_API_KEY`). The default STT and TTS run through LiveKit
Inference.

## Example: a patient reading buddy

The agent should listen while a child slowly reads the page aloud, ignore background noise and
unrelated talk, and answer questions about the book.

```python
from voice_kit import chain, llm_gate, min_words_gate, reference_text_gate, init_chat_model_from_spec
from voice_kit.livekit import start_graph_session

gate = chain(
    min_words_gate(2),                          # coughs, "um", single stray words -> IGNORE
    reference_text_gate("page_text"),           # reading the page aloud -> LISTEN (kept in history)
    llm_gate(                                   # everything else: is it for us and about the book?
        init_chat_model_from_spec("openai:gpt-5.4-nano"),
        "A reading buddy for a 6-year-old. Answer only questions about the picture book on screen. "
        "Adults talking to each other, TV, and unrelated chatter should be IGNORED.",
        context_keys=["book_title", "page", "page_text"],
    ),
)


async def entrypoint(ctx):
    await start_graph_session(
        ctx,
        book_graph,                  # your LangGraph graph
        preset="patient",            # long pauses OK, background voices cancelled
        gate=gate,
        greeting="Hi! I'm ready when you are. Read to me, or ask me anything about the story.",
    )
```

The app pushes context in either of two ways:

- **Participant attributes**: `localParticipant.setAttributes({"page": "3", "page_text": "..."})`.
  An empty string deletes a key, and `lk.*` attributes are ignored.
- **A data or text message on topic `"context"`**: a JSON object is merged into the context; any
  other payload is stored under `"message"`.

Initial attributes can also go into the token (see below). The gate and the reply both see the
latest context. By default it is added as a system note for that reply only and is not kept in
history.

Tune the preset per app:

```python
from voice_kit import get_preset, register_preset

register_preset(get_preset("patient", name="kids", min_endpointing_delay=2.0, tts_voice="..."))
```

## Minting tokens on an app backend

```python
from voice_kit.tokens import mint_token

jwt = mint_token(
    room=f"book-{session_id}",
    identity=device_id,
    attributes={"book_title": "Little Bear", "page": "1"},
    agent_name="reading-buddy",           # dispatched automatically when the room is created
    agent_metadata={"book_id": "little-bear"},
)
# return {"url": LIVEKIT_URL, "token": jwt} to the app
```

`dispatch_agent(room=..., agent_name=...)` dispatches into an existing room explicitly. Credentials
come from the arguments or from the environment variables `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`,
and `LIVEKIT_URL`. Inside the agent, `job_metadata(ctx)` returns the dispatch metadata as a dict.

## API surface

### `voice_kit` (core, framework-free)

| Area | Symbols |
|---|---|
| Model specs | `LLMModelSpec`, `parse_llm_model_spec("provider:model[@effort]")`, `llm_model_init_args(spec, temperature=0.3, service_tier=None)`, `init_chat_model_from_spec(spec, **overrides)` |
| Presets | `VoicePreset` (STT, TTS, voice, language, turn detection, endpointing, interruption, preemptive generation, noise cancellation, user-away timeout; `.with_overrides()`, `.turn_handling()`), `DEFAULT`, `TELEPHONY`, `PATIENT` (in `voice_kit.presets`), `PRESETS`, `get_preset(name, **overrides)`, `register_preset`, `preset_from_env(default, prefix="")` |
| Turn gates | `TurnDecision` (RESPOND/LISTEN/IGNORE), `UserTurn(text, context, history)`, `TurnGate`, `decide`, `always_respond`, `chain`, `min_words_gate`, `keyword_gate`, `question_gate`, `is_question`, `reference_text_gate`, `text_overlap`, `llm_gate` |
| Context | `ContextStore` (`update`, `handle_data`, `on_change`, `snapshot`, `get`) |
| Streaming | `ToolMarkupFilter`, `SayStreamer(tool_name, field)`, `OpenerDropper`, `strip_opener`, `WordGate`, `FILLER`, `strip_leading_filler`, `message_text`, `chunk_text`, `is_farewell` |
| Env | `env_bool`, `env_int`, `env_float`, `env_str`, `env_list` |

A gate is any `(UserTurn) -> bool | TurnDecision`, sync or async. `True` means respond and
`False` means ignore. LISTEN keeps the utterance in history without replying.

`preset_from_env` reads these optional variables, each with the given prefix: `VOICE_PRESET`,
`STT_MODEL`, `STT_LANGUAGE`, `TTS_MODEL`, `TTS_VOICE`, `MIN_ENDPOINTING_DELAY`,
`MAX_ENDPOINTING_DELAY`, and `NOISE_CANCELLATION`.

### `voice_kit.livekit` (extra `livekit`)

| Symbol | Purpose |
|---|---|
| `start_graph_session(ctx, graph, *, preset, gate, instructions, context, context_topic="context", context_participant, context_message, on_decision, greeting, stream_mode="custom", graph_config, graph_context, agent, room_kwargs, **session_kwargs)` | Connect, wire up client context, and start the session. Returns `GraphSession(session, agent, context, detach_context)`. |
| `build_session(preset, *, llm=None, graph=None, stream_mode, stt, tts, vad, turn_handling, **kwargs)` | `AgentSession` from a preset. |
| `build_stt`, `build_tts`, `build_turn_handling`, `noise_canceller`, `room_options(preset, **kw)`, `graph_llm(graph, ...)` | Lower-level builders. |
| `GraphAgent(instructions, gate, context, context_message, on_decision)` | `Agent` that applies the gate in `on_user_turn_completed` and injects context. |
| `attach_room_context(store, room, *, topic="context", attributes=True, participant_identity=None)` | Feeds a `ContextStore` from attributes, data, and text streams. Returns `detach()`. |
| `create_server(entrypoint, agent_name="")`, `run_app(server_or_entrypoint, agent_name="")`, `job_metadata(ctx)` | Worker plumbing. |

Noise cancellation (`nc`, `bvc`, `bvc_telephony`) uses `livekit-plugins-noise-cancellation`, which
requires LiveKit Cloud. If the plugin is missing, voice-kit logs a warning and continues without it.

### `voice_kit.tokens` (extra `tokens`)

| Symbol | Purpose |
|---|---|
| `mint_token(*, room, identity, name, metadata, attributes, agent_name, agent_metadata, ttl, can_publish, can_subscribe, can_publish_data, api_key, api_secret)` | Returns a JWT. |
| `async dispatch_agent(*, room, agent_name, metadata, url, api_key, api_secret)` | Dispatches an agent into a room. |

### `voice_kit.retell` (extra `retell`)

| Symbol | Purpose |
|---|---|
| `RetellWebsocketRouter(app, *, secret, handler_factory, path_prefix="/retell/llm/", session_kwargs)` | ASGI wrapper. Serves `{prefix}{secret}/{call_id}` and passes everything else to `app`. |
| `RetellLLMSession(ws, call_id, handler, *, begin_message, filler_text, filler_seconds, error_reply, bind_timeout, auto_reconnect)` | The protocol: config, ping_pong, call_details, update_only, response/reminder, filler, errors. Exposes `interrupt(messages)`, `update_agent(cfg)`, `send`, `close`, `call`, `last_transcript`, `turn_active`, and `heard_response_request`. |
| `RetellCallHandler` | Override `respond(turn)` (an async iterator of `str` or `RetellChunk`). Optional hooks: `on_call_details`, `on_transcript`, `filler_wait`, `after_turn`, `on_close`. |
| `GraphRetellHandler(graph, stream_mode="custom", config=None)` | Streams a LangGraph graph. |
| `RetellTurn` | Per-turn data. Set `end_call` or `digit_to_press` from `respond`. |
| `RetellChunk(content, no_interruption_allowed)` | A response chunk with an optional no-interruption flag. |
| `transcript_to_messages` | Converts a Retell transcript to LangChain messages. |
| `RetellClient(api_key)`, `RetellError` | REST client for calls, agents, and phone numbers. |
| `retell_llm_websocket_url(base, secret)`, `sync_agent_websocket_url(client, agent_id, url)` | Websocket URL helpers. |

### `voice_kit.keepalive` (extra `server`)

`keep_awake(url, interval)` and `KeepAliveMiddleware(app, url, interval)` keep free web hosts from
sleeping by pinging themselves.

## Migrating MuseVoiceAgent

| Muse symbol | voice-kit replacement |
|---|---|
| `config._bool/_int/_str/_list` | `voice_kit.env.env_bool/env_int/env_str/env_list` |
| `config.LLMModelSpec`, `parse_llm_model_spec`, `llm_model_init_args` | `voice_kit.llm` (same names). `init_chat_model_from_spec` wraps `init_chat_model`. |
| `config` STT/TTS model and voice settings | `VoicePreset` / `preset_from_env(prefix="MUSE_")` |
| `Settings.retell_llm_websocket_url()` | `voice_kit.retell.retell_llm_websocket_url(public_base_url, secret)` |
| `graph._ToolMarkupFilter`, `_SayStreamer`, `_OpenerDropper`, `_strip_opener` | `voice_kit.streaming.ToolMarkupFilter`, `SayStreamer()`, `OpenerDropper`, `strip_opener` |
| `graph.is_farewell`, `_text` / `_content_text` | `voice_kit.streaming.is_farewell`, `message_text` |
| `retell.FILLER`, `_LEADING_FILLER` | `voice_kit.streaming.FILLER`, `strip_leading_filler` |
| `retell.RetellClient`, `RetellError`, `transcript_to_messages`, `sync_agent_websocket_url` | `voice_kit.retell` (same names) |
| `retell.RetellLLMSession` (protocol parts) | `voice_kit.retell.RetellLLMSession` plus a Muse `RetellCallHandler` that owns call binding, `REMINDER_NUDGE`, quick closing replies (set `turn.end_call`, yield nothing), pickup/answer watchers (via `session.interrupt`), and the reminder `update_agent` (in `after_turn`). |
| `retell.RetellWebsocketRouter` | `voice_kit.retell.RetellWebsocketRouter(app, secret=..., handler_factory=...)` |
| `keepalive.keep_awake`, `KeepAliveMiddleware` | `voice_kit.keepalive` (same names) |
| `agent.py` `AgentSession(...)` construction | `voice_kit.livekit.build_session("default", graph=graph)`, or `start_graph_session` |

Muse behavior differences:

- `strip_opener` strips leading whitespace first.
- `OpenerDropper("")` passes text through.
- `SayStreamer` checks the next tool call instead of returning early.

## Development

```bash
uv sync --all-extras --dev
uv run pytest
```
