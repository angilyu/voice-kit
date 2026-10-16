"""``provider:model[@reasoning_effort]`` model specs and LangChain chat-model construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

__all__ = ["LLMModelSpec", "init_chat_model_from_spec", "llm_model_init_args", "parse_llm_model_spec"]


@dataclass(frozen=True)
class LLMModelSpec:
    model: str
    reasoning_effort: str | None = None


def parse_llm_model_spec(raw: str) -> LLMModelSpec:
    """Parse ``provider:model[@reasoning_effort]``, e.g. ``openai:gpt-5.4@low``."""
    value = (raw or "").strip()
    if not value:
        raise ValueError("LLM model must not be empty")
    model, sep, effort = value.rpartition("@")
    if not sep:
        return LLMModelSpec(model=value)
    model = model.strip()
    effort = effort.strip()
    if not model or not effort:
        raise ValueError(f"Invalid LLM model spec {raw!r}; expected provider:model[@effort]")
    return LLMModelSpec(model=model, reasoning_effort=effort)


def llm_model_init_args(
    raw: str,
    *,
    temperature: float | None = 0.3,
    service_tier: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Return the model name and ``init_chat_model`` kwargs for a model spec.

    Reasoning-effort specs omit temperature because OpenAI reasoning models reject non-default
    temperature. ``service_tier`` (e.g. "priority") only applies to OpenAI models.
    """
    spec = parse_llm_model_spec(raw)
    kwargs: dict[str, Any] = {}
    if spec.reasoning_effort:
        kwargs["reasoning_effort"] = spec.reasoning_effort
        if spec.model.startswith("openai:"):
            # Chat Completions rejects reasoning_effort together with tools for some models;
            # LangChain's Responses API path supports streaming + tool calling.
            kwargs["use_responses_api"] = True
    elif temperature is not None:
        kwargs["temperature"] = temperature
    tier = (service_tier or "").strip().lower()
    if tier and tier != "default" and spec.model.startswith("openai:"):
        kwargs["service_tier"] = tier
    return spec.model, kwargs


def init_chat_model_from_spec(
    raw: str,
    *,
    temperature: float | None = 0.3,
    service_tier: str | None = None,
    **overrides: Any,
) -> BaseChatModel:
    """Build a LangChain chat model from a spec. Extra kwargs override the derived ones.

    The provider package (e.g. ``langchain-openai`` via the ``openai`` extra) must be installed.
    """
    from langchain.chat_models import init_chat_model

    model, kwargs = llm_model_init_args(raw, temperature=temperature, service_tier=service_tier)
    kwargs.update(overrides)
    return init_chat_model(model, **kwargs)
