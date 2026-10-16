import pytest

from voice_kit.llm import LLMModelSpec, llm_model_init_args, parse_llm_model_spec


def test_parse_plain_and_effort_specs():
    assert parse_llm_model_spec("openai:gpt-5.4-mini") == LLMModelSpec("openai:gpt-5.4-mini")
    assert parse_llm_model_spec(" openai:gpt-5.4 @ low ") == LLMModelSpec("openai:gpt-5.4", "low")


@pytest.mark.parametrize("raw", ["", "  ", "@low", "openai:gpt-5.4@"])
def test_parse_rejects_invalid_specs(raw):
    with pytest.raises(ValueError):
        parse_llm_model_spec(raw)


def test_init_args_temperature_for_plain_models():
    assert llm_model_init_args("openai:gpt-4.1") == ("openai:gpt-4.1", {"temperature": 0.3})
    assert llm_model_init_args("openai:gpt-4.1", temperature=None) == ("openai:gpt-4.1", {})


def test_init_args_reasoning_effort_uses_responses_api_and_no_temperature():
    model, kwargs = llm_model_init_args("openai:gpt-5.4@low")
    assert model == "openai:gpt-5.4"
    assert kwargs == {"reasoning_effort": "low", "use_responses_api": True}
    _, other = llm_model_init_args("anthropic:claude@high")
    assert other == {"reasoning_effort": "high"}


def test_service_tier_only_for_openai():
    assert llm_model_init_args("openai:gpt-4.1", service_tier="Priority")[1]["service_tier"] == "priority"
    assert "service_tier" not in llm_model_init_args("openai:gpt-4.1", service_tier="default")[1]
    assert "service_tier" not in llm_model_init_args("anthropic:claude", service_tier="priority")[1]
