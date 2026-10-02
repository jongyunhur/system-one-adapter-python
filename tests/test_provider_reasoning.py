"""Reasoning-effort options must reach each provider's request, and only when set.

Each provider names the controls its own API does — `reasoning_effort` for OpenAI,
`thinking` and `effort` for Anthropic, `thinking_level` for Gemini — so these tests
check the request envelope per provider rather than a shared shape. The transport cases
assert on the JSON that actually leaves the SDK, since an option that stops at the
provider object would otherwise look configured while changing nothing.
"""

import asyncio
import json
from typing import Any, Literal

import httpx
import httpx2
import pytest

from system_one_adapter import AsyncSystemOneAdapterClient, Noul, SystemOneAdapterClient, SystemOneResponse
from system_one_adapter.providers import Message
from system_one_adapter.providers.anthropic import AnthropicProvider, AsyncAnthropicProvider
from system_one_adapter.providers.anthropic import _request_kwargs as anthropic_request_kwargs
from system_one_adapter.providers.anthropic import _result as anthropic_result
from system_one_adapter.providers.gemini import AsyncGeminiProvider, GeminiProvider
from system_one_adapter.providers.gemini import _request_kwargs as gemini_request_kwargs
from system_one_adapter.providers.openai import AsyncOpenAIProvider, OpenAIProvider, _responses_request_kwargs
from tests.http import json_response

SCHEMA = {"type": "object", "properties": {"answers": {"type": "object"}}}
MESSAGES = [
    Message(role="system", content="system prompt"),
    Message(role="user", content="the document"),
]
ANSWER = '{"answers":{"positive":true}}'
QUESTIONS = {"positive": Noul(instructions="The review is positive.")}


@pytest.fixture
def http_requests(monkeypatch: pytest.MonkeyPatch) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Capture request bodies and reply with a caller-populated payload."""
    payload: dict[str, Any] = {}
    requests: list[dict[str, Any]] = []

    def respond(request: httpx.Request | httpx2.Request) -> httpx.Response | httpx2.Response:
        requests.append(json.loads(request.content))
        return json_response(request, payload)

    def send(client: httpx.Client | httpx2.Client, request: httpx.Request | httpx2.Request, **kwargs: Any) -> httpx.Response | httpx2.Response:
        return respond(request)

    async def send_async(
        client: httpx.AsyncClient | httpx2.AsyncClient, request: httpx.Request | httpx2.Request, **kwargs: Any
    ) -> httpx.Response | httpx2.Response:
        return respond(request)

    for client_class, handler in [
        (httpx.Client, send),
        (httpx2.Client, send),
        (httpx.AsyncClient, send_async),
        (httpx2.AsyncClient, send_async),
    ]:
        monkeypatch.setattr(client_class, "send", handler)
    return payload, requests


def _evaluate(provider: Any, *, structured: bool = True) -> SystemOneResponse:
    """Run one evaluation through the provider, closing whatever it owns."""
    if isinstance(provider, (AsyncOpenAIProvider, AsyncAnthropicProvider, AsyncGeminiProvider)):

        async def run() -> SystemOneResponse:
            try:
                async with AsyncSystemOneAdapterClient(structured_outputs=structured, llm_answer_mode="discrete", model=provider) as client:
                    return await client.system_one("A delightful book.", QUESTIONS)
            finally:
                await provider.aclose()

        return asyncio.run(run())
    try:
        with SystemOneAdapterClient(structured_outputs=structured, llm_answer_mode="discrete", model=provider) as client:
            return client.system_one("A delightful book.", QUESTIONS)
    finally:
        provider.close()


# OpenAI


@pytest.mark.parametrize("structured", [False, True])
def test_openai_responses_request_omits_reasoning_by_default(structured: bool) -> None:
    kwargs = _responses_request_kwargs("gpt-5-mini", MESSAGES, SCHEMA, structured=structured)
    assert "reasoning" not in kwargs


@pytest.mark.parametrize("structured", [False, True])
def test_openai_responses_request_carries_reasoning_effort(structured: bool) -> None:
    kwargs = _responses_request_kwargs("gpt-5-mini", MESSAGES, SCHEMA, structured=structured, reasoning_effort="low")
    assert kwargs["reasoning"] == {"effort": "low"}


@pytest.mark.parametrize("provider_class", [OpenAIProvider, AsyncOpenAIProvider])
def test_openai_provider_defaults_reasoning_effort_to_unset(provider_class: Any) -> None:
    assert provider_class("gpt-5-mini").reasoning_effort is None


@pytest.mark.parametrize("provider_class", [OpenAIProvider, AsyncOpenAIProvider])
@pytest.mark.parametrize(
    "api,base_url,endpoint",
    [
        ("responses", None, "/v1/responses"),
        ("chat_completions", "https://compatible.test/v1", "/v1/chat/completions"),
    ],
)
@pytest.mark.parametrize("reasoning_effort", [None, "minimal", "high"])
def test_openai_transport_sends_reasoning_effort(
    http_requests: tuple[dict[str, Any], list[dict[str, Any]]],
    monkeypatch: pytest.MonkeyPatch,
    provider_class: Any,
    api: Literal["responses", "chat_completions"],
    base_url: str | None,
    endpoint: str,
    reasoning_effort: str | None,
) -> None:
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    payload, requests = http_requests
    if endpoint == "/v1/responses":
        payload.update(
            status="completed",
            output=[{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": ANSWER}]}],
            usage={"input_tokens": 12, "output_tokens": 7},
        )
    else:
        payload.update(choices=[{"message": {"role": "assistant", "content": ANSWER}, "finish_reason": "stop"}], usage={})

    provider = provider_class("gpt-5-mini", base_url=base_url, api=api, reasoning_effort=reasoning_effort)
    response = _evaluate(provider)

    assert response.nouls["positive"].noul == 1.0
    sent = requests[0]
    if reasoning_effort is None:
        assert "reasoning" not in sent and "reasoning_effort" not in sent
    elif endpoint == "/v1/responses":
        assert sent["reasoning"] == {"effort": reasoning_effort}
    else:
        assert sent["reasoning_effort"] == reasoning_effort
    # The attempt trace is what callers inspect to confirm what was requested.
    assert response.debug["llm_attempts"][0]["request"] == sent


# Anthropic


@pytest.mark.parametrize("structured", [False, True])
def test_anthropic_request_omits_thinking_by_default(structured: bool) -> None:
    kwargs = anthropic_request_kwargs("claude-haiku-4-5", MESSAGES, SCHEMA, structured=structured, max_tokens=4096)
    assert "thinking" not in kwargs


@pytest.mark.parametrize("structured", [False, True])
def test_anthropic_request_carries_thinking(structured: bool) -> None:
    thinking: Any = {"type": "enabled", "budget_tokens": 2048}
    kwargs = anthropic_request_kwargs("claude-haiku-4-5", MESSAGES, SCHEMA, structured=structured, max_tokens=4096, thinking=thinking)
    assert kwargs["thinking"] == thinking
    assert kwargs["max_tokens"] == 4096


@pytest.mark.parametrize("structured", [False, True])
def test_anthropic_request_carries_effort(structured: bool) -> None:
    kwargs = anthropic_request_kwargs(
        "claude-sonnet-5-5", MESSAGES, SCHEMA, structured=structured, max_tokens=4096, effort="low"
    )
    assert kwargs["output_config"]["effort"] == "low"
    if structured:
        assert kwargs["output_config"]["format"] == {"type": "json_schema", "schema": SCHEMA}
    else:
        assert set(kwargs["output_config"]) == {"effort"}


@pytest.mark.parametrize("provider_class", [AnthropicProvider, AsyncAnthropicProvider])
@pytest.mark.parametrize("budget_tokens", [4096, 8192])
def test_anthropic_rejects_thinking_budget_above_output_limit(provider_class: Any, budget_tokens: int) -> None:
    with pytest.raises(ValueError, match=f"budget_tokens \\({budget_tokens}\\) must be below max_tokens \\(4096\\)"):
        provider_class("claude-haiku-4-5", thinking={"type": "enabled", "budget_tokens": budget_tokens})


@pytest.mark.parametrize("provider_class", [AnthropicProvider, AsyncAnthropicProvider])
@pytest.mark.parametrize("thinking", [{"type": "enabled", "budget_tokens": 4095}, {"type": "disabled"}, {"type": "adaptive"}])
def test_anthropic_accepts_thinking_that_fits_the_output_limit(provider_class: Any, thinking: Any) -> None:
    provider = provider_class("claude-haiku-4-5", thinking=thinking)
    assert provider.thinking == thinking
    assert provider.max_tokens == 4096


@pytest.mark.parametrize("provider_class", [AnthropicProvider, AsyncAnthropicProvider])
def test_anthropic_provider_defaults_thinking_to_unset(provider_class: Any) -> None:
    provider = provider_class("claude-haiku-4-5")
    assert provider.thinking is None
    assert provider.effort is None


@pytest.mark.parametrize("provider_class", [AnthropicProvider, AsyncAnthropicProvider])
@pytest.mark.parametrize(
    "thinking,effort",
    [(None, None), ({"type": "enabled", "budget_tokens": 1024}, None), ({"type": "adaptive"}, "low")],
)
def test_anthropic_transport_sends_reasoning_controls(
    http_requests: tuple[dict[str, Any], list[dict[str, Any]]],
    provider_class: Any,
    thinking: Any,
    effort: str | None,
) -> None:
    payload, requests = http_requests
    model_name = "claude-sonnet-5-5" if effort is not None else "claude-haiku-4-5"
    payload.update(
        id="message-test",
        type="message",
        role="assistant",
        model=model_name,
        stop_reason="end_turn",
        # A thinking block precedes the answer whenever thinking is enabled.
        content=([{"type": "thinking", "thinking": "Weighing the review.", "signature": "sig"}] if thinking else [])
        + [{"type": "text", "text": ANSWER}],
        usage={"input_tokens": 12, "output_tokens": 7},
    )

    response = _evaluate(provider_class(model_name, thinking=thinking, effort=effort))

    assert response.nouls["positive"].noul == 1.0
    sent = requests[0]
    if thinking is None:
        assert "thinking" not in sent
    else:
        assert sent["thinking"] == thinking
    if effort is None:
        assert "effort" not in sent.get("output_config", {})
    else:
        assert sent["output_config"]["effort"] == effort
        assert sent["output_config"]["format"]["type"] == "json_schema"
        assert sent["output_config"]["format"]["schema"]["type"] == "object"


def test_anthropic_result_reads_the_answer_past_thinking_blocks() -> None:
    """Thinking text shares the content list with the answer and must not be parsed."""
    response = anthropic_result(
        type(
            "Response",
            (),
            {
                "stop_reason": "end_turn",
                "content": [
                    type("Thinking", (), {"type": "thinking", "text": "not the answer"})(),
                    type("Text", (), {"type": "text", "text": ANSWER})(),
                ],
                "usage": type("Usage", (), {"input_tokens": 12, "output_tokens": 7})(),
                "model_dump": lambda mode="json": {},
            },
        )()
    )
    assert response.text == ANSWER


# Gemini


@pytest.mark.parametrize("structured", [False, True])
def test_gemini_request_omits_generation_config_by_default(structured: bool) -> None:
    kwargs = gemini_request_kwargs("gemini-3.8-flash", MESSAGES, SCHEMA, structured=structured)
    assert "generation_config" not in kwargs


@pytest.mark.parametrize("structured", [False, True])
def test_gemini_request_carries_thinking_level(structured: bool) -> None:
    kwargs = gemini_request_kwargs("gemini-3.8-flash", MESSAGES, SCHEMA, structured=structured, thinking_level="high")
    assert kwargs["generation_config"] == {"thinking_level": "high"}


@pytest.mark.parametrize("provider_class", [GeminiProvider, AsyncGeminiProvider])
def test_gemini_provider_defaults_thinking_level_to_unset(provider_class: Any) -> None:
    provider = provider_class("gemini-3.8-flash")
    try:
        assert provider.thinking_level is None
    finally:
        provider._client.close()


@pytest.mark.parametrize("provider_class", [GeminiProvider, AsyncGeminiProvider])
@pytest.mark.parametrize("thinking_level", [None, "low", "high"])
def test_gemini_transport_sends_thinking_level(
    http_requests: tuple[dict[str, Any], list[dict[str, Any]]],
    provider_class: Any,
    thinking_level: str | None,
) -> None:
    payload, requests = http_requests
    payload.update(
        id="interaction-test",
        status="completed",
        model="gemini-3.8-flash",
        steps=[{"type": "model_output", "content": [{"type": "text", "text": ANSWER}]}],
        usage={"total_input_tokens": 12, "total_output_tokens": 7, "total_tokens": 19},
    )

    provider = provider_class("gemini-3.8-flash", thinking_level=thinking_level)
    response = _evaluate(provider)

    assert response.nouls["positive"].noul == 1.0
    sent = requests[0]
    if thinking_level is None:
        assert "generation_config" not in sent
    else:
        assert sent["generation_config"] == {"thinking_level": thinking_level}
