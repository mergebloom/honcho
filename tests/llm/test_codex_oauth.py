from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import httpx
import jwt
import pytest
from pydantic import BaseModel

from src.config import (
    ConfiguredEmbeddingModelSettings,
    ConfiguredModelSettings,
    FallbackModelSettings,
    ModelConfig,
    ModelOverrideSettings,
    ResolvedFallbackConfig,
    resolve_model_config,
    settings,
)
from src.exceptions import ValidationException
from src.llm import registry
from src.llm.backend import CompletionResult
from src.llm.codex_oauth import (
    CodexOAuthBackend,
    CodexOAuthClient,
    CodexOAuthError,
    get_codex_oauth_client,
)
from src.llm.request_builder import execute_completion

ENDPOINT = "https://chatgpt.com/backend-api/codex/responses"
SENTINEL = "PRIVATE-PROVIDER-SENTINEL"


def write_credentials(
    path: Path, *, account: str = "account-one", residency: str | None = "us"
) -> str:
    claims: dict[str, Any] = {"chatgpt_account_id": account}
    if residency is not None:
        claims["chatgpt_account_residency"] = residency
    token = jwt.encode(
        {"https://api.openai.com/auth": claims, "exp": int(time.time()) + 3600},
        "fixture-signing-key-not-a-real-credential",
        algorithm="HS256",
    )
    replacement = path.with_suffix(".new")
    replacement.write_text(
        json.dumps({"access_token": token, "expires_at": time.time() + 3600})
    )
    replacement.chmod(0o600)
    replacement.replace(path)
    return token


def response_events(
    *,
    text: str = "ok",
    status: str = "completed",
    output: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    response: dict[str, Any] = {
        "id": "resp_fixture",
        "object": "response",
        "created_at": 1,
        "status": "in_progress",
        "output": [],
        "model": "fixture-model",
    }
    events: list[dict[str, Any]] = [
        {"type": "response.created", "response": response.copy(), "sequence_number": 0}
    ]
    if text:
        events.extend(
            [
                {
                    "type": "response.output_item.added",
                    "item": {
                        "type": "message",
                        "id": "msg_fixture",
                        "role": "assistant",
                        "status": "in_progress",
                        "content": [],
                    },
                    "output_index": 0,
                    "sequence_number": 1,
                },
                {
                    "type": "response.content_part.added",
                    "part": {"type": "output_text", "text": "", "annotations": []},
                    "item_id": "msg_fixture",
                    "output_index": 0,
                    "content_index": 0,
                    "sequence_number": 2,
                },
                {
                    "type": "response.output_text.delta",
                    "delta": text,
                    "item_id": "msg_fixture",
                    "output_index": 0,
                    "content_index": 0,
                    "sequence_number": 3,
                    "logprobs": [],
                },
            ]
        )
    response["status"] = status
    response["output"] = (
        output
        if output is not None
        else [
            {
                "type": "message",
                "id": "msg_fixture",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ]
    )
    if status == "incomplete":
        response["incomplete_details"] = {"reason": "max_output_tokens"}
    events.append(
        {"type": f"response.{status}", "response": response, "sequence_number": 4}
    )
    return events


def sse(events: list[dict[str, Any]]) -> httpx.Response:
    return httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        content="".join(f"data: {json.dumps(event)}\n\n" for event in events),
    )


@pytest.fixture
def token_file(tmp_path: Path) -> Path:
    path = tmp_path / "access.json"
    write_credentials(path)
    return path


def make_client(
    token_file: Path, handler: Callable[[httpx.Request], httpx.Response], **kwargs: Any
) -> CodexOAuthClient:
    return CodexOAuthClient(
        token_file=str(token_file), transport=httpx.MockTransport(handler), **kwargs
    )


def make_backend(client: CodexOAuthClient) -> CodexOAuthBackend:
    backend = registry.backend_for_provider("openai", client)
    assert isinstance(backend, CodexOAuthBackend)
    return backend


async def complete(backend: CodexOAuthBackend, **kwargs: Any) -> CompletionResult:
    return await backend.complete(
        model="fixture-model",
        messages=[{"role": "user", "content": "Answer."}],
        max_tokens=42,
        **kwargs,
    )


@pytest.mark.parametrize(
    "config_type",
    [
        ConfiguredModelSettings,
        FallbackModelSettings,
        ModelConfig,
        ResolvedFallbackConfig,
    ],
)
@pytest.mark.parametrize("transport", ["anthropic", "gemini"])
def test_codex_rejects_non_openai(config_type: Any, transport: str) -> None:
    params: dict[str, Any] = {"model": "fixture-model", "transport": transport}
    if config_type in (ConfiguredModelSettings, FallbackModelSettings):
        params["overrides"] = {"provider_params": {"codex_oauth": True}}
    else:
        params["provider_params"] = {"codex_oauth": True}
    with pytest.raises(ValueError, match="codex_oauth"):
        config_type(**params)


@pytest.mark.parametrize(
    "overrides",
    [
        {"base_url": "https://example.org"},
        {"api_key": "fixture-key"},
        {"api_key_env": "NEVER_READ_THIS_SECRET"},
    ],
)
def test_codex_rejects_credential_overrides(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="codex_oauth"):
        ConfiguredModelSettings.model_validate(
            {
                "model": "fixture-model",
                "transport": "openai",
                "overrides": {**overrides, "provider_params": {"codex_oauth": True}},
            }
        )


@pytest.mark.parametrize(
    "unsafe",
    [
        {"extra_headers": {"Authorization": SENTINEL}},
        {"extra_body": {"store": True}},
        {"extra_query": {"api_key": SENTINEL}},
        {"top_p": 0.5},
        {"unknown_control": True},
        {"timeout": 601},
        {"api_mode": "chat_completions"},
    ],
)
def test_codex_rejects_unsafe_provider_params(unsafe: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        ModelOverrideSettings(provider_params={"codex_oauth": True, **unsafe})


@pytest.mark.parametrize("value", ["true", 1, None, [], {}])
def test_codex_opt_in_requires_boolean(value: Any) -> None:
    with pytest.raises(ValueError, match="codex_oauth"):
        ModelOverrideSettings(provider_params={"codex_oauth": value})


@pytest.mark.parametrize(
    "control",
    [
        {"stop_sequences": ["END"]},
        {"top_k": 1},
        {"thinking_budget_tokens": 0},
        {"fallback": {"model": "billed", "transport": "openai"}},
    ],
)
def test_codex_rejects_unsupported_model_controls(control: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="codex_oauth"):
        ModelConfig(
            model="fixture-model",
            transport="openai",
            provider_params={"codex_oauth": True},
            **control,
        )


def test_embeddings_cannot_opt_in() -> None:
    with pytest.raises(ValueError, match="codex_oauth"):
        ConfiguredEmbeddingModelSettings(
            overrides=ModelOverrideSettings(provider_params={"codex_oauth": True})
        )


async def test_real_sdk_body_headers_rotation_and_structured_output(
    token_file: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    class Answer(BaseModel):
        answer: str

    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return sse(response_events(text='{"answer":"yes"}'))

    caplog.set_level(logging.DEBUG)
    async with make_client(token_file, handle, client_version="0.146.0") as client:
        backend = make_backend(client)
        result = await complete(
            backend, response_format=Answer, temperature=0.7, thinking_effort="low"
        )
        assert result.content == Answer(answer="yes")
        write_credentials(token_file, account="account-two", residency="eu")
        await complete(backend)
        token = write_credentials(token_file, account="account-three", residency=None)
        await complete(backend)
        assert client.max_retries == 0
        assert isinstance(client.timeout, httpx.Timeout)
        assert client.timeout.read is not None and client.timeout.read <= 600
    assert len(requests) == 3
    for request in requests:
        assert str(request.url) == ENDPOINT
        assert request.method == "POST"
        body = json.loads(request.content)
        assert body["stream"] is True
        assert body["store"] is False
        assert body["instructions"].strip()
        assert body["include"] == ["reasoning.encrypted_content"]
        assert "max_output_tokens" not in body
        assert "temperature" not in body
        assert "codex_oauth" not in body
        assert request.headers["originator"] == "honcho-fork"
        assert request.headers["user-agent"].startswith("HonchoFork/")
        assert request.headers["version"] == "0.146.0"
    assert requests[0].headers["chatgpt-account-id"] == "account-one"
    assert requests[1].headers["x-openai-internal-codex-residency"] == "eu"
    assert requests[2].headers["authorization"] == f"Bearer {token}"
    assert "x-openai-internal-codex-residency" not in requests[2].headers
    assert json.loads(requests[0].content)["text"]["format"]["type"] == "json_schema"
    for request in requests:
        assert request.headers["authorization"] not in caplog.text


@pytest.mark.parametrize(
    "damage",
    [
        "missing",
        "expired",
        "malformed",
        "refresh",
        "oversized",
        "permissions",
        "symlink",
        "directory",
        "bad-jwt",
        "bad-claims",
        "nan-expiry",
        "jwt-expired",
    ],
)
async def test_credentials_fail_closed(
    token_file: Path, damage: str, caplog: pytest.LogCaptureFixture
) -> None:
    data = json.loads(token_file.read_text())
    if damage == "missing":
        token_file.unlink()
    elif damage == "expired":
        data["expires_at"] = time.time() - 1
    elif damage == "refresh":
        data["refresh_token"] = SENTINEL
    elif damage == "nan-expiry":
        data["expires_at"] = float("nan")
    elif damage in {"bad-jwt", "bad-claims", "jwt-expired"}:
        data["access_token"] = (
            SENTINEL
            if damage == "bad-jwt"
            else jwt.encode(
                {"exp": 1 if damage == "jwt-expired" else time.time() + 3600},
                "fixture-key-long-enough-for-safe-tests",
                algorithm="HS256",
            )
        )
    if damage in {
        "expired",
        "refresh",
        "bad-jwt",
        "bad-claims",
        "nan-expiry",
        "jwt-expired",
    }:
        token_file.write_text(json.dumps(data))
    elif damage == "malformed":
        token_file.write_text(SENTINEL)
    elif damage == "oversized":
        token_file.write_text(SENTINEL * 2000)
    elif damage == "permissions":
        token_file.chmod(0o644)
    elif damage == "symlink":
        target = token_file.with_suffix(".target")
        token_file.rename(target)
        token_file.symlink_to(target)
    elif damage == "directory":
        token_file.unlink()
        token_file.mkdir()
    handler = Mock(return_value=sse(response_events()))
    caplog.set_level(logging.DEBUG)
    async with make_client(token_file, handler) as client:
        with pytest.raises(ValidationException, match="Codex OAuth") as error:
            await complete(make_backend(client))
    handler.assert_not_called()
    assert SENTINEL not in str(error.value)
    assert SENTINEL not in caplog.text


@pytest.mark.parametrize(
    "extra",
    [
        {"extra_headers": {"host": "evil.example", "authorization": SENTINEL}},
        {"extra_body": {"model": "billed", "stream": False}},
        {"codex_oauth": False},
        {"api_mode": "chat_completions"},
        {"top_p": 0.9},
        {"seed": 1},
    ],
)
async def test_runtime_overrides_fail_before_network(
    token_file: Path, extra: dict[str, Any]
) -> None:
    handler = Mock(return_value=sse(response_events()))
    async with make_client(token_file, handler) as client:
        with pytest.raises(ValidationException):
            await complete(make_backend(client), extra_params=extra)
    handler.assert_not_called()


@pytest.mark.parametrize(
    "url",
    [
        "https://evil.example/responses",
        "https://chatgpt.com/backend-api/codex/responses?token=bad",
        "http://chatgpt.com/backend-api/codex/responses",
        "https://chatgpt.com/backend-api/codex/models",
    ],
)
async def test_token_client_rejects_all_other_urls(token_file: Path, url: str) -> None:
    handler = Mock(return_value=sse(response_events()))
    async with make_client(token_file, handler) as client:
        with pytest.raises(CodexOAuthError):
            await client.post(url, cast_to=object, body={"stream": True})
    handler.assert_not_called()


@pytest.mark.parametrize("status", [302, 400, 401, 429, 500])
async def test_http_errors_never_leak_or_retry(
    token_file: Path, status: int, caplog: pytest.LogCaptureFixture
) -> None:
    handler = Mock(
        return_value=httpx.Response(
            status,
            headers={"location": "https://evil.example", "x-request-id": SENTINEL},
            json={"error": {"message": SENTINEL}},
        )
    )
    caplog.set_level(logging.DEBUG)
    async with make_client(token_file, handler) as client:
        with pytest.raises(ValidationException) as error:
            await complete(make_backend(client))
    assert handler.call_count == 1
    assert SENTINEL not in str(error.value)
    assert SENTINEL not in caplog.text


@pytest.mark.parametrize(
    "terminal", ["failed", "incomplete", "cancelled", "refusal", "missing", "error"]
)
@pytest.mark.parametrize("streaming", [False, True])
async def test_terminal_failures_propagate_safely(
    token_file: Path, terminal: str, streaming: bool, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    events = response_events()
    if terminal == "missing":
        events = events[:-1]
    elif terminal == "refusal":
        events = response_events(
            text="",
            output=[
                {
                    "type": "message",
                    "id": "msg_fixture",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "refusal", "refusal": SENTINEL}],
                }
            ],
        )
    elif terminal == "error":
        events = [
            {
                "type": "error",
                "code": "server_error",
                "message": SENTINEL,
                "sequence_number": 0,
            }
        ]
    else:
        events = response_events(status=terminal)
    async with make_client(token_file, lambda _: sse(events)) as client:
        backend = make_backend(client)
        with pytest.raises(ValidationException) as error:
            if streaming:
                _ = [
                    chunk
                    async for chunk in backend.stream(
                        model="fixture-model", messages=[], max_tokens=42
                    )
                ]
            else:
                await complete(backend)
    assert SENTINEL not in str(error.value)

    assert SENTINEL not in caplog.text


async def test_successful_text_stream(token_file: Path) -> None:
    async with make_client(token_file, lambda _: sse(response_events())) as client:
        chunks = [
            chunk
            async for chunk in make_backend(client).stream(
                model="fixture-model", messages=[], max_tokens=42
            )
        ]
    assert "".join(chunk.content for chunk in chunks) == "ok"
    assert sum(chunk.is_done for chunk in chunks) == 1


@pytest.mark.parametrize("terminal_output", [True, False])
async def test_tool_calls_and_encrypted_reasoning_round_trip(
    token_file: Path, terminal_output: bool
) -> None:
    requests: list[dict[str, Any]] = []
    reasoning: dict[str, Any] = {
        "type": "reasoning",
        "id": "rs_fixture",
        "summary": [],
        "encrypted_content": "fixture-encrypted-reasoning",
        "status": None,
    }
    output: list[dict[str, Any]] = [
        reasoning,
        {
            "type": "function_call",
            "id": "fc_fixture",
            "call_id": "call_fixture",
            "name": "lookup",
            "arguments": '{"query":"hi"}',
            "status": "completed",
        },
    ]

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        events = response_events(text="", output=output)
        if not terminal_output:
            events[-1]["response"].pop("output")
            events[-1:-1] = [
                {
                    "type": "response.output_item.done",
                    "item": item,
                    "output_index": index,
                    "sequence_number": index + 1,
                }
                for index, item in enumerate(output)
            ]
        return sse(events)

    async with make_client(token_file, handle) as client:
        backend = make_backend(client)
        result = await complete(
            backend,
            tools=[
                {
                    "name": "lookup",
                    "description": "Look up",
                    "input_schema": {"type": "object", "properties": {}},
                }
            ],
            tool_choice="any",
        )
        assert result.tool_calls[0].id == "call_fixture"
        assert result.tool_calls[0].input == {"query": "hi"}
        assert (
            result.reasoning_details[0]["encrypted_content"]
            == "fixture-encrypted-reasoning"
        )
        await backend.complete(
            model="fixture-model",
            max_tokens=42,
            messages=[
                {
                    "role": "assistant",
                    "content": "",
                    "reasoning_details": result.reasoning_details,
                    "tool_calls": [
                        {
                            "id": "call_fixture",
                            "type": "function",
                            "function": {
                                "name": "lookup",
                                "arguments": '{"query":"hi"}',
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_fixture", "content": "found"},
            ],
        )
    assert requests[0]["tool_choice"] == "required"
    assert requests[1]["input"][0]["encrypted_content"] == "fixture-encrypted-reasoning"
    assert "status" not in requests[1]["input"][0]
    assert requests[1]["input"][-1] == {
        "type": "function_call_output",
        "call_id": "call_fixture",
        "output": "found",
    }


async def test_streaming_with_tools_rejected(token_file: Path) -> None:
    handler = Mock(return_value=sse(response_events()))
    async with make_client(token_file, handler) as client:
        with pytest.raises(ValidationException):
            _ = [
                chunk
                async for chunk in make_backend(client).stream(
                    model="fixture-model",
                    messages=[],
                    max_tokens=42,
                    tools=[{"name": "lookup"}],
                )
            ]
    handler.assert_not_called()


async def test_registry_opt_in_does_not_touch_normal_client(
    token_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings.LLM, "CODEX_OAUTH_TOKEN_FILE", str(token_file))
    ordinary = Mock()
    monkeypatch.setitem(registry.CLIENTS, "openai", ordinary)
    config = resolve_model_config(
        ConfiguredModelSettings(
            model="fixture-model",
            transport="openai",
            overrides=ModelOverrideSettings(provider_params={"codex_oauth": True}),
        )
    )
    client = registry.client_for_model_config("openai", config)
    try:
        assert isinstance(client, CodexOAuthClient)
        assert isinstance(registry.get_backend(config), CodexOAuthBackend)
        assert (
            registry.client_for_model_config(
                "openai", ModelConfig(model="normal", transport="openai")
            )
            is ordinary
        )
        assert registry.default_client("openai") is ordinary
    finally:
        assert isinstance(client, CodexOAuthClient)
        await client.close()
        get_codex_oauth_client.cache_clear()


async def test_request_builder_uses_codex_backend(token_file: Path) -> None:
    config = ModelConfig(
        model="fixture-model", transport="openai", provider_params={"codex_oauth": True}
    )
    async with make_client(token_file, lambda _: sse(response_events())) as client:
        result = await execute_completion(
            make_backend(client),
            config,
            messages=[{"role": "user", "content": "Answer."}],
            max_tokens=42,
        )
    assert result.content == "ok"


@pytest.mark.parametrize(
    "extra",
    [
        {"timeout": False},
        {"timeout": 0},
        {"timeout": float("inf")},
        {"timeout": float("nan")},
        {"structured_output_mode": {}},
        {"structured_output_mode": "unsupported"},
    ],
)
def test_codex_rejects_malformed_allowed_controls(extra: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        ModelConfig(
            model="fixture-model",
            transport="openai",
            provider_params={"codex_oauth": True, **extra},
        )


@pytest.mark.parametrize("control", [{"stop": ["END"]}, {"thinking_budget_tokens": 0}])
async def test_runtime_unsupported_controls_do_not_send(
    token_file: Path, control: dict[str, Any]
) -> None:
    handler = Mock(return_value=sse(response_events()))
    async with make_client(token_file, handler) as client:
        with pytest.raises(ValidationException):
            await complete(make_backend(client), **control)
    handler.assert_not_called()


@pytest.mark.parametrize(
    "control",
    [
        {"store": True},
        {"stream": False},
        {"temperature": 0.5},
        {"max_output_tokens": 100},
        {"instructions": ""},
    ],
)
async def test_direct_sdk_body_controls_cannot_bypass_guard(
    token_file: Path, control: dict[str, Any]
) -> None:
    handler = Mock(return_value=sse(response_events()))
    body: dict[str, Any] = {
        "model": "fixture-model",
        "input": [],
        "instructions": "Be helpful.",
        "store": False,
        "stream": True,
        **control,
    }
    async with make_client(token_file, handler) as client:
        with pytest.raises(CodexOAuthError):
            await client.post("/responses", cast_to=object, body=body)
    handler.assert_not_called()


async def test_direct_sdk_http_error_is_safe(
    token_file: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    handler = Mock(return_value=httpx.Response(400, text=SENTINEL))
    async with make_client(token_file, handler) as client:
        with pytest.raises(CodexOAuthError) as error:
            await client.responses.create(
                model="fixture-model",
                input=[],
                instructions="Be helpful.",
                store=False,
                stream=True,
            )
    assert handler.call_count == 1
    assert SENTINEL not in str(error.value)
    assert SENTINEL not in caplog.text


async def test_network_error_is_safe(
    token_file: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    handler = Mock(side_effect=httpx.ReadTimeout(SENTINEL))
    async with make_client(token_file, handler) as client:
        with pytest.raises(ValidationException) as error:
            await complete(make_backend(client))
    assert handler.call_count == 1
    assert SENTINEL not in str(error.value)
    assert SENTINEL not in caplog.text


async def test_per_request_timeout_is_bounded(token_file: Path) -> None:
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return sse(response_events())

    async with make_client(token_file, handle) as client:
        await complete(make_backend(client), extra_params={"timeout": 5})
    assert requests[0].extensions["timeout"] == {
        "connect": 5,
        "read": 5,
        "write": 5,
        "pool": 5,
    }
    assert "version" not in requests[0].headers


@pytest.mark.parametrize("version", ["bad\r\nheader", "official-codex", ""])
def test_invalid_protocol_version_rejected(token_file: Path, version: str) -> None:
    with pytest.raises(ValidationException):
        make_client(token_file, Mock(), client_version=version)


def test_missing_token_file_setting_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings.LLM, "CODEX_OAUTH_TOKEN_FILE", None)
    with pytest.raises(ValidationException, match="CODEX_OAUTH_TOKEN_FILE"):
        registry.get_backend(
            ModelConfig(
                model="fixture-model",
                transport="openai",
                provider_params={"codex_oauth": True},
            )
        )


@pytest.mark.parametrize("value", [True, False])
def test_for_model_revalidates_codex_transport(value: bool) -> None:
    config = ModelConfig(
        model="fixture-model",
        transport="openai",
        provider_params={"codex_oauth": value},
    )
    if value:
        with pytest.raises(ValueError, match="codex_oauth"):
            config.for_model("other", transport_override="anthropic")
    else:
        assert (
            config.for_model("other", transport_override="anthropic").transport
            == "anthropic"
        )


async def test_residency_claim_alias_and_conflicts(token_file: Path) -> None:
    claims = {"chatgpt_account_id": "account-one", "residency": "us"}
    data = {
        "access_token": jwt.encode(
            {"https://api.openai.com/auth": claims, "exp": time.time() + 3600},
            "fixture-signing-key-not-a-real-credential",
            algorithm="HS256",
        ),
        "expires_at": time.time() + 3600,
    }
    token_file.write_text(json.dumps(data))
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return sse(response_events())

    async with make_client(token_file, handle) as client:
        await complete(make_backend(client))
    assert requests[0].headers["x-openai-internal-codex-residency"] == "us"
    claims["chatgpt_account_residency"] = "eu"
    data["access_token"] = jwt.encode(
        {"https://api.openai.com/auth": claims, "exp": time.time() + 3600},
        "fixture-signing-key-not-a-real-credential",
        algorithm="HS256",
    )
    token_file.write_text(json.dumps(data))
    handler = Mock(return_value=sse(response_events()))
    async with make_client(token_file, handler) as client:
        with pytest.raises(ValidationException):
            await complete(make_backend(client))
    handler.assert_not_called()


@pytest.mark.parametrize("terminal_output", ["missing", "null", "empty"])
@pytest.mark.parametrize("streaming", [False, True])
async def test_codex_terminal_without_output_uses_streamed_items(
    token_file: Path, terminal_output: str, streaming: bool
) -> None:
    output: list[dict[str, Any]] = [
        {
            "type": "message",
            "id": "msg_fixture",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": "ok", "annotations": []}],
        }
    ]
    events = response_events(output=output)
    events.insert(
        -1,
        {
            "type": "response.output_item.done",
            "item": output[0],
            "output_index": 0,
            "sequence_number": 4,
        },
    )
    if terminal_output == "missing":
        events[-1]["response"].pop("output")
    else:
        events[-1]["response"]["output"] = None if terminal_output == "null" else []
    async with make_client(token_file, lambda _: sse(events)) as client:
        backend = make_backend(client)
        if streaming:
            chunks = [
                chunk
                async for chunk in backend.stream(
                    model="fixture-model", messages=[], max_tokens=42
                )
            ]
            assert "".join(chunk.content for chunk in chunks) == "ok"
            assert chunks[-1].is_done
        else:
            result = await complete(backend)
            assert result.content == "ok"
