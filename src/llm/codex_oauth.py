"""Access-only Codex OAuth transport; credential refresh belongs to its owner.

Tokens are injected below the SDK, only after the exact request destination is
checked. The SDK and its request-option logger never receive the access token.
No discovery requests are made: this client has exactly one allowed endpoint.
"""

from __future__ import annotations

import json
import math
import os
import re
import stat
import time
from collections.abc import AsyncIterator
from functools import lru_cache
from types import TracebackType
from typing import Any, cast

import httpx
import jwt
from openai import APIConnectionError, AsyncOpenAI, AsyncStream, OpenAIError
from openai.types.responses import Response, ResponseStreamEvent

from src._version import HONCHO_VERSION
from src.config import (
    CODEX_OAUTH_MAX_TIMEOUT,
    CODEX_OAUTH_VERSION_PATTERN,
    validate_codex_oauth_params,
)
from src.exceptions import UpstreamLLMError, ValidationException
from src.llm.backend import CompletionResult, StreamChunk
from src.llm.backends.openai import OpenAIBackend

CODEX_RESPONSES_URL = "https://chatgpt.com/backend-api/codex/responses"
_MAX_CREDENTIAL_BYTES = 16_384
_TIMEOUT = httpx.Timeout(
    connect=10.0, read=CODEX_OAUTH_MAX_TIMEOUT, write=30.0, pool=10.0
)
_USER_AGENT = f"HonchoFork/{HONCHO_VERSION}"


class CodexOAuthError(OpenAIError):
    """A safe error that the SDK propagates without wrapping provider content."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code: int | None = status_code


def _safe_failure(error: Exception, message: str) -> Exception:
    status = getattr(error, "status_code", None)
    if (isinstance(status, int) and status >= 500) or isinstance(
        error, (APIConnectionError, httpx.TransportError, TimeoutError)
    ):
        return UpstreamLLMError("Codex OAuth model provider is unavailable")
    return ValidationException(message)


def _header_claim(value: Any) -> str:
    if not isinstance(value, str) or not 1 <= len(value) <= 256:
        raise ValueError("Invalid credential metadata")
    if not all(32 < ord(character) < 127 for character in value):
        raise ValueError("Invalid credential metadata")
    return value


def _expiry(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError("Invalid credential expiry")
    expiry = float(value)
    if not math.isfinite(expiry) or expiry <= time.time():
        raise ValueError("Expired credential")
    return expiry


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate credential field")
        result[key] = value
    return result


def _credential_headers(token_file: str) -> dict[str, str]:
    """Read one bounded regular file through its descriptor after every rotation.

    JWT claims are unverified routing metadata, not an authorization decision.
    The remote service validates the externally owned bearer credential.
    """
    try:
        descriptor = os.open(
            token_file, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
        )
        with os.fdopen(descriptor, "rb") as credential_file:
            metadata = os.fstat(credential_file.fileno())
            if not stat.S_ISREG(metadata.st_mode) or stat.S_IMODE(
                metadata.st_mode
            ) not in {0o400, 0o600}:
                raise ValueError("Unsafe credential permissions")
            if metadata.st_size > _MAX_CREDENTIAL_BYTES:
                raise ValueError("Credential file too large")
            encoded = credential_file.read(_MAX_CREDENTIAL_BYTES + 1)
        if len(encoded) > _MAX_CREDENTIAL_BYTES:
            raise ValueError("Credential file too large")
        parsed: Any = json.loads(encoded, object_pairs_hook=_unique_object)
        if not isinstance(parsed, dict) or parsed.keys() != {
            "access_token",
            "expires_at",
        }:
            raise ValueError("Invalid access-only credential shape")
        data = cast(dict[str, Any], parsed)
        _expiry(data["expires_at"])
        token = data["access_token"]
        if (
            not isinstance(token, str)
            or not token
            or not token.isascii()
            or any(character.isspace() for character in token)
        ):
            raise ValueError("Invalid access token")
        payload = jwt.decode(token, options={"verify_signature": False})
        _expiry(payload["exp"])
        raw_claims: Any = payload["https://api.openai.com/auth"]
        if not isinstance(raw_claims, dict):
            raise ValueError("Invalid account metadata")
        claims = cast(dict[str, Any], raw_claims)
        headers = {
            "Authorization": f"Bearer {token}",
            "ChatGPT-Account-ID": _header_claim(claims["chatgpt_account_id"]),
        }
        residency = (
            claims.get("chatgpt_data_residency")
            or claims.get("chatgpt_compute_residency")
            or claims.get("residency", claims.get("chatgpt_account_residency"))
        )
        if (
            "residency" in claims
            and "chatgpt_account_residency" in claims
            and claims["residency"] != claims["chatgpt_account_residency"]
        ):
            raise ValueError("Conflicting residency metadata")
        if residency is not None:
            headers["x-openai-internal-codex-residency"] = _header_claim(residency)
        return headers
    except Exception:
        raise CodexOAuthError(
            "Codex OAuth access credentials unavailable or invalid"
        ) from None


class _CodexOAuthTransport(httpx.AsyncBaseTransport):
    """Guard the destination and keep credential/provider errors below the SDK."""

    def __init__(
        self,
        token_file: str,
        client_version: str | None,
        transport: httpx.AsyncBaseTransport | None,
    ) -> None:
        self._token_file: str = token_file
        self._client_version: str | None = client_version
        self._transport: httpx.AsyncBaseTransport = (
            transport
            if transport is not None
            else httpx.AsyncHTTPTransport(retries=0, trust_env=False)
        )

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if request.method != "POST" or str(request.url) != CODEX_RESPONSES_URL:
            raise CodexOAuthError("Codex OAuth request destination is not allowed")
        content = await request.aread()
        try:
            raw_body: Any = json.loads(content)
            allowed = {
                "model",
                "input",
                "store",
                "instructions",
                "reasoning",
                "tools",
                "tool_choice",
                "text",
                "include",
                "stream",
            }
            if not isinstance(raw_body, dict) or raw_body.keys() - allowed:
                raise ValueError("Unsupported request body")
            body = cast(dict[str, Any], raw_body)
            if body.get("stream") is not True or body.get("store") is not False:
                raise ValueError("Invalid streaming or storage controls")
            instructions: Any = body.get("instructions")
            if not isinstance(instructions, str) or not instructions.strip():
                raise ValueError("Missing instructions")
        except Exception:
            raise CodexOAuthError("Codex OAuth request body is not allowed") from None
        headers = httpx.Headers(request.headers)
        for name in (
            "authorization",
            "chatgpt-account-id",
            "x-openai-internal-codex-residency",
            "originator",
            "user-agent",
            "version",
            "host",
        ):
            headers.pop(name, None)
        headers.update(_credential_headers(self._token_file))
        headers.update(
            {
                "Host": "chatgpt.com",
                "originator": "honcho-fork",
                "User-Agent": _USER_AGENT,
            }
        )
        if self._client_version is not None:
            headers["version"] = self._client_version
        extensions = dict(request.extensions)
        requested_timeout = cast(dict[str, Any], extensions.get("timeout", {}))
        extensions["timeout"] = {
            key: min(value, requested_timeout[key])
            if isinstance(requested_timeout.get(key), int | float)
            and requested_timeout[key] > 0
            else value
            for key, value in _TIMEOUT.as_dict().items()
        }
        wire_request = httpx.Request(
            request.method,
            request.url,
            headers=headers,
            content=content,
            extensions=extensions,
        )
        try:
            response = await self._transport.handle_async_request(wire_request)
            if response.status_code != 200:
                status = response.status_code
                await response.aclose()
                raise CodexOAuthError(
                    f"Codex OAuth HTTP request failed (status {status})",
                    status_code=status,
                )
            response.headers = httpx.Headers({"content-type": "text/event-stream"})
            return response
        except CodexOAuthError:
            raise
        except Exception:
            raise CodexOAuthError(
                "Codex OAuth network request failed", status_code=503
            ) from None

    async def aclose(self) -> None:
        await self._transport.aclose()


class CodexOAuthClient(AsyncOpenAI):
    """An isolated SDK client, never a replacement for the default OpenAI client."""

    def __init__(
        self,
        *,
        token_file: str,
        client_version: str | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if not token_file:
            raise ValidationException("Codex OAuth requires LLM_CODEX_OAUTH_TOKEN_FILE")
        if (
            client_version is not None
            and re.fullmatch(CODEX_OAUTH_VERSION_PATTERN, client_version) is None
        ):
            raise ValidationException(
                "Codex OAuth client version must be a version string"
            )
        super().__init__(
            api_key="honcho-codex-access-only-placeholder",
            base_url="https://chatgpt.com/backend-api/codex/",
            max_retries=0,
            timeout=_TIMEOUT,
            http_client=httpx.AsyncClient(
                transport=_CodexOAuthTransport(token_file, client_version, transport),
                follow_redirects=False,
                trust_env=False,
                timeout=_TIMEOUT,
            ),
        )


class _CodexResponseStream:
    """Use SDK SSE decoding without its full-terminal-output accumulator.

    Codex can omit terminal output; the shared backend already reconstructs it
    from output-item events. The SDK's higher-level stream manager instead tries
    to iterate that absent output before the backend can see the terminal event.
    """

    def __init__(self, client: AsyncOpenAI, params: dict[str, Any]) -> None:
        self._client: AsyncOpenAI = client
        self._params: dict[str, Any] = params
        self._stream: AsyncStream[ResponseStreamEvent] | None = None
        self._final_response: Response | None = None

    async def __aenter__(self) -> _CodexResponseStream:
        self._stream = await self._client.responses.create(**self._params, stream=True)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._stream is not None:
            await self._stream.close()

    async def __aiter__(self) -> AsyncIterator[ResponseStreamEvent]:
        if self._stream is None:
            raise CodexOAuthError("Codex OAuth stream is not open")
        async for event in self._stream:
            if getattr(event, "type", None) in {
                "response.completed",
                "response.incomplete",
                "response.failed",
                "response.cancelled",
            }:
                self._final_response = getattr(event, "response", None)
                if (
                    self._final_response is not None
                    and getattr(self._final_response, "output", None) is None
                ):
                    self._final_response.output = []
            yield event

    async def get_final_response(self) -> Response:
        if self._final_response is None:
            raise CodexOAuthError(
                "Codex OAuth stream ended without a terminal response"
            )
        return self._final_response


@lru_cache(maxsize=8)
def get_codex_oauth_client(
    token_file: str | None, client_version: str | None
) -> CodexOAuthClient:
    if not token_file:
        raise ValidationException("Codex OAuth requires LLM_CODEX_OAUTH_TOKEN_FILE")
    return CodexOAuthClient(token_file=token_file, client_version=client_version)


class CodexOAuthBackend(OpenAIBackend):
    """Reuse generic Responses normalization with a strict Codex request builder.

    max_tokens/max_output_tokens remain interface/accounting hints, NOT server-side
    output caps. The endpoint does not accept them or temperature.
    """

    @staticmethod
    def _codex_extra_params(extra_params: dict[str, Any] | None) -> dict[str, Any]:
        params = dict(extra_params or {})
        # The shared executor always supplies its inactive per-call toggles.
        # These are not requested provider controls; active/unknown values still
        # fail closed below rather than being silently discarded.
        if params.get("json_mode") is False or params.get("json_mode") is None:
            params.pop("json_mode", None)
        if params.get("verbosity") is None:
            params.pop("verbosity", None)
        if "codex_oauth" in params and params["codex_oauth"] is not True:
            raise ValidationException("Codex OAuth cannot be disabled on this client")
        params["codex_oauth"] = True
        try:
            validate_codex_oauth_params(params)
        except ValueError:
            raise ValidationException(
                "Codex OAuth unsupported request controls"
            ) from None
        params["api_mode"] = "responses"
        return params

    async def complete(self, **kwargs: Any) -> CompletionResult:
        kwargs["extra_params"] = self._codex_extra_params(kwargs.get("extra_params"))
        try:
            return await super().complete(**kwargs)
        except Exception as error:
            raise _safe_failure(
                error, "Codex OAuth request failed or response was not complete"
            ) from None

    async def stream(self, **kwargs: Any) -> AsyncIterator[StreamChunk]:
        kwargs["extra_params"] = self._codex_extra_params(kwargs.get("extra_params"))
        if kwargs.get("tools") or kwargs.get("tool_choice") is not None:
            raise ValidationException(
                "Codex OAuth text streaming does not support tools"
            )
        try:
            async for chunk in super().stream(**kwargs):
                yield chunk
        except Exception as error:
            raise _safe_failure(
                error, "Codex OAuth stream failed or response was not complete"
            ) from None

    def _responses_stream(self, params: dict[str, Any]) -> _CodexResponseStream:
        return _CodexResponseStream(self._client, params)

    def _build_responses_params(self, **kwargs: Any) -> dict[str, Any]:
        kwargs["extra_params"] = self._codex_extra_params(kwargs.get("extra_params"))
        params = super()._build_responses_params(**kwargs)
        params.pop("max_output_tokens", None)
        params.pop("temperature", None)
        instructions = params.get("instructions")
        params["instructions"] = (
            instructions
            if isinstance(instructions, str) and instructions.strip()
            else "You are a helpful assistant."
        )
        params["store"] = False
        params["include"] = ["reasoning.encrypted_content"]
        return params

    @classmethod
    def _validate_responses_status(cls, response: Any) -> None:
        if cls._response_value(response, "status") != "completed":
            raise ValidationException("Codex OAuth response was not complete")
        super()._validate_responses_status(response)
