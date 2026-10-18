"""The opt-in Codex profile is public config, not credentials or global defaults."""

from __future__ import annotations

import json
import os
from pathlib import Path

import httpx
import pytest
from dotenv import dotenv_values

from src.config import (
    DeriverSettings,
    DialecticSettings,
    DreamSettings,
    LLMSettings,
    SummarySettings,
    resolve_model_config,
)
from tests.llm.test_codex_oauth import (
    make_backend,
    make_client,
    response_events,
    sse,
    write_credentials,
)

PROFILE = Path(__file__).resolve().parents[2] / "fork" / "codex-tiered.env.example"
EFFORTS = {
    "DERIVER_MODEL_CONFIG": "low",
    "SUMMARY_MODEL_CONFIG": "low",
    "DREAM_DEDUCTION_MODEL_CONFIG": "medium",
    "DREAM_INDUCTION_MODEL_CONFIG": "medium",
    "DIALECTIC_LEVELS__MINIMAL__MODEL_CONFIG": "low",
    "DIALECTIC_LEVELS__LOW__MODEL_CONFIG": "low",
    "DIALECTIC_LEVELS__MEDIUM__MODEL_CONFIG": "medium",
    "DIALECTIC_LEVELS__HIGH__MODEL_CONFIG": "high",
    "DIALECTIC_LEVELS__MAX__MODEL_CONFIG": "max",
}


def load_profile() -> dict[str, str | None]:
    assert PROFILE.is_file(), "The release must ship its non-secret tier profile"
    return dict(dotenv_values(PROFILE, interpolate=False))


def test_profile_has_exact_model_effort_keys_only() -> None:
    expected = {}
    for prefix, effort in EFFORTS.items():
        expected[prefix + "__MODEL"] = "gpt-6.1-sol"
        expected[prefix + "__THINKING_EFFORT"] = effort
    assert load_profile() == expected


def test_profile_resolves_real_settings_without_changing_workflows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in list(os.environ):
        if any(name.startswith(prefix + "__") for prefix in EFFORTS):
            monkeypatch.delenv(name)
    monkeypatch.delenv("LLM_CODEX_OAUTH_CLIENT_VERSION", raising=False)
    for prefix in EFFORTS:
        monkeypatch.setenv(prefix + "__TRANSPORT", "openai")
        monkeypatch.setenv(
            prefix + "__OVERRIDES__PROVIDER_PARAMS", '{"codex_oauth":true}'
        )
    before = DialecticSettings(_env_file=None)
    for name, value in load_profile().items():
        assert value is not None
        monkeypatch.setenv(name, value)
    dialectic = DialecticSettings(_env_file=None)
    dream = DreamSettings(_env_file=None)
    configs = {
        "DERIVER_MODEL_CONFIG": DeriverSettings(_env_file=None).MODEL_CONFIG,
        "SUMMARY_MODEL_CONFIG": SummarySettings(_env_file=None).MODEL_CONFIG,
        "DREAM_DEDUCTION_MODEL_CONFIG": dream.DEDUCTION_MODEL_CONFIG,
        "DREAM_INDUCTION_MODEL_CONFIG": dream.INDUCTION_MODEL_CONFIG,
        **{
            f"DIALECTIC_LEVELS__{level.upper()}__MODEL_CONFIG": cfg.MODEL_CONFIG
            for level, cfg in dialectic.LEVELS.items()
        },
    }
    for prefix, config in configs.items():
        resolved = resolve_model_config(config)
        assert resolved.model == "gpt-6.1-sol"
        assert resolved.thinking_effort == EFFORTS[prefix]
        assert resolved.provider_params["codex_oauth"] is True
        assert resolved.fallback is None
    for level, config in dialectic.LEVELS.items():
        assert config.MAX_TOOL_ITERATIONS == before.LEVELS[level].MAX_TOOL_ITERATIONS
        assert config.MAX_OUTPUT_TOKENS == before.LEVELS[level].MAX_OUTPUT_TOKENS
        assert config.TOOL_CHOICE == before.LEVELS[level].TOOL_CHOICE
    assert LLMSettings(_env_file=None).CODEX_OAUTH_CLIENT_VERSION is None


@pytest.mark.parametrize("prefix", list(EFFORTS))
async def test_profile_effort_reaches_real_sdk_without_forced_version(
    tmp_path: Path, prefix: str
) -> None:
    profile = load_profile()
    token_file = tmp_path / "access.json"
    write_credentials(token_file)
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return sse(response_events())

    async with make_client(token_file, handle) as client:
        backend = make_backend(client)
        await backend.complete(
            model=profile[prefix + "__MODEL"],
            messages=[{"role": "user", "content": "Synthetic fixture."}],
            max_tokens=64,
            thinking_effort=profile[prefix + "__THINKING_EFFORT"],
            extra_params={"codex_oauth": True},
        )
    assert len(requests) == 1
    request = requests[0]
    body = json.loads(request.content)
    assert body["model"] == "gpt-6.1-sol"
    assert body["reasoning"]["effort"] == EFFORTS[prefix]
    assert "version" not in request.headers
    assert request.headers["originator"] == "honcho-fork"
    assert body["store"] is False and body["stream"] is True
