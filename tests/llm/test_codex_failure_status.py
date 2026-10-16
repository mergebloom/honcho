"""Provider outages must not masquerade as invalid caller requests."""

import httpx
import pytest

from src.config import ModelConfig
from src.exceptions import UpstreamLLMError
from src.llm.request_builder import execute_completion
from tests.llm.test_codex_oauth import (
    make_backend,
    make_client,
    response_events,
    sse,
    write_credentials,
)


@pytest.fixture
def token_file(tmp_path):
    path = tmp_path / "access.json"
    write_credentials(path)
    return path


async def test_real_executor_default_toggles_are_accepted(token_file):
    from src.llm.executor import honcho_llm_call_inner

    cfg = ModelConfig(
        model="fixture-model", transport="openai", provider_params={"codex_oauth": True}
    )
    async with make_client(token_file, lambda _: sse(response_events())) as client:
        result = await honcho_llm_call_inner(
            provider="openai",
            model=cfg.model,
            prompt="Synthetic fixture",
            max_tokens=32,
            client_override=client,
            selected_config=cfg,
        )
    assert result.content == "ok"


@pytest.mark.parametrize("streaming", [False, True])
async def test_provider_outage_is_503(token_file, streaming):
    def handler(request):
        return httpx.Response(
            503, json={"error": {"message": "PRIVATE-ERROR-SENTINEL"}}
        )

    async with make_client(token_file, handler) as client:
        backend = make_backend(client)
        with pytest.raises(UpstreamLLMError) as exc:
            if streaming:
                _ = [
                    chunk
                    async for chunk in backend.stream(
                        model="fixture-model", messages=[], max_tokens=32
                    )
                ]
            else:
                cfg = ModelConfig(
                    model="fixture-model",
                    transport="openai",
                    provider_params={"codex_oauth": True},
                )
                await execute_completion(backend, cfg, messages=[], max_tokens=32)
        assert exc.value.status_code == 503
        assert "PRIVATE-ERROR-SENTINEL" not in str(exc.value)
