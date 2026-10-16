"""Canonical Codex JWT residency claims match the installed credential owner."""

import json
import time

import jwt
import pytest

from src.llm.codex_oauth import _credential_headers


@pytest.mark.parametrize(
    "metadata,expected",
    [
        ({"chatgpt_data_residency": "eu"}, "eu"),
        ({"chatgpt_compute_residency": "us"}, "us"),
        ({"chatgpt_data_residency": "eu", "chatgpt_compute_residency": "us"}, "eu"),
    ],
)
def test_canonical_residency_claims(tmp_path, metadata, expected):
    claims = {"chatgpt_account_id": "fixture-account", **metadata}
    token = jwt.encode(
        {"https://api.openai.com/auth": claims, "exp": time.time() + 3600},
        "fixture-signing-key-not-a-real-credential",
        algorithm="HS256",
    )
    path = tmp_path / "access.json"
    path.write_text(
        json.dumps({"access_token": token, "expires_at": time.time() + 3600})
    )
    path.chmod(0o600)
    assert (
        _credential_headers(str(path))["x-openai-internal-codex-residency"] == expected
    )
