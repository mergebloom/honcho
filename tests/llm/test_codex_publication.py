"""Access-token publication protocol tests (no real credentials)."""

import base64
import importlib.util
import json
import time
from pathlib import Path

import pytest


def module():
    path = Path(__file__).parents[2] / "fork" / "receive_access_token.py"
    spec = importlib.util.spec_from_file_location("receiver", path)
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def payload(expires=None):
    expires = int(time.time()) + 3600 if expires is None else expires
    claims = (
        base64.urlsafe_b64encode(json.dumps({"exp": expires}).encode())
        .decode()
        .rstrip("=")
    )
    return {"access_token": "e30." + claims + ".signature", "expires_at": expires}


def test_publication_atomic_and_permissions(tmp_path):
    m = module()
    p = payload()
    result = m.publish(json.dumps(p).encode(), tmp_path)
    assert result["status"] == "published"
    target = tmp_path / "access.json"
    assert json.loads(target.read_text()) == p
    assert target.stat().st_mode & 0o777 == 0o600
    assert len(list(tmp_path.iterdir())) == 1
    assert m.publish(json.dumps(p).encode(), tmp_path)["status"] == "unchanged"


@pytest.mark.parametrize(
    "bad",
    [
        {"refresh_token": "SECRET"},
        {"access_token": "bad", "expires_at": 1},
        {"access_token": "bad", "expires_at": "tomorrow"},
        {"access_token": "bad", "expires_at": True},
    ],
)
def test_invalid_publication_is_safe_and_preserves_old_file(tmp_path, bad):
    m = module()
    good = payload()
    m.publish(json.dumps(good).encode(), tmp_path)
    with pytest.raises(ValueError) as exc:
        m.publish(json.dumps(bad).encode(), tmp_path)
    assert "SECRET" not in str(exc.value)
    assert json.loads((tmp_path / "access.json").read_text()) == good


def test_expired_and_mismatched_expiry_rejected(tmp_path):
    m = module()
    with pytest.raises(ValueError):
        m.publish(json.dumps(payload(int(time.time()) - 1)).encode(), tmp_path)
    p = payload()
    p["expires_at"] += 100
    with pytest.raises(ValueError):
        m.publish(json.dumps(p).encode(), tmp_path)


def test_oversized_and_symlink_rejected(tmp_path):
    m = module()
    with pytest.raises(ValueError):
        m.publish(b"x" * 65537, tmp_path)
    protected = tmp_path / "protected"
    protected.write_text("original")
    (tmp_path / "access.json").symlink_to(protected)
    with pytest.raises(ValueError):
        m.publish(json.dumps(payload()).encode(), tmp_path)
    assert protected.read_text() == "original"
