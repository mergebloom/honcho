#!/usr/bin/env python3
"""Restricted SSH receiver: stdin -> atomic access-only credential file.

Install as root-owned, non-writable executable. Authorize its SSH public key
with restrict,from=...,command="python3 ... --directory ... --uid ... --gid ...".
Never grant the publication key a shell, forwarding, PTY or arbitrary arguments.
"""

import argparse
import base64
import json
import os
import stat
import sys
import tempfile
import time
from pathlib import Path

MAX_BYTES = 65536


def publish(raw: bytes, directory: Path, uid: int = -1, gid: int = -1) -> dict:
    """Validate and publish one access token; errors contain no input values."""
    try:
        data = json.loads(raw) if len(raw) <= MAX_BYTES else None
        if not isinstance(data, dict) or set(data) != {"access_token", "expires_at"}:
            raise ValueError
        token, expiry = data["access_token"], data["expires_at"]
        if not isinstance(token, str) or not token or len(token) > MAX_BYTES // 2:
            raise ValueError
        if type(expiry) is not int or expiry <= time.time() + 60:
            raise ValueError
        encoded = token.split(".")[1]
        claims = json.loads(
            base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4))
        )
        if type(claims.get("exp")) is not int or claims["exp"] != expiry:
            raise ValueError
    except Exception:
        raise ValueError("invalid access-only credential publication") from None
    directory = Path(directory)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("credential directory is not an ordinary directory")
    if directory.stat().st_mode & 0o022:
        raise ValueError("credential directory is writable by another principal")
    target = directory / "access.json"
    if target.is_symlink() or (
        target.exists() and not stat.S_ISREG(target.stat().st_mode)
    ):
        raise ValueError("credential target is not an ordinary file")
    if target.exists() and target.stat().st_mode & 0o077 == 0:
        try:
            existing = json.loads(target.read_bytes())
        except Exception:
            existing = None
        if (
            existing == data
            and (uid == -1 or target.stat().st_uid == uid)
            and (gid == -1 or target.stat().st_gid == gid)
        ):
            return {"status": "unchanged", "expires_at": expiry}
    temporary = None
    try:
        fd, name = tempfile.mkstemp(prefix=".access-", dir=directory)
        temporary = Path(name)
        with os.fdopen(fd, "wb") as out:
            os.fchmod(out.fileno(), 0o600)
            if uid != -1 or gid != -1:
                os.fchown(out.fileno(), uid, gid)
            out.write(json.dumps(data, separators=(",", ":")).encode())
            out.flush()
            os.fsync(out.fileno())
        os.replace(temporary, target)
        fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return {"status": "published", "expires_at": expiry}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--uid", type=int, default=-1)
    parser.add_argument("--gid", type=int, default=-1)
    args = parser.parse_args()
    try:
        result = publish(
            sys.stdin.buffer.read(MAX_BYTES + 1), args.directory, args.uid, args.gid
        )
        print(json.dumps(result))
        return 0
    except Exception:
        print("access-token publication rejected", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
