#!/usr/bin/env python3
"""Publish a short-lived access token using the credential owner's resolver.

Run with Hermes' Python, HERMES_HOME pinned to the intended profile, and a
restricted SSH publication key. The refresh token never leaves Hermes.
No endpoint/proxy is added. This helper depends on the installed Hermes
resolver; revalidate it after Hermes upgrades. Does not force a refresh.
"""

import argparse
import base64
import json
import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hermes-source", type=Path, required=True)
    parser.add_argument("--ssh-target", required=True)
    parser.add_argument("--ssh-key", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("HERMES_HOME"):
        parser.error("HERMES_HOME must explicitly identify the credential owner")
    phase = "credential-resolution"
    try:
        sys.path.insert(0, str(args.hermes_source))
        from hermes_cli.auth import resolve_codex_runtime_credentials

        credential = resolve_codex_runtime_credentials()
        if (
            credential["base_url"].rstrip("/")
            != "https://chatgpt.com/backend-api/codex"
        ):
            raise ValueError("unexpected credential origin")
        token = credential["api_key"]
        encoded = token.split(".")[1]
        claims = json.loads(
            base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4))
        )
        payload = json.dumps(
            {"access_token": token, "expires_at": claims["exp"]}
        ).encode()
        phase = "ssh-publication"
        result = subprocess.run(
            [
                "ssh",
                "-T",
                "-F",
                "/dev/null",
                "-i",
                str(args.ssh_key),
                "-o",
                "BatchMode=yes",
                "-o",
                "IdentitiesOnly=yes",
                "-o",
                "StrictHostKeyChecking=yes",
                "-o",
                "ConnectTimeout=10",
                args.ssh_target,
            ],
            input=payload,
            capture_output=True,
            timeout=30,
            check=False,
        )
        if result.returncode:
            raise ValueError("publication transport failed")
        phase = "acknowledgement"
        response = json.loads(result.stdout)
        if (
            response.get("status") not in ("published", "unchanged")
            or response.get("expires_at") != claims["exp"]
        ):
            raise ValueError("publication acknowledgement invalid")
        print(json.dumps({"status": response["status"], "expires_at": claims["exp"]}))
        return 0
    except Exception as error:
        print(
            f"Codex access publication failed phase={phase} class={type(error).__name__}; "
            "inspect credential-owner auth and receiver availability",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
