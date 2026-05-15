"""Small API smoke test against a running server."""

from __future__ import annotations

import argparse
import os

import requests


EXPECTED_KEYS = {"reply", "recommendations", "end_of_conversation"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=os.getenv("BASE_URL", "http://localhost:8000"))
    args = parser.parse_args()
    base = args.base_url.rstrip("/")

    health = requests.get(f"{base}/health", timeout=10)
    health.raise_for_status()
    health_data = health.json()
    if health_data != {"status": "ok"}:
        raise AssertionError(f"Unexpected /health response: {health_data}")
    print("GET /health", health_data)

    payload = {
        "messages": [
            {
                "role": "user",
                "content": "Hiring a mid-level Java developer who works with stakeholders",
            }
        ]
    }
    chat = requests.post(f"{base}/chat", json=payload, timeout=30)
    chat.raise_for_status()
    data = chat.json()
    if set(data) != EXPECTED_KEYS:
        raise AssertionError(f"Unexpected /chat keys: {sorted(data)}")
    if not isinstance(data["reply"], str):
        raise AssertionError("/chat reply must be a string")
    if not isinstance(data["recommendations"], list):
        raise AssertionError("/chat recommendations must be a list")
    if not isinstance(data["end_of_conversation"], bool):
        raise AssertionError("/chat end_of_conversation must be a boolean")
    print("POST /chat reply:", data["reply"][:300])
    print("recommendations:", data["recommendations"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
