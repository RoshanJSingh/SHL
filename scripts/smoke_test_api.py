"""Small API smoke test against a running server."""

from __future__ import annotations

import argparse

import requests


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    args = parser.parse_args()
    base = args.base_url.rstrip("/")

    health = requests.get(f"{base}/health", timeout=10)
    health.raise_for_status()
    print("GET /health", health.json())

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
    print("POST /chat reply:", data["reply"][:300])
    print("recommendations:", data["recommendations"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
