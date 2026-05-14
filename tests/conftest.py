from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def client() -> TestClient:
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="session")
def catalog_urls() -> set[str]:
    data = json.loads((ROOT / "data/catalog.json").read_text(encoding="utf-8"))
    return {item["url"].rstrip("/") for item in data}


def post_chat(client: TestClient, messages: list[dict[str, str]]) -> dict:
    response = client.post("/chat", json={"messages": messages})
    assert response.status_code == 200, response.text
    return response.json()
