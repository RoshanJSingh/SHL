import json
from pathlib import Path

from tests.conftest import post_chat


ROOT = Path(__file__).resolve().parents[1]


def test_public_traces_behavior_and_expected_names(client):
    traces = json.loads((ROOT / "data/evaluation/public_traces.json").read_text(encoding="utf-8"))
    assert len(traces) == 10
    for trace in traces:
        data = post_chat(client, trace["messages"])
        names = {item["name"] for item in data["recommendations"]}
        behavior = trace["behavior"]
        if behavior == "clarify":
            assert data["recommendations"] == []
            assert data["end_of_conversation"] is False
            assert "?" in data["reply"]
        elif behavior == "compare":
            assert data["recommendations"] == []
            assert data["end_of_conversation"] is False
            assert "catalog" in data["reply"].lower() or "test type" in data["reply"].lower()
        elif behavior == "refine":
            assert any("java" in name.lower() for name in names)
            assert any("P" in item["test_type"].split() for item in data["recommendations"])
        else:
            assert 1 <= len(data["recommendations"]) <= 10
            for expected_name in trace.get("expected_names", []):
                assert expected_name in names, f"{trace['id']} missed {expected_name}: {names}"
