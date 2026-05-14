from tests.conftest import post_chat


def test_vague_query_asks_one_clarifying_question(client):
    data = post_chat(client, [{"role": "user", "content": "I need an assessment."}])
    assert data["recommendations"] == []
    assert data["end_of_conversation"] is False
    assert "?" in data["reply"]
    assert "role" in data["reply"].lower() or "job" in data["reply"].lower()
