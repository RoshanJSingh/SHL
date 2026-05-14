from tests.conftest import post_chat


def test_chat_exact_schema_for_clarification(client):
    data = post_chat(client, [{"role": "user", "content": "I need an assessment."}])
    assert set(data.keys()) == {"reply", "recommendations", "end_of_conversation"}
    assert isinstance(data["reply"], str)
    assert data["recommendations"] == []
    assert data["end_of_conversation"] is False


def test_recommendation_item_schema(client):
    data = post_chat(
        client,
        [{"role": "user", "content": "Hiring a mid-level Java developer who works with stakeholders"}],
    )
    assert 1 <= len(data["recommendations"]) <= 10
    for item in data["recommendations"]:
        assert set(item.keys()) == {"name", "url", "test_type"}
        assert item["name"]
        assert item["url"]
        assert item["test_type"]
