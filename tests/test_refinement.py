from tests.conftest import post_chat


def test_refinement_adds_personality_without_losing_java_context(client):
    data = post_chat(
        client,
        [
            {"role": "user", "content": "We are hiring a mid-level Java developer who works with stakeholders."},
            {"role": "assistant", "content": "1. Java 8 (New)\n2. Java Frameworks (New)"},
            {"role": "user", "content": "Actually add personality tests too."},
        ],
    )
    assert data["end_of_conversation"] is True
    assert any("java" in item["name"].lower() for item in data["recommendations"])
    assert any("P" in item["test_type"].split() for item in data["recommendations"])


def test_exclusion_downranks_or_excludes_personality(client):
    data = post_chat(client, [{"role": "user", "content": "Need tests for Java developer, no personality."}])
    assert data["recommendations"]
    assert not any("P" in item["test_type"].split() for item in data["recommendations"])
