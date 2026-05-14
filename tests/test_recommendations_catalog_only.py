from tests.conftest import post_chat


def test_specific_technical_role_recommends_catalog_items(client, catalog_urls):
    data = post_chat(
        client,
        [{"role": "user", "content": "We are hiring a mid-level Java developer who works with stakeholders."}],
    )
    assert data["end_of_conversation"] is True
    assert 1 <= len(data["recommendations"]) <= 10
    assert any("java" in item["name"].lower() for item in data["recommendations"])
    assert any("K" in item["test_type"].split() for item in data["recommendations"])
    for item in data["recommendations"]:
        assert item["url"].rstrip("/") in catalog_urls


def test_catalog_only_across_multiple_queries(client, catalog_urls):
    queries = [
        "Need Python tests for a backend engineer.",
        "Recommend sales representative assessments.",
        "I want personality assessments only.",
        "Need Excel tests for an analyst.",
    ]
    for query in queries:
        data = post_chat(client, [{"role": "user", "content": query}])
        for item in data["recommendations"]:
            assert item["url"].rstrip("/") in catalog_urls
