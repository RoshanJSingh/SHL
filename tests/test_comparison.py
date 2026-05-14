from tests.conftest import post_chat


def test_compare_opq_and_gsa_is_grounded_and_not_recommendation(client):
    data = post_chat(client, [{"role": "user", "content": "What is the difference between OPQ and GSA?"}])
    assert data["recommendations"] == []
    assert data["end_of_conversation"] is False
    reply = data["reply"].lower()
    assert "opq" in reply
    assert "gsa" in reply or "global skills assessment" in reply
    assert "catalog" in reply
    assert "not available" in reply or "test type" in reply
