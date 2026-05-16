from tests.conftest import post_chat


def names(data):
    return [item["name"] for item in data["recommendations"]]


def test_sales_representative_expected_items(client):
    data = post_chat(
        client,
        [{"role": "user", "content": "Recommend SHL assessments for a sales representative who handles customer conversations."}],
    )
    got = names(data)
    assert len(got) == 10
    assert "Sales & Service Phone Simulation" in got
    assert "Sales & Service Phone Solution" in got
    assert "Entry Level Sales Solution" in got


def test_contact_center_expected_items(client):
    data = post_chat(
        client,
        [{"role": "user", "content": "We need contact center and customer service assessments for frontline agents."}],
    )
    got = names(data)
    assert len(got) == 10
    assert "Contact Center Call Simulation (New)" in got
    assert "Customer Service Phone Simulation" in got
    assert "Customer Service Phone Solution" in got


def test_graduate_cognitive_expected_items(client):
    data = post_chat(
        client,
        [{"role": "user", "content": "Graduate hiring program: looking for cognitive ability and broad potential tests."}],
    )
    got = names(data)
    assert len(got) == 10
    assert "Verify - General Ability Screen" in got
    assert "SHL Verify Interactive G+" in got
    assert "Global Skills Assessment" in got


def test_manager_leadership_expected_items(client):
    data = post_chat(
        client,
        [{"role": "user", "content": "Assess a manager for leadership potential and workplace personality."}],
    )
    got = names(data)
    assert len(got) == 10
    assert "OPQ Leadership Report" in got
    assert "Enterprise Leadership Report 1.0" in got
    assert "Occupational Personality Questionnaire OPQ32r" in got


def test_opq_personality_expected_items(client):
    data = post_chat(
        client,
        [{"role": "user", "content": "I only want personality assessments, preferably OPQ-style reports."}],
    )
    got = names(data)
    assert len(got) == 10
    assert "Occupational Personality Questionnaire OPQ32r" in got
    assert "OPQ Candidate Plus Report" in got
    assert "OPQ Profile Report" in got


def test_python_developer_avoids_obviously_unrelated_domains_in_top_five(client):
    data = post_chat(client, [{"role": "user", "content": "Need Python tests for a backend developer."}])
    top_five = " ".join(names(data)[:5]).lower()
    assert "python (new)" in top_five
    for bad_term in ["food", "beverage", "seo", "search engine optimization", "cardiology", "civil engineering"]:
        assert bad_term not in top_five
