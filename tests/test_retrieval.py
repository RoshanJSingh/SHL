from app.catalog import load_catalog
from app.retrieval import build_retriever


def test_retrieval_finds_java_assessments():
    catalog = load_catalog("data/catalog.json", allow_emergency=False)
    retriever = build_retriever(catalog)
    results = retriever.recommend(
        "mid-level Java developer technical programming",
        {"skills": ["java"], "desired_test_types": ["K"], "excluded_test_types": []},
        max_items=5,
    )
    assert results
    assert any("java" in item.assessment.name.lower() for item in results)
    assert all(item.assessment.url.startswith("https://www.shl.com/") for item in results)


def test_retrieval_respects_no_personality():
    catalog = load_catalog("data/catalog.json", allow_emergency=False)
    retriever = build_retriever(catalog)
    results = retriever.recommend(
        "Java developer no personality",
        {"skills": ["java"], "desired_test_types": ["K"], "excluded_test_types": ["P"]},
        max_items=5,
    )
    assert results
    assert not any("P" in item.assessment.test_type_codes for item in results)
