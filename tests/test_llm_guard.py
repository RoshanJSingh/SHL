from app.agent import AssessmentAgent
from app.catalog import load_catalog
from app.config import get_settings
from app.retrieval import build_retriever
from app.schemas import Message


def test_llm_reply_with_fake_url_is_rejected(monkeypatch):
    catalog = load_catalog("data/catalog.json", allow_emergency=False)
    agent = AssessmentAgent(catalog, build_retriever(catalog), get_settings())

    def fake_complete(*args, **kwargs):
        return "Use Fake SHL Engineer Test at https://fake.example/test. It takes 99 minutes."

    monkeypatch.setattr(agent.llm, "complete", fake_complete)
    response = agent.chat([Message(role="user", content="Hiring a mid-level Java developer")])
    assert "fake.example" not in response.reply
    assert "Fake SHL Engineer Test" not in response.reply
    assert response.recommendations
    assert all(item.url.startswith("https://www.shl.com/") for item in response.recommendations)


def test_safe_llm_reply_can_be_used(monkeypatch):
    catalog = load_catalog("data/catalog.json", allow_emergency=False)
    agent = AssessmentAgent(catalog, build_retriever(catalog), get_settings())

    def safe_complete(*args, **kwargs):
        return "This shortlist emphasizes Java knowledge tests and a communication-related assessment from the provided catalog evidence."

    monkeypatch.setattr(agent.llm, "complete", safe_complete)
    response = agent.chat([Message(role="user", content="Hiring a mid-level Java developer who works with stakeholders")])
    assert response.reply.startswith("This shortlist emphasizes")
    assert response.recommendations
