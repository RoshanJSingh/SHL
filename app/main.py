"""FastAPI entrypoint for the stateless SHL assessment recommender."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.agent import AssessmentAgent
from app.catalog import load_catalog
from app.config import get_settings
from app.logging_utils import configure_logging
from app.retrieval import build_retriever
from app.schemas import ChatRequest, ChatResponse, HealthResponse

configure_logging()
LOGGER = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    catalog = load_catalog(settings.catalog_path, allow_emergency=not settings.is_production)
    retriever = build_retriever(catalog, use_semantic=False)
    app.state.settings = settings
    app.state.catalog = catalog
    app.state.retriever = retriever
    app.state.agent = AssessmentAgent(catalog, retriever, settings)
    LOGGER.info("Application startup complete")
    yield


app = FastAPI(title="SHL Assessment Recommender", version="1.0.0", lifespan=lifespan)


def _get_agent() -> AssessmentAgent:
    agent = getattr(app.state, "agent", None)
    if agent is not None:
        return agent
    # TestClient normally runs lifespan, but this fallback keeps direct calls robust.
    settings = get_settings()
    catalog = load_catalog(settings.catalog_path, allow_emergency=not settings.is_production)
    retriever = build_retriever(catalog, use_semantic=False)
    app.state.agent = AssessmentAgent(catalog, retriever, settings)
    app.state.catalog = catalog
    app.state.retriever = retriever
    return app.state.agent


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    agent = _get_agent()
    return agent.chat(request.messages)
