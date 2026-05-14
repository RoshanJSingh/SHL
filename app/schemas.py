"""API schemas with strict top-level response validation."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator


class Message(BaseModel):
    """Single conversation message supplied by the caller."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=20000)

    @field_validator("content")
    @classmethod
    def strip_content(cls, value: str) -> str:
        return value.strip()


class ChatRequest(BaseModel):
    """POST /chat request."""

    model_config = ConfigDict(extra="forbid")

    messages: list[Message] = Field(min_length=1, max_length=16)


class RecommendationItem(BaseModel):
    """Public recommendation record returned by POST /chat."""

    model_config = ConfigDict(extra="forbid")

    name: str
    url: str
    test_type: str

    @field_validator("name", "url", "test_type")
    @classmethod
    def not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value


class ChatResponse(BaseModel):
    """Exact assignment response shape."""

    model_config = ConfigDict(extra="forbid")

    reply: str
    recommendations: list[RecommendationItem]
    end_of_conversation: bool

    @field_validator("reply")
    @classmethod
    def reply_not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            return "I can help select or compare SHL assessments from the catalog."
        return value

    @field_validator("recommendations")
    @classmethod
    def recommendation_count_is_valid(
        cls, value: list[RecommendationItem]
    ) -> list[RecommendationItem]:
        if len(value) > 10:
            raise ValueError("recommendations must contain at most 10 items")
        return value


class HealthResponse(BaseModel):
    """GET /health response."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["ok"]
