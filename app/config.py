"""Environment-backed application configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _first_present(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


@dataclass(frozen=True)
class Settings:
    catalog_path: Path
    llm_provider: str | None
    model_name: str | None
    openai_api_key: str | None
    gemini_api_key: str | None
    groq_api_key: str | None
    openrouter_api_key: str | None
    openai_base_url: str | None
    use_llm: bool
    app_env: str
    port: int
    request_timeout_seconds: float

    @property
    def has_llm_key(self) -> bool:
        return bool(
            self.openai_api_key
            or self.gemini_api_key
            or self.groq_api_key
            or self.openrouter_api_key
        )

    @property
    def is_production(self) -> bool:
        return self.app_env.lower() in {"prod", "production", "render", "railway", "fly"}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    provider = os.getenv("LLM_PROVIDER")
    openai_key = os.getenv("OPENAI_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY")
    groq_key = os.getenv("GROQ_API_KEY")
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    default_use_llm = bool(openai_key or gemini_key or groq_key or openrouter_key)
    return Settings(
        catalog_path=Path(os.getenv("CATALOG_PATH", "data/catalog.json")),
        llm_provider=provider.lower().strip() if provider else None,
        model_name=os.getenv("MODEL_NAME"),
        openai_api_key=openai_key,
        gemini_api_key=gemini_key,
        groq_api_key=groq_key,
        openrouter_api_key=openrouter_key,
        openai_base_url=os.getenv("OPENAI_BASE_URL"),
        use_llm=_env_bool("USE_LLM", default_use_llm),
        app_env=os.getenv("APP_ENV", os.getenv("ENV", "development")),
        port=int(os.getenv("PORT", "8000")),
        request_timeout_seconds=float(os.getenv("REQUEST_TIMEOUT_SECONDS", "8")),
    )
