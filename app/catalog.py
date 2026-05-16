"""Catalog data model and catalog-only validation helpers."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlunparse

from pydantic import BaseModel, ConfigDict, Field, field_validator

LOGGER = logging.getLogger(__name__)
SHL_HOSTS = {"www.shl.com", "shl.com"}

TEST_TYPE_LABELS = {
    "A": "Ability / cognitive",
    "B": "Biodata / behavioral",
    "C": "Competency",
    "D": "Development",
    "E": "Assessment exercises",
    "K": "Knowledge / skills",
    "P": "Personality",
    "S": "Situational judgment",
}


def normalize_space(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", " ", value).strip()


def normalize_name(value: str) -> str:
    value = normalize_space(value).lower()
    value = re.sub(r"\([^)]*\)", "", value)
    value = re.sub(r"[^a-z0-9+#.]+", " ", value)
    return normalize_space(value)


def dedupe_name_key(value: str) -> str:
    value = normalize_space(value).lower()
    value = re.sub(r"[^a-z0-9+#.]+", " ", value)
    return normalize_space(value)


def canonical_url(url: str) -> str:
    parsed = urlparse(url)
    scheme = "https"
    netloc = parsed.netloc.lower()
    path = re.sub(r"/+", "/", parsed.path)
    if path != "/" and path.endswith("/"):
        path = path[:-1]
    return urlunparse((scheme, netloc, path, "", "", ""))


def is_valid_shl_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
    except ValueError:
        return False
    return parsed.scheme in {"http", "https"} and parsed.netloc.lower() in SHL_HOSTS


def normalize_test_type(value: str | list[str] | None) -> str:
    if isinstance(value, list):
        tokens = value
    else:
        tokens = re.findall(r"[A-Z]", value or "")
    cleaned: list[str] = []
    for token in tokens:
        token = token.upper()
        if token in TEST_TYPE_LABELS and token not in cleaned:
            cleaned.append(token)
    return " ".join(cleaned) if cleaned else "Other"


def stable_id(name: str, url: str) -> str:
    digest = hashlib.sha1(f"{normalize_name(name)}|{canonical_url(url)}".encode()).hexdigest()
    return digest[:12]


class AssessmentModel(BaseModel):
    """Pydantic representation stored in data/catalog.json."""

    model_config = ConfigDict(extra="allow")

    id: str | None = None
    name: str
    url: str
    test_type: str = "Other"
    description: str = ""
    raw_text: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name", "url")
    @classmethod
    def required_text(cls, value: str) -> str:
        value = normalize_space(value)
        if not value:
            raise ValueError("required field is blank")
        return value


@dataclass(frozen=True)
class Assessment:
    id: str
    name: str
    url: str
    test_type: str
    description: str = ""
    raw_text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def test_type_codes(self) -> set[str]:
        return set(re.findall(r"[A-Z]", self.test_type))

    @property
    def is_placeholder(self) -> bool:
        return bool(self.metadata.get("placeholder"))

    def to_public_dict(self) -> dict[str, str]:
        return {"name": self.name, "url": self.url, "test_type": self.test_type}


def _coerce_assessment(record: dict[str, Any]) -> Assessment | None:
    try:
        model = AssessmentModel.model_validate(record)
    except Exception as exc:  # pydantic error carries detailed context
        LOGGER.warning("Skipping invalid catalog record: %s", exc)
        return None

    url = model.url
    if not is_valid_shl_url(url):
        LOGGER.warning("Skipping non-SHL catalog URL: %s", url)
        return None

    metadata = dict(model.metadata or {})
    raw_test_type = metadata.get("original_test_type") or model.test_type
    test_type = normalize_test_type(raw_test_type)
    name = normalize_space(model.name)
    description = normalize_space(model.description)
    raw_text = normalize_space(model.raw_text or " ".join([name, description]))
    return Assessment(
        id=model.id or stable_id(name, url),
        name=name,
        url=url,
        test_type=test_type,
        description=description,
        raw_text=raw_text,
        metadata=metadata,
    )


def emergency_catalog() -> list[Assessment]:
    """Tiny non-production fallback so local boot errors are diagnosable."""

    return [
        Assessment(
            id="emergency-dev",
            name="Catalog unavailable",
            url="https://www.shl.com/solutions/products/product-catalog/",
            test_type="Other",
            description="Local development placeholder. Run scripts/scrape_catalog.py.",
            raw_text="Catalog unavailable. Run scripts/scrape_catalog.py.",
            metadata={"placeholder": True},
        )
    ]


def load_catalog(path: str | Path, allow_emergency: bool = True) -> list[Assessment]:
    """Load, validate, normalize, and deduplicate SHL catalog records."""

    catalog_path = Path(path)
    if not catalog_path.exists():
        LOGGER.error("Catalog file does not exist: %s", catalog_path)
        return emergency_catalog() if allow_emergency else []

    try:
        payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    except Exception as exc:
        LOGGER.exception("Failed to read catalog file %s: %s", catalog_path, exc)
        return emergency_catalog() if allow_emergency else []

    if isinstance(payload, dict):
        records = payload.get("items") or payload.get("assessments") or []
    else:
        records = payload
    if not isinstance(records, list):
        LOGGER.error("Catalog JSON root must be a list or contain an items list")
        return emergency_catalog() if allow_emergency else []

    seen_urls: set[str] = set()
    seen_names: set[str] = set()
    catalog: list[Assessment] = []
    for raw in records:
        if not isinstance(raw, dict):
            continue
        assessment = _coerce_assessment(raw)
        if assessment is None:
            continue
        key_url = canonical_url(assessment.url)
        key_name = dedupe_name_key(assessment.name)
        if key_url in seen_urls or key_name in seen_names:
            continue
        seen_urls.add(key_url)
        seen_names.add(key_name)
        catalog.append(assessment)

    if not catalog:
        LOGGER.error("Catalog loaded zero valid records from %s", catalog_path)
        return emergency_catalog() if allow_emergency else []

    LOGGER.info("Loaded %d catalog records from %s", len(catalog), catalog_path)
    return catalog


def catalog_url_set(catalog: list[Assessment]) -> set[str]:
    return {canonical_url(item.url) for item in catalog if not item.is_placeholder}


def find_by_url(catalog: list[Assessment], url: str) -> Assessment | None:
    target = canonical_url(url)
    for item in catalog:
        if canonical_url(item.url) == target:
            return item
    return None
