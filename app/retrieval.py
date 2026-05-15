"""Deterministic hybrid retrieval and ranking over the local SHL catalog."""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize as sk_normalize

from app.catalog import (
    Assessment,
    TEST_TYPE_LABELS,
    normalize_name,
    normalize_space,
    normalize_test_type,
)

LOGGER = logging.getLogger(__name__)

SKILL_PATTERNS = {
    ".net": [r"\.net\b", r"\basp\.net\b", r"\bc#\b"],
    "aws": [r"\baws\b", r"\bamazon web services\b"],
    "azure": [r"\bazure\b"],
    "c++": [r"\bc\+\+\b", r"\bc plus plus\b"],
    "customer service": [r"\bcustomer service\b", r"\bcustomer support\b"],
    "contact center": [r"\bcontact center\b", r"\bcall center\b"],
    "excel": [r"\bexcel\b", r"\bspreadsheet\b"],
    "java": [r"\bjava\b", r"\bj2ee\b", r"\bspring\b"],
    "javascript": [r"\bjavascript\b", r"\bjs\b", r"\bnode\.?js\b"],
    "python": [r"\bpython\b", r"\bdjango\b", r"\bflask\b"],
    "react": [r"\breact\b", r"\breactjs\b"],
    "sales": [r"\bsales\b", r"\baccount manager\b"],
    "sql": [r"\bsql\b", r"\bdata analyst\b", r"\bdatabase\b"],
    "stakeholder": [r"\bstakeholders?\b", r"\bcommunication\b", r"\bcollaboration\b"],
}

ROLE_PATTERNS = {
    "analyst": [r"\banalyst\b", r"\banalytics\b"],
    "developer": [r"\bdeveloper\b", r"\bengineer\b", r"\bprogrammer\b"],
    "graduate": [r"\bgraduate\b", r"\bentry[ -]?level\b", r"\bearly careers?\b"],
    "leadership": [r"\bleadership\b", r"\bleader\b", r"\bmanager\b", r"\bmanagement\b"],
    "frontline": [r"\bfrontline\b", r"\bretail\b", r"\bcashier\b"],
    "sales": [r"\bsales\b", r"\baccount executive\b", r"\baccount manager\b"],
    "support": [r"\bcustomer service\b", r"\bcontact center\b", r"\bsupport\b"],
}

TYPE_KEYWORDS = {
    "K": [
        "technical",
        "coding",
        "programming",
        "knowledge",
        "skills",
        "skill",
        "java",
        "python",
        "sql",
        "excel",
        ".net",
        "c++",
    ],
    "A": ["cognitive", "ability", "aptitude", "reasoning", "numerical", "verbal", "inductive"],
    "P": ["personality", "opq", "behavioral style", "work style"],
    "B": ["biodata", "behavioral", "behavioural"],
    "S": ["situational", "judgment", "judgement", "sjt", "scenario"],
}

ASSESSMENT_FAMILIES = ["opq", "verify", "g+", "gsa", "mq", "ceb", "shl", "java", "python", "excel"]

PREFERRED_COMPARISON_MATCHES = {
    "opq": "occupational personality questionnaire opq32r",
    "gsa": "global skills assessment",
    "g+": "shl verify interactive g+",
}

AMBIGUOUS_COMPARISON_TERMS = {"verify", "java", "excel", "sql", "opq report"}


@dataclass(frozen=True)
class ScoredAssessment:
    assessment: Assessment
    score: float
    lexical_score: float
    semantic_score: float = 0.0
    boosts: dict[str, float] = field(default_factory=dict)
    evidence: list[str] = field(default_factory=list)


def _contains_any(text: str, patterns: Iterable[str]) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def extract_skills(text: str) -> list[str]:
    lowered = text.lower()
    skills = [skill for skill, patterns in SKILL_PATTERNS.items() if _contains_any(lowered, patterns)]
    return skills


def extract_roles(text: str) -> list[str]:
    lowered = text.lower()
    return [role for role, patterns in ROLE_PATTERNS.items() if _contains_any(lowered, patterns)]


def infer_desired_test_types(text: str) -> list[str]:
    lowered = text.lower()
    desired: list[str] = []
    for code, keywords in TYPE_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            desired.append(code)
    return desired


def infer_excluded_test_types(text: str) -> list[str]:
    lowered = text.lower()
    excluded: list[str] = []
    for code, keywords in TYPE_KEYWORDS.items():
        for keyword in keywords:
            if re.search(rf"\b(no|without|exclude|not|avoid)\s+(?:\w+\s+){{0,2}}{re.escape(keyword)}", lowered):
                excluded.append(code)
                break
    return list(dict.fromkeys(excluded))


def _metadata_text(metadata: dict[str, Any]) -> str:
    parts: list[str] = []
    for key, value in sorted(metadata.items()):
        if isinstance(value, (str, int, float, bool)):
            parts.append(f"{key}: {value}")
        elif isinstance(value, list):
            parts.append(f"{key}: {' '.join(str(v) for v in value)}")
    return " ".join(parts)


def assessment_text(item: Assessment) -> str:
    return normalize_space(
        " ".join(
            [
                item.name,
                item.description,
                item.raw_text,
                _metadata_text(item.metadata),
                " ".join(TEST_TYPE_LABELS.get(code, code) for code in item.test_type_codes),
            ]
        )
    )


def _normalize_scores(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    max_value = float(values.max())
    min_value = float(values.min())
    if math.isclose(max_value, min_value):
        return np.zeros_like(values) if math.isclose(max_value, 0.0) else np.ones_like(values)
    return (values - min_value) / (max_value - min_value)


def _extract_duration_minutes(item: Assessment) -> int | None:
    candidates = [
        item.metadata.get("duration"),
        item.metadata.get("assessment_length"),
        item.metadata.get("completion_time_minutes"),
        item.raw_text,
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        text = str(candidate).lower()
        match = re.search(r"(\d{1,3})\s*(?:minutes?|mins?)", text)
        if not match:
            match = re.search(r"minutes\s*=\s*(\d{1,3})", text)
        if match:
            return int(match.group(1))
    return None


class CatalogRetriever:
    """Hybrid retriever with lexical search, optional semantic search, and rule boosts."""

    def __init__(self, catalog: list[Assessment], use_semantic: bool = False) -> None:
        self.catalog = [item for item in catalog if not item.is_placeholder]
        self.documents = [assessment_text(item) for item in self.catalog]
        self.word_vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
            token_pattern=r"(?u)\b[\w+#.]{1,}\b",
        )
        self.char_vectorizer = TfidfVectorizer(
            lowercase=True,
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=1,
            max_df=0.95,
        )
        if self.documents:
            self.word_matrix = sk_normalize(self.word_vectorizer.fit_transform(self.documents))
            self.char_matrix = sk_normalize(self.char_vectorizer.fit_transform(self.documents))
        else:
            self.word_matrix = None
            self.char_matrix = None
        self.semantic_model = None
        self.semantic_embeddings = None
        if use_semantic:
            self._try_load_semantic_model()

    def _try_load_semantic_model(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            self.semantic_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            self.semantic_embeddings = self.semantic_model.encode(
                self.documents, normalize_embeddings=True, show_progress_bar=False
            )
            LOGGER.info("Loaded optional sentence-transformers semantic retriever")
        except Exception as exc:
            LOGGER.info("Semantic retriever unavailable, using lexical only: %s", exc)
            self.semantic_model = None
            self.semantic_embeddings = None

    def _lexical_scores(self, query: str) -> np.ndarray:
        if not self.documents or self.word_matrix is None or self.char_matrix is None:
            return np.array([])
        word_query = self.word_vectorizer.transform([query])
        char_query = self.char_vectorizer.transform([query])
        word_scores = cosine_similarity(word_query, self.word_matrix).ravel()
        char_scores = cosine_similarity(char_query, self.char_matrix).ravel()
        return 0.72 * _normalize_scores(word_scores) + 0.28 * _normalize_scores(char_scores)

    def _semantic_scores(self, query: str) -> np.ndarray:
        if self.semantic_model is None or self.semantic_embeddings is None:
            return np.zeros(len(self.catalog))
        try:
            query_embedding = self.semantic_model.encode([query], normalize_embeddings=True)
            return _normalize_scores(np.matmul(self.semantic_embeddings, query_embedding[0]))
        except Exception as exc:
            LOGGER.warning("Semantic scoring failed, continuing lexical only: %s", exc)
            return np.zeros(len(self.catalog))

    def _rule_boosts(self, item: Assessment, query: str, constraints: dict[str, Any]) -> dict[str, float]:
        text = assessment_text(item).lower()
        name = item.name.lower()
        boosts: dict[str, float] = {}

        skills = list(dict.fromkeys((constraints.get("skills") or []) + extract_skills(query)))
        for skill in skills:
            patterns = SKILL_PATTERNS.get(skill, [rf"\b{re.escape(skill)}\b"])
            if _contains_any(text, patterns):
                boosts[f"skill:{skill}"] = 0.32 if skill in name else 0.22

        roles = list(dict.fromkeys((constraints.get("roles") or []) + extract_roles(query)))
        for role in roles:
            patterns = ROLE_PATTERNS.get(role, [rf"\b{re.escape(role)}\b"])
            if _contains_any(text, patterns):
                boosts[f"role:{role}"] = 0.12

        for family in ASSESSMENT_FAMILIES:
            if family in query.lower() and family in text:
                boosts[f"family:{family}"] = 0.26 if family in name else 0.16

        desired_types = set(constraints.get("desired_test_types") or infer_desired_test_types(query))
        excluded_types = set(constraints.get("excluded_test_types") or infer_excluded_test_types(query))
        for code in desired_types:
            if code in item.test_type_codes:
                boosts[f"type:{code}"] = 0.24
        for code in excluded_types:
            if code in item.test_type_codes:
                boosts[f"excluded:{code}"] = -1.25

        seniority = constraints.get("seniority")
        if seniority:
            seniority_text = str(seniority).lower()
            if seniority_text in text:
                boosts[f"seniority:{seniority_text}"] = 0.10
            if seniority_text in {"entry", "graduate"} and any(t in text for t in ["graduate", "entry", "early career"]):
                boosts["seniority:entry"] = 0.14
            if seniority_text in {"manager", "senior", "leadership"} and any(
                t in text for t in ["manager", "leadership", "supervisor"]
            ):
                boosts["seniority:leader"] = 0.14

        duration = _extract_duration_minutes(item)
        if constraints.get("short_tests_only") and duration is not None:
            if duration <= 30:
                boosts["duration:short"] = 0.12
            elif duration > 45:
                boosts["duration:long_penalty"] = -0.18
        return boosts

    def _evidence_for(self, item: Assessment, query: str, max_snippets: int = 3) -> list[str]:
        snippets: list[str] = []
        candidates = [
            ("name", item.name),
            ("test_type", item.test_type),
            ("description", item.description),
            ("job_levels", str(item.metadata.get("job_levels", ""))),
            ("languages", str(item.metadata.get("languages", ""))),
            ("duration", str(item.metadata.get("duration") or item.metadata.get("assessment_length") or "")),
            ("raw", item.raw_text),
        ]
        terms = set(re.findall(r"[a-zA-Z+#.]{3,}", query.lower()))
        for field, value in candidates:
            clean = normalize_space(value)
            if not clean:
                continue
            lowered = clean.lower()
            if field in {"name", "test_type"} or any(term in lowered for term in terms):
                snippets.append(f"{field}: {clean[:240]}")
            if len(snippets) >= max_snippets:
                break
        if not snippets and item.description:
            snippets.append(f"description: {item.description[:240]}")
        return snippets

    def retrieve(self, query: str, constraints: dict[str, Any] | None = None, k: int = 20) -> list[ScoredAssessment]:
        constraints = constraints or {}
        if not self.catalog:
            return []
        lexical = self._lexical_scores(query)
        semantic = self._semantic_scores(query)
        combined = 0.82 * lexical + 0.18 * semantic
        scored: list[ScoredAssessment] = []
        for idx, item in enumerate(self.catalog):
            boosts = self._rule_boosts(item, query, constraints)
            boost_total = sum(boosts.values())
            score = float(combined[idx] + boost_total)
            if constraints.get("excluded_test_types"):
                excluded = set(constraints["excluded_test_types"])
                if item.test_type_codes and item.test_type_codes.issubset(excluded):
                    score -= 1.0
            scored.append(
                ScoredAssessment(
                    assessment=item,
                    score=score,
                    lexical_score=float(lexical[idx]),
                    semantic_score=float(semantic[idx]),
                    boosts=boosts,
                    evidence=self._evidence_for(item, query),
                )
            )
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[: max(k, 0)]

    def recommend(
        self, query: str, constraints: dict[str, Any] | None = None, max_items: int = 10
    ) -> list[ScoredAssessment]:
        constraints = constraints or {}
        max_items = max(1, min(max_items, 10))
        candidates = self.retrieve(query, constraints, k=len(self.catalog))
        excluded_types = set(constraints.get("excluded_test_types") or [])
        filtered = [
            item
            for item in candidates
            if not (excluded_types and item.assessment.test_type_codes.issubset(excluded_types))
            and item.score > -0.5
        ]
        selected = self._diverse_select(filtered, max_items=max_items, query=query, constraints=constraints)
        return selected[:max_items]

    def _diverse_select(
        self,
        candidates: list[ScoredAssessment],
        max_items: int,
        query: str,
        constraints: dict[str, Any],
    ) -> list[ScoredAssessment]:
        selected: list[ScoredAssessment] = []
        seen_family: set[str] = set()
        explicit_many = bool(re.search(r"\b(10|ten|many|all|versions?)\b", query.lower()))

        def family_key(item: Assessment) -> str:
            name = normalize_name(item.name)
            name = re.sub(r"\b(new|short form|solution|assessment|test)\b", "", name)
            first_tokens = " ".join(name.split()[:3])
            for family in ASSESSMENT_FAMILIES:
                if family in name:
                    return family
            return first_tokens

        def append_unique(scored: ScoredAssessment) -> bool:
            if len(selected) >= max_items:
                return False
            if any(existing.assessment.url == scored.assessment.url for existing in selected):
                return False
            key = family_key(scored.assessment)
            if not explicit_many and key in seen_family and len(selected) >= 3:
                return False
            selected.append(scored)
            seen_family.add(key)
            return True

        def recompute_seen_family() -> None:
            seen_family.clear()
            for scored_item in selected:
                seen_family.add(family_key(scored_item.assessment))

        skills = constraints.get("skills") or extract_skills(query)
        desired_types = constraints.get("desired_test_types") or infer_desired_test_types(query)
        for skill in skills:
            added_for_skill = 0
            for scored in candidates:
                if f"skill:{skill}" in scored.boosts:
                    if append_unique(scored):
                        added_for_skill += 1
                    if added_for_skill >= min(3, max_items):
                        break
                    if len(selected) >= max_items:
                        break
            if len(selected) >= max_items:
                break

        # Ensure explicit type requests are represented, useful for refinements like
        # "add personality tests too" without losing the earlier technical context.
        for code in desired_types:
            if any(code in item.assessment.test_type_codes for item in selected):
                continue
            for scored in candidates:
                if code in scored.assessment.test_type_codes:
                    while len(selected) >= max_items:
                        selected.pop()
                        recompute_seen_family()
                    append_unique(scored)
                    break

        for scored in candidates:
            if len(selected) >= max_items:
                break
            append_unique(scored)
        return selected

    def find_assessment(self, term: str, limit: int = 5) -> list[ScoredAssessment]:
        clean = normalize_space(term)
        if not clean:
            return []
        normalized = normalize_name(clean)
        exact: list[ScoredAssessment] = []
        for item in self.catalog:
            item_norm = normalize_name(item.name)
            acronym = "".join(token[0] for token in re.findall(r"[A-Za-z0-9]+", item.name)).lower()
            if normalized == item_norm or normalized in item_norm or normalized == acronym:
                exact.append(
                    ScoredAssessment(
                        assessment=item,
                        score=2.0,
                        lexical_score=1.0,
                        evidence=self._evidence_for(item, clean),
                    )
                )
        if exact:
            preferred = PREFERRED_COMPARISON_MATCHES.get(normalized)
            if preferred:
                exact.sort(
                    key=lambda scored: (
                        normalize_name(scored.assessment.name) != preferred,
                        -scored.score,
                    )
                )
                return exact[:limit]
            exact.sort(key=lambda scored: (normalize_name(scored.assessment.name) != normalized, -len(scored.assessment.name)))
            return exact[:limit]
        return self.retrieve(clean, {}, k=limit)

    def compare_assessments(self, names_or_terms: list[str], conversation_context: str) -> dict[str, Any]:
        matches = []
        missing = []
        ambiguous = []
        for term in names_or_terms:
            found = self.find_assessment(term, limit=3)
            if found and found[0].score > 0.08:
                normalized = normalize_name(term)
                if normalized in AMBIGUOUS_COMPARISON_TERMS and len(found) > 1:
                    ambiguous.append(
                        {
                            "term": term,
                            "options": [item.assessment.name for item in found[:5]],
                        }
                    )
                    continue
                matches.append({"term": term, "matches": found})
            else:
                missing.append(term)
        return {"matches": matches, "missing": missing, "ambiguous": ambiguous, "context": conversation_context}


def build_retriever(catalog: list[Assessment], use_semantic: bool = False) -> CatalogRetriever:
    return CatalogRetriever(catalog, use_semantic=use_semantic)


def retrieve(
    query: str, constraints: dict[str, Any], k: int = 20, catalog: list[Assessment] | None = None
) -> list[ScoredAssessment]:
    if catalog is None:
        raise ValueError("catalog is required when using module-level retrieve")
    return build_retriever(catalog).retrieve(query, constraints, k=k)


def recommend(
    query: str,
    constraints: dict[str, Any],
    max_items: int = 10,
    catalog: list[Assessment] | None = None,
) -> list[Assessment]:
    if catalog is None:
        raise ValueError("catalog is required when using module-level recommend")
    scored = build_retriever(catalog).recommend(query, constraints, max_items=max_items)
    return [item.assessment for item in scored]


def compare_assessments(
    names_or_terms: list[str],
    conversation_context: str,
    catalog: list[Assessment] | None = None,
) -> dict[str, Any]:
    if catalog is None:
        raise ValueError("catalog is required when using module-level compare_assessments")
    return build_retriever(catalog).compare_assessments(names_or_terms, conversation_context)
