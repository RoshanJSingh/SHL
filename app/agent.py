"""Stateless conversation understanding and response assembly."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from app.catalog import Assessment, TEST_TYPE_LABELS, canonical_url, catalog_url_set, normalize_name
from app.config import Settings
from app.llm import LLMClient
from app.retrieval import (
    CatalogRetriever,
    ScoredAssessment,
    extract_roles,
    extract_skills,
    infer_intents,
    infer_desired_test_types,
    infer_excluded_test_types,
    is_broad_intent_query,
)
from app.safety import detect_off_topic, detect_prompt_injection, refusal_reply_for
from app.schemas import ChatResponse, Message, RecommendationItem


@dataclass
class ConversationState:
    user_goal: str = ""
    role_title: str | None = None
    seniority: str | None = None
    skills: list[str] = field(default_factory=list)
    job_description: str | None = None
    desired_test_types: list[str] = field(default_factory=list)
    excluded_test_types: list[str] = field(default_factory=list)
    constraints: dict[str, Any] = field(default_factory=dict)
    mentioned_assessments: list[str] = field(default_factory=list)
    last_recommendations: list[str] = field(default_factory=list)
    is_vague: bool = False
    is_off_topic: bool = False
    is_prompt_injection: bool = False
    wants_comparison: bool = False
    wants_refinement: bool = False
    confidence: float = 0.0


def _latest_user_message(messages: list[Message]) -> str:
    for message in reversed(messages):
        if message.role == "user":
            return message.content
    return ""


def _user_messages(messages: list[Message]) -> list[str]:
    return [message.content for message in messages if message.role == "user"]


def _assistant_messages(messages: list[Message]) -> list[str]:
    return [message.content for message in messages if message.role == "assistant"]


def _unique_extend(values: list[str], additions: list[str]) -> list[str]:
    for value in additions:
        if value and value not in values:
            values.append(value)
    return values


def _remove_excluded(values: list[str], text: str) -> list[str]:
    lowered = text.lower()
    kept = []
    for value in values:
        if re.search(rf"\b(no|without|exclude|avoid|not)\s+(?:\w+\s+){{0,2}}{re.escape(value)}\b", lowered):
            continue
        kept.append(value)
    return kept


def _extract_seniority(text: str) -> str | None:
    lowered = text.lower()
    if re.search(r"\b(entry[ -]?level|graduate|early careers?|junior)\b", lowered):
        return "graduate"
    if re.search(r"\b(mid[ -]?level|intermediate)\b", lowered):
        return "mid"
    if re.search(r"\b(senior|lead|principal)\b", lowered):
        return "senior"
    if re.search(r"\b(manager|leadership|supervisor|executive)\b", lowered):
        return "manager"
    return None


def _extract_role_title(text: str) -> str | None:
    patterns = [
        r"\bhiring (?:a|an|for a|for an)?\s+([^.,;\n]{3,90})",
        r"\bfor (?:a|an|the)?\s+([^.,;\n]{3,80})\s+(?:role|position|job)\b",
        r"\bneed (?:tests?|assessments?) for (?:a|an)?\s+([^.,;\n]{3,90})",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            role = re.sub(r"\b(who|with|that|and)\b.*$", "", match.group(1), flags=re.IGNORECASE)
            role = " ".join(role.split())
            if role:
                return role[:90]
    role_terms = extract_roles(text)
    if role_terms:
        return ", ".join(role_terms)
    return None


def _extract_count(text: str) -> int | None:
    match = re.search(r"\b(?:top|best)?\s*(10|[1-9])\b", text.lower())
    if match:
        return max(1, min(int(match.group(1)), 10))
    return None


def _looks_like_job_description(text: str) -> bool:
    lowered = text.lower()
    return len(text.split()) >= 25 or any(
        marker in lowered for marker in ["job description", "responsibilities", "requirements", "must have", "jd:"]
    )


def _extract_comparison_terms(text: str) -> list[str]:
    clean = text.strip().strip("?")
    lowered = clean.lower()
    for pattern in [
        r"difference between\s+(.+?)\s+and\s+(.+)$",
        r"compare\s+(.+?)\s+(?:and|vs\.?|versus)\s+(.+)$",
        r"(.+?)\s+(?:vs\.?|versus)\s+(.+)$",
    ]:
        match = re.search(pattern, lowered, flags=re.IGNORECASE)
        if match:
            left = clean[match.start(1) : match.end(1)].strip(" .,:;?")
            right = clean[match.start(2) : match.end(2)].strip(" .,:;?")
            right = re.sub(r"^(the|a|an)\s+", "", right, flags=re.IGNORECASE)
            return [left, right]
    # Fallback: known acronyms and capitalized product-ish terms.
    terms = re.findall(r"\b(OPQ|GSA|G\+|MQ|Verify|Java|Python|Excel|SQL)\b", text, flags=re.IGNORECASE)
    return list(dict.fromkeys(terms))[:3]


def _assessment_mentions(text: str, catalog: list[Assessment], limit: int = 12) -> list[str]:
    mentions: list[str] = []
    lowered = text.lower()
    for item in catalog:
        name_norm = normalize_name(item.name)
        if len(name_norm) < 4:
            continue
        if item.name.lower() in lowered or name_norm in normalize_name(lowered):
            mentions.append(item.name)
        if len(mentions) >= limit:
            break
    for acronym in ["OPQ", "GSA", "G+", "MQ", "Verify"]:
        if re.search(rf"\b{re.escape(acronym)}\b", text, re.IGNORECASE) and acronym not in mentions:
            mentions.append(acronym)
    return mentions


def parse_conversation(messages: list[Message], catalog: list[Assessment]) -> ConversationState:
    state = ConversationState()
    latest = _latest_user_message(messages)
    user_texts = _user_messages(messages)
    full_user_context = "\n".join(user_texts)
    state.user_goal = latest or full_user_context
    state.is_prompt_injection = detect_prompt_injection(latest)
    state.is_off_topic = detect_off_topic(latest)
    state.wants_comparison = bool(re.search(r"\b(compare|difference|vs\.?|versus)\b", latest, re.IGNORECASE))
    state.wants_refinement = bool(
        re.search(r"\b(actually|instead|add|remove|also|too|no |without|exclude|change|refine)\b", latest, re.IGNORECASE)
    )

    for text in user_texts:
        skills = extract_skills(text)
        state.skills = _remove_excluded(_unique_extend(state.skills, skills), text)

        desired = infer_desired_test_types(text)
        excluded = infer_excluded_test_types(text)
        state.desired_test_types = _unique_extend(state.desired_test_types, desired)
        state.excluded_test_types = _unique_extend(state.excluded_test_types, excluded)
        state.desired_test_types = [code for code in state.desired_test_types if code not in state.excluded_test_types]

        seniority = _extract_seniority(text)
        if seniority:
            state.seniority = seniority
        role = _extract_role_title(text)
        if role:
            state.role_title = role
        if _looks_like_job_description(text):
            state.job_description = text
        state.mentioned_assessments = _unique_extend(
            state.mentioned_assessments, _assessment_mentions(text, catalog)
        )

    assistant_text = "\n".join(_assistant_messages(messages))
    state.last_recommendations = _assessment_mentions(assistant_text, catalog)
    signal_count = sum(
        [
            bool(state.role_title),
            bool(state.skills),
            bool(state.desired_test_types),
            bool(state.mentioned_assessments),
            bool(state.job_description),
        ]
    )
    state.confidence = min(1.0, signal_count / 3)
    vague_uncertainty = bool(
        re.search(r"\b(not sure|unsure|something|whatever|any assessment|some assessment)\b", latest, re.IGNORECASE)
    )
    state.is_vague = (signal_count == 0 and not state.wants_comparison) or (
        signal_count <= 1 and vague_uncertainty and not state.wants_comparison
    )
    intents = infer_intents(full_user_context)
    state.constraints = {
        "role_title": state.role_title,
        "roles": extract_roles(full_user_context),
        "skills": state.skills,
        "intents": intents,
        "seniority": state.seniority,
        "desired_test_types": state.desired_test_types,
        "excluded_test_types": state.excluded_test_types,
        "mentioned_assessments": state.mentioned_assessments,
        "short_tests_only": bool(re.search(r"\b(short|quick|under\s+30|30 minutes|brief)\b", full_user_context, re.I)),
        "max_items": _extract_count(latest),
    }
    return state


def _conversation_summary(state: ConversationState, messages: list[Message]) -> str:
    facts = []
    if state.role_title:
        facts.append(f"role={state.role_title}")
    if state.seniority:
        facts.append(f"seniority={state.seniority}")
    if state.skills:
        facts.append(f"skills={', '.join(state.skills)}")
    if state.desired_test_types:
        facts.append(f"desired_types={' '.join(state.desired_test_types)}")
    if state.excluded_test_types:
        facts.append(f"excluded_types={' '.join(state.excluded_test_types)}")
    latest = _latest_user_message(messages)
    return f"Latest user request: {latest}\nExtracted facts: {'; '.join(facts) or 'none'}"


def _build_query(state: ConversationState, messages: list[Message]) -> str:
    pieces = _user_messages(messages)
    if state.role_title:
        pieces.append(f"role {state.role_title}")
    if state.seniority:
        pieces.append(f"seniority {state.seniority}")
    if state.skills:
        pieces.append("skills " + " ".join(state.skills))
    if state.desired_test_types:
        labels = [TEST_TYPE_LABELS.get(code, code) for code in state.desired_test_types]
        pieces.append("desired assessment types " + " ".join(labels))
    return " ".join(pieces)


def _clarification_response() -> ChatResponse:
    return ChatResponse(
        reply=(
            "What role or job family are you hiring for, and are you looking for "
            "technical, cognitive, personality, or mixed assessments?"
        ),
        recommendations=[],
        end_of_conversation=False,
    )


def _safe_response(reply: str) -> ChatResponse:
    return ChatResponse(reply=reply, recommendations=[], end_of_conversation=False)


def _recommendation_items(scored: list[ScoredAssessment], catalog: list[Assessment]) -> list[RecommendationItem]:
    valid_urls = catalog_url_set(catalog)
    items: list[RecommendationItem] = []
    seen: set[str] = set()
    for scored_item in scored:
        item = scored_item.assessment
        url_key = canonical_url(item.url)
        if item.is_placeholder or url_key not in valid_urls or url_key in seen:
            continue
        items.append(RecommendationItem(name=item.name, url=item.url, test_type=item.test_type))
        seen.add(url_key)
        if len(items) >= 10:
            break
    return items


def _fallback_recommendation_reply(state: ConversationState, scored: list[ScoredAssessment]) -> str:
    constraints = []
    if state.role_title:
        constraints.append(state.role_title)
    if state.seniority:
        constraints.append(f"{state.seniority} level")
    if state.skills:
        constraints.append("skills: " + ", ".join(state.skills))
    if state.desired_test_types:
        constraints.append("types: " + ", ".join(state.desired_test_types))
    if state.excluded_test_types:
        constraints.append("excluding: " + ", ".join(state.excluded_test_types))
    interpreted = "; ".join(constraints) if constraints else "the available role and assessment signals"
    lines = [f"Based on the catalog fields available, I ranked this shortlist for {interpreted}:"]
    for idx, scored_item in enumerate(scored[:10], start=1):
        item = scored_item.assessment
        evidence = "; ".join(scored_item.evidence[:2]) if scored_item.evidence else item.description[:160]
        lines.append(f"{idx}. {item.name} ({item.test_type}) - {evidence}")
    return "\n".join(lines)


def _comparison_value(item: Assessment, key: str) -> str:
    if key == "test_type":
        return item.test_type or "not available in the catalog data"
    if key == "description":
        return item.description or "not available in the catalog data"
    if key == "duration":
        return str(
            item.metadata.get("duration")
            or item.metadata.get("assessment_length")
            or item.metadata.get("completion_time_minutes")
            or "not available in the catalog data"
        )
    if key == "remote_testing":
        value = item.metadata.get("remote_testing")
        return "not available in the catalog data" if value in {None, ""} else str(value)
    if key == "adaptive_irt":
        value = item.metadata.get("adaptive_irt")
        return "not available in the catalog data" if value in {None, ""} else str(value)
    if key == "job_levels":
        value = item.metadata.get("job_levels") or item.metadata.get("job_family")
        return "not available in the catalog data" if not value else str(value)
    return "not available in the catalog data"


def _fallback_comparison_reply(compare_obj: dict[str, Any]) -> str:
    matches = compare_obj["matches"]
    missing = compare_obj["missing"]
    ambiguous = compare_obj.get("ambiguous", [])
    if ambiguous:
        lines = [
            "I need a more specific catalog assessment to make a grounded comparison. "
            "These terms match multiple SHL catalog records:"
        ]
        for item in ambiguous:
            options = ", ".join(item["options"][:5])
            lines.append(f"- {item['term']}: {options}")
        lines.append("Which exact assessment should I compare?")
        return "\n".join(lines)
    if len(matches) < 2:
        available = []
        for match in matches:
            options = ", ".join(item.assessment.name for item in match["matches"][:3])
            available.append(f"{match['term']}: {options}")
        missing_text = ", ".join(missing) if missing else "one of the requested assessments"
        found_text = "; ".join(available) if available else "no exact catalog match"
        return (
            f"I could not ground a full comparison because {missing_text} was not clearly available "
            f"in the catalog data. Closest grounded matches: {found_text}."
        )

    chosen = [(match["term"], match["matches"][0].assessment) for match in matches[:2]]
    lines = ["Using only the catalog fields available:"]
    for term, item in chosen:
        description = _comparison_value(item, "description")
        if len(description) > 220:
            description = description[:217].rstrip() + "..."
        lines.append(
            f"- {term} -> {item.name}: test type {item.test_type}; duration "
            f"{_comparison_value(item, 'duration')}; remote testing "
            f"{_comparison_value(item, 'remote_testing')}; adaptive/IRT "
            f"{_comparison_value(item, 'adaptive_irt')}; role fit/job levels "
            f"{_comparison_value(item, 'job_levels')}. Description: {description}"
        )
    lines.append("Any field marked unavailable was missing from the scraped catalog record.")
    return "\n".join(lines)


def _llm_evidence(scored: list[ScoredAssessment]) -> list[dict[str, Any]]:
    return [
        {
            "name": item.assessment.name,
            "url": item.assessment.url,
            "test_type": item.assessment.test_type,
            "description": item.assessment.description,
            "metadata": item.assessment.metadata,
            "evidence": item.evidence,
        }
        for item in scored[:10]
    ]


def _guard_llm_reply(reply: str | None, scored: list[ScoredAssessment]) -> str | None:
    """Reject LLM prose that introduces unsupported factual claims."""

    if not reply:
        return None
    text = reply.strip()
    lowered = text.lower()
    if not text or "http://" in lowered or "https://" in lowered:
        return None
    if any(marker in lowered for marker in ["fake assessment", "fake url", "not in the catalog"]):
        return None

    evidence_parts: list[str] = []
    for scored_item in scored[:10]:
        item = scored_item.assessment
        evidence_parts.extend([item.description, item.raw_text])
        evidence_parts.extend(str(value) for value in item.metadata.values())
    joined_evidence = " ".join(evidence_parts).lower()

    allowed_minutes = {
        match.group(1)
        for match in re.finditer(r"\b(\d{1,3})\s*(?:minutes?|mins?)\b", joined_evidence)
    }
    for match in re.finditer(r"\b(\d{1,3})\s*(?:minutes?|mins?)\b", lowered):
        if match.group(1) not in allowed_minutes:
            return None

    for sensitive in ["remote testing", "adaptive", "irt"]:
        if sensitive in lowered and sensitive not in joined_evidence:
            return None
    return text


def _validate_final_response(response: ChatResponse, catalog: list[Assessment]) -> ChatResponse:
    if not response.recommendations:
        return ChatResponse(reply=response.reply, recommendations=[], end_of_conversation=False)
    valid_urls = catalog_url_set(catalog)
    catalog_by_url = {canonical_url(item.url): item for item in catalog if not item.is_placeholder}
    repaired: list[RecommendationItem] = []
    seen: set[str] = set()
    for rec in response.recommendations[:10]:
        key = canonical_url(rec.url)
        if key not in valid_urls or key in seen:
            continue
        catalog_item = catalog_by_url[key]
        repaired.append(
            RecommendationItem(
                name=catalog_item.name,
                url=catalog_item.url,
                test_type=catalog_item.test_type,
            )
        )
        seen.add(key)
    if not repaired:
        return ChatResponse(
            reply="I need a bit more role or skill context before I can return a catalog-grounded shortlist.",
            recommendations=[],
            end_of_conversation=False,
        )
    return ChatResponse(reply=response.reply, recommendations=repaired, end_of_conversation=True)


class AssessmentAgent:
    """Stateless per-request recommender."""

    def __init__(self, catalog: list[Assessment], retriever: CatalogRetriever, settings: Settings) -> None:
        self.catalog = catalog
        self.retriever = retriever
        self.llm = LLMClient(settings)

    def chat(self, messages: list[Message]) -> ChatResponse:
        state = parse_conversation(messages, self.catalog)
        latest = _latest_user_message(messages)
        refusal = refusal_reply_for(latest)
        if refusal:
            return _safe_response(refusal)

        if state.wants_comparison:
            terms = _extract_comparison_terms(latest)
            compare_obj = self.retriever.compare_assessments(terms, _conversation_summary(state, messages))
            reply = _fallback_comparison_reply(compare_obj)
            return ChatResponse(reply=reply, recommendations=[], end_of_conversation=False)

        if state.is_vague:
            return _clarification_response()

        if not self.retriever.catalog:
            return ChatResponse(
                reply=(
                    "The SHL catalog data is unavailable, so I can't return a catalog-grounded "
                    "shortlist yet. Run the catalog scraper or provide a valid catalog.json."
                ),
                recommendations=[],
                end_of_conversation=False,
            )

        query = _build_query(state, messages)
        requested_items = state.constraints.get("max_items")
        max_items = int(requested_items) if requested_items else (10 if is_broad_intent_query(query, state.constraints) else 5)
        scored = self.retriever.recommend(query, state.constraints, max_items=max_items)
        recommendations = _recommendation_items(scored, self.catalog)
        if not recommendations:
            return ChatResponse(
                reply=(
                    "I could not find enough catalog-grounded matches for that request. "
                    "Which role, skill, or SHL assessment family should I focus on?"
                ),
                recommendations=[],
                end_of_conversation=False,
            )

        deterministic_reply = _fallback_recommendation_reply(state, scored)
        llm_reply = self.llm.complete(
            task="recommendation_explanation",
            user_context=_conversation_summary(state, messages),
            evidence=_llm_evidence(scored),
        )
        guarded_llm_reply = _guard_llm_reply(llm_reply, scored)
        response = ChatResponse(
            reply=guarded_llm_reply or deterministic_reply,
            recommendations=recommendations,
            end_of_conversation=True,
        )
        return _validate_final_response(response, self.catalog)
