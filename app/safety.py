"""Scope and prompt-injection detection for the recommender."""

from __future__ import annotations

import re


PROMPT_INJECTION_PATTERNS = [
    r"\bignore (all )?(previous|prior|above|system|developer) instructions\b",
    r"\bignore (all )?(above|previous|prior)\b",
    r"\bdisregard (previous|prior|above|system|developer)\s*(directions|instructions)?\b",
    r"\breveal (the )?(system|developer) (prompt|message|instructions)\b",
    r"\bshow (the )?(system|developer) (prompt|message|instructions)\b",
    r"\btell me (your|the) (hidden )?(instructions|prompt)\b",
    r"\bhidden instructions\b",
    r"\bjailbreak\b",
    r"\bdeveloper message\b",
    r"\bsystem prompt\b",
    r"\bact as (?:dan|an unrestricted|a different|a fake)\b",
    r"\byou are now\b",
    r"\breturn assessments? not in (the )?catalog\b",
    r"\bmake up (fake )?(urls?|assessments?|tests?)\b",
    r"\bfake shl (tests?|assessments?|urls?)\b",
]

OFF_TOPIC_PATTERNS = [
    r"\blegal advice\b",
    r"\bemployment law\b",
    r"\bhow (can|should) i reject\b",
    r"\bshould i reject\b",
    r"\breject (this|the) candidate\b",
    r"\breject(?:ing)? a candidate\b",
    r"\bfire (an employee|someone)\b",
    r"\bgeneral hiring lawyer\b",
    r"\bhiring lawyer\b",
    r"\bsalary\b",
    r"\bcompensation\b",
    r"\bprotected class\b",
    r"\brace\b",
    r"\breligion\b",
    r"\bpregnan(?:t|cy)\b",
    r"\bdisabilit(?:y|ies)\b",
    r"\bmedical advice\b",
    r"\bdiagnos(?:e|is)\b",
    r"\bpolitics?\b",
    r"\bwho should i hire\b",
    r"\bshould i hire\b",
]

ASSESSMENT_SCOPE_TERMS = {
    "assessment",
    "assessments",
    "test",
    "tests",
    "shl",
    "catalog",
    "opq",
    "gsa",
    "verify",
    "personality",
    "cognitive",
    "ability",
    "skills",
    "skill",
    "situational",
    "judgment",
    "java",
    "python",
    "sql",
    "excel",
    "developer",
    "sales",
    "customer service",
    "contact center",
    "graduate",
    "manager",
    "leadership",
}


def _matches_any(text: str, patterns: list[str]) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def detect_prompt_injection(text: str) -> bool:
    return _matches_any(text or "", PROMPT_INJECTION_PATTERNS)


def detect_off_topic(text: str) -> bool:
    """Refuse clearly out-of-scope requests without blocking normal assessment queries."""

    lowered = (text or "").lower()
    if not _matches_any(lowered, OFF_TOPIC_PATTERNS):
        return False
    has_scope_term = any(term in lowered for term in ASSESSMENT_SCOPE_TERMS)
    # Legal/hiring-decision requests remain out of scope even if they mention candidates.
    strong_off_topic = any(
        phrase in lowered
        for phrase in [
            "legal advice",
            "employment law",
            "rejecting a candidate",
            "reject a candidate",
            "reject this candidate",
            "reject the candidate",
            "should i reject",
            "who should i hire",
            "should i hire",
            "hiring lawyer",
        ]
    )
    return strong_off_topic or not has_scope_term


def refusal_reply_for(text: str) -> str | None:
    if detect_prompt_injection(text):
        return (
            "I can't follow instructions that override the catalog-only assessment "
            "selection rules. I can help select or compare SHL assessments."
        )
    if detect_off_topic(text):
        return "I can only help with selecting or comparing SHL assessments from the catalog."
    return None
