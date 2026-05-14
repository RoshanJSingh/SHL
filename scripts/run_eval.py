"""Offline evaluation for schema, catalog validity, retrieval recall, and behavior probes."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.agent import AssessmentAgent  # noqa: E402
from app.catalog import Assessment, canonical_url, catalog_url_set, load_catalog, normalize_name  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.retrieval import assessment_text, build_retriever  # noqa: E402
from app.schemas import Message  # noqa: E402

EVAL_DIR = ROOT / "data/evaluation"
RESULTS_PATH = EVAL_DIR / "results.json"
CASES_PATH = EVAL_DIR / "eval_cases.json"
PUBLIC_TRACES_PATH = EVAL_DIR / "public_traces.json"


def load_cases() -> list[dict[str, Any]]:
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    if PUBLIC_TRACES_PATH.exists():
        try:
            traces = json.loads(PUBLIC_TRACES_PATH.read_text(encoding="utf-8"))
            if isinstance(traces, list):
                for idx, trace in enumerate(traces):
                    if isinstance(trace, dict) and trace.get("messages"):
                        cases.append(
                            {
                                "id": f"public_trace_{idx}",
                                "messages": trace["messages"],
                                "behavior": trace.get("behavior", "recommend"),
                                "expected_names": trace.get("expected_names", []),
                                "expected_urls": trace.get("expected_urls", []),
                            }
                        )
        except Exception as exc:
            print(f"Warning: could not load public traces: {exc}")
    return cases


def schema_ok(payload: dict[str, Any]) -> bool:
    if set(payload.keys()) != {"reply", "recommendations", "end_of_conversation"}:
        return False
    if not isinstance(payload["reply"], str):
        return False
    if not isinstance(payload["recommendations"], list):
        return False
    if not isinstance(payload["end_of_conversation"], bool):
        return False
    count = len(payload["recommendations"])
    if count > 10:
        return False
    for item in payload["recommendations"]:
        if set(item.keys()) != {"name", "url", "test_type"}:
            return False
        if not all(isinstance(item[key], str) and item[key] for key in item):
            return False
    return True


def catalog_validity(payload: dict[str, Any], valid_urls: set[str]) -> tuple[bool, list[str]]:
    bad = []
    for item in payload["recommendations"]:
        if canonical_url(item["url"]) not in valid_urls:
            bad.append(item["url"])
    return not bad, bad


def recall_at_10(payload: dict[str, Any], case: dict[str, Any]) -> float | None:
    expected_urls = {canonical_url(url) for url in case.get("expected_urls", []) if url}
    expected_names = {normalize_name(name) for name in case.get("expected_names", []) if name}
    if not expected_urls and not expected_names:
        return None
    hits = 0
    total = len(expected_urls) + len(expected_names)
    rec_urls = {canonical_url(item["url"]) for item in payload["recommendations"][:10]}
    rec_names = [normalize_name(item["name"]) for item in payload["recommendations"][:10]]
    hits += len(expected_urls & rec_urls)
    for expected in expected_names:
        if any(expected == name or expected in name or name in expected for name in rec_names):
            hits += 1
    return hits / max(total, 1)


def tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-zA-Z0-9+#.]{3,}", text.lower())
        if token not in {"the", "and", "for", "with", "that", "this", "from", "assessment"}
    }


def relevance_score(payload: dict[str, Any], messages: list[dict[str, str]], catalog_by_url: dict[str, Assessment]) -> float:
    query_tokens = tokens(" ".join(message["content"] for message in messages if message["role"] == "user"))
    if not payload["recommendations"] or not query_tokens:
        return 1.0 if not payload["recommendations"] else 0.0
    scores = []
    for item in payload["recommendations"]:
        assessment = catalog_by_url.get(canonical_url(item["url"]))
        if not assessment:
            scores.append(0.0)
            continue
        evidence_tokens = tokens(assessment_text(assessment))
        scores.append(len(query_tokens & evidence_tokens) / max(len(query_tokens), 1))
    return mean(scores) if scores else 0.0


def groundedness(payload: dict[str, Any], catalog_by_url: dict[str, Assessment]) -> float:
    if not payload["recommendations"]:
        return 1.0
    reply = payload["reply"].lower()
    checks = 0
    supported = 0
    for item in payload["recommendations"]:
        assessment = catalog_by_url.get(canonical_url(item["url"]))
        if not assessment:
            continue
        checks += 2
        if assessment.name.lower() in reply:
            supported += 1
        if assessment.test_type.lower() in reply:
            supported += 1
    return supported / checks if checks else 1.0


def behavior_pass(payload: dict[str, Any], case: dict[str, Any], valid_urls: set[str]) -> tuple[bool, str]:
    behavior = case.get("behavior")
    reply_lower = payload["reply"].lower()
    recs = payload["recommendations"]
    if len(recs) > 10:
        return False, "more than 10 recommendations"
    if any(canonical_url(item["url"]) not in valid_urls for item in recs):
        return False, "hallucinated URL"
    if behavior == "clarify":
        return (not recs and not payload["end_of_conversation"] and "?" in payload["reply"], "expected clarification")
    if behavior == "recommend":
        return (1 <= len(recs) <= 10 and payload["end_of_conversation"], "expected recommendations")
    if behavior == "refine":
        has_personality = any("P" in item["test_type"].split() for item in recs)
        keeps_java = any("java" in item["name"].lower() for item in recs)
        return (has_personality and keeps_java, "expected Java context plus personality")
    if behavior == "exclude_personality":
        return (recs and not any("P" in item["test_type"].split() for item in recs), "expected no personality")
    if behavior == "compare":
        return (not recs and any(term in reply_lower for term in ["difference", "catalog", "test type"]), "expected comparison")
    if behavior == "refuse":
        return (not recs and "catalog" in reply_lower, "expected refusal")
    return True, ""


def evaluate() -> dict[str, Any]:
    catalog = load_catalog(ROOT / "data/catalog.json", allow_emergency=False)
    if not catalog:
        raise RuntimeError("No catalog records. Run python scripts/scrape_catalog.py first.")
    valid_urls = catalog_url_set(catalog)
    catalog_by_url = {canonical_url(item.url): item for item in catalog}
    agent = AssessmentAgent(catalog, build_retriever(catalog), get_settings())
    cases = load_cases()
    rows = []
    for case in cases:
        messages = [Message.model_validate(message) for message in case["messages"]]
        response = agent.chat(messages).model_dump()
        ok_schema = schema_ok(response)
        ok_catalog, bad_urls = catalog_validity(response, valid_urls)
        recall = recall_at_10(response, case)
        behavior_ok, behavior_reason = behavior_pass(response, case, valid_urls)
        row = {
            "id": case["id"],
            "schema_ok": ok_schema,
            "catalog_valid": ok_catalog,
            "bad_urls": bad_urls,
            "recall_at_10": recall,
            "relevance": relevance_score(response, case["messages"], catalog_by_url),
            "groundedness": groundedness(response, catalog_by_url),
            "behavior_ok": behavior_ok,
            "failure_reason": "" if behavior_ok and ok_schema and ok_catalog else behavior_reason,
            "response": response,
        }
        rows.append(row)

    summary = {
        "case_count": len(rows),
        "schema_compliance_rate": mean(1.0 if row["schema_ok"] else 0.0 for row in rows),
        "catalog_validity_rate": mean(1.0 if row["catalog_valid"] else 0.0 for row in rows),
        "recall_at_10": mean(row["recall_at_10"] for row in rows if row["recall_at_10"] is not None),
        "mean_relevance": mean(row["relevance"] for row in rows),
        "mean_groundedness": mean(row["groundedness"] for row in rows),
        "behavior_probe_pass_rate": mean(1.0 if row["behavior_ok"] else 0.0 for row in rows),
        "failures": [
            {"id": row["id"], "reason": row["failure_reason"], "response": row["response"]}
            for row in rows
            if not (row["schema_ok"] and row["catalog_valid"] and row["behavior_ok"])
        ],
    }
    RESULTS_PATH.write_text(json.dumps({"summary": summary, "cases": rows}, indent=2), encoding="utf-8")
    return {"summary": summary, "cases": rows}


def main() -> int:
    results = evaluate()
    summary = results["summary"]
    print("Metric                         Value")
    print("-----------------------------  -----")
    for key in [
        "schema_compliance_rate",
        "catalog_validity_rate",
        "recall_at_10",
        "mean_relevance",
        "mean_groundedness",
        "behavior_probe_pass_rate",
    ]:
        print(f"{key:29}  {summary[key]:.3f}")
    print(f"Results saved to {RESULTS_PATH}")
    if summary["failures"]:
        print("Failures:")
        for failure in summary["failures"]:
            print(f"- {failure['id']}: {failure['reason']}")
    return 0 if not summary["failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
