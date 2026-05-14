"""Scrape SHL Individual Test Solutions into data/catalog.json.

The app never scrapes during request handling. This script is an offline catalog
preparation step with retries, polite pacing, pagination discovery, detail-page
enrichment, and catalog-only normalization.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlencode, urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup, Tag

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.catalog import normalize_space, normalize_test_type, stable_id  # noqa: E402

LOGGER = logging.getLogger("scrape_catalog")
BASE_URL = "https://www.shl.com"
CATALOG_URL = "https://www.shl.com/solutions/products/product-catalog/"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "Chrome/124.0 Safari/537.36 SHLAssessmentRecommender/1.0"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


def absolute_url(href: str) -> str:
    return urljoin(BASE_URL, href)


def canonical_listing_url(start: int) -> str:
    query = urlencode({"start": start, "type": 1})
    return f"{CATALOG_URL}?{query}"


def fetch(session: requests.Session, url: str, timeout: float, retries: int, delay: float) -> str:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            response = session.get(url, headers=HEADERS, timeout=timeout)
            response.raise_for_status()
            if "text/html" not in response.headers.get("Content-Type", ""):
                LOGGER.warning("Unexpected content type for %s: %s", url, response.headers.get("Content-Type"))
            time.sleep(delay + random.uniform(0, delay / 3 if delay else 0))
            return response.text
        except Exception as exc:
            last_error = exc
            sleep_for = min(8.0, (2**attempt) * 0.75)
            LOGGER.warning("Fetch failed (%s/%s) for %s: %s", attempt + 1, retries + 1, url, exc)
            time.sleep(sleep_for)
    raise RuntimeError(f"Failed to fetch {url}: {last_error}")


def circle_state(cell: Tag | None) -> bool | None:
    if cell is None:
        return None
    classes = " ".join(cell.get("class") or [])
    if "-yes" in classes:
        return True
    if "-no" in classes:
        return False
    for span in cell.find_all(True):
        span_classes = " ".join(span.get("class") or [])
        if "-yes" in span_classes:
            return True
        if "-no" in span_classes:
            return False
    text = cell.get_text(" ", strip=True).lower()
    if text in {"yes", "y", "true"}:
        return True
    if text in {"no", "n", "false"}:
        return False
    return None


def parse_test_type(cell: Tag | None) -> str:
    if cell is None:
        return "Other"
    keys = [span.get_text(" ", strip=True) for span in cell.select(".product-catalogue__key")]
    if not keys:
        keys = re.findall(r"\b[A-Z]\b", cell.get_text(" ", strip=True))
    return normalize_test_type(keys)


def parse_listing_page(html: str, source_page: str) -> tuple[list[dict[str, Any]], list[int]]:
    soup = BeautifulSoup(html, "html.parser")
    table = None
    for candidate in soup.find_all("table"):
        header_text = normalize_space(candidate.get_text(" ", strip=True))
        if "Individual Test Solutions" in header_text:
            table = candidate
            break
    if table is None:
        LOGGER.warning("No Individual Test Solutions table found on %s", source_page)
        return [], []

    rows: list[dict[str, Any]] = []
    for tr in table.find_all("tr"):
        cells = tr.find_all(["th", "td"])
        if len(cells) < 4:
            continue
        first_text = normalize_space(cells[0].get_text(" ", strip=True))
        if not first_text or first_text == "Individual Test Solutions":
            continue
        link = cells[0].find("a", href=True)
        if link is None:
            continue
        name = normalize_space(link.get_text(" ", strip=True))
        url = absolute_url(link["href"])
        rows.append(
            {
                "name": name,
                "url": url,
                "test_type": parse_test_type(cells[3]),
                "remote_testing": circle_state(cells[1]),
                "adaptive_irt": circle_state(cells[2]),
                "source_page": source_page,
                "listing_text": normalize_space(tr.get_text(" ", strip=True)),
            }
        )

    starts: set[int] = set()
    for link in soup.select("a[href*='product-catalog'][href*='type=1']"):
        href = absolute_url(link.get("href", ""))
        parsed = urlparse(href)
        qs = parse_qs(parsed.query)
        try:
            if qs.get("type", [""])[0] == "1":
                starts.add(int(qs.get("start", ["0"])[0] or "0"))
        except ValueError:
            continue
    return rows, sorted(starts)


def section_rows(soup: BeautifulSoup) -> dict[str, str]:
    sections: dict[str, str] = {}
    for row in soup.select(".product-catalogue-training-calendar__row"):
        text = normalize_space(row.get_text(" ", strip=True))
        if not text:
            continue
        heading_tag = row.find(["h2", "h3", "h4", "strong"])
        if heading_tag:
            label = normalize_space(heading_tag.get_text(" ", strip=True)).rstrip(":")
            value = normalize_space(text.replace(label, "", 1))
            if label:
                sections[label.lower()] = value
        else:
            for label in ["Description", "Job levels", "Languages", "Assessment length", "Downloads"]:
                if text.lower().startswith(label.lower()):
                    sections[label.lower()] = normalize_space(text[len(label) :])
    return sections


def parse_detail_page(html: str, url: str) -> dict[str, Any]:
    soup = BeautifulSoup(html, "html.parser")
    main = soup.select_one(".product-catalogue") or soup.body or soup
    title = soup.find("h1")
    name = normalize_space(title.get_text(" ", strip=True)) if title else ""
    sections = section_rows(soup)
    description = sections.get("description", "")
    raw_text = normalize_space(main.get_text(" ", strip=True))

    test_type = ""
    remote_testing: bool | None = None
    adaptive_irt: bool | None = None
    for p in soup.select(".product-catalogue__small-text"):
        text = normalize_space(p.get_text(" ", strip=True))
        lowered = text.lower()
        if "test type" in lowered:
            test_type = normalize_test_type(text)
        elif "remote testing" in lowered:
            remote_testing = circle_state(p)
        elif "adaptive" in lowered or "irt" in lowered:
            adaptive_irt = circle_state(p)

    duration = sections.get("assessment length", "")
    minutes = None
    match = re.search(r"(?:minutes\s*=\s*|completion time.*?)(\d{1,3})", duration, re.IGNORECASE)
    if match:
        minutes = int(match.group(1))

    return {
        "detail_name": name,
        "description": description,
        "job_levels": sections.get("job levels", ""),
        "languages": sections.get("languages", ""),
        "duration": duration,
        "completion_time_minutes": minutes,
        "remote_testing": remote_testing,
        "adaptive_irt": adaptive_irt,
        "test_type": test_type,
        "raw_text": raw_text,
        "detail_url": url,
    }


def clean_record(raw: dict[str, Any], scraped_at: str) -> dict[str, Any]:
    detail = raw.get("detail") or {}
    name = normalize_space(detail.get("detail_name") or raw.get("name") or "")
    url = raw["url"]
    original_test_type = detail.get("test_type") or raw.get("test_type") or "Other"
    metadata = {
        "original_test_type": original_test_type,
        "test_type_labels": {
            code: label
            for code, label in {
                "A": "Ability / cognitive",
                "B": "Biodata / behavioral",
                "C": "Competency",
                "D": "Development",
                "E": "Assessment exercises",
                "K": "Knowledge / skills",
                "P": "Personality",
                "S": "Situational judgment",
            }.items()
            if code in normalize_test_type(original_test_type).split()
        },
        "job_levels": normalize_space(detail.get("job_levels") or ""),
        "job_family": "",
        "category": "Individual Test Solutions",
        "duration": normalize_space(detail.get("duration") or ""),
        "completion_time_minutes": detail.get("completion_time_minutes"),
        "languages": normalize_space(detail.get("languages") or ""),
        "remote_testing": detail.get("remote_testing")
        if detail.get("remote_testing") is not None
        else raw.get("remote_testing"),
        "adaptive_irt": detail.get("adaptive_irt")
        if detail.get("adaptive_irt") is not None
        else raw.get("adaptive_irt"),
        "keywords": keywords_for(name, detail.get("description") or "", detail.get("raw_text") or ""),
        "source_page": raw.get("source_page"),
        "scraped_at": scraped_at,
    }
    description = normalize_space(detail.get("description") or "")
    raw_text = normalize_space(detail.get("raw_text") or raw.get("listing_text") or "")
    return {
        "id": stable_id(name, url),
        "name": name,
        "url": url,
        "test_type": normalize_test_type(original_test_type),
        "description": description,
        "raw_text": raw_text,
        "metadata": metadata,
    }


def keywords_for(name: str, description: str, raw_text: str) -> list[str]:
    text = f"{name} {description} {raw_text}".lower()
    vocab = [
        ".net",
        "java",
        "python",
        "sql",
        "excel",
        "javascript",
        "c++",
        "aws",
        "azure",
        "react",
        "sales",
        "customer service",
        "contact center",
        "graduate",
        "manager",
        "leadership",
        "personality",
        "cognitive",
        "situational",
        "judgment",
        "verify",
        "opq",
        "g+",
        "mq",
    ]
    return [term for term in vocab if term in text]


def deduplicate(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen_urls: set[str] = set()
    seen_names: set[str] = set()
    cleaned: list[dict[str, Any]] = []
    for record in records:
        parsed = urlparse(record["url"])
        path = parsed.path.rstrip("/")
        canonical = urlunparse(("https", parsed.netloc.lower(), path, "", "", ""))
        name_key = re.sub(r"[^a-z0-9+#.]+", " ", record["name"].lower()).strip()
        if canonical in seen_urls or name_key in seen_names:
            continue
        seen_urls.add(canonical)
        seen_names.add(name_key)
        cleaned.append(record)
    return cleaned


def fetch_detail_record(raw: dict[str, Any], timeout: float, retries: int, delay: float) -> dict[str, Any]:
    session = requests.Session()
    enriched = dict(raw)
    try:
        html = fetch(session, raw["url"], timeout=timeout, retries=retries, delay=delay)
        enriched["detail"] = parse_detail_page(html, raw["url"])
    except Exception as exc:
        LOGGER.warning("Detail fetch failed for %s: %s", raw["url"], exc)
        enriched["detail_error"] = str(exc)
    return enriched


def scrape(
    max_pages: int | None = None,
    max_details: int | None = None,
    delay: float = 0.25,
    workers: int = 4,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    session = requests.Session()
    timeout = 30.0
    retries = 3
    to_visit = [0]
    visited: set[int] = set()
    listing_rows: list[dict[str, Any]] = []

    while to_visit:
        start = to_visit.pop(0)
        if start in visited:
            continue
        if max_pages is not None and len(visited) >= max_pages:
            break
        visited.add(start)
        url = canonical_listing_url(start)
        LOGGER.info("Fetching listing page start=%s", start)
        html = fetch(session, url, timeout=timeout, retries=retries, delay=delay)
        rows, starts = parse_listing_page(html, url)
        listing_rows.extend(rows)
        for discovered in starts:
            if discovered not in visited and discovered not in to_visit:
                to_visit.append(discovered)
        to_visit.sort()

    by_url: dict[str, dict[str, Any]] = {}
    for row in listing_rows:
        by_url.setdefault(row["url"], row)
    raw_records = list(by_url.values())
    LOGGER.info("Discovered %d individual test listing rows", len(raw_records))

    detail_limit = len(raw_records) if max_details is None else min(max_details, len(raw_records))
    detail_inputs = raw_records[:detail_limit]
    untouched = raw_records[detail_limit:]
    workers = max(1, min(workers, 8))
    LOGGER.info("Fetching %d detail pages with %d workers", len(detail_inputs), workers)
    enriched_details: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {
            executor.submit(fetch_detail_record, raw, timeout, retries, delay): raw
            for raw in detail_inputs
        }
        for idx, future in enumerate(as_completed(future_map), start=1):
            enriched = future.result()
            enriched_details.append(enriched)
            if idx % 25 == 0 or idx == len(detail_inputs):
                LOGGER.info("Fetched detail pages %d/%d", idx, len(detail_inputs))

    raw_records = enriched_details + untouched

    scraped_at = datetime.now(timezone.utc).isoformat()
    cleaned = deduplicate([clean_record(raw, scraped_at) for raw in raw_records])
    return raw_records, cleaned


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-output", default="data/catalog_raw.json")
    parser.add_argument("--output", default="data/catalog.json")
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument("--max-details", type=int, default=None)
    parser.add_argument("--delay", type=float, default=0.25)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raw, cleaned = scrape(
        max_pages=args.max_pages,
        max_details=args.max_details,
        delay=args.delay,
        workers=args.workers,
    )
    raw_path = ROOT / args.raw_output
    output_path = ROOT / args.output
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(json.dumps(raw, indent=2, ensure_ascii=False), encoding="utf-8")
    output_path.write_text(json.dumps(cleaned, indent=2, ensure_ascii=False), encoding="utf-8")
    LOGGER.info("Wrote %d raw and %d cleaned records", len(raw), len(cleaned))
    return 0 if cleaned else 1


if __name__ == "__main__":
    raise SystemExit(main())
