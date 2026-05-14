"""Build and sanity-check the local retrieval index."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.catalog import load_catalog  # noqa: E402
from app.retrieval import build_retriever  # noqa: E402


def main() -> int:
    catalog = load_catalog(ROOT / "data/catalog.json", allow_emergency=False)
    if not catalog:
        print("No real catalog records found. Run python scripts/scrape_catalog.py first.")
        return 1
    retriever = build_retriever(catalog, use_semantic=False)
    output = ROOT / "data/retriever_index.joblib"
    joblib.dump(
        {
            "word_vectorizer": retriever.word_vectorizer,
            "char_vectorizer": retriever.char_vectorizer,
            "word_matrix": retriever.word_matrix,
            "char_matrix": retriever.char_matrix,
            "catalog_size": len(catalog),
        },
        output,
    )
    manifest = ROOT / "data/retriever_index_manifest.json"
    manifest.write_text(json.dumps({"catalog_size": len(catalog), "index_path": str(output)}, indent=2), encoding="utf-8")
    print(f"Built retrieval index for {len(catalog)} catalog records at {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
