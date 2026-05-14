# SHL Conversational Assessment Recommender

Production-oriented FastAPI service for the SHL AI Intern take-home: a stateless conversational recommender over the scraped SHL Individual Test Solutions catalog.

## Architecture

```text
Client
  -> POST /chat full message history
  -> FastAPI + Pydantic schema validation
  -> safety detector: off-topic and prompt injection
  -> stateless conversation parser
  -> local catalog retriever
       word TF-IDF + char n-grams
       optional embeddings
       skill/type/seniority boosts
       diversity reranking
  -> optional LLM wording over retrieved evidence only
  -> catalog-only output guardrails
  -> exact JSON response
```

The local `data/catalog.json` is the only source of recommendation truth. The app does not scrape live SHL pages during requests.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Environment Variables

- `CATALOG_PATH`: defaults to `data/catalog.json`
- `USE_LLM`: defaults to true only when a supported key exists, otherwise false
- `LLM_PROVIDER`: `openai`, `gemini`, `groq`, or `openrouter`
- `MODEL_NAME`: optional model override
- `OPENAI_API_KEY`, `GEMINI_API_KEY`, `GROQ_API_KEY`, `OPENROUTER_API_KEY`: optional
- `OPENAI_BASE_URL`: optional OpenAI-compatible endpoint
- `PORT`: defaults to `8000`
- `APP_ENV`: set to `production` in deployment

Copy `.env.example` for local reference. The deterministic fallback works without any LLM key.

## Catalog Scraping

```bash
python scripts/scrape_catalog.py
python scripts/build_index.py
```

The scraper crawls `https://www.shl.com/solutions/products/product-catalog/?type=1`, follows pagination, enriches records from detail pages, writes `data/catalog_raw.json`, and writes the cleaned catalog to `data/catalog.json`.

## Run Locally

```bash
uvicorn app.main:app --reload
```

The deployment command is:

```bash
uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

## API Examples

```bash
curl http://localhost:8000/health
```

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hiring a mid-level Java developer who works with stakeholders"}]}'
```

## Test And Evaluate

```bash
pytest -q
python scripts/run_eval.py
python scripts/smoke_test_api.py
python scripts/make_approach_pdf.py
```

Evaluation saves `data/evaluation/results.json` and reports schema compliance, catalog validity, Recall@10, relevance, groundedness, and behavior probe pass rate.

## Deployment

### Render

1. Push this repository to GitHub.
2. Create a Render Web Service from the repo, or use `render.yaml`.
3. Set `APP_ENV=production`.
4. Optional: set `USE_LLM=true`, `LLM_PROVIDER`, `MODEL_NAME`, and the provider API key.
5. Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`.

### Railway / Fly / Hugging Face Spaces

Use the Dockerfile. Set `PORT` if the platform does not inject it automatically. Do not run scraping at startup; commit `data/catalog.json`.

## Cold Start

Cold start is usually a few seconds: loading about 377 catalog records and building TF-IDF matrices in memory. Optional sentence-transformers is not enabled by default to keep startup predictable.

## Known Limitations

- The catalog only reflects the last successful scrape.
- Some SHL detail pages omit duration, adaptive/IRT, languages, or job-level metadata; the agent says unavailable rather than inventing.
- The retriever is deterministic and strong for catalog terms, but exact suitability still depends on the user providing role or skill context.
- Optional LLM text can improve phrasing, but final recommendations remain catalog-enforced.

## Interview Defense Notes

- Stateless design fits the assignment contract and evaluator: every `/chat` call reconstructs state from the full message history.
- Catalog-only validation prevents hallucinated products and fake URLs.
- Hybrid retrieval handles both exact skills and noisy job descriptions; character n-grams help with Java, C++, .NET, OPQ, GSA, and product variants.
- Prompt injection is detected before retrieval and the LLM never receives authority to create recommendations.
- Without LLM keys, the deterministic agent still clarifies, recommends, refines, compares, refuses, and passes tests.
- Recall@10 is measured by comparing recommended names/URLs against labeled eval cases in `data/evaluation/eval_cases.json`.
