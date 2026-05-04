# Meta-Agent — Persistent Student Modelling & Learning Path Generation

Part of an AI Tutoring System (Final Year Research Project, SLIIT).

The Meta-Agent operates between tutoring sessions to build and maintain a
structured model of each student's knowledge. It parses session transcripts,
extracts mathematical concepts and mastery signals, runs Bayesian Knowledge
Tracing (BKT) to update a per-student knowledge graph, detects regressions,
and produces a personalised learning path.

See `docs/requirements.md` for the full specification and `docs/risks.md` for
the risk register.

---

## Project structure

```
meta_agent_pro/
├── api/          FastAPI route definitions
├── core/         Business logic (parser, extractor, graph, path, new-student)
├── bkt/          BKT training, inference, and evaluation
├── db/           SQLite access layer
├── data/
│   ├── raw/      Source datasets (not version-controlled)
│   └── processed/
├── models/       Serialised trained models (*.pkl not version-controlled)
├── tests/        pytest test suite
├── notebooks/    Exploration and evaluation notebooks
├── docs/         Requirements and risk register
├── config.py     Centralised configuration (env-var driven)
├── main.py       FastAPI application entry point
└── requirements.txt
```

---

## Setup

**Requirements:** Python 3.10+

```bash
# 1. Clone the repository
git clone <repo-url>
cd meta_agent_pro

# 2. Create and activate a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
#    Copy the template and fill in your values
cp .env.example .env   # create this file if it doesn't exist yet
```

Minimum `.env` content:

```
LLM_API_KEY=your_llm_api_key_here
MASTERY_THRESHOLD=0.8
DB_PATH=data/meta_agent.db
BKT_MODEL_PATH=models/bkt_model.pkl
```

---

## Running the API

```bash
uvicorn main:app --reload
```

The API will be available at `http://127.0.0.1:8000`.

Interactive docs (Swagger UI): `http://127.0.0.1:8000/docs`

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/session/process` | Ingest a transcript and run the full update pipeline |
| GET | `/student/{id}/profile` | Current knowledge graph for a student |
| GET | `/student/{id}/path` | Current personalised learning path |
| POST | `/student/new` | Initialise a new student profile (cold-start) |

All endpoints currently return mock data. The `_mock: true` flag in each
response indicates that the real pipeline has not yet been wired in.

---

## Running tests

```bash
pytest
```

To see verbose output:

```bash
pytest -v
```

The test suite includes:
- **API smoke tests** (`tests/test_api.py`) — all four endpoints return HTTP 200
  with the expected response shape.
- **Core module smoke tests** — each core module stub is verified to be
  importable and to raise `NotImplementedError` (confirming it is pending
  implementation, not silently broken).

---

## Development status

| Module | Status |
|--------|--------|
| `api/routes.py` | Mock responses — wiring pending |
| `core/transcript_parser.py` | Stub |
| `core/concept_extractor.py` | Stub |
| `core/knowledge_graph.py` | Stub |
| `core/learning_path.py` | Stub |
| `core/new_student.py` | Stub |
| `bkt/train.py` | Stub |
| `bkt/predict.py` | Stub |
| `bkt/evaluate.py` | Stub |
| `db/database.py` | Stub |
