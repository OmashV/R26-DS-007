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



What Meta-Agent Does — Plain Explanation
The Big Picture
Imagine a student using an AI tutoring system to learn math. They ask questions, get explanations, make mistakes, and come back the next day for more. The tutoring system handles the live conversation — but nobody is watching what the student actually understands over time.
That's exactly what your Meta-Agent does. It sits in the background, watches every conversation, and builds up a picture of what each student knows, what they struggle with, and what they should focus on next.
The Three Things It Does
1. It listens and interprets
After every tutoring session ends, the Meta-Agent reads through the full conversation. It doesn't just store it — it actually interprets it. It figures out which concepts came up, whether the student understood them, got confused, or had a specific misconception. So instead of just saving a chat history, it extracts meaning from it.
2. It builds a knowledge map for each student
Based on what it interprets, it builds and updates a concept graph — think of it like a map of everything the student has encountered, where each concept is marked as strong, partial, or weak. Every time the student comes back for a new session, this map gets updated. If a student who previously understood fractions suddenly starts making mistakes with them, the map catches that too — that's the regression detection.
3. It tells the student what to do next
Using the knowledge map, it generates a personalised learning path — what to revise urgently, what's ready to be learned next, and what the student already has a good grip on. This learning path updates after every session as the student improves.
The Extra Role — New Students
For brand new students with no history, the Meta-Agent does one more thing. It looks at how the student writes their very first message — their vocabulary, how they phrase questions, how confident they sound — and uses that to set an initial profile. Things like how simple or complex the tutor's language should be, how fast explanations should go, and how much support to offer. This profile gets replaced by real data once the student has completed a session.
Why It's Different From Just Saving Chat History
Most systems either store raw conversations and do nothing with them, or track simple right/wrong scores from quizzes. Your Meta-Agent does something in between that neither approach does — it takes messy, unstructured conversation and turns it into a structured, evolving model of what that specific student knows. And it does this continuously, getting smarter about each student the more they interact with the system.
One-Line Version
It watches students learn, figures out what they know and don't know, and keeps updating a personalised map of their knowledge so the system always knows what to teach them next.