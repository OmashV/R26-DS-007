"""
FastAPI route definitions for the Meta-Agent API.

Implements the four endpoints specified in requirements.md §7:
  POST /session/process   — FR1, FR2, FR3, and the full update pipeline
  GET  /student/{id}/profile — FR11, FR12, FR13
  GET  /student/{id}/path    — FR16, FR18
  POST /student/new          — FR19, FR20

All handlers currently return mock data so teammates can develop against a
stable interface (R3 mitigation). Replace mock returns with real pipeline
calls once core modules are implemented.
"""

import logging
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------

class SessionProcessRequest(BaseModel):
    student_id: str
    transcript: dict[str, Any]


class NewStudentRequest(BaseModel):
    first_message: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/session/process")
async def process_session(body: SessionProcessRequest) -> dict[str, Any]:
    """
    Ingest a session transcript and run the full update pipeline.

    Full pipeline (pending): parse → extract concepts → BKT predict →
    update knowledge graph → detect regressions → regenerate learning path.
    FR1, FR2, FR3 (ingestion); FR4–FR7 (extraction); FR8–FR10 (BKT);
    FR11, FR12 (graph); FR14–FR17 (regression + path).
    """
    logger.info("POST /session/process | student_id=%s [MOCK]", body.student_id)
    return {
        "status": "processed",
        "student_id": body.student_id,
        "concepts_extracted": [
            {
                "concept_id": "c_fractions",
                "concept_name": "Fractions",
                "mastery_signal": "partial",
                "is_misconception": False,
            }
        ],
        "graph_updated": True,
        "regressions_detected": [],
        "_mock": True,
    }


@router.get("/student/{student_id}/profile")
async def get_student_profile(student_id: str) -> dict[str, Any]:
    """
    Return the current knowledge graph for a student.

    FR11, FR12, FR13 — per-student concept map with timestamped history.
    NFR2 — target response time < 500 ms.
    NFR7 — student_id is an opaque identifier; no PII stored.
    """
    logger.info("GET /student/%s/profile [MOCK]", student_id)
    return {
        "student_id": student_id,
        "concepts": [
            {
                "concept_id": "c_fractions",
                "concept_name": "Fractions",
                "domain": "mathematics",
                "mastery_prob": 0.72,
                "mastered": False,
                "history": [
                    {"timestamp": "2026-04-27T10:00:00Z", "mastery_prob": 0.55},
                    {"timestamp": "2026-04-28T09:00:00Z", "mastery_prob": 0.72},
                ],
            },
            {
                "concept_id": "c_addition",
                "concept_name": "Basic Addition",
                "domain": "mathematics",
                "mastery_prob": 0.95,
                "mastered": True,
                "history": [
                    {"timestamp": "2026-04-26T10:00:00Z", "mastery_prob": 0.95}
                ],
            },
        ],
        "last_updated": "2026-04-28T09:00:00Z",
        "_mock": True,
    }


@router.get("/student/{student_id}/path")
async def get_student_path(student_id: str) -> dict[str, Any]:
    """
    Return the current personalised learning path for a student.

    FR16, FR18 — three categories: revise urgently, learn next, already strong.
    UR1 — student receives a path indicating status of each concept.
    """
    logger.info("GET /student/%s/path [MOCK]", student_id)
    return {
        "student_id": student_id,
        "revise_urgently": ["Equivalent Fractions", "Fraction Comparison"],
        "learn_next": ["Mixed Numbers", "Fraction Addition", "Fraction Division"],
        "already_strong": ["Basic Addition", "Multiplication Tables", "Whole Number Division"],
        "_mock": True,
    }


@router.post("/student/new")
async def create_new_student(body: NewStudentRequest) -> dict[str, Any]:
    """
    Initialise a new student profile from their first message (cold-start).

    FR19 — infer vocabulary level, support need, pace preference from first message.
    FR20 — this cold-start profile will be replaced after the first full session.
    R5  — profile is a soft prior, not a hard label.
    UR3 — new student receives appropriately calibrated starting profile.
    """
    logger.info("POST /student/new | first_message_len=%d [MOCK]", len(body.first_message))
    return {
        "student_id": "new_student_placeholder",
        "profile": {
            "vocabulary_level": "intermediate",
            "support_need": "medium",
            "pace_preference": "moderate",
            "is_cold_start": True,
        },
        "message": "Cold-start profile created; will be replaced after first full session (FR20).",
        "_mock": True,
    }
