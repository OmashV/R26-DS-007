from __future__ import annotations

import math
import sqlite3
from contextlib import contextmanager
from pathlib import Path

import pytest

import core.knowledge_graph as knowledge_graph_module
from bkt.predict import BKTPredictor
from core.detector_service import DetectorService
from core.knowledge_graph import KnowledgeGraph
from core.pipeline_adapter import resolve_extracted_events


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class NoTransferColdStart:
    """
    Keep this integration test focused on the real BKT model.

    KnowledgeGraph's cold-start interface is still exercised, but the
    returned prior is exactly the BKT population prior, so no external
    similarity-matrix artifact is required.
    """

    def compute_prior(
        self,
        *,
        skill: str,
        student_masteries: dict,
        population_prior: float,
    ) -> dict:
        return {
            "prior": population_prior,
            "used_transfer": False,
            "related_skills_used": [],
            "transfer_evidence": [],
        }


def _schema_sql() -> str:
    """
    Minimal schema required by KnowledgeGraph's public methods used here.

    This mirrors the fields KnowledgeGraph reads/writes:
    students, sessions, attempts, and mastery.
    """
    return """
    CREATE TABLE IF NOT EXISTS students (
        student_id TEXT PRIMARY KEY,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS sessions (
        session_id TEXT PRIMARY KEY,
        student_id TEXT NOT NULL,
        concept_count INTEGER NOT NULL DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS attempts (
        attempt_id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id TEXT NOT NULL,
        skill_name TEXT NOT NULL,
        correct INTEGER NOT NULL,
        confidence REAL NOT NULL,
        signal_type TEXT,
        session_id TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS mastery (
        student_id TEXT NOT NULL,
        skill_name TEXT NOT NULL,
        mastery_probability REAL NOT NULL,
        mastery_label TEXT NOT NULL,
        previous_mastery_probability REAL,
        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (student_id, skill_name)
    );
    """


@pytest.fixture()
def temporary_knowledge_graph(monkeypatch, tmp_path):
    """
    Actual KnowledgeGraph + actual BKTPredictor, but every SQL operation is
    redirected to a pytest temporary SQLite file.

    The production database is never opened.
    """
    db_path = tmp_path / "cross_session_integration.sqlite3"

    @contextmanager
    def temp_get_connection():
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def temp_initialise_database():
        with temp_get_connection() as conn:
            conn.executescript(_schema_sql())

    # KnowledgeGraph imported these functions directly, so patch the names
    # in the KnowledgeGraph module itself.
    monkeypatch.setattr(
        knowledge_graph_module,
        "get_connection",
        temp_get_connection,
    )
    monkeypatch.setattr(
        knowledge_graph_module,
        "initialise_database",
        temp_initialise_database,
    )

    predictor = BKTPredictor.load()

    kg = KnowledgeGraph(
        predictor=predictor,
        cold_start=NoTransferColdStart(),
    )

    return kg, predictor, db_path


@pytest.fixture(scope="module")
def detectors():
    return DetectorService.from_project_defaults(PROJECT_ROOT)


def _supported_skill(predictor: BKTPredictor) -> str:
    if predictor.has_skill("Percent Of"):
        return "Percent Of"

    params = getattr(predictor, "params", None)

    if not params:
        pytest.fail(
            "Loaded BKT predictor exposes no skills in predictor.params"
        )

    return sorted(params.keys())[0]


def test_real_bkt_artifact_loads():
    predictor = BKTPredictor.load()

    params = getattr(predictor, "params", None)

    assert params
    assert len(params) >= 1


def test_real_detectors_to_temp_sqlite_to_cross_session_bkt(
    temporary_knowledge_graph,
    detectors,
):
    kg, predictor, db_path = temporary_knowledge_graph

    student_id = "cross_session_real_001"
    skill = _supported_skill(predictor)

    # ------------------------------------------------------------------
    # SESSION 1
    # One evaluator-confirmed correct event + one eligible unknown
    # clarification. Resolver v2 may turn the latter into one weak proxy.
    # ------------------------------------------------------------------
    session_1 = "real_e2e_session_001"

    transcript_1 = [
        {
            "role": "student",
            "text": "I think it is 10.",
        },
        {
            "role": "tutor",
            "text": "Why?",
        },
        {
            "role": "student",
            "text": "Because 20 percent of 50 is 10.",
        },
        {
            "role": "tutor",
            "text": "Now use the same idea on the next step.",
        },
        {
            "role": "student",
            "text": "Can you explain what that step means?",
        },
    ]

    extraction_1 = {
        "events": [
            {
                "turn_index": 2,
                "skill": skill,
                "correctness": "correct",
            },
            {
                "turn_index": 4,
                "skill": skill,
                "correctness": "unknown",
            },
        ]
    }

    resolved_1 = resolve_extracted_events(
        extraction=extraction_1,
        transcript=transcript_1,
        student_id=student_id,
        session_id=session_1,
        reasoning_predictor=detectors.reasoning_predictor,
        uncertainty_predictor=detectors.uncertainty_predictor,
        clarification_predictor=detectors.clarification_predictor,
        history_getter=kg.get_attempts,
    )

    assert len(resolved_1) == 2

    # Each event creates at most one observation: one evaluator observation
    # and one weak behavioural proxy.
    assert sum(
        event.bkt_update.should_update
        for event in resolved_1
    ) == 2

    assert resolved_1[0].bkt_update.outcome == 1
    assert resolved_1[1].bkt_update.should_update is True
    assert resolved_1[1].bkt_update.outcome == 0
    assert 0.0 < resolved_1[1].bkt_update.update_confidence <= 0.25

    result_1 = kg.process_resolved_events(
        student_id=student_id,
        resolved_events=resolved_1,
        session_id=session_1,
    )

    attempts_after_1 = kg.get_attempts(student_id, skill)

    assert len(attempts_after_1) == 2
    assert [correct for correct, _ in attempts_after_1] == [1, 0]
    assert attempts_after_1[0][1] == pytest.approx(
        resolved_1[0].bkt_update.update_confidence
    )
    assert attempts_after_1[1][1] == pytest.approx(
        resolved_1[1].bkt_update.update_confidence
    )
    assert len(result_1["behaviour_events"]) == 2

    mastery_1 = kg.get_mastery(student_id, skill)

    assert mastery_1 is not None
    assert 0.0 <= mastery_1["mastery_probability"] <= 1.0

    expected_1 = predictor.predict(
        skill,
        attempts_after_1,
        initial_prior=predictor.params[skill]["prior"],
    )

    assert mastery_1["mastery_probability"] == pytest.approx(expected_1)

    # ------------------------------------------------------------------
    # SESSION 2
    # One genuine incorrect event. History must come from SESSION 1.
    # ------------------------------------------------------------------
    session_2 = "real_e2e_session_002"

    transcript_2 = [
        {
            "role": "student",
            "text": "I think the answer is 25.",
        }
    ]

    extraction_2 = {
        "events": [
            {
                "turn_index": 0,
                "skill": skill,
                "correctness": "incorrect",
            }
        ]
    }

    resolved_2 = resolve_extracted_events(
        extraction=extraction_2,
        transcript=transcript_2,
        student_id=student_id,
        session_id=session_2,
        reasoning_predictor=detectors.reasoning_predictor,
        uncertainty_predictor=detectors.uncertainty_predictor,
        clarification_predictor=detectors.clarification_predictor,
        history_getter=kg.get_attempts,
    )

    assert len(resolved_2) == 1
    assert resolved_2[0].bkt_update.should_update is True
    assert resolved_2[0].bkt_update.outcome == 0

    # The immediately preceding stored observation is the weak negative proxy,
    # so the current genuine incorrect answer is consecutive negative evidence.
    assert resolved_2[0].history.repeated_misunderstanding is True

    kg.process_resolved_events(
        student_id=student_id,
        resolved_events=resolved_2,
        session_id=session_2,
    )

    attempts_after_2 = kg.get_attempts(student_id, skill)

    assert len(attempts_after_2) == 3
    assert [correct for correct, _ in attempts_after_2] == [1, 0, 0]

    mastery_2 = kg.get_mastery(student_id, skill)

    assert mastery_2 is not None
    assert 0.0 <= mastery_2["mastery_probability"] <= 1.0

    # KnowledgeGraph must persist the prior stored mastery value.
    assert mastery_2["previous_mastery_probability"] == pytest.approx(
        mastery_1["mastery_probability"]
    )

    expected_2 = predictor.predict(
        skill,
        attempts_after_2,
        initial_prior=None,
    )

    assert mastery_2["mastery_probability"] == pytest.approx(expected_2)

    # ------------------------------------------------------------------
    # SESSION 3
    # A second consecutive incorrect event should be marked as repeated
    # misunderstanding, but still produce only ONE new negative attempt.
    # ------------------------------------------------------------------
    session_3 = "real_e2e_session_003"

    transcript_3 = [
        {
            "role": "student",
            "text": "I still think it is 25.",
        }
    ]

    extraction_3 = {
        "events": [
            {
                "turn_index": 0,
                "skill": skill,
                "correctness": "incorrect",
            }
        ]
    }

    resolved_3 = resolve_extracted_events(
        extraction=extraction_3,
        transcript=transcript_3,
        student_id=student_id,
        session_id=session_3,
        reasoning_predictor=detectors.reasoning_predictor,
        uncertainty_predictor=detectors.uncertainty_predictor,
        clarification_predictor=detectors.clarification_predictor,
        history_getter=kg.get_attempts,
    )

    assert len(resolved_3) == 1
    assert resolved_3[0].history.repeated_misunderstanding is True
    assert resolved_3[0].bkt_update.should_update is True
    assert resolved_3[0].bkt_update.outcome == 0

    kg.process_resolved_events(
        student_id=student_id,
        resolved_events=resolved_3,
        session_id=session_3,
    )

    attempts_after_3 = kg.get_attempts(student_id, skill)

    # One correct + one weak proxy + two genuine incorrect events = exactly
    # four attempts. repeated_misunderstanding must NOT create a fifth.
    assert len(attempts_after_3) == 4
    assert [correct for correct, _ in attempts_after_3] == [1, 0, 0, 0]

    mastery_3 = kg.get_mastery(student_id, skill)

    assert mastery_3 is not None
    assert math.isfinite(mastery_3["mastery_probability"])
    assert 0.0 <= mastery_3["mastery_probability"] <= 1.0

    expected_3 = predictor.predict(
        skill,
        attempts_after_3,
        initial_prior=None,
    )

    assert mastery_3["mastery_probability"] == pytest.approx(expected_3)

    # Confirm all three sessions were written to the temporary DB.
    with sqlite3.connect(db_path) as conn:
        session_count = conn.execute(
            """
            SELECT COUNT(*)
            FROM sessions
            WHERE student_id = ?
            """,
            (student_id,),
        ).fetchone()[0]

        attempt_count = conn.execute(
            """
            SELECT COUNT(*)
            FROM attempts
            WHERE student_id = ?
              AND skill_name = ?
            """,
            (student_id, skill),
        ).fetchone()[0]

    assert session_count == 3
    assert attempt_count == 4
