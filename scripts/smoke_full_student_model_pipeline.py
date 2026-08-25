from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

# When a script is launched as:
#     python scripts/smoke_full_student_model_pipeline.py
# Python puts scripts/ (not the repository root) on sys.path.
# Add the repository root before importing the root-level core package.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import core.knowledge_graph as knowledge_graph_module
from bkt.predict import BKTPredictor
from core.concept_extractor import ConceptExtractor
from core.cross_session_pipeline import CrossSessionStudentModelPipeline
from core.detector_service import DetectorService
from core.knowledge_graph import KnowledgeGraph
from core.student_answer_evaluator import (
    EvaluatorContext,
    StudentAnswerEvaluator,
)



def schema_sql() -> str:
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


class NoTransferColdStart:
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


def dump_resolved(event) -> dict:
    return {
        "event_id": event.event_id,
        "skill": event.skill_id,
        "primary_signal": event.primary_signal.value,
        "bkt_update": {
            "should_update": event.bkt_update.should_update,
            "outcome": event.bkt_update.outcome,
            "evidence_weight": event.bkt_update.evidence_weight,
            "evaluator_confidence": event.bkt_update.evaluator_confidence,
            "update_confidence": event.bkt_update.update_confidence,
        },
        "behaviour": {
            "reasoning_probability": event.behaviour.reasoning_probability,
            "reasoning_present": event.behaviour.reasoning_present,
            "uncertainty_probability": event.behaviour.uncertainty_probability,
            "uncertainty_present": event.behaviour.uncertainty_present,
            "clarification_probability": event.behaviour.clarification_probability,
            "clarification_present": event.behaviour.clarification_present,
        },
        "history": {
            "repeated_misunderstanding": (
                event.history.repeated_misunderstanding
            )
        },
    }


def main() -> None:
    predictor = BKTPredictor.load()
    required_skill = "Percent Of"

    if not predictor.has_skill(required_skill):
        raise RuntimeError(
            "This smoke script uses the problem '20 percent of 50'. "
            f"The loaded BKT model does not contain {required_skill!r}. "
            "Edit required_skill/problem/reference_answer together before running."
        )

    transcript = [
        {
            "role": "tutor",
            "text": "What is 20 percent of 50? Explain your reasoning.",
        },
        {
            "role": "student",
            "text": "Because 20 percent is 0.2, I multiply 0.2 by 50 and get 10.",
        },
    ]

    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "live_full_pipeline.sqlite3"

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
                conn.executescript(schema_sql())

        knowledge_graph_module.get_connection = temp_get_connection
        knowledge_graph_module.initialise_database = temp_initialise_database

        extractor = ConceptExtractor(
            allowed_skills=list(predictor.params.keys())
        )

        evaluator = StudentAnswerEvaluator(
            EvaluatorContext(
                problem="What is 20 percent of 50?",
                reference_answer="10",
                rubric=(
                    "A fully correct response should conclude 10. "
                    "A correct method is 0.2 x 50 = 10."
                ),
                assessed_skills=("Percent Of",),
            )
        )

        detectors = DetectorService.from_project_defaults(
            PROJECT_ROOT
        )

        knowledge_graph = KnowledgeGraph(
            predictor=predictor,
            cold_start=NoTransferColdStart(),
        )

        pipeline = CrossSessionStudentModelPipeline(
            concept_extractor=extractor,
            evaluator=evaluator,
            detectors=detectors,
            knowledge_graph=knowledge_graph,
        )

        result = pipeline.process_transcript(
            transcript=transcript,
            student_id="live_full_pipeline_student",
            session_id="live_full_pipeline_session_001",
        )

        raw = result["raw_extraction"]
        evaluated = result["evaluated_extraction"]
        resolved = result["resolved_events"]
        kg_result = result["knowledge_graph_result"]

        print("=" * 78)
        print("FULL LIVE STUDENT-MODEL PIPELINE SMOKE TEST")
        print("=" * 78)

        print("\n[RAW CONCEPT EXTRACTOR OUTPUT]")
        print(json.dumps(raw, indent=2))

        print("\n[EVALUATED EXTRACTION]")
        print(json.dumps(evaluated, indent=2))

        print("\n[RESOLVED EVENTS]")
        print(json.dumps([dump_resolved(event) for event in resolved], indent=2))

        print("\n[KNOWLEDGE GRAPH RESULT]")
        print(json.dumps(kg_result, indent=2, default=str))

        if not raw.get("events"):
            raise AssertionError(
                "ConceptExtractor returned no events for the clear student answer."
            )

        if not evaluated.get("events"):
            raise AssertionError("No evaluated events were produced.")

        if len(resolved) != len(evaluated["events"]):
            raise AssertionError(
                "Resolved event count differs from evaluated event count."
            )

        matching = [
            event
            for event in evaluated["events"]
            if event["skill"] == required_skill
        ]

        if not matching:
            raise AssertionError(
                "ConceptExtractor did not map the student answer to 'Percent Of'."
            )

        if not any(event["correctness"] == "correct" for event in matching):
            raise AssertionError(
                "Objective evaluator did not produce a correct verdict."
            )

        attempts = knowledge_graph.get_attempts(
            "live_full_pipeline_student",
            required_skill,
        )

        if len(attempts) != 1:
            raise AssertionError(
                "Expected exactly one Percent Of attempt; "
                f"got {len(attempts)}: {attempts!r}"
            )

        if attempts[0][0] != 1:
            raise AssertionError(
                f"Expected positive BKT outcome, got {attempts!r}"
            )

        # Stronger invariant: across ALL skills, this one student turn must
        # produce exactly one stored BKT attempt in total.
        conn = sqlite3.connect(db_path)
        try:
            total_attempts = conn.execute(
                """
                SELECT COUNT(*)
                FROM attempts
                WHERE student_id = ?
                """,
                ("live_full_pipeline_student",),
            ).fetchone()[0]
        finally:
            conn.close()

        if total_attempts != 1:
            raise AssertionError(
                "Expected exactly one BKT attempt across all extracted skills; "
                f"got {total_attempts}"
            )

        non_assessed = [
            event
            for event in evaluated["events"]
            if event["skill"] != required_skill
        ]

        if any(
            event["correctness"] != "unknown"
            or event["evaluator_confidence"] != 0.0
            for event in non_assessed
        ):
            raise AssertionError(
                "Non-assessed extracted skills must remain metadata-only "
                "(unknown, confidence 0.0)."
            )

        mastery = knowledge_graph.get_mastery(
            "live_full_pipeline_student",
            required_skill,
        )

        if mastery is None:
            raise AssertionError("No mastery record was persisted.")

        print("\n[FINAL ASSERTIONS]")
        print("ConceptExtractor live call       : PASS")
        print("Evaluator authority              : PASS")
        print("Real detector inference          : PASS")
        print("Signal Resolver                  : PASS")
        print("Exactly one total BKT observation: PASS")
        print("Temporary SQLite persistence     : PASS")
        print("Real BKT mastery update          : PASS")
        print(f"Temporary DB                     : {db_path}")
        print(
            "Mastery probability              : "
            f"{mastery['mastery_probability']:.6f}"
        )
        print("\nFULL LIVE PIPELINE: PASS")


if __name__ == "__main__":
    main()
