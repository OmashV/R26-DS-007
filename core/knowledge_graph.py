"""
Knowledge graph: per-student concept mastery, backed by SQLite + BKT.

Implements:
    FR8  — Maintains mastery probability per (student, skill)
    FR10 — Classifies mastery into strong/partial/weak using thresholds
    FR11 — Persistently stores per-student concept map
    FR12 — Records timestamped attempt history
    FR13 — Exposes current knowledge graph via the API
    FR16 — Provides data for learning path generation

Architecture: this module orchestrates BKT (bkt/predict.py) and the
database (db/database.py). It is the canonical entry point for any
external code that needs to read or update student mastery state.
"""

import logging
import uuid
from typing import Optional

from bkt.predict import BKTPredictor
from db.database import get_connection, initialise_database
from config import (
    MASTERY_STRONG_THRESHOLD,
    MASTERY_WEAK_THRESHOLD,
)

logger = logging.getLogger(__name__)


def label_mastery(probability: float) -> str:
    """Map a mastery probability to a discrete label."""
    if probability >= MASTERY_STRONG_THRESHOLD:
        return "strong"
    if probability < MASTERY_WEAK_THRESHOLD:
        return "weak"
    return "partial"


class KnowledgeGraph:
    """
    Per-student concept mastery, persisted to SQLite and updated via BKT.

    Typical usage:
        kg = KnowledgeGraph()
        kg.process_session(student_id="u_42", attempts=[
            {"skill": "Percent Of", "correct": 0},
            {"skill": "Percent Of", "correct": 1},
        ])
        graph = kg.get_student_graph("u_42")
    """

    def __init__(self, predictor: Optional[BKTPredictor] = None) -> None:
        """
        Args:
            predictor: BKT predictor (loaded from default location if None).
        """
        initialise_database()
        self.predictor = predictor or BKTPredictor.load()
        logger.info("KnowledgeGraph initialised")

    # ----- Student management -----

    def ensure_student(self, student_id: str) -> None:
        """Create a student record if one doesn't already exist."""
        with get_connection() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO students (student_id) VALUES (?)",
                (student_id,),
            )

    # ----- Attempts -----

    def record_attempt(
        self,
        student_id: str,
        skill: str,
        correct: int,
        session_id: Optional[str] = None,
    ) -> None:
        """Log a single attempt to the database."""
        if correct not in (0, 1):
            raise ValueError(f"correct must be 0 or 1, got {correct}")

        self.ensure_student(student_id)

        with get_connection() as conn:
            conn.execute(
                """
                INSERT INTO attempts (student_id, skill_name, correct, session_id)
                VALUES (?, ?, ?, ?)
                """,
                (student_id, skill, correct, session_id),
            )

    def get_attempts(self, student_id: str, skill: str) -> list[int]:
        """Return chronological list of correct/incorrect for a (student, skill) pair."""
        with get_connection() as conn:
            rows = conn.execute(
                """
                SELECT correct FROM attempts
                WHERE student_id = ? AND skill_name = ?
                ORDER BY created_at ASC, attempt_id ASC
                """,
                (student_id, skill),
            ).fetchall()
        return [r["correct"] for r in rows]

    # ----- Mastery -----

    def update_mastery(self, student_id: str, skill: str) -> float:
        """
        Recalculate mastery for (student, skill) using full attempt history,
        persist the new value, and store the previous value for regression detection.
        Returns the new mastery probability.
        """
        if not self.predictor.has_skill(skill):
            logger.warning(f"Skill '{skill}' not in BKT model — skipping update")
            return 0.0

        attempts = self.get_attempts(student_id, skill)
        if not attempts:
            logger.warning(f"No attempts recorded for {student_id}/{skill}")
            return 0.0

        probability = self.predictor.predict(skill, attempts)
        label = label_mastery(probability)

        with get_connection() as conn:
            existing = conn.execute(
                "SELECT mastery_probability FROM mastery WHERE student_id = ? AND skill_name = ?",
                (student_id, skill),
            ).fetchone()
            previous = existing["mastery_probability"] if existing else None

            conn.execute(
                """
                INSERT INTO mastery 
                    (student_id, skill_name, mastery_probability, mastery_label, previous_mastery_probability)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(student_id, skill_name) DO UPDATE SET
                    previous_mastery_probability = mastery.mastery_probability,
                    mastery_probability = excluded.mastery_probability,
                    mastery_label = excluded.mastery_label,
                    last_updated = CURRENT_TIMESTAMP
                """,
                (student_id, skill, probability, label, previous),
            )

        logger.info(
            f"Updated mastery: student={student_id} skill='{skill}' "
            f"P={probability:.3f} ({label})"
        )
        return probability

    def get_mastery(self, student_id: str, skill: str) -> Optional[dict]:
        """Return current stored mastery for (student, skill), or None if not present."""
        with get_connection() as conn:
            row = conn.execute(
                """
                SELECT mastery_probability, mastery_label,
                       previous_mastery_probability, last_updated
                FROM mastery
                WHERE student_id = ? AND skill_name = ?
                """,
                (student_id, skill),
            ).fetchone()

        if not row:
            return None

        return {
            "skill": skill,
            "mastery_probability": row["mastery_probability"],
            "mastery_label": row["mastery_label"],
            "previous_mastery_probability": row["previous_mastery_probability"],
            "last_updated": row["last_updated"],
        }

    def get_student_graph(self, student_id: str) -> list[dict]:
        """Return all mastery records for a student, sorted by skill name."""
        with get_connection() as conn:
            rows = conn.execute(
                """
                SELECT skill_name, mastery_probability, mastery_label,
                       previous_mastery_probability, last_updated
                FROM mastery
                WHERE student_id = ?
                ORDER BY skill_name
                """,
                (student_id,),
            ).fetchall()

        return [
            {
                "skill": r["skill_name"],
                "mastery_probability": r["mastery_probability"],
                "mastery_label": r["mastery_label"],
                "previous_mastery_probability": r["previous_mastery_probability"],
                "last_updated": r["last_updated"],
            }
            for r in rows
        ]

    # ----- Session orchestration -----

    def process_session(
        self,
        student_id: str,
        attempts: list[dict],
        session_id: Optional[str] = None,
    ) -> dict:
        """
        Process a full session: record all attempts, then update mastery
        for each affected skill.

        Args:
            student_id: Student identifier.
            attempts: List of {"skill": str, "correct": 0|1} dicts.
            session_id: Optional session identifier (auto-generated if None).

        Returns:
            Summary dict with session_id, skills_updated, and updated graph.
        """
        session_id = session_id or f"sess_{uuid.uuid4().hex[:12]}"
        self.ensure_student(student_id)

        # Log session start
        with get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO sessions (session_id, student_id, concept_count)
                VALUES (?, ?, ?)
                """,
                (session_id, student_id, len({a["skill"] for a in attempts})),
            )

        # Record every attempt
        for a in attempts:
            self.record_attempt(student_id, a["skill"], a["correct"], session_id)

        # Update mastery for each unique skill touched in this session
        affected_skills = sorted({a["skill"] for a in attempts})
        for skill in affected_skills:
            self.update_mastery(student_id, skill)

        return {
            "session_id": session_id,
            "skills_updated": affected_skills,
            "graph": self.get_student_graph(student_id),
        }