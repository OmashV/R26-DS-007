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
from bkt.predict import BKTPredictor, ColdStartPriorCalculator
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

    def __init__(
        self,
        predictor: Optional[BKTPredictor] = None,
        cold_start: Optional[ColdStartPriorCalculator] = None,
    ) -> None:
        """
        Args:
            predictor: BKT predictor (loaded from default location if None).
            cold_start: Cold-start prior calculator. If None, attempts to load
                        from default path. Falls back gracefully if unavailable.
        """
        initialise_database()
        self.predictor = predictor or BKTPredictor.load()

        # Cold-start is optional — system works without it
        try:
            self.cold_start = cold_start or ColdStartPriorCalculator()
        except FileNotFoundError:
            logger.warning(
                "Skill similarity matrix not found — cold-start transfer disabled. "
                "System will fall back to population priors for new (student, skill) pairs."
            )
            self.cold_start = None

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

    def update_mastery(self, student_id: str, skill: str) -> dict:
        """
        Recalculate mastery for (student, skill) using full attempt history,
        persist the new value, and store the previous value for regression detection.

        On a student's first encounter with a skill, uses cross-skill knowledge
        transfer (cold-start prior) to set an informed initial mastery probability,
        rather than the default population prior.

        Returns:
            Dict with 'probability', 'label', 'cold_start_used', and
            'cold_start_details' fields.
        """
        if not self.predictor.has_skill(skill):
            logger.warning(f"Skill '{skill}' not in BKT model — skipping update")
            return {
                "probability": 0.0,
                "label": "weak",
                "cold_start_used": False,
                "cold_start_details": None,
            }

        attempts = self.get_attempts(student_id, skill)
        if not attempts:
            logger.warning(f"No attempts recorded for {student_id}/{skill}")
            return {
                "probability": 0.0,
                "label": "weak",
                "cold_start_used": False,
                "cold_start_details": None,
            }

        previous_record = self.get_mastery(student_id, skill)
        is_first_encounter = previous_record is None
        cold_start_used = False
        cold_start_prior = None
        cold_start_details = None

        if is_first_encounter and self.cold_start is not None:
            student_graph = self.get_student_graph(student_id)
            student_masteries = {
                e["skill"]: e["mastery_probability"]
                for e in student_graph
                if e["skill"] != skill
            }
            population_prior = self.predictor.params[skill]["prior"]
            cold_start_result = self.cold_start.compute_prior(
                skill=skill,
                student_masteries=student_masteries,
                population_prior=population_prior,
            )
            cold_start_prior = cold_start_result["prior"]
            cold_start_used = cold_start_result["used_transfer"]
            cold_start_details = {
                "population_prior": population_prior,
                "transferred_prior": cold_start_prior,
                "related_skills_used": cold_start_result["related_skills_used"],
                "evidence": cold_start_result["transfer_evidence"],
            }

            if cold_start_used:
                logger.info(
                    f"Cold-start transfer for {student_id}/{skill}: "
                    f"prior {population_prior:.3f} → {cold_start_prior:.3f} "
                    f"(transferred from {cold_start_result['related_skills_used']})"
                )

        probability = self.predictor.predict(
            skill, attempts, initial_prior=cold_start_prior
        )
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
            + (" [cold-start transfer applied]" if cold_start_used else "")
        )

        return {
            "probability": probability,
            "label": label,
            "cold_start_used": cold_start_used,
            "cold_start_details": cold_start_details,
        }
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
            Summary dict with session_id, skills_updated (with cold-start details),
            and the updated student graph.
        """
        session_id = session_id or f"sess_{uuid.uuid4().hex[:12]}"
        self.ensure_student(student_id)

        with get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO sessions (session_id, student_id, concept_count)
                VALUES (?, ?, ?)
                """,
                (session_id, student_id, len({a["skill"] for a in attempts})),
            )

        for a in attempts:
            self.record_attempt(student_id, a["skill"], a["correct"], session_id)

        affected_skills = sorted({a["skill"] for a in attempts})
        skill_updates = []
        for skill in affected_skills:
            update_result = self.update_mastery(student_id, skill)
            skill_updates.append({
                "skill": skill,
                **update_result,
            })

        return {
            "session_id": session_id,
            "skills_updated": skill_updates,
            "graph": self.get_student_graph(student_id),
        }