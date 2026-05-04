"""
Handles cold-start profiling for students with no session history.

FR19 — analyse a new student's first message to infer initial profile
        parameters: vocabulary_level, support_need, pace_preference.
FR20 — replace the inferred cold-start profile with empirical data after
        the student completes their first full session.
UR3  — new students receive an appropriately calibrated starting profile.

Risk cross-references:
  R5 — cold-start profiling unreliable risk (Low/2): mitigated by treating
       the profile as a soft prior (is_cold_start flag) that is replaced
       after session 1 (FR20).  Tutor agents may override if the profile
       conflicts with observed behaviour.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def create_cold_start_profile(first_message: str) -> dict[str, Any]:
    """
    Analyse first_message to infer vocabulary level, support need, and
    pace preference (FR19).

    Returns a profile dict:
        {
            student_id: None,          # assigned by the caller after DB insert
            vocabulary_level: str,     # "basic" | "intermediate" | "advanced"
            support_need: str,         # "low" | "medium" | "high"
            pace_preference: str,      # "slow" | "moderate" | "fast"
            is_cold_start: True
        }
    Stub: LLM-based analysis not yet implemented — returns NotImplementedError.
    The fallback design returns conservative defaults (R5 mitigation).
    """
    raise NotImplementedError


def upgrade_profile(student_id: str, session_data: dict[str, Any]) -> dict[str, Any]:
    """
    Replace the cold-start prior with empirical parameters derived from
    the student's first complete session (FR20).

    session_data is the output of the full session processing pipeline.
    Returns the updated profile dict with is_cold_start set to False.
    """
    raise NotImplementedError


def assign_student_id() -> str:
    """
    Generate a new opaque student identifier (NFR7 — no PII in the graph).
    Returns a UUID4 string.
    """
    raise NotImplementedError
