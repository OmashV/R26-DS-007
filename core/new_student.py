"""
New-student cold-start profiling.

When a student arrives with no BKT history, the Meta-Agent analyses their
first message linguistically to infer an initial profile. The profile is
used by live tutor agents to adapt their language, pace, and support level
until enough empirical data is collected to replace it.

Implements:
    FR19 — Analyse first message for initial profile parameters
    FR20 — Profile is replaced/augmented after first complete session
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional

from google import genai
from google.genai import types
from dotenv import load_dotenv

from db.database import get_connection, initialise_database

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-2.5-flash"


SYSTEM_PROMPT = """You are an educational profiling assistant. Your job is to analyse a student's first message to a tutoring system and infer four profile parameters from their language. The system uses these parameters to adapt its tone, pace, and support level for the student.

Look for signals such as:
- Vocabulary used (simple words vs technical terms)
- Sentence structure (short and direct vs detailed and complex)
- Confidence markers (hedging like "I think", "maybe", "I'm not sure" vs assertive statements)
- Specificity of the question (vague "I don't get it" vs precise "how do I solve for x in 3x+5=20")
- Self-reported state ("I'm confused", "I struggle with", "I want to learn faster")

Output strict JSON with this exact structure:
{
  "vocabulary_level": "basic" | "intermediate" | "advanced",
  "pace_preference": "slow" | "moderate" | "fast",
  "support_need": "high" | "moderate" | "low",
  "confidence": "tentative" | "moderate" | "confident",
  "reasoning": "Short explanation of the signals you used (one sentence per parameter)."
}

Output ONLY the JSON. No prose, no markdown fences, no explanation outside the reasoning field."""


VALID_VALUES = {
    "vocabulary_level": {"basic", "intermediate", "advanced"},
    "pace_preference": {"slow", "moderate", "fast"},
    "support_need": {"high", "moderate", "low"},
    "confidence": {"tentative", "moderate", "confident"},
}


class NewStudentProfiler:
    """
    Infers initial student profiles from first messages using a prompted LLM.
    """

    def __init__(self, api_key: Optional[str] = None) -> None:
        key = api_key or GEMINI_API_KEY
        if not key:
            raise RuntimeError("Gemini API key not found. Set GEMINI_API_KEY in .env.")

        self.client = genai.Client(api_key=key)
        logger.info("NewStudentProfiler initialised")

    def profile_from_message(self, first_message: str, max_retries: int = 3) -> dict:
        """
        Infer a starter profile from a student's first message.

        Args:
            first_message: The student's opening message.
            max_retries: How many times to retry on transient API errors.

        Returns:
            Dict with vocabulary_level, pace_preference, support_need,
            confidence, and reasoning fields.
        """
        if not first_message.strip():
            raise ValueError("first_message cannot be empty")

        import time
        last_error = None
        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=f"Student's first message:\n\n{first_message}",
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        response_mime_type="application/json",
                        temperature=0.1,
                    ),
                )
                break
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    wait = 2 ** attempt
                    logger.warning(
                        f"Gemini API error (attempt {attempt + 1}/{max_retries}): {e}. "
                        f"Retrying in {wait}s..."
                    )
                    time.sleep(wait)
                else:
                    logger.error(f"Gemini API failed after {max_retries} attempts")
                    raise

        try:
            result = json.loads(response.text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM JSON: {response.text}")
            raise ValueError(f"Invalid JSON from LLM: {e}") from e

        return self._validate(result)

    def _validate(self, result: dict) -> dict:
        """Ensure all required fields are present and have valid values."""
        validated = {}
        for field, allowed in VALID_VALUES.items():
            value = result.get(field)
            if value not in allowed:
                logger.warning(f"Invalid value for {field}: '{value}', defaulting to moderate")
                value = "moderate" if "moderate" in allowed else next(iter(allowed))
            validated[field] = value

        validated["reasoning"] = result.get("reasoning", "")
        return validated

    def initialise_student(self, student_id: str, first_message: str) -> dict:
        """
        Profile a new student from their first message and persist the result.

        Args:
            student_id: Identifier for the new student.
            first_message: Their opening message.

        Returns:
            The inferred profile dict.
        """
        initialise_database()
        profile = self.profile_from_message(first_message)

        with get_connection() as conn:
            conn.execute(
                """
                INSERT INTO students (student_id, profile_json)
                VALUES (?, ?)
                ON CONFLICT(student_id) DO UPDATE SET profile_json = excluded.profile_json
                """,
                (student_id, json.dumps(profile)),
            )

        logger.info(f"Initialised profile for {student_id}: {profile}")
        return profile

    def get_profile(self, student_id: str) -> Optional[dict]:
        """Retrieve a student's stored profile, or None if not set."""
        with get_connection() as conn:
            row = conn.execute(
                "SELECT profile_json FROM students WHERE student_id = ?",
                (student_id,),
            ).fetchone()

        if not row or not row["profile_json"]:
            return None
        return json.loads(row["profile_json"])