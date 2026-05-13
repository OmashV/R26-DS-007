"""
Concept extractor: turns a raw tutoring session transcript into structured
attempts that the knowledge graph can consume.

Implements:
    FR4 — Identifies concepts present in a transcript
    FR5 — Classifies each concept by mastery signal
    FR6 — Distinguishes gaps from active misconceptions
    FR7 — Produces structured JSON output

Uses Google Gemini for extraction. The list of valid skills is constrained
to what the trained BKT model knows, ensuring downstream compatibility.
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional

from google import genai
from google.genai import types
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-2.5-flash"


SYSTEM_PROMPT = """You are an educational data extraction assistant. Your job is to read a tutoring conversation between a student and a tutor, and identify learning evidence signals about specific concepts. The system uses these signals to estimate student mastery.

Rules:
1. Only use skill names from the provided list of allowed skills.
2. For each piece of learning evidence, output one signal entry.
3. Multiple signals on the same skill in chronological order are normal — the system models sequences.
4. Don't invent signals. Only emit a signal when there is clear textual evidence.
5. Optionally, identify any misconceptions — specific wrong beliefs the student revealed.

Signal types and their meanings:

POSITIVE EVIDENCE (label=1):
- correct_answer       — Student gave a correct, definitive answer to a problem (confidence 1.0)
- correct_explanation  — Student explained the concept correctly in their own words (confidence 0.7)
- partial_correct      — Student got there with significant scaffolding/hints from the tutor (confidence 0.4)
- evaluator_positive   — Explicit positive feedback from an evaluator agent (confidence 1.0)

NEGATIVE EVIDENCE (label=0):
- incorrect_answer            — Student gave a wrong answer to a problem (confidence 1.0)
- repeated_misunderstanding   — Student showed the same misconception multiple times (confidence 0.8)
- confusion                   — Student expressed clear confusion ("I don't get it") (confidence 0.6)
- clarification_request       — Student asked for an explanation (confidence 0.3)
- evaluator_negative          — Explicit negative feedback from an evaluator agent (confidence 1.0)

Output strict JSON with this exact structure:
{
  "signals": [
    {
      "skill": "<skill name from allowed list>",
      "signal_type": "<one of the types above>",
      "label": 0 or 1,
      "confidence": <float matching the type's confidence>
    }
  ],
  "misconceptions": [
    "Specific wrong belief expressed by the student"
  ]
}

Use the exact label and confidence values from the lists above. Don't invent your own. Output ONLY the JSON, no prose, no markdown fences."""

class ConceptExtractor:
    """
    Extracts structured concept-mastery data from raw conversation transcripts
    using a constrained LLM prompt.
    """

    def __init__(self, allowed_skills: list[str], api_key: Optional[str] = None) -> None:
        """
        Args:
            allowed_skills: List of skill names the BKT model knows.
                            Extractor will constrain output to this list.
            api_key: Gemini API key. Defaults to GEMINI_API_KEY env var.
        """
        if not allowed_skills:
            raise ValueError("allowed_skills cannot be empty")

        key = api_key or GEMINI_API_KEY
        if not key:
            raise RuntimeError(
                "Gemini API key not found. Set GEMINI_API_KEY in .env file."
            )

        self.client = genai.Client(api_key=key)
        self.allowed_skills = allowed_skills
        logger.info(f"ConceptExtractor initialised with {len(allowed_skills)} allowed skills")

    def extract(self, transcript: list[dict], max_retries: int = 3) -> dict:
        """
        Extract structured attempts from a transcript.

        Args:
            transcript: List of {"role": "student"|"tutor", "text": str} turns.
            max_retries: How many times to retry on transient API errors.

        Returns:
            Dict with "attempts" (list of {skill, correct}) and "misconceptions" (list of str).
        """
        if not transcript:
            return {"attempts": [], "misconceptions": []}

        formatted_transcript = "\n".join(
            f"{turn['role'].upper()}: {turn['text']}" for turn in transcript
        )
        skills_list = "\n".join(f"- {s}" for s in self.allowed_skills)

        user_message = (
            f"ALLOWED SKILLS (use only these):\n{skills_list}\n\n"
            f"TRANSCRIPT:\n{formatted_transcript}\n\n"
            f"Extract structured attempts and misconceptions as JSON."
        )

        import time
        last_error = None
        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=user_message,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        response_mime_type="application/json",
                        temperature=0.1,
                    ),
                )
                break  # success
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # exponential backoff: 1s, 2s, 4s
                    logger.warning(
                        f"Gemini API error (attempt {attempt + 1}/{max_retries}): {e}. "
                        f"Retrying in {wait_time}s..."
                    )
                    time.sleep(wait_time)
                else:
                    logger.error(f"Gemini API failed after {max_retries} attempts")
                    raise

        try:
            result = json.loads(response.text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM JSON response: {response.text}")
            raise ValueError(f"Invalid JSON from LLM: {e}") from e

        result = self._validate_output(result)
        logger.info(
            f"Extracted {len(result['signals'])} signals, "
            f"{len(result['misconceptions'])} misconceptions"
        )
        return result

    # Allowed signal types and their canonical (label, confidence) values
    SIGNAL_TYPES = {
        # positive
        "correct_answer":       (1, 1.0),
        "correct_explanation":  (1, 0.7),
        "partial_correct":      (1, 0.4),
        "evaluator_positive":   (1, 1.0),
        # negative
        "incorrect_answer":           (0, 1.0),
        "repeated_misunderstanding":  (0, 0.8),
        "confusion":                  (0, 0.6),
        "clarification_request":      (0, 0.3),
        "evaluator_negative":         (0, 1.0),
    }

    def _validate_output(self, result: dict) -> dict:
        """Validate LLM output structure and filter invalid entries."""
        signals = result.get("signals", [])
        misconceptions = result.get("misconceptions", [])

        valid_signals = []
        for s in signals:
            if not isinstance(s, dict):
                continue
            skill = s.get("skill")
            signal_type = s.get("signal_type")
            label = s.get("label")
            confidence = s.get("confidence")

            if skill not in self.allowed_skills:
                logger.warning(f"LLM returned skill not in allowed list: '{skill}' — skipping")
                continue
            if signal_type not in self.SIGNAL_TYPES:
                logger.warning(f"Unknown signal type: '{signal_type}' — skipping")
                continue

            # Enforce canonical label/confidence for the signal type
            # (LLMs can drift; we trust the taxonomy, not the LLM's numbers)
            canonical_label, canonical_confidence = self.SIGNAL_TYPES[signal_type]
            valid_signals.append({
                "skill": skill,
                "signal_type": signal_type,
                "label": canonical_label,
                "confidence": canonical_confidence,
            })

        valid_misconceptions = [m for m in misconceptions if isinstance(m, str)]

        return {
            "signals": valid_signals,
            "misconceptions": valid_misconceptions,
        }