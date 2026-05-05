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


SYSTEM_PROMPT = """You are an educational data extraction assistant. Your job is to read a tutoring conversation between a student and a tutor, and output structured data about which mathematical concepts the student attempted and whether they answered correctly.

Rules:
1. Only use skill names from the provided list of allowed skills. If a concept appears that isn't in the list, choose the closest match or omit it.
2. For each distinct attempt by the student at solving a problem, output one entry: {"skill": "<skill name>", "correct": 0 or 1}.
3. "correct": 1 means the student got the answer right. "correct": 0 means they got it wrong, gave up, or only got it after the tutor revealed the answer.
4. Multiple attempts on the same problem each get their own entry, in chronological order.
5. Only count actual problem-solving attempts. Don't count clarifying questions, expressions of confusion, or off-topic chat as attempts.
6. Optionally, identify any misconceptions — specific wrong beliefs the student revealed (e.g. "thinks multiplying always increases size"). These are distinct from just getting an answer wrong.

Output strict JSON with this exact structure:
{
  "attempts": [
    {"skill": "Percent Of", "correct": 0},
    {"skill": "Percent Of", "correct": 1}
  ],
  "misconceptions": [
    "Specific wrong belief expressed by the student"
  ]
}

If no attempts can be identified, return {"attempts": [], "misconceptions": []}.
Output ONLY the JSON. No prose, no markdown fences, no explanation."""


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
            f"Extracted {len(result['attempts'])} attempts, "
            f"{len(result['misconceptions'])} misconceptions"
        )
        return result

    def _validate_output(self, result: dict) -> dict:
        """Validate LLM output structure and filter invalid entries."""
        attempts = result.get("attempts", [])
        misconceptions = result.get("misconceptions", [])

        valid_attempts = []
        for a in attempts:
            if not isinstance(a, dict):
                continue
            skill = a.get("skill")
            correct = a.get("correct")
            if skill not in self.allowed_skills:
                logger.warning(f"LLM returned skill not in allowed list: '{skill}' — skipping")
                continue
            if correct not in (0, 1):
                logger.warning(f"Invalid 'correct' value: {correct} — skipping")
                continue
            valid_attempts.append({"skill": skill, "correct": correct})

        valid_misconceptions = [m for m in misconceptions if isinstance(m, str)]

        return {
            "attempts": valid_attempts,
            "misconceptions": valid_misconceptions,
        }