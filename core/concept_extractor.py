import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-2.5-flash"


SYSTEM_PROMPT = """You are an educational data extraction assistant.

Your job is to read a tutoring conversation and identify event-level learning evidence.

IMPORTANT:
You are NOT assigning BKT-ready signals.
You are NOT assigning uncertainty, clarification, confusion, or repeated misunderstanding.
You are ONLY extracting:
1. which skill is involved,
2. which student turn provides evidence,
3. the student's correctness state for that skill,
4. any explicit misconception.

Rules:

1. Only use skill names from the provided list of allowed skills.
2. Output one event for each student-turn × skill pair where there is clear evidence.
3. Use the exact transcript turn index for the student utterance that produced the evidence.
4. Only use these correctness values:
   - "correct"
   - "partial"
   - "incorrect"
   - "unknown"
5. Use:
   - "correct" when the student clearly demonstrates correct understanding or a correct answer,
   - "partial" when the student is partly right or reaches the answer with substantial tutor support,
   - "incorrect" when the student gives a clearly wrong answer or explanation,
   - "unknown" when the turn is relevant to a skill but correctness cannot be judged.
6. Do not invent events. Only emit an event when there is clear textual evidence.
7. Do not emit behavioural signals such as confusion or clarification_request.
8. If a student expresses a specific wrong belief, also include it in misconceptions.

Output strict JSON with this exact structure:

{
  "events": [
    {
      "turn_index": <integer>,
      "skill": "<skill name from allowed list>",
      "correctness": "correct" | "partial" | "incorrect" | "unknown",
      "evidence_span": "<short quote from the relevant student turn>"
    }
  ],
  "misconceptions": [
    "Specific wrong belief expressed by the student"
  ]
}

Output ONLY the JSON. No prose. No markdown fences.
"""


class ConceptExtractor:
    """
    Extracts structured event candidates from raw tutoring transcripts.

    This extractor is intentionally limited to:
    - skill identification
    - correctness state
    - misconception extraction

    It does NOT produce BKT-ready behavioural signals.
    """

    VALID_CORRECTNESS = {
        "correct",
        "partial",
        "incorrect",
        "unknown",
    }

    def __init__(
        self,
        allowed_skills: list[str],
        api_key: Optional[str] = None,
    ) -> None:
        if not allowed_skills:
            raise ValueError("allowed_skills cannot be empty")

        key = api_key or GEMINI_API_KEY
        if not key:
            raise RuntimeError(
                "Gemini API key not found. Set GEMINI_API_KEY in .env file."
            )

        self.client = genai.Client(api_key=key)
        self.allowed_skills = allowed_skills

        logger.info(
            f"ConceptExtractor initialised with {len(allowed_skills)} allowed skills"
        )

    def extract(
        self,
        transcript: list[dict],
        max_retries: int = 3,
    ) -> dict:
        """
        Extract structured event candidates from a transcript.

        Args:
            transcript: List of {"role": "student"|"tutor", "text": str} turns.
            max_retries: Number of retries on transient API failure.

        Returns:
            Dict:
            {
                "events": [...],
                "misconceptions": [...]
            }
        """

        if not transcript:
            return {
                "events": [],
                "misconceptions": [],
            }

        formatted_transcript_lines = []
        for i, turn in enumerate(transcript):
            role = str(turn["role"]).upper()
            text = str(turn["text"])
            formatted_transcript_lines.append(
                f"[{i}] {role}: {text}"
            )

        formatted_transcript = "\n".join(formatted_transcript_lines)
        skills_list = "\n".join(f"- {s}" for s in self.allowed_skills)

        user_message = (
            f"ALLOWED SKILLS (use only these):\n{skills_list}\n\n"
            f"TRANSCRIPT:\n{formatted_transcript}\n\n"
            f"Extract event candidates and misconceptions as JSON."
        )

        response = None
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
                break
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.warning(
                        f"Gemini API error (attempt {attempt + 1}/{max_retries}): {e}. "
                        f"Retrying in {wait_time}s..."
                    )
                    time.sleep(wait_time)
                else:
                    logger.error(
                        f"Gemini API failed after {max_retries} attempts"
                    )
                    raise

        if response is None:
            raise RuntimeError(
                f"Gemini API returned no response. Last error: {last_error}"
            )

        try:
            result = json.loads(response.text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM JSON response: {response.text}")
            raise ValueError(f"Invalid JSON from LLM: {e}") from e

        result = self._validate_output(
            result=result,
            transcript=transcript,
        )

        logger.info(
            f"Extracted {len(result['events'])} events, "
            f"{len(result['misconceptions'])} misconceptions"
        )

        return result

    def _validate_output(
        self,
        result: dict,
        transcript: list[dict],
    ) -> dict:
        """
        Validate LLM output structure and filter invalid entries.
        """

        events = result.get("events", [])
        misconceptions = result.get("misconceptions", [])

        valid_events = []

        for event in events:
            if not isinstance(event, dict):
                continue

            turn_index = event.get("turn_index")
            skill = event.get("skill")
            correctness = event.get("correctness")
            evidence_span = event.get("evidence_span", "")

            # ------------------------------
            # turn index checks
            # ------------------------------

            if not isinstance(turn_index, int):
                logger.warning(
                    f"Event missing valid turn_index: {event} — skipping"
                )
                continue

            if turn_index < 0 or turn_index >= len(transcript):
                logger.warning(
                    f"turn_index out of range: {turn_index} — skipping"
                )
                continue

            turn = transcript[turn_index]

            if turn.get("role") != "student":
                logger.warning(
                    f"turn_index {turn_index} is not a student turn — skipping"
                )
                continue

            # ------------------------------
            # skill checks
            # ------------------------------

            if skill not in self.allowed_skills:
                logger.warning(
                    f"LLM returned skill not in allowed list: '{skill}' — skipping"
                )
                continue

            # ------------------------------
            # correctness checks
            # ------------------------------

            if correctness not in self.VALID_CORRECTNESS:
                logger.warning(
                    f"Invalid correctness '{correctness}' — skipping"
                )
                continue

            # ------------------------------
            # normalize evidence span
            # ------------------------------

            if not isinstance(evidence_span, str):
                evidence_span = ""

            evidence_span = evidence_span.strip()

            if not evidence_span:
                # fallback to full student turn
                evidence_span = str(turn.get("text", "")).strip()

            valid_events.append(
                {
                    "turn_index": turn_index,
                    "skill": skill,
                    "correctness": correctness,
                    "evidence_span": evidence_span,
                    "student_text": str(turn.get("text", "")).strip(),
                }
            )

        valid_misconceptions = [
            m for m in misconceptions if isinstance(m, str)
        ]

        return {
            "events": valid_events,
            "misconceptions": valid_misconceptions,
        }