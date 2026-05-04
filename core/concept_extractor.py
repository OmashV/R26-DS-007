"""
Extracts mathematical concepts and mastery signals from session transcripts.

FR4 — identify mathematical concepts present in a transcript.
FR5 — classify each concept by mastery signal (strong, partial, weak).
FR6 — distinguish knowledge gaps from active misconceptions.
FR7 — produce concept extraction output as structured JSON.
NFR1 — extraction shall complete within 30 s for a <50-turn transcript.

Risk cross-references:
  R1 — concept extraction accuracy risk (High/9): mitigated by structured
       JSON output + strict schema validation in validate_extraction_output().
  R4 — LLM API availability risk: a fallback rule-based path should be wired
       in before the demo (R7 mitigation).
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def extract_concepts(transcript: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Call the LLM API with a few-shot prompt to extract concepts and mastery
    signals from a normalised transcript dict.

    Returns a list of concept dicts:
        {concept_id, concept_name, mastery_signal, is_misconception, domain}
    Stub: LLM call not yet implemented — returns NotImplementedError (R4).
    When implemented, output must be validated through validate_extraction_output().
    """
    raise NotImplementedError


def validate_extraction_output(raw_output: str) -> list[dict[str, Any]]:
    """
    Parse and schema-validate the raw JSON string returned by the LLM (R1 mitigation).

    Required keys per item: concept_id, concept_name, mastery_signal, is_misconception.
    Raises ValueError if any item violates the schema.
    Returns the validated list of concept dicts (FR7).
    """
    raise NotImplementedError


def build_extraction_prompt(transcript: dict[str, Any]) -> str:
    """
    Build the few-shot LLM prompt for concept extraction from a transcript.

    Prompt design follows R1 mitigation: concrete examples + JSON output instruction.
    Returns the prompt string.
    """
    raise NotImplementedError
