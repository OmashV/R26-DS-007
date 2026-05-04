"""
Smoke tests for core.concept_extractor.
NFR6 — each core module shall have unit test coverage for its primary code path.
"""

import pytest
from core.concept_extractor import (
    extract_concepts,
    validate_extraction_output,
    build_extraction_prompt,
)


def test_extract_concepts_is_stubbed():
    with pytest.raises(NotImplementedError):
        extract_concepts({"student_id": "s1", "turns": []})


def test_validate_extraction_output_is_stubbed():
    with pytest.raises(NotImplementedError):
        validate_extraction_output("[]")


def test_build_extraction_prompt_is_stubbed():
    with pytest.raises(NotImplementedError):
        build_extraction_prompt({"student_id": "s1", "turns": []})
