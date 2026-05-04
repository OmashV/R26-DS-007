"""
Smoke tests for core.transcript_parser.
NFR6 — each core module shall have unit test coverage for its primary code path.
"""

import pytest
from core.transcript_parser import parse_transcript, extract_turns, validate_student_id


def test_parse_transcript_is_stubbed():
    """Confirms the stub is in place and correctly raises NotImplementedError."""
    with pytest.raises(NotImplementedError):
        parse_transcript({"student_id": "s1", "turns": []})


def test_extract_turns_is_stubbed():
    with pytest.raises(NotImplementedError):
        extract_turns({"student_id": "s1", "turns": []})


def test_validate_student_id_is_stubbed():
    with pytest.raises(NotImplementedError):
        validate_student_id("s1")
