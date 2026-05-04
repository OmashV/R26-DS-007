"""
Smoke tests for core.learning_path.
NFR6 — each core module shall have unit test coverage for its primary code path.
"""

import pytest
from core.learning_path import generate_path, prioritise_regressions


def test_generate_path_is_stubbed():
    with pytest.raises(NotImplementedError):
        generate_path({"student_id": "s1", "concepts": []})


def test_prioritise_regressions_is_stubbed():
    mock_path = {
        "student_id": "s1",
        "revise_urgently": [],
        "learn_next": [],
        "already_strong": [],
    }
    with pytest.raises(NotImplementedError):
        prioritise_regressions(mock_path, ["c_fractions"])
