"""
Smoke tests for core.new_student.
NFR6 — each core module shall have unit test coverage for its primary code path.
"""

import pytest
from core.new_student import create_cold_start_profile, upgrade_profile, assign_student_id


def test_create_cold_start_profile_is_stubbed():
    with pytest.raises(NotImplementedError):
        create_cold_start_profile("I need help with fractions.")


def test_upgrade_profile_is_stubbed():
    with pytest.raises(NotImplementedError):
        upgrade_profile("s1", {})


def test_assign_student_id_is_stubbed():
    with pytest.raises(NotImplementedError):
        assign_student_id()
