from unittest.mock import MagicMock

import pytest

import core.knowledge_graph as kg_module
from core.knowledge_graph import KnowledgeGraph

from core.signal_resolver import (
    Correctness,
    ResolverInput,
    resolve_signal,
)


# ============================================================
# FAKE DATABASE CONNECTION
# ============================================================


class FakeConnection:
    """
    Prevent process_resolved_events() from touching the real SQLite DB.
    """

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def execute(self, *args, **kwargs):
        return None


# ============================================================
# FIXTURE
# ============================================================


@pytest.fixture
def kg(monkeypatch):
    """
    Build a KnowledgeGraph without running __init__.

    This avoids:
    - loading the BKT model
    - loading cold-start files
    - initialising the real database
    """

    graph = object.__new__(KnowledgeGraph)

    # Mock KG methods used by process_resolved_events()
    graph.ensure_student = MagicMock()

    graph.record_attempt = MagicMock()

    graph.update_mastery = MagicMock(
        return_value={
            "probability": 0.75,
            "label": "strong",
            "cold_start_used": False,
            "cold_start_details": None,
        }
    )

    graph.get_student_graph = MagicMock(
        return_value=[
            {
                "skill": "algebra",
                "mastery_probability": 0.75,
                "mastery_label": "strong",
            }
        ]
    )

    # Replace real DB connection
    monkeypatch.setattr(
        kg_module,
        "get_connection",
        lambda: FakeConnection(),
    )

    return graph


# ============================================================
# TEST 1
# Incorrect + uncertainty + clarification
# must still be ONE BKT attempt
# ============================================================


def test_multiple_behaviours_create_only_one_bkt_attempt(kg):

    resolved = resolve_signal(
        ResolverInput(
            event_id="event_1",
            student_id="student_1",
            skill_id="algebra",

            correctness=Correctness.INCORRECT,

            uncertainty_probability=0.95,
            clarification_probability=0.90,
        )
    )

    result = kg.process_resolved_events(
        student_id="student_1",
        resolved_events=[resolved],
        session_id="session_1",
    )

    # Most important assertion:
    # ONE event => ONE BKT attempt
    kg.record_attempt.assert_called_once()

    kg.record_attempt.assert_called_once_with(
        student_id="student_1",
        skill="algebra",
        correct=0,
        confidence=1.0,
        signal_type="incorrect_answer",
        session_id="session_1",
    )

    # Behaviour must still be preserved
    behaviour = result["behaviour_events"][0]

    assert behaviour["uncertainty_probability"] == 0.95
    assert behaviour["clarification_probability"] == 0.90

    # Mastery recalculated once
    kg.update_mastery.assert_called_once_with(
        "student_1",
        "algebra",
    )


# ============================================================
# TEST 2
# Unknown correctness with strong behaviour creates one weak proxy
# ============================================================


def test_unknown_correctness_with_strong_behaviour_creates_weak_proxy(kg):

    resolved = resolve_signal(
        ResolverInput(
            event_id="event_2",
            student_id="student_1",
            skill_id="algebra",

            correctness=Correctness.UNKNOWN,

            uncertainty_probability=0.85,
            clarification_probability=0.91,
        )
    )

    result = kg.process_resolved_events(
        student_id="student_1",
        resolved_events=[resolved],
        session_id="session_2",
    )

    assert resolved.bkt_update.should_update is True
    assert resolved.bkt_update.outcome == 0
    assert 0.0 < resolved.bkt_update.update_confidence <= 0.25

    kg.record_attempt.assert_called_once_with(
        student_id="student_1",
        skill="algebra",
        correct=0,
        confidence=resolved.bkt_update.update_confidence,
        signal_type="behavioural_difficulty",
        session_id="session_2",
    )

    kg.update_mastery.assert_called_once_with(
        "student_1",
        "algebra",
    )

    # Behaviour must NOT disappear
    assert len(result["behaviour_events"]) == 1

    behaviour = result["behaviour_events"][0]

    assert behaviour["uncertainty_probability"] == 0.85
    assert behaviour["clarification_probability"] == 0.91

    assert len(result["skills_updated"]) == 1


# ============================================================
# TEST 3
# Correct + uncertain must remain a positive BKT observation
# ============================================================


def test_correct_uncertain_event_stays_positive(kg):

    resolved = resolve_signal(
        ResolverInput(
            event_id="event_3",
            student_id="student_1",
            skill_id="algebra",

            correctness=Correctness.CORRECT,

            uncertainty_probability=0.90,
        )
    )

    kg.process_resolved_events(
        student_id="student_1",
        resolved_events=[resolved],
        session_id="session_3",
    )

    kg.record_attempt.assert_called_once_with(
        student_id="student_1",
        skill="algebra",
        correct=1,
        confidence=resolved.bkt_update.update_confidence,
        signal_type="correct_answer",
        session_id="session_3",
    )
    assert 0.0 < resolved.bkt_update.update_confidence < 1.0


# ============================================================
# TEST 4
# Partial correctness should pass weighted confidence
# ============================================================


def test_partial_correct_uses_resolver_confidence(kg):

    resolved = resolve_signal(
        ResolverInput(
            event_id="event_4",
            student_id="student_1",
            skill_id="algebra",

            correctness=Correctness.PARTIAL,

            evaluator_confidence=0.80,
        )
    )

    kg.process_resolved_events(
        student_id="student_1",
        resolved_events=[resolved],
        session_id="session_4",
    )

    # partial weight = 0.40
    # evaluator confidence = 0.80
    #
    # update confidence = 0.40 * 0.80 = 0.32

    kg.record_attempt.assert_called_once_with(
        student_id="student_1",
        skill="algebra",
        correct=1,
        confidence=pytest.approx(0.32),
        signal_type="partial_correct",
        session_id="session_4",
    )


# ============================================================
# TEST 5
# Two different student events on same skill:
# two observations, but one mastery recalculation
# ============================================================


def test_multiple_events_same_skill_update_mastery_once(kg):

    first = resolve_signal(
        ResolverInput(
            event_id="event_5",
            student_id="student_1",
            skill_id="algebra",
            correctness=Correctness.INCORRECT,
        )
    )

    second = resolve_signal(
        ResolverInput(
            event_id="event_6",
            student_id="student_1",
            skill_id="algebra",
            correctness=Correctness.CORRECT,
        )
    )

    kg.process_resolved_events(
        student_id="student_1",
        resolved_events=[
            first,
            second,
        ],
        session_id="session_5",
    )

    # These are two genuine student events,
    # so two BKT observations are valid.
    assert kg.record_attempt.call_count == 2

    # But mastery only needs recalculating once after
    # both observations have been stored.
    kg.update_mastery.assert_called_once_with(
        "student_1",
        "algebra",
    )


# ============================================================
# TEST 6
# Two skills should each get mastery recalculation
# ============================================================


def test_multiple_skills_are_updated_separately(kg):

    algebra = resolve_signal(
        ResolverInput(
            event_id="event_7",
            student_id="student_1",
            skill_id="algebra",
            correctness=Correctness.CORRECT,
        )
    )

    fractions = resolve_signal(
        ResolverInput(
            event_id="event_8",
            student_id="student_1",
            skill_id="fractions",
            correctness=Correctness.INCORRECT,
        )
    )

    result = kg.process_resolved_events(
        student_id="student_1",
        resolved_events=[
            algebra,
            fractions,
        ],
        session_id="session_6",
    )

    assert kg.record_attempt.call_count == 2

    assert kg.update_mastery.call_count == 2

    kg.update_mastery.assert_any_call(
        "student_1",
        "algebra",
    )

    kg.update_mastery.assert_any_call(
        "student_1",
        "fractions",
    )

    assert len(result["skills_updated"]) == 2


# ============================================================
# TEST 7
# Repeated misunderstanding is metadata,
# not an extra BKT attempt
# ============================================================


def test_repeated_misunderstanding_does_not_duplicate_attempt(kg):

    resolved = resolve_signal(
        ResolverInput(
            event_id="event_9",
            student_id="student_1",
            skill_id="algebra",

            correctness=Correctness.INCORRECT,

            recent_same_skill_outcomes=[
                Correctness.INCORRECT,
            ],
        )
    )

    assert (
        resolved.history.repeated_misunderstanding
        is True
    )

    result = kg.process_resolved_events(
        student_id="student_1",
        resolved_events=[resolved],
        session_id="session_7",
    )

    # Current incorrect answer only.
    kg.record_attempt.assert_called_once()

    behaviour = result["behaviour_events"][0]

    assert (
        behaviour["repeated_misunderstanding"]
        is True
    )
