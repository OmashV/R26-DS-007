import pytest

from core.signal_resolver import (
    Correctness,
    PrimarySignal,
    ResolverInput,
    resolve_signal,
)


def test_correct_answer():
    result = resolve_signal(ResolverInput(
        event_id="e1", student_id="s1", skill_id="algebra",
        correctness=Correctness.CORRECT,
        reasoning_probability=0.10,
        uncertainty_probability=0.10,
        clarification_probability=0.10,
    ))
    assert result.primary_signal == PrimarySignal.CORRECT_ANSWER
    assert result.bkt_update.should_update
    assert result.bkt_update.outcome == 1
    assert result.bkt_update.evidence_weight == 1.0


def test_correct_explanation():
    result = resolve_signal(ResolverInput(
        event_id="e2", student_id="s1", skill_id="algebra",
        correctness=Correctness.CORRECT,
        reasoning_probability=0.31,
    ))
    assert result.primary_signal == PrimarySignal.CORRECT_EXPLANATION
    assert result.behaviour.reasoning_present
    assert result.bkt_update.outcome == 1


def test_uncertain_correct_answer_stays_correct():
    result = resolve_signal(ResolverInput(
        event_id="e3", student_id="s1", skill_id="algebra",
        correctness=Correctness.CORRECT,
        uncertainty_probability=0.90,
    ))
    assert result.bkt_update.outcome == 1
    assert result.behaviour.uncertainty_present is True


def test_incorrect_with_multiple_behaviours_is_one_update():
    result = resolve_signal(ResolverInput(
        event_id="e4", student_id="s1", skill_id="algebra",
        correctness=Correctness.INCORRECT,
        uncertainty_probability=0.95,
        clarification_probability=0.90,
    ))
    assert result.primary_signal == PrimarySignal.INCORRECT_ANSWER
    assert result.bkt_update.outcome == 0
    assert result.behaviour.uncertainty_present is True
    assert result.behaviour.clarification_present is True


def test_unknown_correctness_with_high_clarification_creates_weak_proxy():
    result = resolve_signal(ResolverInput(
        event_id="e5",
        student_id="s1",
        skill_id="algebra",
        correctness=Correctness.UNKNOWN,
        evaluator_confidence=0.0,
        clarification_probability=0.95,
    ))

    assert (
        result.primary_signal
        == PrimarySignal.BEHAVIOURAL_DIFFICULTY
    )

    assert result.bkt_update.should_update is True
    assert result.bkt_update.outcome == 0

    assert (
        0.0
        < result.bkt_update.update_confidence
        <= 0.25
    )

    assert result.behaviour.clarification_present is True


def test_behavioural_proxy_can_be_disallowed():
    event = ResolverInput(
        event_id="evt-1",
        student_id="student-1",
        skill_id="Ratio",
        correctness=Correctness.UNKNOWN,
        evaluator_confidence=0.0,
        uncertainty_probability=0.95,
        clarification_probability=0.95,
        allow_behavioural_proxy=False,
    )

    resolved = resolve_signal(event)

    assert resolved.primary_signal == PrimarySignal.NO_UPDATE
    assert resolved.bkt_update.should_update is False
    assert resolved.bkt_update.outcome is None

    # Behaviour remains observable metadata even though it is not authorized
    # to create a mastery observation for this skill.
    assert resolved.behaviour.uncertainty_present is True
    assert resolved.behaviour.clarification_present is True


@pytest.mark.parametrize(
    ("correctness", "expected_outcome"),
    [
        (Correctness.CORRECT, 1),
        (Correctness.PARTIAL, 1),
        (Correctness.INCORRECT, 0),
    ],
)
def test_proxy_flag_does_not_suppress_evaluator_backed_observations(
    correctness,
    expected_outcome,
):
    resolved = resolve_signal(
        ResolverInput(
            event_id="evt-evaluator",
            student_id="student-1",
            skill_id="Ratio",
            correctness=correctness,
            allow_behavioural_proxy=False,
            uncertainty_probability=0.95,
            clarification_probability=0.95,
        )
    )

    assert resolved.bkt_update.should_update is True
    assert resolved.bkt_update.outcome == expected_outcome


def test_partial_correct():
    result = resolve_signal(ResolverInput(
        event_id="e6", student_id="s1", skill_id="algebra",
        correctness=Correctness.PARTIAL,
    ))
    assert result.primary_signal == PrimarySignal.PARTIAL_CORRECT
    assert result.bkt_update.outcome == 1
    assert result.bkt_update.evidence_weight == 0.40


def test_repeated_misunderstanding():
    result = resolve_signal(ResolverInput(
        event_id="e7", student_id="s1", skill_id="algebra",
        correctness=Correctness.INCORRECT,
        recent_same_skill_outcomes=[Correctness.INCORRECT],
    ))
    assert result.history.repeated_misunderstanding is True
    assert result.bkt_update.outcome == 0


def test_not_repeated_after_correct_attempt():
    result = resolve_signal(ResolverInput(
        event_id="e8", student_id="s1", skill_id="algebra",
        correctness=Correctness.INCORRECT,
        recent_same_skill_outcomes=[
            Correctness.INCORRECT,
            Correctness.CORRECT,
        ],
    ))
    assert result.history.repeated_misunderstanding is False


def test_threshold_boundaries_are_inclusive():
    result = resolve_signal(ResolverInput(
        event_id="e9", student_id="s1", skill_id="algebra",
        correctness=Correctness.UNKNOWN,
        reasoning_probability=0.30,
        uncertainty_probability=0.25,
        clarification_probability=0.40,
    ))
    assert result.behaviour.reasoning_present
    assert result.behaviour.uncertainty_present
    assert result.behaviour.clarification_present


def test_evidence_weight_and_confidence_are_separate():
    result = resolve_signal(ResolverInput(
        event_id="e10", student_id="s1", skill_id="algebra",
        correctness=Correctness.PARTIAL,
        evaluator_confidence=0.80,
    ))
    assert result.bkt_update.evidence_weight == 0.40
    assert result.bkt_update.evaluator_confidence == 0.80
    assert result.bkt_update.update_confidence == pytest.approx(0.32)


def test_missing_probabilities_are_not_positive():
    result = resolve_signal(ResolverInput(
        event_id="e11", student_id="s1", skill_id="algebra",
        correctness=Correctness.UNKNOWN,
    ))
    assert result.behaviour.reasoning_present is False
    assert result.behaviour.uncertainty_present is False
    assert result.behaviour.clarification_present is False


def test_single_incorrect_attempt_is_not_repeated_misunderstanding():
    result = resolve_signal(ResolverInput(
        event_id="e12", student_id="s1", skill_id="algebra",
        correctness=Correctness.INCORRECT,
    ))
    assert result.history.repeated_misunderstanding is False
