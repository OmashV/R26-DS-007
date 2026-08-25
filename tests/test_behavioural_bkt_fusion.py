import pytest

from core.signal_resolver import (
    Correctness,
    ObservationSource,
    PrimarySignal,
    ResolverConfig,
    ResolverInput,
    resolve_signal,
)


def make_event(
    *,
    correctness=Correctness.UNKNOWN,
    evaluator_confidence=0.0,
    reasoning=None,
    uncertainty=None,
    clarification=None,
    history=None,
):
    return ResolverInput(
        event_id="evt-1",
        student_id="student-1",
        skill_id="Percent Of",
        correctness=correctness,
        evaluator_confidence=evaluator_confidence,
        reasoning_probability=reasoning,
        uncertainty_probability=uncertainty,
        clarification_probability=clarification,
        recent_same_skill_outcomes=history or [],
    )


def test_unknown_high_uncertainty_creates_one_weak_negative_proxy():
    event = resolve_signal(
        make_event(
            uncertainty=0.85,
        )
    )

    assert event.primary_signal == (
        PrimarySignal.BEHAVIOURAL_DIFFICULTY
    )
    assert event.bkt_update.should_update is True
    assert event.bkt_update.outcome == 0
    assert event.bkt_update.observation_source == (
        ObservationSource.BEHAVIOURAL_PROXY
    )
    assert event.bkt_update.evaluator_confidence == 0.0
    assert 0.0 < event.bkt_update.update_confidence <= 0.25
    assert event.bkt_update.contributors == ["uncertainty"]


def test_unknown_high_clarification_creates_one_weak_negative_proxy():
    event = resolve_signal(
        make_event(
            clarification=0.90,
        )
    )

    assert event.primary_signal == (
        PrimarySignal.BEHAVIOURAL_DIFFICULTY
    )
    assert event.bkt_update.should_update is True
    assert event.bkt_update.outcome == 0
    assert 0.0 < event.bkt_update.update_confidence <= 0.25
    assert event.bkt_update.contributors == ["clarification"]


def test_unknown_uncertainty_and_clarification_are_fused_not_duplicated():
    uncertainty_only = resolve_signal(
        make_event(uncertainty=0.80)
    )
    clarification_only = resolve_signal(
        make_event(clarification=0.80)
    )
    both = resolve_signal(
        make_event(
            uncertainty=0.80,
            clarification=0.80,
        )
    )

    assert both.bkt_update.should_update is True
    assert both.bkt_update.outcome == 0

    # There is still only one BKTUpdate object for the event.
    assert both.primary_signal == (
        PrimarySignal.BEHAVIOURAL_DIFFICULTY
    )

    assert set(both.bkt_update.contributors) == {
        "uncertainty",
        "clarification",
    }

    # Fused evidence can be stronger than either detector alone, but remains
    # bounded by one weak behavioural cap.
    assert both.bkt_update.update_confidence >= (
        uncertainty_only.bkt_update.update_confidence
    )
    assert both.bkt_update.update_confidence >= (
        clarification_only.bkt_update.update_confidence
    )
    assert both.bkt_update.update_confidence <= 0.25


def test_unknown_reasoning_alone_does_not_invent_positive_mastery():
    event = resolve_signal(
        make_event(
            reasoning=0.95,
        )
    )

    assert event.primary_signal == PrimarySignal.NO_UPDATE
    assert event.bkt_update.should_update is False
    assert event.bkt_update.outcome is None
    assert event.behaviour.reasoning_present is True


def test_unknown_below_behaviour_thresholds_does_not_update():
    event = resolve_signal(
        make_event(
            uncertainty=0.20,
            clarification=0.30,
        )
    )

    assert event.primary_signal == PrimarySignal.NO_UPDATE
    assert event.bkt_update.should_update is False


def test_correct_uncertainty_stays_positive_but_is_weaker():
    baseline = resolve_signal(
        make_event(
            correctness=Correctness.CORRECT,
            evaluator_confidence=0.80,
        )
    )

    uncertain = resolve_signal(
        make_event(
            correctness=Correctness.CORRECT,
            evaluator_confidence=0.80,
            uncertainty=0.90,
        )
    )

    assert uncertain.bkt_update.outcome == 1
    assert uncertain.bkt_update.should_update is True
    assert uncertain.bkt_update.update_confidence < (
        baseline.bkt_update.update_confidence
    )
    assert uncertain.bkt_update.observation_source == (
        ObservationSource.EVALUATOR
    )
    assert "uncertainty" in uncertain.bkt_update.contributors


def test_correct_reasoning_can_strengthen_evaluator_backed_positive():
    baseline = resolve_signal(
        make_event(
            correctness=Correctness.PARTIAL,
            evaluator_confidence=0.80,
        )
    )

    reasoned = resolve_signal(
        make_event(
            correctness=Correctness.PARTIAL,
            evaluator_confidence=0.80,
            reasoning=0.95,
        )
    )

    assert reasoned.bkt_update.outcome == 1
    assert reasoned.bkt_update.update_confidence > (
        baseline.bkt_update.update_confidence
    )
    assert "reasoning" in reasoned.bkt_update.contributors


def test_incorrect_repeated_misunderstanding_strengthens_without_duplicate():
    baseline = resolve_signal(
        make_event(
            correctness=Correctness.INCORRECT,
            evaluator_confidence=0.60,
            history=[Correctness.CORRECT],
        )
    )

    repeated = resolve_signal(
        make_event(
            correctness=Correctness.INCORRECT,
            evaluator_confidence=0.60,
            history=[Correctness.INCORRECT],
        )
    )

    assert repeated.history.repeated_misunderstanding is True
    assert repeated.primary_signal == PrimarySignal.INCORRECT_ANSWER
    assert repeated.bkt_update.outcome == 0
    assert repeated.bkt_update.should_update is True
    assert repeated.bkt_update.update_confidence > (
        baseline.bkt_update.update_confidence
    )
    assert "repeated_misunderstanding" in (
        repeated.bkt_update.contributors
    )


def test_behavioural_proxy_can_be_disabled_for_ablation():
    config = ResolverConfig(
        enable_behavioural_bkt=False,
    )

    event = resolve_signal(
        make_event(
            uncertainty=0.95,
            clarification=0.95,
        ),
        config=config,
    )

    assert event.primary_signal == PrimarySignal.NO_UPDATE
    assert event.bkt_update.should_update is False


def test_existing_partial_confidence_semantics_still_hold_without_behaviour():
    event = resolve_signal(
        make_event(
            correctness=Correctness.PARTIAL,
            evaluator_confidence=0.80,
        )
    )

    assert event.bkt_update.evidence_weight == pytest.approx(0.40)
    assert event.bkt_update.evaluator_confidence == pytest.approx(0.80)
    assert event.bkt_update.behaviour_factor == pytest.approx(1.0)
    assert event.bkt_update.update_confidence == pytest.approx(0.32)
