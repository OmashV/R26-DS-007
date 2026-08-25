import pytest

from core.pipeline_adapter import (
    build_resolver_inputs,
    resolve_extracted_events,
)
from core.signal_resolver import (
    Correctness,
    PrimarySignal,
)


def test_builds_resolver_input_with_detector_probabilities():
    transcript = [
        {"role": "student", "text": "I think it is 20."},
    ]

    extraction = {
        "events": [
            {
                "turn_index": 0,
                "skill": "Percent Of",
                "correctness": "correct",
                "student_text": "I think it is 20.",
            }
        ]
    }

    events = build_resolver_inputs(
        extraction=extraction,
        transcript=transcript,
        student_id="s1",
        session_id="sess1",
        reasoning_predictor=lambda text, previous: 0.72,
        uncertainty_predictor=lambda text: 0.64,
        clarification_predictor=lambda text: 0.05,
    )

    assert len(events) == 1

    event = events[0]

    assert event.student_id == "s1"
    assert event.skill_id == "Percent Of"
    assert event.correctness == Correctness.CORRECT
    assert event.reasoning_probability == pytest.approx(0.72)
    assert event.uncertainty_probability == pytest.approx(0.64)
    assert event.clarification_probability == pytest.approx(0.05)


def test_reasoning_detector_receives_previous_student_turn():
    transcript = [
        {"role": "student", "text": "Maybe 10?"},
        {"role": "tutor", "text": "Explain why."},
        {"role": "student", "text": "Because 20 percent of 50 is 10."},
    ]

    extraction = {
        "events": [
            {
                "turn_index": 2,
                "skill": "Percent Of",
                "correctness": "correct",
            }
        ]
    }

    received = {}

    def reasoning(text, previous):
        received["text"] = text
        received["previous"] = previous
        return 0.80

    build_resolver_inputs(
        extraction=extraction,
        transcript=transcript,
        student_id="s1",
        session_id="sess1",
        reasoning_predictor=reasoning,
    )

    assert received["text"] == (
        "Because 20 percent of 50 is 10."
    )

    assert received["previous"] == "Maybe 10?"


def test_history_is_converted_for_repeated_misunderstanding():
    transcript = [
        {"role": "student", "text": "I still think it is 30."},
    ]

    extraction = {
        "events": [
            {
                "turn_index": 0,
                "skill": "Percent Of",
                "correctness": "incorrect",
            }
        ]
    }

    def history_getter(student_id, skill):
        assert student_id == "s1"
        assert skill == "Percent Of"

        return [
            (1, 1.0),
            (0, 1.0),
        ]

    resolved = resolve_extracted_events(
        extraction=extraction,
        transcript=transcript,
        student_id="s1",
        session_id="sess1",
        history_getter=history_getter,
    )

    assert len(resolved) == 1
    assert (
        resolved[0].history.repeated_misunderstanding
        is True
    )

    # Still only one current negative BKT observation.
    assert resolved[0].bkt_update.outcome == 0


def test_unknown_correctness_with_behaviour_creates_weak_proxy():
    transcript = [
        {
            "role": "student",
            "text": "Can you explain that again?",
        },
    ]

    extraction = {
        "events": [
            {
                "turn_index": 0,
                "skill": "Percent Of",
                "correctness": "unknown",
            }
        ]
    }

    resolved = resolve_extracted_events(
        extraction=extraction,
        transcript=transcript,
        student_id="s1",
        session_id="sess1",
        uncertainty_predictor=lambda text: 0.70,
        clarification_predictor=lambda text: 0.95,
    )

    event = resolved[0]

    assert (
        event.primary_signal
        == PrimarySignal.BEHAVIOURAL_DIFFICULTY
    )

    assert event.bkt_update.should_update is True
    assert event.bkt_update.outcome == 0

    assert (
        0.0
        < event.bkt_update.update_confidence
        <= 0.25
    )

    assert event.behaviour.uncertainty_present is True
    assert event.behaviour.clarification_present is True

    assert set(event.bkt_update.contributors) == {
        "uncertainty",
        "clarification",
    }


def test_unauthorized_skill_cannot_create_behavioural_proxy():
    transcript = [
        {"role": "tutor", "text": "What is the ratio?"},
        {"role": "student", "text": "I do not understand."},
    ]
    extraction = {
        "events": [
            {
                "turn_index": 1,
                "skill": "Ratio",
                "correctness": "unknown",
                "evaluator_confidence": 0.0,
                "evaluator_source": (
                    "skill_not_authorized_for_assessment"
                ),
            }
        ]
    }

    resolved = resolve_extracted_events(
        extraction=extraction,
        transcript=transcript,
        student_id="student-1",
        session_id="session-1",
        uncertainty_predictor=lambda text: 0.95,
        clarification_predictor=lambda text: 0.95,
    )[0]

    assert resolved.primary_signal == PrimarySignal.NO_UPDATE
    assert resolved.bkt_update.should_update is False
    assert resolved.bkt_update.outcome is None
    assert resolved.behaviour.uncertainty_present is True
    assert resolved.behaviour.clarification_present is True


@pytest.mark.parametrize(
    "evaluator_source",
    [
        "behaviour_only_turn",
        "gemini_structured_non_answer",
        "missing_evaluation_context",
    ],
)
def test_other_unknown_sources_remain_behavioural_proxy_eligible(
    evaluator_source,
):
    transcript = [
        {"role": "student", "text": "Can you explain that again?"},
    ]
    extraction = {
        "events": [
            {
                "turn_index": 0,
                "skill": "Percent Of",
                "correctness": "unknown",
                "evaluator_confidence": 0.0,
                "evaluator_source": evaluator_source,
            }
        ]
    }

    resolved = resolve_extracted_events(
        extraction=extraction,
        transcript=transcript,
        student_id="student-1",
        session_id="session-1",
        uncertainty_predictor=lambda text: 0.95,
        clarification_predictor=lambda text: 0.95,
    )[0]

    assert resolved.primary_signal == PrimarySignal.BEHAVIOURAL_DIFFICULTY
    assert resolved.bkt_update.should_update is True


def test_correct_plus_reasoning_becomes_correct_explanation():
    transcript = [
        {
            "role": "student",
            "text": "It is 10 because 20 percent of 50 is 10.",
        },
    ]

    extraction = {
        "events": [
            {
                "turn_index": 0,
                "skill": "Percent Of",
                "correctness": "correct",
            }
        ]
    }

    resolved = resolve_extracted_events(
        extraction=extraction,
        transcript=transcript,
        student_id="s1",
        session_id="sess1",
        reasoning_predictor=lambda text, previous: 0.81,
    )

    assert (
        resolved[0].primary_signal
        == PrimarySignal.CORRECT_EXPLANATION
    )

    assert resolved[0].bkt_update.outcome == 1


def test_evaluator_confidence_is_kept_separate():
    transcript = [
        {"role": "student", "text": "Maybe half."},
    ]

    extraction = {
        "events": [
            {
                "turn_index": 0,
                "skill": "Fractions",
                "correctness": "partial",
                "evaluator_confidence": 0.80,
            }
        ]
    }

    resolved = resolve_extracted_events(
        extraction=extraction,
        transcript=transcript,
        student_id="s1",
        session_id="sess1",
    )

    update = resolved[0].bkt_update

    assert update.evidence_weight == pytest.approx(0.40)
    assert update.evaluator_confidence == pytest.approx(0.80)
    assert update.update_confidence == pytest.approx(0.32)


def test_event_id_is_deterministic():
    transcript = [
        {"role": "student", "text": "10"},
    ]

    extraction = {
        "events": [
            {
                "turn_index": 0,
                "skill": "Percent Of",
                "correctness": "correct",
            }
        ]
    }

    first = build_resolver_inputs(
        extraction=extraction,
        transcript=transcript,
        student_id="s1",
        session_id="sess_xyz",
    )

    second = build_resolver_inputs(
        extraction=extraction,
        transcript=transcript,
        student_id="s1",
        session_id="sess_xyz",
    )

    assert first[0].event_id == second[0].event_id
    assert first[0].event_id == (
        "sess_xyz:turn_0:percent_of"
    )
