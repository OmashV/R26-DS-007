import pytest

from core.evaluation_contract import (
    EvaluationVerdict,
    apply_evaluator_to_extraction,
)
from core.pipeline_adapter import (
    build_resolver_inputs,
    resolve_extracted_events,
)
from core.signal_resolver import (
    Correctness,
    PrimarySignal,
)


def test_evaluator_is_authoritative_over_extractor_correctness():
    transcript = [
        {
            "role": "student",
            "text": "Because 20 percent of 50 is 10.",
        }
    ]

    extraction = {
        "events": [
            {
                "turn_index": 0,
                "skill": "Percent Of",
                # Deliberately wrong extractor judgment.
                "correctness": "incorrect",
            }
        ]
    }

    evaluated = apply_evaluator_to_extraction(
        extraction=extraction,
        transcript=transcript,
        evaluator=lambda event, transcript: {
            "correctness": "correct",
            "confidence": 0.94,
            "source": "objective_math_evaluator",
        },
    )

    event = evaluated["events"][0]

    assert event["correctness"] == "correct"
    assert event["evaluator_confidence"] == pytest.approx(0.94)
    assert event["evaluator_source"] == "objective_math_evaluator"


def test_evaluator_never_receives_extractor_correctness():
    transcript = [
        {"role": "student", "text": "10"},
    ]

    extraction = {
        "events": [
            {
                "turn_index": 0,
                "skill": "Percent Of",
                "correctness": "incorrect",
                "confusion": True,
                "signal_type": "evaluator_negative",
            }
        ]
    }

    captured = {}

    def evaluator(event, transcript):
        captured.update(event)

        return {
            "correctness": "correct",
            "confidence": 1.0,
        }

    apply_evaluator_to_extraction(
        extraction=extraction,
        transcript=transcript,
        evaluator=evaluator,
    )

    assert "correctness" not in captured
    assert "confusion" not in captured
    assert "signal_type" not in captured


def test_legacy_flat_signal_fields_are_not_propagated():
    transcript = [
        {"role": "student", "text": "Maybe half."},
    ]

    extraction = {
        "events": [
            {
                "turn_index": 0,
                "skill": "Fractions",
                "correctness": "incorrect",
                "correct_answer": True,
                "partial_correct": True,
                "confusion": True,
                "clarification_request": True,
                "repeated_misunderstanding": True,
                "evaluator_positive": True,
            }
        ],
        "misconceptions": [
            "Student is unsure about fraction equivalence."
        ],
    }

    evaluated = apply_evaluator_to_extraction(
        extraction=extraction,
        transcript=transcript,
        evaluator=lambda event, transcript: {
            "correctness": "partial",
            "confidence": 0.80,
        },
    )

    event = evaluated["events"][0]

    assert set(event) == {
        "turn_index",
        "skill",
        "correctness",
        "evaluator_confidence",
        "evaluator_source",
        "evidence_span",
        "student_text",
    }

    assert evaluated["misconceptions"] == [
        "Student is unsure about fraction equivalence."
    ]


def test_none_evaluator_verdict_becomes_unknown_no_update():
    transcript = [
        {
            "role": "student",
            "text": "Can you explain that again?",
        }
    ]

    extraction = {
        "events": [
            {
                "turn_index": 0,
                "skill": "Percent Of",
            }
        ]
    }

    evaluated = apply_evaluator_to_extraction(
        extraction=extraction,
        transcript=transcript,
        evaluator=lambda event, transcript: None,
    )

    event = evaluated["events"][0]

    assert event["correctness"] == "unknown"
    assert event["evaluator_confidence"] == pytest.approx(0.0)

    resolved = resolve_extracted_events(
        extraction=evaluated,
        transcript=transcript,
        student_id="s1",
        session_id="sess1",
        clarification_predictor=lambda text: 0.95,
    )

    assert resolved[0].primary_signal == PrimarySignal.NO_UPDATE
    assert resolved[0].bkt_update.should_update is False
    assert resolved[0].behaviour.clarification_present is True


def test_partial_evaluator_confidence_reaches_resolver_unchanged():
    transcript = [
        {
            "role": "student",
            "text": "Maybe one half.",
        }
    ]

    extraction = {
        "events": [
            {
                "turn_index": 0,
                "skill": "Fractions",
            }
        ]
    }

    evaluated = apply_evaluator_to_extraction(
        extraction=extraction,
        transcript=transcript,
        evaluator=lambda event, transcript: EvaluationVerdict(
            correctness=Correctness.PARTIAL,
            confidence=0.80,
            source="structured_evaluator",
        ),
    )

    resolver_inputs = build_resolver_inputs(
        extraction=evaluated,
        transcript=transcript,
        student_id="s1",
        session_id="sess1",
    )

    assert resolver_inputs[0].correctness == Correctness.PARTIAL
    assert (
        resolver_inputs[0].evaluator_confidence
        == pytest.approx(0.80)
    )

    resolved = resolve_extracted_events(
        extraction=evaluated,
        transcript=transcript,
        student_id="s1",
        session_id="sess1",
    )

    update = resolved[0].bkt_update

    assert update.evidence_weight == pytest.approx(0.40)
    assert update.evaluator_confidence == pytest.approx(0.80)
    assert update.update_confidence == pytest.approx(0.32)


def test_evidence_span_falls_back_to_student_text():
    transcript = [
        {
            "role": "student",
            "text": "I multiplied 20 by 0.5 and got 10.",
        }
    ]

    evaluated = apply_evaluator_to_extraction(
        extraction={
            "events": [
                {
                    "turn_index": 0,
                    "skill": "Percent Of",
                }
            ]
        },
        transcript=transcript,
        evaluator=lambda event, transcript: {
            "correctness": "correct",
            "confidence": 1.0,
        },
    )

    event = evaluated["events"][0]

    assert event["student_text"] == (
        "I multiplied 20 by 0.5 and got 10."
    )
    assert event["evidence_span"] == event["student_text"]


def test_non_student_extractor_event_is_rejected():
    transcript = [
        {
            "role": "tutor",
            "text": "What is 20 percent of 50?",
        }
    ]

    with pytest.raises(
        ValueError,
        match="not a student turn",
    ):
        apply_evaluator_to_extraction(
            extraction={
                "events": [
                    {
                        "turn_index": 0,
                        "skill": "Percent Of",
                    }
                ]
            },
            transcript=transcript,
            evaluator=lambda event, transcript: {
                "correctness": "correct",
            },
        )


def test_multiple_events_are_evaluated_independently():
    transcript = [
        {"role": "student", "text": "10"},
        {"role": "tutor", "text": "And the fraction?"},
        {"role": "student", "text": "Two fourths."},
    ]

    extraction = {
        "events": [
            {
                "turn_index": 0,
                "skill": "Percent Of",
            },
            {
                "turn_index": 2,
                "skill": "Fractions",
            },
        ]
    }

    verdicts = iter(
        [
            {
                "correctness": "correct",
                "confidence": 0.99,
            },
            {
                "correctness": "incorrect",
                "confidence": 0.91,
            },
        ]
    )

    evaluated = apply_evaluator_to_extraction(
        extraction=extraction,
        transcript=transcript,
        evaluator=lambda event, transcript: next(verdicts),
    )

    assert [
        event["correctness"]
        for event in evaluated["events"]
    ] == ["correct", "incorrect"]

    assert [
        event["evaluator_confidence"]
        for event in evaluated["events"]
    ] == pytest.approx([0.99, 0.91])
