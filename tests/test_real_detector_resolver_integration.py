from pathlib import Path

import pytest

from core.detector_service import DetectorService
from core.pipeline_adapter import resolve_extracted_events
from core.signal_resolver import PrimarySignal


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def detectors():
    return DetectorService.from_project_defaults(PROJECT_ROOT)


def test_real_detectors_do_not_duplicate_bkt_observations(detectors):
    transcript = [
        {"role": "student", "text": "I think it is 10."},
        {"role": "tutor", "text": "Why?"},
        {"role": "student", "text": "Because 20 percent of 50 is 10."},
        {"role": "tutor", "text": "Now try the next one."},
        {"role": "student", "text": "Can you explain what denominator means?"},
        {"role": "tutor", "text": "It is the bottom number of a fraction."},
        {
            "role": "student",
            "text": "Then I think one half plus one half is two fourths.",
        },
    ]

    extraction = {
        "events": [
            {
                "turn_index": 2,
                "skill": "Percent Of",
                "correctness": "correct",
            },
            {
                "turn_index": 4,
                "skill": "Fractions",
                "correctness": "unknown",
            },
            {
                "turn_index": 6,
                "skill": "Fractions",
                "correctness": "incorrect",
            },
        ]
    }

    resolved = resolve_extracted_events(
        extraction=extraction,
        transcript=transcript,
        student_id="integration_student",
        session_id="real_detector_resolver_001",
        reasoning_predictor=detectors.reasoning_predictor,
        uncertainty_predictor=detectors.uncertainty_predictor,
        clarification_predictor=detectors.clarification_predictor,
        history_getter=lambda student_id, skill: [],
    )

    assert len(resolved) == 3

    for event in resolved:
        assert 0.0 <= event.behaviour.reasoning_probability <= 1.0
        assert 0.0 <= event.behaviour.uncertainty_probability <= 1.0
        assert 0.0 <= event.behaviour.clarification_probability <= 1.0

    updates = [
        event
        for event in resolved
        if event.bkt_update.should_update
    ]

    assert len(updates) == 2
    assert [event.bkt_update.outcome for event in updates] == [1, 0]

    unknown_event = resolved[1]
    assert unknown_event.primary_signal == PrimarySignal.NO_UPDATE
    assert unknown_event.bkt_update.should_update is False


def test_real_detector_overlap_still_produces_one_positive_update(detectors):
    transcript = [
        {"role": "student", "text": "I think it is 10."},
        {"role": "tutor", "text": "Why?"},
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

    resolved = resolve_extracted_events(
        extraction=extraction,
        transcript=transcript,
        student_id="integration_student",
        session_id="real_detector_overlap_001",
        reasoning_predictor=detectors.reasoning_predictor,
        uncertainty_predictor=detectors.uncertainty_predictor,
        clarification_predictor=detectors.clarification_predictor,
        history_getter=lambda student_id, skill: [],
    )

    assert len(resolved) == 1

    event = resolved[0]

    assert event.bkt_update.should_update is True
    assert event.bkt_update.outcome == 1

    behavioural_flags = [
        event.behaviour.reasoning_present,
        event.behaviour.uncertainty_present,
        event.behaviour.clarification_present,
    ]

    assert sum(behavioural_flags) >= 1
