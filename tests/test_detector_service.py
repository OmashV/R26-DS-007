import numpy as np
import pytest

from core.detector_service import (
    DetectorService,
    DetectorThresholds,
    format_talkmoves_reasoning,
    normalize_itspoke_text,
    format_cima_clarification,
)


class FakeProbabilityModel:
    def __init__(self, positive_probability):
        self.positive_probability = positive_probability
        self.classes_ = np.array([0, 1])
        self.received = []

    def predict_proba(self, features):
        self.received.append(features)
        p = self.positive_probability
        return np.array([[1.0 - p, p]])


def make_service(
    reasoning=0.80,
    uncertainty=0.60,
    clarification=0.10,
):
    return DetectorService(
        reasoning_model=FakeProbabilityModel(reasoning),
        uncertainty_model=FakeProbabilityModel(uncertainty),
        clarification_model=FakeProbabilityModel(clarification),
    )


def test_talkmoves_formatter_matches_training_contract():
    formatted = format_talkmoves_reasoning(
        "Because 20 percent of 50 is 10.",
        "Maybe 10?",
    )

    assert formatted == (
        "[PREVIOUS_STUDENT] Maybe 10? "
        "[CURRENT_STUDENT] Because 20 percent of 50 is 10."
    )


def test_talkmoves_formatter_supports_no_previous_turn():
    formatted = format_talkmoves_reasoning(
        "The answer is 4.",
        None,
    )

    assert formatted == (
        "[PREVIOUS_STUDENT]  "
        "[CURRENT_STUDENT] The answer is 4."
    )


def test_reasoning_predictor_uses_talkmoves_formatter():
    service = make_service(reasoning=0.81)

    probability = service.reasoning_predictor(
        "current",
        "previous",
    )

    assert probability == pytest.approx(0.81)

    assert service.reasoning_model.received[0] == [
        "[PREVIOUS_STUDENT] previous [CURRENT_STUDENT] current"
    ]


def test_itspoke_normalizer_applies_documented_operations():
    text = "<nspn> I+ think*** it's, maybe... 12?!"

    normalized = normalize_itspoke_text(text)

    assert normalized == "I think it's maybe 12"


def test_uncertainty_predictor_uses_normalized_text():
    service = make_service(uncertainty=0.67)

    result = service.uncertainty_predictor(
        "<nspn> I+ think*** it's 12."
    )

    assert result == pytest.approx(0.67)

    assert service.uncertainty_model.received[0] == [
        "I think it's 12"
    ]


def test_cima_formatter_preserves_raw_target_text_except_outer_space():
    assert format_cima_clarification(
        "  Can you explain that again?  "
    ) == "Can you explain that again?"


def test_clarification_predictor_uses_target_text():
    service = make_service(clarification=0.92)

    result = service.clarification_predictor(
        "Can you explain that again?"
    )

    assert result == pytest.approx(0.92)


def test_frozen_thresholds():
    thresholds = DetectorThresholds()

    assert thresholds.reasoning == pytest.approx(0.30)
    assert thresholds.uncertainty == pytest.approx(0.25)
    assert thresholds.clarification == pytest.approx(0.40)
