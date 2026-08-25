from pathlib import Path

import pytest

from core.detector_service import DetectorService


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REASONING_MODEL = (
    PROJECT_ROOT / "models" / "detectors" / "reasoning.joblib"
)
UNCERTAINTY_MODEL = (
    PROJECT_ROOT / "models" / "detectors" / "uncertainty.joblib"
)
CLARIFICATION_MODEL = (
    PROJECT_ROOT / "models" / "detectors" / "clarification.joblib"
)


@pytest.fixture(scope="module")
def detectors():
    missing = [
        path
        for path in (
            REASONING_MODEL,
            UNCERTAINTY_MODEL,
            CLARIFICATION_MODEL,
        )
        if not path.exists()
    ]

    if missing:
        pytest.fail(
            "Missing detector artifacts: "
            + ", ".join(str(path) for path in missing)
        )

    return DetectorService.from_project_defaults(
        PROJECT_ROOT
    )


@pytest.mark.parametrize(
    "current,previous",
    [
        (
            "Because 20 percent of 50 is 10.",
            "I think it is 10.",
        ),
        (
            "I'm not sure, maybe 12.",
            None,
        ),
        (
            "Can you explain that again?",
            None,
        ),
        (
            "10",
            None,
        ),
    ],
)
def test_real_detectors_return_valid_probabilities(
    detectors,
    current,
    previous,
):
    result = detectors.predict_all(
        current_text=current,
        previous_student_text=previous,
    )

    for key in (
        "reasoning_probability",
        "uncertainty_probability",
        "clarification_probability",
    ):
        assert 0.0 <= result[key] <= 1.0


def test_real_detector_models_expose_binary_classes(detectors):
    for model in (
        detectors.reasoning_model,
        detectors.uncertainty_model,
        detectors.clarification_model,
    ):
        assert hasattr(model, "classes_")
        assert 1 in list(model.classes_)


def test_real_models_support_predict_proba(detectors):
    for model in (
        detectors.reasoning_model,
        detectors.uncertainty_model,
        detectors.clarification_model,
    ):
        assert hasattr(model, "predict_proba")
