from types import SimpleNamespace

import pytest

from core.evaluation_contract import apply_evaluator_to_extraction
from core.student_answer_evaluator import (
    EvaluatorContext,
    StudentAnswerEvaluator,
    exact_or_numeric_objective_comparator,
)


class FakeModels:
    def __init__(self, payload: str):
        self.payload = payload
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(text=self.payload)


class FakeClient:
    def __init__(self, payload: str):
        self.models = FakeModels(payload)


def test_objective_numeric_correct():
    evaluator = StudentAnswerEvaluator(
        EvaluatorContext(
            problem="What is 20 percent of 50?",
            reference_answer="10",
        ),
        client=FakeClient(
            '{"correctness":"incorrect","confidence":0.99}'
        ),
    )

    verdict = evaluator(
        {
            "turn_index": 0,
            "skill": "Percent Of",
            "student_text": "I think it is 10.",
            "evidence_span": "10",
        },
        [],
    )

    assert verdict.correctness.value == "correct"
    assert verdict.confidence == pytest.approx(1.0)
    assert verdict.source == "objective_numeric_match"


def test_objective_numeric_incorrect():
    evaluator = StudentAnswerEvaluator(
        EvaluatorContext(
            problem="What is 20 percent of 50?",
            reference_answer="10",
        ),
    )

    verdict = evaluator(
        {
            "turn_index": 0,
            "skill": "Percent Of",
            "student_text": "25",
            "evidence_span": "25",
        },
        [],
    )

    assert verdict.correctness.value == "incorrect"
    assert verdict.confidence == pytest.approx(1.0)


def test_missing_context_returns_unknown():
    evaluator = StudentAnswerEvaluator(
        EvaluatorContext(
            problem="Explain the method.",
        ),
    )

    verdict = evaluator(
        {
            "turn_index": 0,
            "skill": "Percent Of",
            "student_text": "Maybe multiply.",
            "evidence_span": "Maybe multiply.",
        },
        [],
    )

    assert verdict.correctness.value == "unknown"
    assert verdict.confidence == pytest.approx(0.0)
    assert verdict.source == "missing_evaluation_context"


def test_structured_fallback_can_return_partial():
    fake_client = FakeClient(
        '{"response_type":"answer_attempt",'
        '"correctness":"partial",'
        '"confidence":0.82}'
    )

    evaluator = StudentAnswerEvaluator(
        EvaluatorContext(
            problem="Explain why 20 percent of 50 is 10.",
            reference_answer="20 percent means 0.2, so 0.2 x 50 = 10.",
            rubric="Award partial for a correct method with an arithmetic slip.",
        ),
        objective_comparator=lambda student, reference: None,
        client=fake_client,
    )

    verdict = evaluator(
        {
            "turn_index": 0,
            "skill": "Percent Of",
            "student_text": "20 percent is 0.2, so I multiply.",
            "evidence_span": "20 percent is 0.2",
        },
        [
            {
                "role": "student",
                "text": "20 percent is 0.2, so I multiply.",
            }
        ],
    )

    assert verdict.correctness.value == "partial"
    assert verdict.confidence == pytest.approx(0.82)
    assert verdict.source == "gemini_structured_fallback"
    assert len(fake_client.models.calls) == 1


def test_invalid_fallback_json_becomes_unknown():
    evaluator = StudentAnswerEvaluator(
        EvaluatorContext(
            problem="Explain.",
            rubric="Judge the explanation.",
        ),
        client=FakeClient("not-json"),
    )

    verdict = evaluator(
        {
            "turn_index": 0,
            "skill": "Some Skill",
            "student_text": "My explanation.",
            "evidence_span": "My explanation.",
        },
        [],
    )

    assert verdict.correctness.value == "unknown"
    assert verdict.confidence == pytest.approx(0.0)


def test_contract_ignores_extractor_correctness_with_real_evaluator():
    transcript = [
        {
            "role": "student",
            "text": "The answer is 10.",
        }
    ]

    raw_extraction = {
        "events": [
            {
                "turn_index": 0,
                "skill": "Percent Of",
                # Deliberately wrong extractor label.
                "correctness": "incorrect",
            }
        ]
    }

    evaluator = StudentAnswerEvaluator(
        EvaluatorContext(
            problem="What is 20 percent of 50?",
            reference_answer="10",
        ),
    )

    evaluated = apply_evaluator_to_extraction(
        extraction=raw_extraction,
        transcript=transcript,
        evaluator=evaluator,
    )

    assert (
        evaluated["events"][0]["correctness"]
        == "correct"
    )
    assert (
        evaluated["events"][0]["evaluator_source"]
        == "objective_numeric_match"
    )


@pytest.mark.parametrize(
    "student,reference,expected",
    [
        ("10", "10", "correct"),
        ("The answer is 10.", "10", "correct"),
        ("9", "10", "incorrect"),
        ("ten", "ten", "correct"),
    ],
)
def test_objective_comparator(student, reference, expected):
    verdict = exact_or_numeric_objective_comparator(
        student,
        reference,
    )

    assert verdict is not None
    assert verdict.correctness.value == expected
