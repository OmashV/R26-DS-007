import json

from core.signal_resolver import Correctness
from core.student_answer_evaluator import (
    EvaluatorContext,
    StudentAnswerEvaluator,
)


class _Response:
    def __init__(self, payload):
        self.text = json.dumps(payload)


class _Models:
    def __init__(self, payload):
        self.payload = payload

    def generate_content(self, **kwargs):
        return _Response(self.payload)


class _Client:
    def __init__(self, payload):
        self.models = _Models(payload)


def _event(text: str) -> dict:
    return {
        "turn_index": 1,
        "skill": "Percent Of",
        "student_text": text,
        "evidence_span": text,
    }


def _transcript(text: str) -> list[dict]:
    return [
        {
            "role": "tutor",
            "text": "What is 20 percent of 50? Explain your reasoning.",
        },
        {
            "role": "student",
            "text": text,
        },
    ]


def _evaluator(payload: dict) -> StudentAnswerEvaluator:
    # Rubric-only forces the fake structured fallback path.
    return StudentAnswerEvaluator(
        EvaluatorContext(
            problem="What is 20 percent of 50? Explain your reasoning.",
            rubric="A correct answer is 10; a correct method is 0.2 x 50.",
            assessed_skills=("Percent Of",),
        ),
        client=_Client(payload),
    )


def test_non_answer_forces_unknown_even_if_model_returns_incorrect():
    evaluator = _evaluator(
        {
            "response_type": "non_answer",
            "correctness": "incorrect",
            "confidence": 1.0,
        }
    )

    verdict = evaluator(
        _event("I don't understand."),
        _transcript("I don't understand."),
    )

    assert verdict.correctness == Correctness.UNKNOWN
    assert verdict.confidence == 0.0
    assert verdict.source == "gemini_structured_non_answer"


def test_help_request_is_non_answer_not_incorrect():
    evaluator = _evaluator(
        {
            "response_type": "non_answer",
            "correctness": "unknown",
            "confidence": 0.9,
        }
    )

    verdict = evaluator(
        _event("Can you explain that again?"),
        _transcript("Can you explain that again?"),
    )

    assert verdict.correctness == Correctness.UNKNOWN
    assert verdict.confidence == 0.0


def test_uncertain_answer_attempt_can_still_be_incorrect():
    evaluator = _evaluator(
        {
            "response_type": "answer_attempt",
            "correctness": "incorrect",
            "confidence": 0.85,
        }
    )

    verdict = evaluator(
        _event("I think maybe it is 15."),
        _transcript("I think maybe it is 15."),
    )

    assert verdict.correctness == Correctness.INCORRECT
    assert verdict.confidence == 0.85


def test_uncertain_answer_attempt_can_still_be_correct():
    evaluator = _evaluator(
        {
            "response_type": "answer_attempt",
            "correctness": "correct",
            "confidence": 0.88,
        }
    )

    verdict = evaluator(
        _event("I'm not sure, but I think it is 10."),
        _transcript("I'm not sure, but I think it is 10."),
    )

    assert verdict.correctness == Correctness.CORRECT
    assert verdict.confidence == 0.88


def test_missing_response_type_is_conservative_unknown():
    evaluator = _evaluator(
        {
            "correctness": "incorrect",
            "confidence": 1.0,
        }
    )

    verdict = evaluator(
        _event("I don't understand."),
        _transcript("I don't understand."),
    )

    assert verdict.correctness == Correctness.UNKNOWN
    assert verdict.confidence == 0.0
    assert verdict.source == (
        "gemini_structured_fallback_invalid_response_type"
    )
