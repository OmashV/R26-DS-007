from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from typing import Any, Callable, Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types

from core.evaluation_contract import EvaluationVerdict
from core.signal_resolver import Correctness


GEMINI_MODEL = "gemini-2.5-flash"


ObjectiveComparator = Callable[
    [str, str],
    Optional[EvaluationVerdict],
]


def _normalise_text(value: str) -> str:
    return " ".join(str(value).strip().lower().split())


def _extract_last_number(text: str) -> Optional[float]:
    """
    Conservative numeric extraction for simple objective-answer matching.

    Returns the final standalone number in the student's response, supporting
    optional sign, decimals, and percentages. Percent signs are ignored for
    comparison because the reference answer is expected to use the same
    semantic unit supplied by the host problem/rubric.
    """
    matches = re.findall(
        r"(?<!\w)[-+]?(?:\d+(?:\.\d+)?|\.\d+)%?(?!\w)",
        str(text),
    )
    if not matches:
        return None

    raw = matches[-1].rstrip("%")

    try:
        value = float(raw)
    except ValueError:
        return None

    if not math.isfinite(value):
        return None

    return value


def exact_or_numeric_objective_comparator(
    student_text: str,
    reference_answer: str,
) -> Optional[EvaluationVerdict]:
    """
    Objective-first comparator for simple answers.

    - Exact normalized string match => CORRECT.
    - If both sides expose a numeric final answer:
        equal within tight tolerance => CORRECT
        otherwise => INCORRECT.
    - Otherwise => None, allowing a structured evaluator fallback.

    This function intentionally does not invent PARTIAL correctness.
    """
    student_norm = _normalise_text(student_text)
    reference_norm = _normalise_text(reference_answer)

    if student_norm and student_norm == reference_norm:
        return EvaluationVerdict(
            correctness=Correctness.CORRECT,
            confidence=1.0,
            source="objective_exact_match",
        )

    student_number = _extract_last_number(student_text)
    reference_number = _extract_last_number(reference_answer)

    if student_number is None or reference_number is None:
        return None

    if math.isclose(
        student_number,
        reference_number,
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        return EvaluationVerdict(
            correctness=Correctness.CORRECT,
            confidence=1.0,
            source="objective_numeric_match",
        )

    return EvaluationVerdict(
        correctness=Correctness.INCORRECT,
        confidence=1.0,
        source="objective_numeric_mismatch",
    )


@dataclass(frozen=True)
class EvaluatorContext:
    """
    Problem-specific information bound into the evaluator.

    `reference_answer` or `rubric` should be supplied whenever possible.
    If neither is available, the evaluator returns UNKNOWN rather than
    turning unconstrained LLM inference into BKT correctness.
    """

    problem: str
    reference_answer: Optional[str] = None
    rubric: Optional[str] = None
    assessed_skills: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.problem, str) or not self.problem.strip():
            raise ValueError("problem must be a non-empty string")

        if (
            self.reference_answer is not None
            and not str(self.reference_answer).strip()
        ):
            raise ValueError(
                "reference_answer must be non-empty when supplied"
            )

        if self.rubric is not None and not str(self.rubric).strip():
            raise ValueError(
                "rubric must be non-empty when supplied"
            )

        for skill in self.assessed_skills:
            if not isinstance(skill, str) or not skill.strip():
                raise ValueError(
                    "assessed_skills must contain non-empty strings"
                )


class StudentAnswerEvaluator:
    """
    Objective-first student correctness evaluator.

    Resolution order:

    1. Run an injected/objective comparator when a reference answer exists.
    2. If the objective comparator resolves the answer, return that verdict.
    3. If objective scoring cannot resolve it and a rubric/reference answer
       exists, use Gemini only as a structured fallback.
    4. If there is no reference answer and no rubric, return UNKNOWN.

    The class implements the existing EventEvaluator callable contract:

        evaluator(event, transcript) -> EvaluationVerdict

    so it can be passed directly to apply_evaluator_to_extraction().
    """

    def __init__(
        self,
        context: EvaluatorContext,
        api_key: Optional[str] = None,
        model: str = GEMINI_MODEL,
        objective_comparator: Optional[
            ObjectiveComparator
        ] = exact_or_numeric_objective_comparator,
        client: Any = None,
    ) -> None:
        self.context = context
        self.model = model
        self.objective_comparator = objective_comparator

        # A client can be injected for tests, avoiding any network call.
        if client is not None:
            self.client = client
            return

        load_dotenv()

        key = api_key or os.getenv("GEMINI_API_KEY")

        # Do not require Gemini if the configured objective comparator can
        # resolve the answer. Client creation is therefore lazy.
        self.client = (
            genai.Client(api_key=key)
            if key
            else None
        )

    def __call__(
        self,
        event: dict[str, Any],
        transcript: list[dict[str, Any]],
    ) -> EvaluationVerdict:
        event_skill = str(
            event.get("skill", "")
        ).strip()

        if (
            self.context.assessed_skills
            and event_skill not in self.context.assessed_skills
        ):
            return EvaluationVerdict(
                correctness=Correctness.UNKNOWN,
                confidence=0.0,
                source="skill_not_authorized_for_assessment",
            )

        student_text = str(
            event.get("student_text", "")
        ).strip()

        if not student_text:
            return EvaluationVerdict(
                correctness=Correctness.UNKNOWN,
                confidence=0.0,
                source="empty_student_text",
            )

        reference_answer = self.context.reference_answer

        if (
            reference_answer is not None
            and self.objective_comparator is not None
        ):
            verdict = self.objective_comparator(
                student_text,
                reference_answer,
            )

            if verdict is not None:
                return verdict

        # No authoritative scoring context -> no BKT correctness.
        if (
            self.context.reference_answer is None
            and self.context.rubric is None
        ):
            return EvaluationVerdict(
                correctness=Correctness.UNKNOWN,
                confidence=0.0,
                source="missing_evaluation_context",
            )

        if self.client is None:
            return EvaluationVerdict(
                correctness=Correctness.UNKNOWN,
                confidence=0.0,
                source="structured_fallback_unavailable",
            )

        return self._gemini_fallback(
            event=event,
            transcript=transcript,
        )

    def _gemini_fallback(
        self,
        *,
        event: dict[str, Any],
        transcript: list[dict[str, Any]],
    ) -> EvaluationVerdict:
        history = "\n".join(
            f"[{index}] {str(turn.get('role', '')).upper()}: "
            f"{str(turn.get('text', '')).strip()}"
            for index, turn in enumerate(transcript)
        )

        prompt = f"""
You are evaluating ONE student's mathematical response for mastery tracking.

Classify only the student's correctness on the named skill.

Allowed correctness labels:
- correct
- partial
- incorrect
- unknown

Use "partial" only when the response contains meaningful correct progress but
is incomplete or also contains a substantive error.
Use "unknown" when the supplied problem/reference/rubric is insufficient to
judge the response reliably.

Do NOT classify uncertainty, help-seeking, clarification requests, tone,
reasoning quality, or tutor quality as correctness.

Problem:
{self.context.problem}

Reference answer:
{self.context.reference_answer or "[not supplied]"}

Rubric:
{self.context.rubric or "[not supplied]"}

Target skill:
{event.get("skill", "")}

Target student response:
{event.get("student_text", "")}

Evidence span:
{event.get("evidence_span", "")}

Transcript:
{history}

Return JSON only:
{{
  "correctness": "correct|partial|incorrect|unknown",
  "confidence": 0.0
}}
""".strip()

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.0,
            ),
        )

        raw_text = getattr(response, "text", "") or ""

        try:
            payload = json.loads(raw_text)
        except json.JSONDecodeError:
            return EvaluationVerdict(
                correctness=Correctness.UNKNOWN,
                confidence=0.0,
                source="gemini_structured_fallback_invalid_json",
            )

        try:
            correctness = Correctness(
                payload.get("correctness", "unknown")
            )
        except ValueError:
            correctness = Correctness.UNKNOWN

        try:
            confidence = float(
                payload.get("confidence", 0.0)
            )
        except (TypeError, ValueError):
            confidence = 0.0

        confidence = min(1.0, max(0.0, confidence))

        if correctness == Correctness.UNKNOWN:
            confidence = 0.0

        return EvaluationVerdict(
            correctness=correctness,
            confidence=confidence,
            source="gemini_structured_fallback",
        )
