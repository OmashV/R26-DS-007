from __future__ import annotations

from typing import Callable, Optional

from core.signal_resolver import (
    Correctness,
    ResolverInput,
    ResolvedSignal,
    resolve_signal,
)


ReasoningPredictor = Callable[[str, Optional[str]], float]
TextPredictor = Callable[[str], float]
HistoryGetter = Callable[[str, str], list[tuple[int, float]]]
EvaluatorConfidenceGetter = Callable[[dict], float]


def _previous_student_text(
    transcript: list[dict],
    turn_index: int,
) -> Optional[str]:
    """
    Return the most recent student utterance before turn_index.
    """
    for i in range(turn_index - 1, -1, -1):
        turn = transcript[i]
        if turn.get("role") == "student":
            text = str(turn.get("text", "")).strip()
            if text:
                return text

    return None


def _history_to_correctness(
    attempts: list[tuple[int, float]],
    limit: int = 5,
) -> list[Correctness]:
    """
    Convert KnowledgeGraph attempt history into the correctness values
    needed by the Signal Resolver.

    Confidence is intentionally ignored here because repeated
    misunderstanding only depends on consecutive incorrect outcomes.
    """
    recent = attempts[-limit:] if limit > 0 else attempts

    return [
        Correctness.CORRECT
        if correct == 1
        else Correctness.INCORRECT
        for correct, _confidence in recent
    ]


def _event_id(
    session_id: str,
    turn_index: int,
    skill: str,
) -> str:
    """
    Build a deterministic ID for one student-turn x skill event.
    """
    safe_skill = (
        skill.strip()
        .lower()
        .replace(" ", "_")
        .replace("/", "_")
    )

    return f"{session_id}:turn_{turn_index}:{safe_skill}"


def build_resolver_inputs(
    *,
    extraction: dict,
    transcript: list[dict],
    student_id: str,
    session_id: str,
    reasoning_predictor: Optional[ReasoningPredictor] = None,
    uncertainty_predictor: Optional[TextPredictor] = None,
    clarification_predictor: Optional[TextPredictor] = None,
    history_getter: Optional[HistoryGetter] = None,
    evaluator_confidence_getter: Optional[
        EvaluatorConfidenceGetter
    ] = None,
    default_evaluator_confidence: float = 1.0,
    history_limit: int = 5,
) -> list[ResolverInput]:
    """
    Convert ConceptExtractor event output into ResolverInput objects.

    Expected extraction format:

    {
        "events": [
            {
                "turn_index": 3,
                "skill": "Percent Of",
                "correctness": "correct",
                "student_text": "...",
                "evidence_span": "..."
            }
        ]
    }

    Detector responsibilities:
    - reasoning_predictor(current_text, previous_student_text)
    - uncertainty_predictor(current_text)
    - clarification_predictor(current_text)

    History responsibilities:
    - history_getter(student_id, skill) should return the same
      (correct, confidence) tuples as KnowledgeGraph.get_attempts().

    Notes:
    - Detector probabilities are passed to the Signal Resolver. They never
      create independent duplicate BKT observations, but uncertainty and
      clarification may contribute to one weak behavioural proxy when
      correctness is UNKNOWN and the skill context is authorized.
    - evaluator confidence is kept separate from evidence weight.
    """
    if not 0.0 <= default_evaluator_confidence <= 1.0:
        raise ValueError(
            "default_evaluator_confidence must be in [0, 1]"
        )

    resolver_inputs: list[ResolverInput] = []

    for event in extraction.get("events", []):
        if not isinstance(event, dict):
            continue

        turn_index = event.get("turn_index")
        skill = event.get("skill")
        correctness = event.get("correctness")

        if not isinstance(turn_index, int):
            raise ValueError(
                f"Invalid event turn_index: {turn_index!r}"
            )

        if turn_index < 0 or turn_index >= len(transcript):
            raise ValueError(
                f"turn_index {turn_index} is outside transcript"
            )

        turn = transcript[turn_index]

        if turn.get("role") != "student":
            raise ValueError(
                f"turn_index {turn_index} is not a student turn"
            )

        if not isinstance(skill, str) or not skill.strip():
            raise ValueError(
                f"Invalid event skill: {skill!r}"
            )

        try:
            correctness_enum = Correctness(correctness)
        except ValueError as exc:
            raise ValueError(
                f"Invalid correctness: {correctness!r}"
            ) from exc

        student_text = str(
            event.get("student_text")
            or turn.get("text", "")
        ).strip()

        previous_student_text = _previous_student_text(
            transcript,
            turn_index,
        )

        reasoning_probability = (
            reasoning_predictor(
                student_text,
                previous_student_text,
            )
            if reasoning_predictor is not None
            else None
        )

        uncertainty_probability = (
            uncertainty_predictor(student_text)
            if uncertainty_predictor is not None
            else None
        )

        clarification_probability = (
            clarification_predictor(student_text)
            if clarification_predictor is not None
            else None
        )

        recent_same_skill_outcomes: list[Correctness] = []

        if history_getter is not None:
            attempts = history_getter(
                student_id,
                skill,
            )

            recent_same_skill_outcomes = (
                _history_to_correctness(
                    attempts,
                    limit=history_limit,
                )
            )

        if evaluator_confidence_getter is not None:
            evaluator_confidence = float(
                evaluator_confidence_getter(event)
            )
        else:
            evaluator_confidence = float(
                event.get(
                    "evaluator_confidence",
                    default_evaluator_confidence,
                )
            )

        if not 0.0 <= evaluator_confidence <= 1.0:
            raise ValueError(
                "evaluator_confidence must be in [0, 1]"
            )

        evaluator_source = str(
            event.get("evaluator_source", "")
        ).strip()
        allow_behavioural_proxy = (
            evaluator_source
            != "skill_not_authorized_for_assessment"
        )

        resolver_inputs.append(
            ResolverInput(
                event_id=_event_id(
                    session_id,
                    turn_index,
                    skill,
                ),
                student_id=student_id,
                skill_id=skill,
                correctness=correctness_enum,
                evaluator_confidence=evaluator_confidence,
                allow_behavioural_proxy=allow_behavioural_proxy,
                reasoning_probability=reasoning_probability,
                uncertainty_probability=uncertainty_probability,
                clarification_probability=clarification_probability,
                recent_same_skill_outcomes=(
                    recent_same_skill_outcomes
                ),
            )
        )

    return resolver_inputs


def resolve_extracted_events(
    *,
    extraction: dict,
    transcript: list[dict],
    student_id: str,
    session_id: str,
    reasoning_predictor: Optional[ReasoningPredictor] = None,
    uncertainty_predictor: Optional[TextPredictor] = None,
    clarification_predictor: Optional[TextPredictor] = None,
    history_getter: Optional[HistoryGetter] = None,
    evaluator_confidence_getter: Optional[
        EvaluatorConfidenceGetter
    ] = None,
    default_evaluator_confidence: float = 1.0,
    history_limit: int = 5,
) -> list[ResolvedSignal]:
    """
    Convenience wrapper:
        ConceptExtractor output -> ResolverInput -> ResolvedSignal
    """
    resolver_inputs = build_resolver_inputs(
        extraction=extraction,
        transcript=transcript,
        student_id=student_id,
        session_id=session_id,
        reasoning_predictor=reasoning_predictor,
        uncertainty_predictor=uncertainty_predictor,
        clarification_predictor=clarification_predictor,
        history_getter=history_getter,
        evaluator_confidence_getter=(
            evaluator_confidence_getter
        ),
        default_evaluator_confidence=(
            default_evaluator_confidence
        ),
        history_limit=history_limit,
    )

    return [
        resolve_signal(event)
        for event in resolver_inputs
    ]
