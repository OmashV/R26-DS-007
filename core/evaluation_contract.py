from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Optional

from core.signal_resolver import Correctness


@dataclass(frozen=True)
class EvaluationVerdict:
    """
    Authoritative evaluator output for one extracted student event.

    Correctness comes from the evaluator/objective scoring layer, not from
    the ConceptExtractor. Confidence is evaluator/model confidence and is
    kept separate from the Signal Resolver's evidence weight.
    """

    correctness: Correctness
    confidence: float = 1.0
    source: str = "structured_evaluator"

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError(
                "EvaluationVerdict.confidence must be in [0, 1]"
            )

        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError(
                "EvaluationVerdict.source must be a non-empty string"
            )


EventEvaluator = Callable[
    [dict[str, Any], list[dict[str, Any]]],
    EvaluationVerdict | Mapping[str, Any] | None,
]


def _normalise_verdict(
    result: EvaluationVerdict | Mapping[str, Any] | None,
) -> EvaluationVerdict:
    """
    Normalize an evaluator adapter's result.

    `None` is deliberately conservative: correctness becomes UNKNOWN,
    which means the downstream Signal Resolver performs no BKT update.
    """
    if result is None:
        return EvaluationVerdict(
            correctness=Correctness.UNKNOWN,
            confidence=0.0,
            source="no_evaluator_verdict",
        )

    if isinstance(result, EvaluationVerdict):
        return result

    if not isinstance(result, Mapping):
        raise TypeError(
            "Evaluator must return EvaluationVerdict, mapping, or None"
        )

    raw_correctness = result.get("correctness", "unknown")

    try:
        correctness = Correctness(raw_correctness)
    except ValueError as exc:
        raise ValueError(
            f"Invalid evaluator correctness: {raw_correctness!r}"
        ) from exc

    raw_confidence = result.get("confidence", 1.0)
    confidence = float(raw_confidence)

    source = str(
        result.get("source", "structured_evaluator")
    ).strip()

    return EvaluationVerdict(
        correctness=correctness,
        confidence=confidence,
        source=source,
    )


def _canonical_extractor_event(
    event: Mapping[str, Any],
    transcript: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Validate and sanitize one ConceptExtractor event.

    The ConceptExtractor is authoritative for:
    - which student turn is an event;
    - which skill/concept the event belongs to;
    - optional evidence span.

    It is NOT authoritative for:
    - correctness;
    - evaluator confidence;
    - behavioural detector labels;
    - repeated misunderstanding;
    - legacy flat signal labels.

    Those fields are intentionally not copied into the canonical event.
    """
    turn_index = event.get("turn_index")
    skill = event.get("skill")

    if not isinstance(turn_index, int):
        raise ValueError(
            f"Invalid extractor turn_index: {turn_index!r}"
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
            f"Invalid extractor skill: {skill!r}"
        )

    transcript_text = str(turn.get("text", "")).strip()

    student_text = str(
        event.get("student_text") or transcript_text
    ).strip()

    if not student_text:
        raise ValueError(
            f"Student text is empty at turn_index {turn_index}"
        )

    evidence_span = str(
        event.get("evidence_span") or student_text
    ).strip()

    return {
        "turn_index": turn_index,
        "skill": skill.strip(),
        "evidence_span": evidence_span,
        "student_text": student_text,
    }


def apply_evaluator_to_extraction(
    *,
    extraction: Mapping[str, Any],
    transcript: list[dict[str, Any]],
    evaluator: EventEvaluator,
) -> dict[str, Any]:
    """
    Convert ConceptExtractor output into the canonical event contract used
    by `core.pipeline_adapter`.

    Flow:

        ConceptExtractor
            -> turn_index / skill / evidence
        Structured evaluator
            -> correctness / evaluator confidence
        Result
            -> pipeline_adapter -> Signal Resolver

    Any correctness emitted by the ConceptExtractor is ignored. This
    prevents an unconstrained extraction LLM from becoming the authoritative
    BKT correctness source.

    Returned event format:

        {
            "turn_index": int,
            "skill": str,
            "correctness": "correct|partial|incorrect|unknown",
            "evaluator_confidence": float,
            "evaluator_source": str,
            "evidence_span": str,
            "student_text": str,
        }
    """
    if not isinstance(extraction, Mapping):
        raise TypeError("extraction must be a mapping")

    raw_events = extraction.get("events", [])

    if not isinstance(raw_events, list):
        raise ValueError("extraction['events'] must be a list")

    canonical_events: list[dict[str, Any]] = []

    for raw_event in raw_events:
        if not isinstance(raw_event, Mapping):
            raise ValueError(
                "Each extraction event must be a mapping"
            )

        canonical = _canonical_extractor_event(
            raw_event,
            transcript,
        )

        # The evaluator receives ONLY sanitized structural evidence.
        # In particular, any LLM-emitted correctness is excluded.
        verdict = _normalise_verdict(
            evaluator(dict(canonical), transcript)
        )

        canonical_events.append(
            {
                **canonical,
                "correctness": verdict.correctness.value,
                "evaluator_confidence": float(
                    verdict.confidence
                ),
                "evaluator_source": verdict.source,
            }
        )

    raw_misconceptions = extraction.get("misconceptions", [])

    misconceptions: list[str] = []

    if isinstance(raw_misconceptions, list):
        misconceptions = [
            str(item).strip()
            for item in raw_misconceptions
            if str(item).strip()
        ]

    return {
        "events": canonical_events,
        "misconceptions": misconceptions,
    }
