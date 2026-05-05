"""
Learning path generation: turns a student's knowledge graph into a
prioritised plan of what to revise, learn next, and what they already know.

Implements:
    FR15 — Flags regressed concepts for prioritisation
    FR16 — Generates personalised learning path with three categories
    FR17 — Path regenerates after every session (called by process_session)
    FR18 — Path exposed via the API
"""

import logging
from typing import Optional

from config import (
    MASTERY_STRONG_THRESHOLD,
    MASTERY_WEAK_THRESHOLD,
    REGRESSION_DROP_THRESHOLD,
)

logger = logging.getLogger(__name__)


def is_regression(entry: dict) -> bool:
    """
    Returns True if a concept's mastery has dropped meaningfully from
    a previously stronger state.
    """
    prev = entry.get("previous_mastery_probability")
    if prev is None:
        return False
    drop = prev - entry["mastery_probability"]
    return (
        drop > REGRESSION_DROP_THRESHOLD
        and prev >= MASTERY_WEAK_THRESHOLD
    )


def generate_learning_path(graph: list[dict]) -> dict:
    """
    Bucket and prioritise a student's concepts into a learning path.

    Args:
        graph: List of mastery entries from KnowledgeGraph.get_student_graph().

    Returns:
        Dict with three lists: 'revise_urgently', 'learn_next', 'already_strong',
        plus a 'regressions' list (subset of revise_urgently flagged for context).
    """
    revise_urgently: list[dict] = []
    learn_next: list[dict] = []
    already_strong: list[dict] = []
    regressions: list[dict] = []

    for entry in graph:
        p = entry["mastery_probability"]
        regressed = is_regression(entry)

        annotated = {**entry, "is_regression": regressed}

        if regressed:
            regressions.append(annotated)
            revise_urgently.append(annotated)
        elif p < MASTERY_WEAK_THRESHOLD:
            revise_urgently.append(annotated)
        elif p >= MASTERY_STRONG_THRESHOLD:
            already_strong.append(annotated)
        else:
            learn_next.append(annotated)

    revise_urgently.sort(
        key=lambda e: (
            not e["is_regression"],
            e["mastery_probability"],
        )
    )
    learn_next.sort(key=lambda e: -e["mastery_probability"])
    already_strong.sort(key=lambda e: -e["mastery_probability"])

    return {
        "revise_urgently": revise_urgently,
        "learn_next": learn_next,
        "already_strong": already_strong,
        "regressions": regressions,
        "summary": {
            "revise_count": len(revise_urgently),
            "learn_next_count": len(learn_next),
            "strong_count": len(already_strong),
            "regression_count": len(regressions),
        },
    }


def format_path_text(path: dict, student_id: Optional[str] = None) -> str:
    """
    Human-readable formatting of a learning path for display/debugging.
    """
    lines: list[str] = []
    if student_id:
        lines.append(f"Learning path for student: {student_id}")
        lines.append("=" * 60)

    lines.append(f"\nREVISE URGENTLY ({len(path['revise_urgently'])}):")
    for e in path["revise_urgently"]:
        flag = "  ↓ REGRESSION" if e.get("is_regression") else ""
        lines.append(
            f"  - {e['skill']:<40} P={e['mastery_probability']:.3f}{flag}"
        )

    lines.append(f"\nLEARN NEXT ({len(path['learn_next'])}):")
    for e in path["learn_next"]:
        lines.append(
            f"  - {e['skill']:<40} P={e['mastery_probability']:.3f}"
        )

    lines.append(f"\nALREADY STRONG ({len(path['already_strong'])}):")
    for e in path["already_strong"]:
        lines.append(
            f"  - {e['skill']:<40} P={e['mastery_probability']:.3f}"
        )

    return "\n".join(lines)