"""Pure presentation helpers for deterministic learning-path output.

These helpers format planner results but never rank, persist, or recalculate
them. They are kept outside Streamlit so formatting can be tested without
executing the application module.
"""

from __future__ import annotations

from typing import Any


SECTION_KEYS: tuple[tuple[str, str], ...] = (
    ("Recommended next", "recommended_order"),
    ("Revise urgently", "revise_urgently"),
    ("Blocked", "blocked"),
    ("Unseen / not started", "unseen"),
    ("Already strong", "already_strong"),
)


def format_mastery_probability(value: Any) -> str:
    """Format stored mastery without turning an unseen value into zero."""
    if value is None:
        return "unseen"
    return f"{float(value):.3f}"


def present_learning_path_entry(entry: dict) -> dict:
    """Return display-safe fields for one planner entry without mutation."""
    return {
        "skill": str(entry.get("skill", "")),
        "mastery_status": str(
            entry.get("mastery_status")
            or entry.get("mastery_label")
            or "unknown"
        ),
        "mastery_probability": format_mastery_probability(
            entry.get("mastery_probability")
        ),
        "planning_status": str(
            entry.get("planning_status") or "unknown"
        ),
        "reason": str(entry.get("reason") or "No reason supplied."),
        "priority_reason": str(
            entry.get("priority_reason") or "No priority reason supplied."
        ),
        "unmet_prerequisites": tuple(
            str(skill)
            for skill in entry.get("unmet_prerequisites", [])
        ),
    }


def learning_path_sections(path: dict) -> tuple[tuple[str, list[dict]], ...]:
    """Expose planner lists in UI order without changing item ordering."""
    return tuple(
        (title, list(path.get(key, [])))
        for title, key in SECTION_KEYS
    )
