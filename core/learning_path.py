"""
Learning path generation: turns a student's knowledge graph into a
prioritised plan of what to revise, learn next, and what they already know.

Implements:
    FR15 — Flags regressed concepts for prioritisation
    FR16 — Generates personalised learning path with three categories
    FR17 — CrossSessionStudentModelPipeline regenerates the path after resolved
           session evidence has updated mastery
    FR18 — Path exposed via the API
"""

import logging
from typing import Optional

from config import (
    MASTERY_STRONG_THRESHOLD,
    MASTERY_WEAK_THRESHOLD,
    REGRESSION_DROP_THRESHOLD,
)
from core.curriculum import Curriculum

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


def _generate_legacy_learning_path(graph: list[dict]) -> dict:
    """Preserve the original mastery-only planner for existing callers."""
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


def _mastery_status(probability: Optional[float]) -> str:
    if probability is None:
        return "unseen"
    if probability < MASTERY_WEAK_THRESHOLD:
        return "weak"
    if probability >= MASTERY_STRONG_THRESHOLD:
        return "strong"
    return "partial"


def _mastery_reason(
    status: str,
    probability: Optional[float],
    previous: Optional[float],
    regressed: bool,
) -> str:
    if status == "unseen":
        reason = "No mastery evidence has been recorded."
    elif status == "weak":
        reason = (
            f"Mastery {probability:.3f} is below the weak threshold "
            f"{MASTERY_WEAK_THRESHOLD:.3f}."
        )
    elif status == "strong":
        reason = (
            f"Mastery {probability:.3f} meets the strong threshold "
            f"{MASTERY_STRONG_THRESHOLD:.3f}."
        )
    else:
        reason = (
            f"Mastery {probability:.3f} is between the weak and strong "
            "thresholds."
        )

    if regressed and previous is not None and probability is not None:
        reason += (
            f" Regression detected: mastery dropped from {previous:.3f} "
            f"to {probability:.3f}."
        )
    return reason


def _generate_curriculum_learning_path(
    graph: list[dict],
    curriculum: Curriculum,
) -> dict:
    graph_by_skill: dict[str, dict] = {}
    for source in graph:
        skill = source["skill"]
        if skill in graph_by_skill:
            raise ValueError(f"graph contains duplicate mastery entries for '{skill}'")
        graph_by_skill[skill] = source

    order = {skill: index for index, skill in enumerate(curriculum.skills)}
    entries: dict[str, dict] = {}

    # First classify mastery independently of prerequisites so every dependency
    # can be evaluated against the same complete state map.
    for skill in curriculum.skills:
        source = graph_by_skill.get(skill)
        probability = None if source is None else source["mastery_probability"]
        status = _mastery_status(probability)
        regressed = source is not None and is_regression(source)
        previous = (
            None if source is None else source.get("previous_mastery_probability")
        )
        base = dict(source) if source is not None else {
            "skill": skill,
            "mastery_probability": None,
            "previous_mastery_probability": None,
        }
        entries[skill] = {
            **base,
            "mastery_status": status,
            "planning_status": "",
            "is_regression": regressed,
            "prerequisites": list(curriculum.prerequisites_for(skill)),
            "unmet_prerequisites": [],
            "reason": _mastery_reason(
                status,
                probability,
                previous,
                regressed,
            ),
            "priority_reason": "",
        }

    # A prerequisite is satisfied only by current strong mastery. Unseen,
    # weak, and partial states all block their direct dependent.
    for skill in curriculum.skills:
        entry = entries[skill]
        unmet = [
            dependency
            for dependency in curriculum.prerequisites_for(skill)
            if entries[dependency]["mastery_status"] != "strong"
        ]
        entry["unmet_prerequisites"] = unmet

        if entry["mastery_status"] == "strong":
            entry["planning_status"] = (
                "revise_urgently"
                if entry["is_regression"]
                else "already_strong"
            )
        elif unmet:
            entry["planning_status"] = "blocked"
        elif entry["is_regression"] or entry["mastery_status"] == "weak":
            entry["planning_status"] = "revise_urgently"
        else:
            entry["planning_status"] = "ready_to_learn"

    def earliest_actionable_prerequisites(skill: str) -> tuple[str, ...]:
        earliest: set[str] = set()

        def descend(candidate: str) -> None:
            candidate_entry = entries[candidate]
            if candidate_entry["planning_status"] != "blocked":
                earliest.add(candidate)
                return
            for dependency in candidate_entry["unmet_prerequisites"]:
                descend(dependency)

        for dependency in entries[skill]["unmet_prerequisites"]:
            descend(dependency)
        return tuple(sorted(earliest, key=lambda item: order[item]))

    # Track which blocked downstream skills each currently actionable
    # prerequisite unlocks. This makes prerequisite priority deterministic and
    # explainable without recommending the blocked target itself.
    unlocks: dict[str, list[str]] = {skill: [] for skill in curriculum.skills}
    earliest_for_blocked: dict[str, tuple[str, ...]] = {}
    for skill in curriculum.skills:
        if entries[skill]["planning_status"] != "blocked":
            continue
        earliest = earliest_actionable_prerequisites(skill)
        earliest_for_blocked[skill] = earliest
        for prerequisite in earliest:
            unlocks[prerequisite].append(skill)

    for skill in curriculum.skills:
        entry = entries[skill]
        planning_status = entry["planning_status"]

        if planning_status == "blocked":
            blocker_details = ", ".join(
                f"{dependency} ({entries[dependency]['mastery_status']})"
                for dependency in entry["unmet_prerequisites"]
            )
            earliest = earliest_for_blocked[skill]
            earliest_text = ", ".join(earliest) if earliest else "none"
            prefix = "Regression detected, but this skill is blocked. " if entry[
                "is_regression"
            ] else ""
            entry["priority_reason"] = (
                prefix
                + f"Blocked by {blocker_details}. Earliest actionable "
                f"prerequisite(s): {earliest_text}."
            )
        elif planning_status == "already_strong":
            entry["priority_reason"] = (
                "Already strong; no immediate learning-path action is needed."
            )
        elif entry["is_regression"]:
            entry["priority_reason"] = (
                "Regression takes priority and all prerequisites are satisfied."
            )
        elif unlocks[skill]:
            downstream = ", ".join(unlocks[skill])
            entry["priority_reason"] = (
                f"Earliest actionable prerequisite for {len(unlocks[skill])} "
                f"blocked downstream skill(s): {downstream}."
            )
        elif planning_status == "revise_urgently":
            entry["priority_reason"] = (
                "Weak mastery and all prerequisites are satisfied."
            )
        elif entry["mastery_status"] == "unseen":
            entry["priority_reason"] = (
                "Unseen skill is ready because all prerequisites are strong."
            )
        else:
            entry["priority_reason"] = (
                "Partial mastery is ready to progress because all prerequisites "
                "are strong."
            )

    def recommendation_key(entry: dict) -> tuple:
        skill = entry["skill"]
        if entry["is_regression"]:
            group = 0
        elif unlocks[skill]:
            group = 1
        elif entry["mastery_status"] == "weak":
            group = 2
        elif entry["mastery_status"] == "partial":
            group = 3
        else:
            group = 4
        return (
            group,
            -len(unlocks[skill]),
            order[skill],
            skill,
        )

    recommended_order = sorted(
        (
            entry
            for entry in entries.values()
            if entry["planning_status"] in {
                "revise_urgently",
                "ready_to_learn",
            }
        ),
        key=recommendation_key,
    )
    revise_urgently = [
        entry
        for entry in recommended_order
        if entry["planning_status"] == "revise_urgently"
    ]
    learn_next = [
        entry
        for entry in recommended_order
        if entry["planning_status"] == "ready_to_learn"
    ]
    already_strong = [
        entries[skill]
        for skill in curriculum.skills
        if entries[skill]["planning_status"] == "already_strong"
    ]
    blocked = [
        entries[skill]
        for skill in curriculum.skills
        if entries[skill]["planning_status"] == "blocked"
    ]
    unseen = [
        entries[skill]
        for skill in curriculum.skills
        if entries[skill]["mastery_status"] == "unseen"
    ]
    regressions = sorted(
        (entry for entry in entries.values() if entry["is_regression"]),
        key=lambda entry: (
            -(
                entry["previous_mastery_probability"]
                - entry["mastery_probability"]
            ),
            order[entry["skill"]],
        ),
    )

    return {
        "revise_urgently": revise_urgently,
        "learn_next": learn_next,
        "already_strong": already_strong,
        "regressions": regressions,
        "blocked": blocked,
        "unseen": unseen,
        "recommended_order": recommended_order,
        "summary": {
            "revise_count": len(revise_urgently),
            "learn_next_count": len(learn_next),
            "strong_count": len(already_strong),
            "regression_count": len(regressions),
            "blocked_count": len(blocked),
            "unseen_count": len(unseen),
            "recommended_count": len(recommended_order),
        },
    }


def generate_learning_path(
    graph: list[dict],
    *,
    curriculum: Optional[Curriculum] = None,
) -> dict:
    """
    Bucket and prioritise a student's concepts into a learning path.

    Args:
        graph: List of mastery entries from KnowledgeGraph.get_student_graph().
        curriculum: Optional explicit prerequisite curriculum. If omitted, the
                    original mastery-only bucketing behaviour is preserved.

    Returns:
        Dict with the original response keys. Curriculum-aware results also
        include 'blocked', 'unseen', and 'recommended_order'.
    """
    if curriculum is None:
        return _generate_legacy_learning_path(graph)
    return _generate_curriculum_learning_path(graph, curriculum)


def format_path_text(path: dict, student_id: Optional[str] = None) -> str:
    """
    Human-readable formatting of a learning path for display/debugging.
    """
    lines: list[str] = []
    if student_id:
        lines.append(f"Learning path for student: {student_id}")
        lines.append("=" * 60)

    def format_probability(entry: dict) -> str:
        probability = entry.get("mastery_probability")
        return "unseen" if probability is None else f"P={probability:.3f}"

    lines.append(f"\nREVISE URGENTLY ({len(path['revise_urgently'])}):")
    for e in path["revise_urgently"]:
        flag = "  ↓ REGRESSION" if e.get("is_regression") else ""
        lines.append(
            f"  - {e['skill']:<40} {format_probability(e)}{flag}"
        )

    lines.append(f"\nLEARN NEXT ({len(path['learn_next'])}):")
    for e in path["learn_next"]:
        lines.append(
            f"  - {e['skill']:<40} {format_probability(e)}"
        )

    lines.append(f"\nALREADY STRONG ({len(path['already_strong'])}):")
    for e in path["already_strong"]:
        lines.append(
            f"  - {e['skill']:<40} {format_probability(e)}"
        )

    if "blocked" in path:
        lines.append(f"\nBLOCKED ({len(path['blocked'])}):")
        for e in path["blocked"]:
            blockers = ", ".join(e["unmet_prerequisites"])
            lines.append(
                f"  - {e['skill']:<40} {format_probability(e)} "
                f"blocked by: {blockers}"
            )

    return "\n".join(lines)
