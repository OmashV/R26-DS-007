"""
Generates personalised learning paths from a student's knowledge graph.

FR16 — generate a learning path with three categories:
         revise_urgently, learn_next, already_strong.
FR17 — regenerate the path after each completed session.
FR18 — expose the current path via the API.
UR1  — student receives a path indicating what to revise, learn next, and
        what is already mastered.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def generate_path(graph: dict[str, Any]) -> dict[str, Any]:
    """
    Build a personalised learning path from the current knowledge graph (FR16, FR17).

    Partitions graph concepts into three lists based on mastery_prob and
    the configured MASTERY_THRESHOLD:
      - already_strong: mastered concepts
      - revise_urgently: below threshold with declining trajectory
      - learn_next: below threshold, stable or improving, ordered by dependency

    Returns a path dict:
        {
            student_id: str,
            revise_urgently: [concept_name, ...],
            learn_next: [concept_name, ...],
            already_strong: [concept_name, ...]
        }
    """
    raise NotImplementedError


def prioritise_regressions(
    path: dict[str, Any],
    regressed_concepts: list[str],
) -> dict[str, Any]:
    """
    Move regressed concept_ids to the front of revise_urgently (FR15, FR16).

    regressed_concepts is the list returned by knowledge_graph.detect_regressions().
    Returns the updated path dict.
    """
    raise NotImplementedError
