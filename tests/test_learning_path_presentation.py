"""Pure presentation tests for Streamlit learning-path rendering helpers."""

from copy import deepcopy

from core.learning_path_presentation import (
    format_mastery_probability,
    learning_path_sections,
    present_learning_path_entry,
)


def test_unseen_mastery_is_rendered_as_unseen_not_zero():
    entry = {
        "skill": "Unseen Skill",
        "mastery_probability": None,
        "mastery_status": "unseen",
        "planning_status": "ready_to_learn",
        "reason": "No mastery evidence has been recorded.",
        "priority_reason": "All prerequisites are satisfied.",
        "unmet_prerequisites": [],
    }

    presented = present_learning_path_entry(entry)

    assert format_mastery_probability(None) == "unseen"
    assert presented["mastery_probability"] == "unseen"
    assert "0.0" not in presented["mastery_probability"]


def test_presentation_preserves_recommended_order_and_does_not_mutate_path():
    path = {
        "recommended_order": [
            {"skill": "Second in alphabet", "mastery_probability": 0.20},
            {"skill": "First in alphabet", "mastery_probability": 0.40},
        ],
        "revise_urgently": [],
        "blocked": [],
        "unseen": [],
        "already_strong": [],
    }
    original = deepcopy(path)

    sections = learning_path_sections(path)

    assert sections[0][0] == "Recommended next"
    assert [entry["skill"] for entry in sections[0][1]] == [
        "Second in alphabet",
        "First in alphabet",
    ]
    assert path == original
