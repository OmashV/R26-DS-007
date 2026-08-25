"""Deterministic tests for legacy and prerequisite-aware learning paths."""

import json

import pytest

from core.curriculum import Curriculum, load_curriculum
from core.learning_path import generate_learning_path


def make_curriculum(records: list[tuple[str, list[str]]]) -> Curriculum:
    data = {
        "version": 1,
        "skills": [
            {"skill": skill, "prerequisites": prerequisites}
            for skill, prerequisites in records
        ],
    }
    return Curriculum.from_dict(
        data,
        bkt_skills=[skill for skill, _ in records],
    )


def mastery(
    skill: str,
    probability: float,
    previous: float | None = None,
    **metadata,
) -> dict:
    return {
        "skill": skill,
        "mastery_probability": probability,
        "previous_mastery_probability": previous,
        **metadata,
    }


def skill_names(entries: list[dict]) -> list[str]:
    return [entry["skill"] for entry in entries]


def test_no_curriculum_preserves_legacy_response_and_ranking():
    graph = [
        mastery("weak", 0.10),
        mastery("partial", 0.50),
        mastery("strong", 0.80),
        mastery("regressed", 0.40, previous=0.70),
    ]

    result = generate_learning_path(graph)

    assert set(result) == {
        "revise_urgently",
        "learn_next",
        "already_strong",
        "regressions",
        "summary",
    }
    assert skill_names(result["revise_urgently"]) == ["regressed", "weak"]
    assert skill_names(result["learn_next"]) == ["partial"]
    assert skill_names(result["already_strong"]) == ["strong"]
    assert skill_names(result["regressions"]) == ["regressed"]
    assert result["summary"] == {
        "revise_count": 2,
        "learn_next_count": 1,
        "strong_count": 1,
        "regression_count": 1,
    }


def test_default_versioned_curriculum_artifact_loads_against_bkt_universe():
    curriculum = load_curriculum()

    assert curriculum.skills[0] == "Addition Whole Numbers"
    assert curriculum.prerequisites_for("Percent Of") == (
        "Percents",
        "Multiplication Fractions",
    )
    assert len(curriculum.skills) == len(set(curriculum.skills))


def test_loader_preserves_explicit_curriculum_order(tmp_path):
    curriculum_path = tmp_path / "curriculum.json"
    bkt_path = tmp_path / "bkt.json"
    curriculum_path.write_text(
        json.dumps(
            {
                "version": 1,
                "skills": [
                    {"skill": "Foundation", "prerequisites": []},
                    {
                        "skill": "Target",
                        "prerequisites": ["Foundation"],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    bkt_path.write_text(
        json.dumps({"Target": {}, "Foundation": {}}),
        encoding="utf-8",
    )

    curriculum = load_curriculum(
        curriculum_path,
        bkt_params_path=bkt_path,
    )

    assert curriculum.skills == ("Foundation", "Target")
    assert curriculum.order_index("Foundation") == 0
    assert curriculum.order_index("Target") == 1


def test_curriculum_rejects_duplicate_skills():
    data = {
        "version": 1,
        "skills": [
            {"skill": "A", "prerequisites": []},
            {"skill": "A", "prerequisites": []},
        ],
    }

    with pytest.raises(ValueError, match="skills must be unique"):
        Curriculum.from_dict(data, bkt_skills=["A"])


def test_curriculum_rejects_unknown_dependency():
    data = {
        "version": 1,
        "skills": [
            {"skill": "A", "prerequisites": ["Missing"]},
        ],
    }

    with pytest.raises(ValueError, match="reference unknown skills: Missing"):
        Curriculum.from_dict(data, bkt_skills=["A"])


def test_curriculum_rejects_self_dependency():
    data = {
        "version": 1,
        "skills": [
            {"skill": "A", "prerequisites": ["A"]},
        ],
    }

    with pytest.raises(ValueError, match="cannot be its own prerequisite"):
        Curriculum.from_dict(data, bkt_skills=["A"])


def test_curriculum_rejects_cycles_with_deterministic_path():
    data = {
        "version": 1,
        "skills": [
            {"skill": "A", "prerequisites": ["B"]},
            {"skill": "B", "prerequisites": ["C"]},
            {"skill": "C", "prerequisites": ["A"]},
        ],
    }

    with pytest.raises(ValueError, match=r"A -> B -> C -> A"):
        Curriculum.from_dict(data, bkt_skills=["A", "B", "C"])


def test_curriculum_rejects_skills_outside_bkt_universe():
    data = {
        "version": 1,
        "skills": [
            {"skill": "A", "prerequisites": []},
            {"skill": "Not Trained", "prerequisites": []},
        ],
    }

    with pytest.raises(ValueError, match="missing from the BKT skill universe"):
        Curriculum.from_dict(data, bkt_skills=["A"])


def test_weak_earliest_prerequisite_precedes_blocked_descendants():
    curriculum = make_curriculum(
        [
            ("A", []),
            ("B", ["A"]),
            ("C", ["B"]),
            ("D", []),
        ]
    )
    graph = [
        mastery("A", 0.10),
        mastery("B", 0.50),
        mastery("D", 0.20),
    ]

    result = generate_learning_path(graph, curriculum=curriculum)

    assert skill_names(result["recommended_order"]) == ["A", "D"]
    assert skill_names(result["blocked"]) == ["B", "C"]
    assert result["blocked"][0]["unmet_prerequisites"] == ["A"]
    assert result["blocked"][1]["unmet_prerequisites"] == ["B"]
    assert "Earliest actionable prerequisite(s): A" in result["blocked"][1][
        "priority_reason"
    ]
    assert "2 blocked downstream skill(s)" in result["recommended_order"][0][
        "priority_reason"
    ]


def test_partial_skill_becomes_ready_only_after_prerequisite_is_strong():
    curriculum = make_curriculum(
        [
            ("A", []),
            ("B", ["A"]),
            ("C", ["B"]),
        ]
    )

    below_threshold = generate_learning_path(
        [mastery("A", 0.699), mastery("B", 0.50)],
        curriculum=curriculum,
    )
    at_threshold = generate_learning_path(
        [mastery("A", 0.70), mastery("B", 0.50)],
        curriculum=curriculum,
    )

    assert skill_names(below_threshold["blocked"]) == ["B", "C"]
    assert skill_names(at_threshold["learn_next"]) == ["B"]
    assert skill_names(at_threshold["blocked"]) == ["C"]
    assert at_threshold["learn_next"][0]["planning_status"] == "ready_to_learn"


def test_unseen_root_is_ready_without_fabricated_mastery_probability():
    curriculum = make_curriculum([("New Skill", [])])

    result = generate_learning_path([], curriculum=curriculum)

    entry = result["unseen"][0]
    assert entry["mastery_probability"] is None
    assert entry["mastery_status"] == "unseen"
    assert entry["planning_status"] == "ready_to_learn"
    assert entry["prerequisites"] == []
    assert entry["unmet_prerequisites"] == []
    assert entry["reason"] == "No mastery evidence has been recorded."
    assert entry["priority_reason"]
    assert skill_names(result["learn_next"]) == ["New Skill"]
    assert skill_names(result["recommended_order"]) == ["New Skill"]


def test_unseen_target_is_blocked_by_non_strong_prerequisite():
    curriculum = make_curriculum(
        [
            ("Foundation", []),
            ("Target", ["Foundation"]),
        ]
    )

    result = generate_learning_path(
        [mastery("Foundation", 0.40)],
        curriculum=curriculum,
    )

    target = result["blocked"][0]
    assert target["skill"] == "Target"
    assert target["mastery_probability"] is None
    assert target["mastery_status"] == "unseen"
    assert target["planning_status"] == "blocked"
    assert target["unmet_prerequisites"] == ["Foundation"]
    assert skill_names(result["recommended_order"]) == ["Foundation"]


def test_actionable_regression_remains_highest_priority():
    curriculum = make_curriculum(
        [
            ("Weak", []),
            ("Regressed", []),
            ("Ready", []),
        ]
    )
    graph = [
        mastery("Weak", 0.10),
        mastery("Regressed", 0.40, previous=0.80),
        mastery("Ready", 0.50),
    ]

    result = generate_learning_path(graph, curriculum=curriculum)

    assert skill_names(result["recommended_order"]) == [
        "Regressed",
        "Weak",
        "Ready",
    ]
    assert skill_names(result["regressions"]) == ["Regressed"]
    assert result["recommended_order"][0]["planning_status"] == "revise_urgently"


def test_curriculum_results_are_independent_of_graph_input_order():
    curriculum = make_curriculum(
        [
            ("A", []),
            ("B", ["A"]),
            ("C", []),
        ]
    )
    graph = [mastery("A", 0.20), mastery("B", 0.50), mastery("C", 0.40)]

    forward = generate_learning_path(graph, curriculum=curriculum)
    reverse = generate_learning_path(list(reversed(graph)), curriculum=curriculum)

    assert forward == reverse


def test_detector_metadata_does_not_affect_curriculum_ranking():
    curriculum = make_curriculum([("A", []), ("B", [])])
    plain = [mastery("A", 0.20), mastery("B", 0.40)]
    with_detector_metadata = [
        mastery(
            "A",
            0.20,
            reasoning_probability=0.99,
            uncertainty_probability=0.01,
        ),
        mastery(
            "B",
            0.40,
            clarification_probability=0.99,
        ),
    ]

    plain_result = generate_learning_path(plain, curriculum=curriculum)
    metadata_result = generate_learning_path(
        with_detector_metadata,
        curriculum=curriculum,
    )

    assert skill_names(plain_result["recommended_order"]) == skill_names(
        metadata_result["recommended_order"]
    )
