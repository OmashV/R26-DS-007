from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path


# Running ``python scripts/smoke_learning_path_inputs.py`` puts scripts/ on
# sys.path. Add the repository root before importing the root-level core package.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.curriculum import Curriculum, load_curriculum
from core.learning_path import generate_learning_path
from core.learning_path_presentation import format_mastery_probability


Graph = list[dict]
Validator = Callable[[dict, Curriculum], None]


def mastery(
    skill: str,
    probability: float,
    previous: float | None = None,
) -> dict:
    return {
        "skill": skill,
        "mastery_probability": probability,
        "previous_mastery_probability": previous,
    }


def skill_names(entries: list[dict]) -> list[str]:
    return [entry["skill"] for entry in entries]


def entry_for(path: dict, skill: str) -> dict:
    for key in (
        "unseen",
        "blocked",
        "recommended_order",
        "already_strong",
        "regressions",
    ):
        for entry in path[key]:
            if entry["skill"] == skill:
                return entry
    raise AssertionError(f"Skill missing from path output: {skill}")


def assert_entry(
    path: dict,
    skill: str,
    *,
    mastery_status: str | None = None,
    planning_status: str | None = None,
    probability: float | None | object = ...,
    unmet: list[str] | None = None,
) -> dict:
    entry = entry_for(path, skill)
    if mastery_status is not None:
        assert entry["mastery_status"] == mastery_status, (
            skill,
            entry["mastery_status"],
        )
    if planning_status is not None:
        assert entry["planning_status"] == planning_status, (
            skill,
            entry["planning_status"],
        )
    if probability is not ...:
        assert entry["mastery_probability"] == probability, (
            skill,
            entry["mastery_probability"],
        )
    if unmet is not None:
        assert entry["unmet_prerequisites"] == unmet, (
            skill,
            entry["unmet_prerequisites"],
        )
    return entry


def print_path(path: dict) -> None:
    print("\nRECOMMENDED ORDER")
    if not path["recommended_order"]:
        print("(none)")
    for index, entry in enumerate(path["recommended_order"], start=1):
        probability = format_mastery_probability(
            entry["mastery_probability"]
        )
        print(
            f"{index}. {entry['skill']} | "
            f"mastery={entry['mastery_status']} ({probability}) | "
            f"planning={entry['planning_status']}"
        )
        print(f"   priority: {entry['priority_reason']}")

    print("\nBLOCKED")
    if not path["blocked"]:
        print("(none)")
    for entry in path["blocked"]:
        print(entry["skill"])
        print("  unmet prerequisites:")
        for prerequisite in entry["unmet_prerequisites"]:
            print(f"  - {prerequisite}")
        print(f"  priority: {entry['priority_reason']}")


def validate_new_student(path: dict, curriculum: Curriculum) -> None:
    assert skill_names(path["recommended_order"]) == [
        "Equivalent Fractions",
        "Addition Whole Numbers",
    ]
    assert len(path["unseen"]) == len(curriculum.skills) == 11
    assert all(
        entry["mastery_probability"] is None
        for entry in path["unseen"]
    )
    assert_entry(
        path,
        "Addition Whole Numbers",
        mastery_status="unseen",
        planning_status="ready_to_learn",
        probability=None,
    )
    assert_entry(
        path,
        "Equivalent Fractions",
        mastery_status="unseen",
        planning_status="ready_to_learn",
        probability=None,
    )
    assert len(path["blocked"]) == 9


def validate_one_foundation(path: dict, curriculum: Curriculum) -> None:
    assert_entry(
        path,
        "Addition Whole Numbers",
        mastery_status="strong",
        planning_status="already_strong",
        probability=0.80,
    )
    assert_entry(
        path,
        "Multiplication Whole Numbers",
        mastery_status="unseen",
        planning_status="ready_to_learn",
        probability=None,
    )
    assert_entry(
        path,
        "Equivalent Fractions",
        mastery_status="unseen",
        planning_status="ready_to_learn",
        probability=None,
    )
    assert_entry(
        path,
        "Multiplication Fractions",
        planning_status="blocked",
        unmet=["Multiplication Whole Numbers", "Equivalent Fractions"],
    )


def validate_partial_prerequisite(path: dict, curriculum: Curriculum) -> None:
    assert_entry(
        path,
        "Equivalent Fractions",
        mastery_status="partial",
        planning_status="ready_to_learn",
        probability=0.69,
    )
    for dependent in (
        "Ordering Fractions",
        "Addition and Subtraction Fractions",
        "Multiplication Fractions",
        "Conversion of Fraction Decimals Percents",
    ):
        assert_entry(path, dependent, planning_status="blocked")

    at_threshold = generate_learning_path(
        [mastery("Equivalent Fractions", 0.70)],
        curriculum=curriculum,
    )
    for direct_dependent in (
        "Ordering Fractions",
        "Addition and Subtraction Fractions",
        "Conversion of Fraction Decimals Percents",
    ):
        assert_entry(
            at_threshold,
            direct_dependent,
            planning_status="ready_to_learn",
        )
    assert_entry(
        at_threshold,
        "Multiplication Fractions",
        planning_status="blocked",
        unmet=["Multiplication Whole Numbers"],
    )


def validate_percent_branch(path: dict, curriculum: Curriculum) -> None:
    assert_entry(path, "Equivalent Fractions", mastery_status="strong")
    assert_entry(
        path,
        "Conversion of Fraction Decimals Percents",
        mastery_status="strong",
    )
    assert_entry(
        path,
        "Percents",
        mastery_status="unseen",
        planning_status="ready_to_learn",
        probability=None,
    )
    assert_entry(
        path,
        "Finding Percents",
        planning_status="blocked",
        unmet=["Percents"],
    )
    assert_entry(path, "Percent Of", planning_status="blocked")


def validate_one_percent_prerequisite_missing(
    path: dict,
    curriculum: Curriculum,
) -> None:
    recommended = skill_names(path["recommended_order"])
    assert recommended[0] == "Multiplication Fractions"
    assert "Finding Percents" in recommended
    assert_entry(
        path,
        "Multiplication Fractions",
        mastery_status="unseen",
        planning_status="ready_to_learn",
        probability=None,
    )
    assert_entry(
        path,
        "Percent Of",
        mastery_status="unseen",
        planning_status="blocked",
        probability=None,
        unmet=["Multiplication Fractions"],
    )


def validate_both_percent_prerequisites(path: dict, curriculum: Curriculum) -> None:
    assert_entry(
        path,
        "Multiplication Fractions",
        mastery_status="strong",
        planning_status="already_strong",
        probability=0.72,
    )
    assert_entry(
        path,
        "Percent Of",
        mastery_status="unseen",
        planning_status="ready_to_learn",
        probability=None,
        unmet=[],
    )
    assert_entry(
        path,
        "Division Fractions",
        mastery_status="unseen",
        planning_status="ready_to_learn",
        probability=None,
        unmet=[],
    )


def validate_regression_priority(path: dict, curriculum: Curriculum) -> None:
    regression = assert_entry(
        path,
        "Equivalent Fractions",
        mastery_status="partial",
        planning_status="revise_urgently",
        probability=0.49,
    )
    assert regression["is_regression"] is True
    assert path["recommended_order"][0]["skill"] == "Equivalent Fractions"
    for downstream in (
        "Ordering Fractions",
        "Addition and Subtraction Fractions",
        "Multiplication Fractions",
        "Conversion of Fraction Decimals Percents",
        "Percents",
        "Finding Percents",
        "Percent Of",
    ):
        assert_entry(path, downstream, planning_status="blocked")

    exact_boundary = generate_learning_path(
        [
            mastery(
                "Equivalent Fractions",
                0.50,
                previous=0.70,
            )
        ],
        curriculum=curriculum,
    )
    assert entry_for(
        exact_boundary,
        "Equivalent Fractions",
    )["is_regression"] is False


def validate_fully_prepared(path: dict, curriculum: Curriculum) -> None:
    expected_actionable = [
        "Ordering Fractions",
        "Addition and Subtraction Fractions",
        "Division Fractions",
        "Finding Percents",
        "Percent Of",
    ]
    assert skill_names(path["recommended_order"]) == expected_actionable
    assert path["blocked"] == []
    for skill in expected_actionable:
        assert_entry(
            path,
            skill,
            mastery_status="unseen",
            planning_status="ready_to_learn",
            probability=None,
            unmet=[],
        )


PERCENT_BRANCH_STRONG = [
    mastery("Addition Whole Numbers", 0.90),
    mastery("Multiplication Whole Numbers", 0.82),
    mastery("Equivalent Fractions", 0.88),
    mastery("Conversion of Fraction Decimals Percents", 0.81),
    mastery("Percents", 0.75),
]


SCENARIOS: tuple[tuple[str, Graph, Validator], ...] = (
    (
        "COMPLETELY NEW STUDENT",
        [],
        validate_new_student,
    ),
    (
        "ONE FOUNDATION MASTERED",
        [mastery("Addition Whole Numbers", 0.80)],
        validate_one_foundation,
    ),
    (
        "PARTIAL PREREQUISITE DOES NOT UNLOCK",
        [mastery("Equivalent Fractions", 0.69)],
        validate_partial_prerequisite,
    ),
    (
        "PERCENT BRANCH PROGRESSES",
        [
            mastery("Equivalent Fractions", 0.85),
            mastery("Conversion of Fraction Decimals Percents", 0.78),
        ],
        validate_percent_branch,
    ),
    (
        "ONE PERCENT-OF PREREQUISITE MISSING",
        PERCENT_BRANCH_STRONG,
        validate_one_percent_prerequisite_missing,
    ),
    (
        "BOTH PERCENT-OF PREREQUISITES SATISFIED",
        PERCENT_BRANCH_STRONG
        + [mastery("Multiplication Fractions", 0.72)],
        validate_both_percent_prerequisites,
    ),
    (
        "REGRESSION BEATS ORDINARY RECOMMENDATIONS",
        [
            mastery("Equivalent Fractions", 0.49, previous=0.82),
            mastery("Addition Whole Numbers", 0.20, previous=0.20),
        ],
        validate_regression_priority,
    ),
    (
        "FULLY PREPARED PERCENT-OF STUDENT",
        [
            mastery("Addition Whole Numbers", 0.90),
            mastery("Multiplication Whole Numbers", 0.85),
            mastery("Equivalent Fractions", 0.90),
            mastery("Multiplication Fractions", 0.80),
            mastery("Conversion of Fraction Decimals Percents", 0.82),
            mastery("Percents", 0.76),
        ],
        validate_fully_prepared,
    ),
)


def run_percent_of_progression(curriculum: Curriculum) -> None:
    print("\n" + "=" * 60)
    print("PERCENT OF UNLOCK PROGRESSION")
    print("=" * 60)

    progression: tuple[tuple[str, Graph, str], ...] = (
        ("Equivalent Fractions unseen", [], "blocked"),
        (
            "Equivalent Fractions = 0.80",
            [mastery("Equivalent Fractions", 0.80)],
            "blocked",
        ),
        (
            "Conversion skill = 0.80",
            [
                mastery("Equivalent Fractions", 0.80),
                mastery("Conversion of Fraction Decimals Percents", 0.80),
            ],
            "blocked",
        ),
        (
            "Percents = 0.80",
            [
                mastery("Equivalent Fractions", 0.80),
                mastery("Conversion of Fraction Decimals Percents", 0.80),
                mastery("Percents", 0.80),
            ],
            "blocked",
        ),
        (
            "Addition Whole Numbers = 0.80",
            [
                mastery("Equivalent Fractions", 0.80),
                mastery("Conversion of Fraction Decimals Percents", 0.80),
                mastery("Percents", 0.80),
                mastery("Addition Whole Numbers", 0.80),
            ],
            "blocked",
        ),
        (
            "Multiplication Whole Numbers = 0.80",
            [
                mastery("Equivalent Fractions", 0.80),
                mastery("Conversion of Fraction Decimals Percents", 0.80),
                mastery("Percents", 0.80),
                mastery("Addition Whole Numbers", 0.80),
                mastery("Multiplication Whole Numbers", 0.80),
            ],
            "blocked",
        ),
        (
            "Multiplication Fractions = 0.69",
            [
                mastery("Equivalent Fractions", 0.80),
                mastery("Conversion of Fraction Decimals Percents", 0.80),
                mastery("Percents", 0.80),
                mastery("Addition Whole Numbers", 0.80),
                mastery("Multiplication Whole Numbers", 0.80),
                mastery("Multiplication Fractions", 0.69),
            ],
            "blocked",
        ),
        (
            "Multiplication Fractions = 0.70",
            [
                mastery("Equivalent Fractions", 0.80),
                mastery("Conversion of Fraction Decimals Percents", 0.80),
                mastery("Percents", 0.80),
                mastery("Addition Whole Numbers", 0.80),
                mastery("Multiplication Whole Numbers", 0.80),
                mastery("Multiplication Fractions", 0.70),
            ],
            "ready_to_learn",
        ),
    )

    for index, (label, graph, expected_status) in enumerate(
        progression,
        start=1,
    ):
        path = generate_learning_path(graph, curriculum=curriculum)
        percent_of = entry_for(path, "Percent Of")
        assert percent_of["planning_status"] == expected_status
        state = (
            "READY"
            if percent_of["planning_status"] == "ready_to_learn"
            else "BLOCKED"
        )
        unmet = ", ".join(percent_of["unmet_prerequisites"]) or "none"
        print(f"Step {index}: {label}")
        print(f"  Percent Of: {state}; unmet: {unmet}")

    print("PROGRESSION PASS")


def main() -> None:
    curriculum = load_curriculum()
    assert len(curriculum.skills) == 11

    passed = 0
    for index, (name, graph, validator) in enumerate(SCENARIOS, start=1):
        print("\n" + "=" * 60)
        print(f"SCENARIO {index} - {name}")
        print("=" * 60)
        path = generate_learning_path(graph, curriculum=curriculum)
        print_path(path)
        validator(path, curriculum)
        passed += 1
        print("\nPASS")

    run_percent_of_progression(curriculum)

    print("\n" + "=" * 60)
    print("LEARNING PATH INPUT SMOKE TEST")
    print(f"{passed}/{len(SCENARIOS)} scenarios passed")
    print("=" * 60)


if __name__ == "__main__":
    main()
