"""Explicit, validated curriculum prerequisite definitions.

Curriculum dependencies are pedagogical data. They are deliberately loaded
from a version-controlled JSON artifact rather than inferred from semantic
skill similarity.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Iterable, Mapping

from config import BKT_PARAMS_PATH, PROJECT_ROOT


DEFAULT_CURRICULUM_PATH = PROJECT_ROOT / "models" / "skill_prerequisites.json"


@dataclass(frozen=True)
class Curriculum:
    """An ordered skill set and its directed prerequisite relationships."""

    skills: tuple[str, ...]
    prerequisites: Mapping[str, tuple[str, ...]]

    def __post_init__(self) -> None:
        skills = tuple(self.skills)
        if not skills:
            raise ValueError("curriculum must contain at least one skill")
        if any(not isinstance(skill, str) or not skill.strip() for skill in skills):
            raise ValueError("curriculum skill names must be non-empty strings")
        if len(skills) != len(set(skills)):
            raise ValueError("curriculum skills must be unique")

        known = set(skills)
        normalised: dict[str, tuple[str, ...]] = {}
        extra_sources = set(self.prerequisites) - known
        if extra_sources:
            raise ValueError(
                "prerequisite definitions contain unknown curriculum skills: "
                + ", ".join(sorted(extra_sources))
            )

        for skill in skills:
            dependencies = tuple(self.prerequisites.get(skill, ()))
            if any(
                not isinstance(dependency, str) or not dependency.strip()
                for dependency in dependencies
            ):
                raise ValueError(
                    f"prerequisites for '{skill}' must be non-empty strings"
                )
            if len(dependencies) != len(set(dependencies)):
                raise ValueError(
                    f"prerequisites for '{skill}' must be unique"
                )
            if skill in dependencies:
                raise ValueError(f"skill '{skill}' cannot be its own prerequisite")

            unknown = [dependency for dependency in dependencies if dependency not in known]
            if unknown:
                raise ValueError(
                    f"prerequisites for '{skill}' reference unknown skills: "
                    + ", ".join(unknown)
                )
            normalised[skill] = dependencies

        object.__setattr__(self, "skills", skills)
        object.__setattr__(
            self,
            "prerequisites",
            MappingProxyType(normalised),
        )
        self._validate_acyclic()

    def _validate_acyclic(self) -> None:
        state = {skill: 0 for skill in self.skills}
        stack: list[str] = []

        def visit(skill: str) -> None:
            if state[skill] == 2:
                return
            if state[skill] == 1:
                cycle_start = stack.index(skill)
                cycle = stack[cycle_start:] + [skill]
                raise ValueError(
                    "curriculum prerequisites contain a cycle: "
                    + " -> ".join(cycle)
                )

            state[skill] = 1
            stack.append(skill)
            for dependency in self.prerequisites[skill]:
                visit(dependency)
            stack.pop()
            state[skill] = 2

        for skill in self.skills:
            visit(skill)

    @classmethod
    def from_dict(
        cls,
        data: dict,
        *,
        bkt_skills: Iterable[str],
    ) -> "Curriculum":
        """Build and validate a curriculum from the JSON artifact shape."""
        if not isinstance(data, dict):
            raise ValueError("curriculum artifact must contain a JSON object")
        if data.get("version") != 1:
            raise ValueError("curriculum artifact version must be 1")

        records = data.get("skills")
        if not isinstance(records, list):
            raise ValueError("curriculum artifact 'skills' must be a list")

        ordered_skills: list[str] = []
        prerequisites: dict[str, tuple[str, ...]] = {}
        for index, record in enumerate(records):
            if not isinstance(record, dict):
                raise ValueError(f"curriculum skill at index {index} must be an object")

            skill = record.get("skill")
            dependencies = record.get("prerequisites", [])
            if not isinstance(skill, str) or not skill.strip():
                raise ValueError(
                    f"curriculum skill at index {index} must have a non-empty name"
                )
            if not isinstance(dependencies, list):
                raise ValueError(
                    f"prerequisites for '{skill}' must be a JSON list"
                )

            ordered_skills.append(skill)
            # Keep the first definition here so Curriculum.__post_init__ can
            # report duplicate skill names deterministically.
            prerequisites.setdefault(skill, tuple(dependencies))

        curriculum = cls(
            skills=tuple(ordered_skills),
            prerequisites=prerequisites,
        )

        bkt_universe = set(bkt_skills)
        outside_bkt = [
            skill for skill in curriculum.skills if skill not in bkt_universe
        ]
        if outside_bkt:
            raise ValueError(
                "curriculum skills are missing from the BKT skill universe: "
                + ", ".join(outside_bkt)
            )
        return curriculum

    def prerequisites_for(self, skill: str) -> tuple[str, ...]:
        """Return direct prerequisites in their explicit artifact order."""
        if skill not in self.prerequisites:
            raise KeyError(f"skill '{skill}' is not in this curriculum")
        return self.prerequisites[skill]

    def order_index(self, skill: str) -> int:
        """Return the stable curriculum position of a skill."""
        try:
            return self.skills.index(skill)
        except ValueError as exc:
            raise KeyError(f"skill '{skill}' is not in this curriculum") from exc


def load_curriculum(
    path: Path = DEFAULT_CURRICULUM_PATH,
    *,
    bkt_params_path: Path = BKT_PARAMS_PATH,
) -> Curriculum:
    """Load a curriculum and validate it against trained BKT skill names."""
    with Path(bkt_params_path).open(encoding="utf-8") as handle:
        bkt_params = json.load(handle)
    if not isinstance(bkt_params, dict):
        raise ValueError("BKT parameter artifact must contain a JSON object")

    with Path(path).open(encoding="utf-8") as handle:
        data = json.load(handle)
    return Curriculum.from_dict(data, bkt_skills=bkt_params.keys())
