"""
BKT inference module.

Loads trained BKT parameters and computes mastery probabilities from
sequences of correct/incorrect attempts.

Implements: FR8, FR9, FR10 (knowledge tracing requirements)
"""

import json
import logging
from pathlib import Path
from typing import Iterable, Optional

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PARAMS_PATH = PROJECT_ROOT / "models" / "bkt_params.json"


class BKTPredictor:
    """
    Bayesian Knowledge Tracing inference.

    Takes a trained set of per-skill parameters and computes the probability
    that a student has mastered a skill given their sequence of correct/
    incorrect attempts on that skill.
    """

    def __init__(self, params: dict[str, dict[str, float]]) -> None:
        """
        Args:
            params: Mapping {skill_name: {prior, learns, guesses, slips, forgets}}
        """
        self.params = params
        logger.info(f"BKTPredictor initialised with {len(params)} skills")

    @classmethod
    def load(cls, path: Path = DEFAULT_PARAMS_PATH) -> "BKTPredictor":
        """Load a predictor from a JSON parameter file."""
        if not path.exists():
            raise FileNotFoundError(
                f"BKT parameters not found at {path}. "
                f"Train the model first (see notebooks/bkt_training_colab.ipynb)."
            )
        with open(path) as f:
            params = json.load(f)
        return cls(params)

    def has_skill(self, skill: str) -> bool:
        """Check whether the model has parameters for a given skill."""
        return skill in self.params

    def predict(
        self,
        skill: str,
        attempts: Iterable[int],
        initial_prior: Optional[float] = None,
    ) -> float:
        """
        Compute mastery probability for a student given their attempt history.

        Args:
            skill: The skill name (must exist in trained parameters).
            attempts: Sequence of 1 (correct) and 0 (incorrect) values, in order.
            initial_prior: Optional override for the starting P(known). If None,
                           uses the population prior P(L₀) from the trained model.
                           Used by cold-start transfer to set an informed prior
                           on a student's first encounter with a skill.

        Returns:
            P(student has mastered skill) — a value in [0, 1].

        Raises:
            KeyError: If the skill is not in the trained model.
            ValueError: If attempts contain values other than 0 or 1.
        """
        if skill not in self.params:
            raise KeyError(f"Skill '{skill}' not found in trained model")

        p = self.params[skill]
        prior   = initial_prior if initial_prior is not None else p["prior"]
        learn   = p["learns"]
        slip    = p["slips"]
        guess   = p["guesses"]
        forget  = p["forgets"]

        p_known = prior

        for attempt in attempts:
            if attempt not in (0, 1):
                raise ValueError(f"Attempts must be 0 or 1, got {attempt}")

            # Step A — posterior update given the observed answer
            if attempt == 1:
                # Correct answer
                numerator   = p_known * (1 - slip)
                denominator = p_known * (1 - slip) + (1 - p_known) * guess
            else:
                # Wrong answer
                numerator   = p_known * slip
                denominator = p_known * slip + (1 - p_known) * (1 - guess)

            # Guard against division by zero (degenerate parameters)
            p_known_posterior = numerator / denominator if denominator > 0 else p_known

            # Step B — learning transition
            p_known = p_known_posterior + (1 - p_known_posterior) * learn

            # (Forgetting term — kept at 0 in standard BKT, included for completeness)
            if forget > 0:
                p_known = p_known * (1 - forget)

        return p_known

    def predict_trajectory(self, skill: str, attempts: Iterable[int]) -> list[float]:
        """
        Like predict(), but returns mastery probability after every attempt.

        Useful for visualisation and demos — shows how P(mastery) evolves
        across the attempt sequence rather than just the final value.

        Returns:
            List of mastery probabilities, one per attempt, plus the prior at index 0.
        """
        if skill not in self.params:
            raise KeyError(f"Skill '{skill}' not found in trained model")

        p = self.params[skill]
        trajectory = [p["prior"]]

        for attempt in attempts:
            current = self.predict(skill, list(attempts)[:len(trajectory)])
            trajectory.append(current)

        # Cleaner implementation — just compute incrementally
        trajectory = [p["prior"]]
        p_known = p["prior"]
        for attempt in attempts:
            if attempt == 1:
                num = p_known * (1 - p["slips"])
                den = p_known * (1 - p["slips"]) + (1 - p_known) * p["guesses"]
            else:
                num = p_known * p["slips"]
                den = p_known * p["slips"] + (1 - p_known) * (1 - p["guesses"])

            posterior = num / den if den > 0 else p_known
            p_known = posterior + (1 - posterior) * p["learns"]
            trajectory.append(p_known)

        return trajectory


class ColdStartPriorCalculator:
    """
    Computes informed initial mastery priors for new (student, skill) pairs
    based on the student's mastery on related skills.

    For a brand-new (student, skill) pair where no attempt history exists,
    standard BKT uses the population prior P(L₀) for that skill. This class
    computes a more informative prior by transferring evidence from skills
    the student has already mastered.

    Implements: cold-start novelty extension to BKT.
    """

    def __init__(
        self,
        similarity_path: Optional[Path] = None,
        similarity_threshold: float = 0.5,
    ) -> None:
        """
        Args:
            similarity_path: Path to JSON file mapping {skill: {related_skill: similarity}}.
            similarity_threshold: Minimum similarity to count as "related" (default 0.5).
        """
        path = similarity_path or (PROJECT_ROOT / "models" / "skill_similarity.json")

        if not path.exists():
            raise FileNotFoundError(
                f"Skill similarity matrix not found at {path}. "
                f"Run notebooks/skill_similarity.ipynb first to generate it."
            )

        with open(path) as f:
            self.similarity = json.load(f)

        self.threshold = similarity_threshold
        logger.info(
            f"ColdStartPriorCalculator initialised with similarities for "
            f"{len(self.similarity)} skills"
        )

    def compute_prior(
        self,
        skill: str,
        student_masteries: dict[str, float],
        population_prior: float,
    ) -> dict:
        """
        Compute an informed initial prior for a (student, skill) pair.

        Args:
            skill: The skill the student is about to encounter.
            student_masteries: The student's current mastery on other skills,
                            as a dict {skill_name: P(known)}.
            population_prior: The default P(L₀) for this skill from the BKT model.

        Returns:
            Dict with 'prior' (the computed prior), 'used_transfer' (bool),
            'related_skills_used' (list), and 'transfer_evidence' (debug info).
        """
        related = self.similarity.get(skill, {})

        # Find related skills the student has actually attempted
        relevant = [
            (rel_skill, sim, student_masteries[rel_skill])
            for rel_skill, sim in related.items()
            if sim >= self.threshold and rel_skill in student_masteries
        ]

        if not relevant:
            # No related evidence — fall back to population prior
            return {
                "prior": population_prior,
                "used_transfer": False,
                "related_skills_used": [],
                "transfer_evidence": "No related skills attempted by student",
            }

        # Weighted average of related-skill masteries, weights = similarities
        total_weight = sum(sim for _, sim, _ in relevant)
        weighted_mastery = sum(sim * mastery for _, sim, mastery in relevant) / total_weight

        # Apply a transfer discount — even strong related mastery doesn't perfectly
        # transfer to a brand-new skill. The discount blends evidence with the
        # population prior, scaled by the strongest similarity used.
        TRANSFER_DISCOUNT = 0.7  # how much weighted mastery contributes vs population prior
        MAX_TRANSFERRED_PRIOR = 0.85  # ceiling — never claim near-certainty pre-evidence

        # Blend the related-skill evidence with the population prior
        blended = (
            TRANSFER_DISCOUNT * weighted_mastery
            + (1 - TRANSFER_DISCOUNT) * population_prior
        )
        # Take the higher of population prior and blended evidence (transfer only helps)
        # Then apply the safety cap
        transferred_prior = min(MAX_TRANSFERRED_PRIOR, max(population_prior, blended))

        return {
            "prior": transferred_prior,
            "used_transfer": transferred_prior > population_prior,
            "related_skills_used": [s for s, _, _ in relevant],
            "transfer_evidence": (
                f"Population prior: {population_prior:.3f}, "
                f"weighted related mastery: {weighted_mastery:.3f}, "
                f"using: {transferred_prior:.3f}"
            ),
        }
