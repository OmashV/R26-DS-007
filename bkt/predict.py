"""
BKT inference module.

Loads trained BKT parameters and computes mastery probabilities from
sequences of correct/incorrect attempts.

Implements: FR8, FR9, FR10 (knowledge tracing requirements)
"""

import json
import logging
from pathlib import Path
from typing import Iterable

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

    def predict(self, skill: str, attempts: Iterable[int]) -> float:
        """
        Compute mastery probability for a student given their attempt history.

        Args:
            skill: The skill name (must exist in trained parameters).
            attempts: Sequence of 1 (correct) and 0 (incorrect) values, in order.

        Returns:
            P(student has mastered skill) — a value in [0, 1].

        Raises:
            KeyError: If the skill is not in the trained model.
            ValueError: If attempts contain values other than 0 or 1.
        """
        if skill not in self.params:
            raise KeyError(f"Skill '{skill}' not found in trained model")

        p = self.params[skill]
        prior   = p["prior"]
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