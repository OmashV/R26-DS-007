"""
Failure Detector — PP1 stage: rule-based labelling functions.

PP1 (current): Rule-based detector using deterministic conditions over
    native ASSISTments fields and rolling features. Serves two roles:
      (a) Operational, dataset-grounded definition of tutoring-turn
          failure (Objective O2 contribution).
      (b) Labelling functions for the weak-supervision pipeline used
          in PP2 / final.

PP2 (planned): Weak-supervision classifier. The rules below become
    Snorkel-style labelling functions whose noisy outputs train a
    neural classifier that incorporates TSRP outputs and trajectory
    features the rules cannot encode. A label model (Snorkel /
    FlyingSquid) estimates per-rule accuracy and resolves
    disagreement across rules.

Final (planned): Multi-source weak supervision with rule-classifier
    agreement statistics, plus integration of misconception data
    (Eedi) and affective signals.

Failure types (mutually exclusive, evaluated in priority order top to bottom):
  compounding_struggle_failure : sustained recent failures + high recent help use
  disengagement_failure        : extreme response time + content struggle signal
  skill_regression_failure     : prior mastery on this skill, now failing
  low_mastery_failure          : wrong AND weak history on this skill
  repair_needed_failure        : wrong with decent history, OR correct with help
  no_failure                   : clean success
"""

from typing import Dict
import numpy as np
import pandas as pd


FAILURE_TYPES = [
    "no_failure",
    "low_mastery_failure",
    "repair_needed_failure",
    "disengagement_failure",
    "skill_regression_failure",
    "compounding_struggle_failure",
]

# Index map for one-hot encoding into the bandit context
FAILURE_TYPE_TO_IDX = {t: i for i, t in enumerate(FAILURE_TYPES)}


# ----- Thresholds (defensible defaults; can be ablated) -----
HINT_HEAVY = 1            # >= this many hints counts as needing help
ATTEMPT_HEAVY = 2         # >= this many attempts counts as needing help
LOW_MASTERY_THR = 0.4     # skill_correct_rate below this is low mastery
PRIOR_MASTERY_THR = 0.6   # skill_correct_rate above this signals prior mastery
SKILL_SEEN_MIN = 5        # need this many prior attempts to claim "regression"
COMPOUNDING_RECENT_RATE_THR = 0.3  # recent correct rate below this is sustained struggle
COMPOUNDING_HINT_RATE_THR = 0.5   # recent hint rate above this is heavy help reliance
DISENGAGE_RT_PERCENTILE = 0.85    # top 15% of response times count as slow
DISENGAGE_RECENT_THR = 0.5        # weakening recent rate amplifies disengagement signal


def _compute_disengage_rt_threshold(df: pd.DataFrame) -> float:
    """Compute the response-time threshold for disengagement from the data itself."""
    if "ms_first_response" in df.columns:
        return float(df["ms_first_response"].quantile(DISENGAGE_RT_PERCENTILE))
    return float("inf")


def detect_dataframe(df: pd.DataFrame, rt_threshold: float = None) -> pd.DataFrame:
    """
    Vectorised rule-based detector. Adds 'failure_type' and 'failure_idx' columns.
    Priority order (first match wins):
      1. compounding_struggle_failure
      2. disengagement_failure
      3. skill_regression_failure
      4. low_mastery_failure
      5. repair_needed_failure
      6. no_failure (default)
    """
    df = df.copy()

    # Compute the disengagement RT threshold from the data if not provided
    if rt_threshold is None:
        rt_threshold = _compute_disengage_rt_threshold(df)

    correct = df["correct"].astype(int).values
    hint = df["hint_count"].astype(int).values
    attempts = df["attempt_count"].astype(int).values
    skill_rate = df["skill_correct_rate"].astype(float).values
    recent_correct = df["recent_correct_rate"].astype(float).values
    help_dep = df["help_dependency"].astype(float).values  # rolling hint rate
    skill_seen = df["skill_seen_count"].astype(float).values
    prev_correct = df["prev_correct"].astype(int).values
    rt = df["ms_first_response"].astype(float).values if "ms_first_response" in df.columns else np.zeros(len(df))

    # ---- Rule masks (highest priority first) ----

    # Rule 1: compounding_struggle_failure
    is_compounding = (
        (correct == 0)
        & (recent_correct < COMPOUNDING_RECENT_RATE_THR)
        & (help_dep >= COMPOUNDING_HINT_RATE_THR)
    )

    # Rule 2: disengagement_failure
    # Slow response time AND (failed attempt OR weakening recent performance).
    # Threshold lowered to 85th percentile to give the class enough support
    # for the bandit to learn from.
    is_disengage = (
        (rt > rt_threshold)
        & ((correct == 0) | (recent_correct < DISENGAGE_RECENT_THR))
    )

    # Rule 3: skill_regression_failure
    is_regression = (
        (correct == 0)
        & (skill_rate >= PRIOR_MASTERY_THR)
        & (prev_correct == 1)
        & (skill_seen >= SKILL_SEEN_MIN)
    )

    # Rule 4: low_mastery_failure
    is_low_mast = (correct == 0) & (skill_rate < LOW_MASTERY_THR)

    # Rule 5: repair_needed_failure
    is_repair = (
        ((correct == 0) & (skill_rate >= LOW_MASTERY_THR))
        | ((correct == 1) & ((hint >= HINT_HEAVY) | (attempts >= ATTEMPT_HEAVY)))
    )

    # ---- Apply in priority order (later rules don't overwrite earlier ones) ----
    label = np.full(len(df), "no_failure", dtype=object)
    label[is_repair] = "repair_needed_failure"
    label[is_low_mast] = "low_mastery_failure"
    label[is_regression] = "skill_regression_failure"
    label[is_disengage] = "disengagement_failure"
    label[is_compounding] = "compounding_struggle_failure"

    df["failure_type"] = label
    df["failure_idx"] = df["failure_type"].map(FAILURE_TYPE_TO_IDX).astype(int)
    return df


def detect_one(row: Dict, rt_threshold: float = None) -> str:
    """Single-row version. For online use during a session."""
    correct = int(row["correct"])
    hint = int(row["hint_count"])
    attempts = int(row["attempt_count"])
    skill_rate = float(row.get("skill_correct_rate", 0.5))
    recent_correct = float(row.get("recent_correct_rate", 0.5))
    help_dep = float(row.get("help_dependency", 0.0))
    skill_seen = float(row.get("skill_seen_count", 0))
    prev_correct = int(row.get("prev_correct", 0))
    rt = float(row.get("ms_first_response", 0.0))

    if rt_threshold is None:
        rt_threshold = float("inf")

    if (correct == 0
            and recent_correct < COMPOUNDING_RECENT_RATE_THR
            and help_dep >= COMPOUNDING_HINT_RATE_THR):
        return "compounding_struggle_failure"

    if (rt > rt_threshold
            and (correct == 0 or recent_correct < DISENGAGE_RECENT_THR)):
        return "disengagement_failure"

    if (correct == 0
            and skill_rate >= PRIOR_MASTERY_THR
            and prev_correct == 1
            and skill_seen >= SKILL_SEEN_MIN):
        return "skill_regression_failure"

    if correct == 0 and skill_rate < LOW_MASTERY_THR:
        return "low_mastery_failure"

    if (correct == 0 and skill_rate >= LOW_MASTERY_THR) or \
       (correct == 1 and (hint >= HINT_HEAVY or attempts >= ATTEMPT_HEAVY)):
        return "repair_needed_failure"

    return "no_failure"


def one_hot(failure_type: str) -> np.ndarray:
    """Return a 6-dim one-hot vector for a failure type."""
    v = np.zeros(len(FAILURE_TYPES), dtype=float)
    v[FAILURE_TYPE_TO_IDX[failure_type]] = 1.0
    return v


def diagnose_boundary_cases(df: pd.DataFrame, margin: float = 0.05) -> pd.DataFrame:
    """
    Identify rows whose label could flip with a small threshold change.
    Useful for showing where a learned model would help most.
    """
    df = df.copy()
    near_low_mastery = (
        (df["correct"] == 0)
        & (df["skill_correct_rate"] >= LOW_MASTERY_THR - margin)
        & (df["skill_correct_rate"] <= LOW_MASTERY_THR + margin)
    )
    near_help_dep = (
        (df["correct"] == 1)
        & (df["hint_count"].between(HINT_HEAVY - 1, HINT_HEAVY))
        & (df["attempt_count"].between(1, ATTEMPT_HEAVY))
    )
    near_prior_mastery = (
        (df["correct"] == 0)
        & (df["skill_correct_rate"] >= PRIOR_MASTERY_THR - margin)
        & (df["skill_correct_rate"] <= PRIOR_MASTERY_THR + margin)
    )
    df["is_boundary"] = near_low_mastery | near_help_dep | near_prior_mastery
    return df


if __name__ == "__main__":
    from src.config import PROCESSED_DIR
    df = pd.read_parquet(PROCESSED_DIR / "assist_test_with_struggle.parquet")
    out = detect_dataframe(df)
    print("Failure type distribution on TEST split (6 classes):")
    counts = out["failure_type"].value_counts()
    print(counts)
    print("\nProportions:")
    print((counts / counts.sum()).round(3))

    # Show only the 5 failure classes (excluding no_failure) to see balance
    print("\nFailure-only proportions (excluding no_failure):")
    failures_only = out[out["failure_type"] != "no_failure"]
    if len(failures_only) > 0:
        fc = failures_only["failure_type"].value_counts()
        print((fc / fc.sum()).round(3))

    out_b = diagnose_boundary_cases(out)
    n_boundary = int(out_b["is_boundary"].sum())
    print(f"\nBoundary cases: {n_boundary:,} rows ({n_boundary / len(out_b):.1%})")
