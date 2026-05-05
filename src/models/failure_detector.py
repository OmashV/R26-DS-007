"""
Rule-based Failure Detector.

Maps each interaction row to exactly one failure type, using deterministic
rules over observed ASSISTments fields plus the rolling skill_correct_rate
feature (computed by build_features).

This is intentionally NOT a trained classifier. There is no manually
labelled ground truth for tutoring-turn failure types in ASSISTments.
The rules below ARE the operational definition of failure, and that
definition is one of the stated contributions of FAPR-LB (Objective O2).

Failure types (mutually exclusive, priority order top to bottom):
  no_failure              : correct == 1 AND hint == 0 AND attempts == 1
  repair_needed_failure   : correct == 1 with help OR (correct == 0 AND skill_rate >= 0.4)
  low_mastery_failure     : correct == 0 AND skill_correct_rate <  0.4
"""

from typing import Dict
import numpy as np
import pandas as pd


FAILURE_TYPES = [
    "no_failure",
    "low_mastery_failure",
    "repair_needed_failure",
]

# Index map for one-hot encoding into the bandit context
FAILURE_TYPE_TO_IDX = {t: i for i, t in enumerate(FAILURE_TYPES)}


# Thresholds (defensible defaults; can be ablated)
HINT_HEAVY = 1          # >= this many hints counts as help-dependent
ATTEMPT_HEAVY = 2       # >= this many attempts counts as help-dependent
LOW_MASTERY_THR = 0.4   # skill_correct_rate below this is low mastery


def detect_one(row: Dict) -> str:
    """Apply rules to a single interaction dict. Returns one failure type."""
    correct = int(row["correct"])
    hint = int(row["hint_count"])
    attempts = int(row["attempt_count"])
    skill_rate = float(row.get("skill_correct_rate", 0.5))

    if correct == 1 and hint == 0 and attempts == 1:
        return "no_failure"
    # Help-dependency cases (correct with help) are merged into repair_needed
    if correct == 1 and (hint >= HINT_HEAVY or attempts >= ATTEMPT_HEAVY):
        return "repair_needed_failure"
    if correct == 0 and skill_rate < LOW_MASTERY_THR:
        return "low_mastery_failure"
    if correct == 0 and skill_rate >= LOW_MASTERY_THR:
        return "repair_needed_failure"
    return "no_failure"

def detect_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorised version. Adds 'failure_type' column to df."""
    df = df.copy()
    correct = df["correct"].astype(int).values
    hint = df["hint_count"].astype(int).values
    attempts = df["attempt_count"].astype(int).values
    skill_rate = df["skill_correct_rate"].astype(float).values

    label = np.full(len(df), "no_failure", dtype=object)

    is_help_or_repair = (
        ((correct == 1) & ((hint >= HINT_HEAVY) | (attempts >= ATTEMPT_HEAVY)))
        | ((correct == 0) & (skill_rate >= LOW_MASTERY_THR))
    )
    is_low_mast = (correct == 0) & (skill_rate < LOW_MASTERY_THR)

    label[is_help_or_repair] = "repair_needed_failure"
    label[is_low_mast] = "low_mastery_failure"

    df["failure_type"] = label
    df["failure_idx"] = df["failure_type"].map(FAILURE_TYPE_TO_IDX).astype(int)
    return df


def one_hot(failure_type: str) -> np.ndarray:
    """Return a 3-dim one-hot vector for a failure type (used by bandit)."""
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
    df["is_boundary"] = near_low_mastery | near_help_dep
    return df


if __name__ == "__main__":
    from src.config import PROCESSED_DIR
    df = pd.read_parquet(PROCESSED_DIR / "assist_test_with_struggle.parquet")
    out = detect_dataframe(df)
    print("Failure type distribution on TEST split:")
    counts = out["failure_type"].value_counts()
    print(counts)
    print("\nProportions:")
    print((counts / counts.sum()).round(3))

    # Boundary diagnosis: rows where rule-based labels are most fragile.
    # These are precisely the cases a PP2 learned classifier would target.
    out_b = diagnose_boundary_cases(out)
    n_boundary = int(out_b["is_boundary"].sum())
    print(f"\nBoundary cases (label fragile under ±0.05 threshold shift): "
          f"{n_boundary:,} rows ({n_boundary / len(out_b):.1%})")
    print("These rows are the target population for the PP2 learned classifier.")
