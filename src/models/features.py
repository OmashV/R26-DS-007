"""
Feature engineering for the Learner State Inferrer (Model 1).

For each interaction row, compute features from the student's history
*strictly before* this row. No leakage of the current row's outcome.

Features:
  - skill_seen_count        : how many times this student saw this skill before
  - skill_correct_rate      : rolling correct rate on this skill (mastery proxy)
  - overall_correct_rate    : rolling correct rate across all skills
  - recent_correct_rate     : last 5 interactions overall accuracy
  - help_dependency         : average hint_count over last 10 interactions
  - mean_attempts           : average attempt_count over last 10 interactions
  - log_response_time       : log(ms_first_response + 1)
  - prev_correct            : whether the immediately previous interaction was correct
  - prev_hint_used          : whether the immediately previous interaction used hints
"""

import numpy as np
import pandas as pd
from tqdm import tqdm


FEATURE_COLS = [
    "skill_seen_count",
    "skill_correct_rate",
    "overall_correct_rate",
    "recent_correct_rate",
    "help_dependency",
    "mean_attempts",
    "log_response_time",
    "prev_correct",
    "prev_hint_used",
    "prev_was_repair_event",   
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add history-based features to each row.
    df must already be sorted by (user_id, order_id).
    Returns a copy of df with feature columns added.
    The first interaction per student gets neutral defaults
    (we drop those rows when training, since they have no history).
    """
    df = df.copy().reset_index(drop=True)

    # Vectorised cumulative computations per user
    grp = df.groupby("user_id", sort=False)

    # Cumulative counts BEFORE current row (shift by 1)
    df["row_idx_in_user"] = grp.cumcount()  # 0-indexed

    # Overall correct rate before current row
    cum_correct = grp["correct"].cumsum().shift(1).fillna(0)
    cum_count = df["row_idx_in_user"]  # already excludes current row
    df["overall_correct_rate"] = np.where(
        cum_count > 0, cum_correct / cum_count, 0.5
    )

    # Per-skill seen count and correct rate before current row
    df["skill_seen_count"] = grp.cumcount()  # placeholder; recompute properly below
    df["skill_correct_rate"] = 0.5            # placeholder

    # Compute per-(user, skill) cumulative stats efficiently
    user_skill_grp = df.groupby(["user_id", "skill_id"], sort=False)
    skill_cum_correct = user_skill_grp["correct"].cumsum().shift(1).fillna(0)
    skill_cum_count = user_skill_grp.cumcount()  # 0-indexed = count BEFORE current
    df["skill_seen_count"] = skill_cum_count
    df["skill_correct_rate"] = np.where(
        skill_cum_count > 0, skill_cum_correct / skill_cum_count, 0.5
    )

    # Recent rolling stats: window of last K interactions per user
    # We compute via rolling on shifted series so current row is excluded.
    K_RECENT = 5
    K_WINDOW = 10

    def rolling_prev_mean(series: pd.Series, k: int) -> pd.Series:
        """Mean of the previous k values, excluding current. NaN if no history."""
        return series.shift(1).rolling(k, min_periods=1).mean()

    df["recent_correct_rate"] = grp["correct"].apply(
        lambda s: rolling_prev_mean(s, K_RECENT)
    ).reset_index(level=0, drop=True)
    df["help_dependency"] = grp["hint_count"].apply(
        lambda s: rolling_prev_mean(s, K_WINDOW)
    ).reset_index(level=0, drop=True)
    df["mean_attempts"] = grp["attempt_count"].apply(
        lambda s: rolling_prev_mean(s, K_WINDOW)
    ).reset_index(level=0, drop=True)

    # Fill defaults for first-row cases
    df["recent_correct_rate"] = df["recent_correct_rate"].fillna(0.5)
    df["help_dependency"] = df["help_dependency"].fillna(0.0)
    df["mean_attempts"] = df["mean_attempts"].fillna(1.0)

    # Response time feature (current row's; this is allowed because
    # response time is observed at the start, not after we know correctness)
    df["log_response_time"] = np.log1p(df["ms_first_response"].clip(lower=0))

    # Previous interaction features
    df["prev_correct"] = grp["correct"].shift(1).fillna(0).astype(int)
    df["prev_hint_used"] = (grp["hint_count"].shift(1).fillna(0) > 0).astype(int)

    # Drop rows with no history (first interaction per user) — model needs context
    df = df[df["row_idx_in_user"] >= 1].reset_index(drop=True)

    # Previous-turn repair-event signal (computed from columns we already have)
    df["prev_was_repair_event"] = (
        (grp["correct"].shift(1).fillna(1) == 0)
        & (
            (grp["hint_count"].shift(1).fillna(0) > 0)
            | (grp["attempt_count"].shift(1).fillna(0) > 1)
        )
    ).astype(int)

    return df


if __name__ == "__main__":
    # Smoke test
    from src.config import PROCESSED_DIR
    df = pd.read_parquet(PROCESSED_DIR / "assist_train.parquet").head(5000)
    out = build_features(df)
    print("Shape with features:", out.shape)
    print("Feature columns:")
    print(out[FEATURE_COLS].describe().round(3))