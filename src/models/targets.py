"""
Target definitions for the Turn-Level Struggle Predictor (TSRP / Model 1).

Each target captures one dimension of the upcoming turn's struggle:

  y_repair_need      : 1 if (correct==0) AND (hint>0 OR attempts>1), this row.
                       Predicted from history features.
  y_effort_cost      : hint_count + attempt_count, this row, capped at 10.
                       Predicted from history features.
  y_next_disengagement : z-scored log(ms_first_response) of the NEXT row
                         (per-user, chronological).
                         Predicted from history-only features.
                         Shifted by one row to avoid leakage from the
                         current row's response time being both a feature
                         and the regression target.

The shift makes Head 3 a forward-looking prediction task. The last
row of each user has no next-row, so it is dropped for Head 3 only.
"""

import numpy as np
import pandas as pd


CAP_EFFORT = 10


def add_struggle_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds:
      y_repair_need
      y_effort_cost
      y_next_disengagement_raw   (= log(ms_first_response of NEXT row))
    df must already be sorted by (user_id, order_id).
    """
    df = df.copy()

    df["y_repair_need"] = (
        (df["correct"] == 0)
        & ((df["hint_count"] > 0) | (df["attempt_count"] > 1))
    ).astype(int)

    df["y_effort_cost"] = (
        (df["hint_count"] + df["attempt_count"]).clip(upper=CAP_EFFORT).astype(float)
    )

    # Next-row response time, per user. Last row of each user becomes NaN.
    next_log_rt = (
        df.groupby("user_id", sort=False)["ms_first_response"]
          .shift(-1)
          .clip(lower=0)
    )
    df["y_next_disengagement_raw"] = np.log1p(next_log_rt)

    return df


def fit_disengagement_zscore(df: pd.DataFrame) -> tuple[float, float]:
    """Mean and std of next-row log RT on training data, NaNs ignored."""
    s = df["y_next_disengagement_raw"].dropna()
    mu = float(s.mean())
    sd = float(s.std())
    if sd < 1e-6:
        sd = 1.0
    return mu, sd


def apply_disengagement_zscore(df: pd.DataFrame, mu: float, sd: float) -> pd.DataFrame:
    df = df.copy()
    df["y_next_disengagement"] = (df["y_next_disengagement_raw"] - mu) / sd
    return df