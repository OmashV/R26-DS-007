"""
Load and preprocess ASSISTments 2009 skill-builder data.

Standard preprocessing for knowledge tracing:
- Keep only rows with a valid skill_id
- Keep only original problems (original == 1) to avoid the
  multi-skill row duplication issue
- Sort by user_id, then order_id (chronological per student)
- Cast types
"""

from pathlib import Path
import pandas as pd
import numpy as np

from src.config import ASSISTMENTS_FILE, PROCESSED_DIR


def load_raw():
    """Load the raw ASSISTments CSV."""
    df = pd.read_csv(ASSISTMENTS_FILE, encoding="ISO-8859-15", low_memory=False)
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Standard KT preprocessing."""
    # 1. Drop rows without skill or correctness label
    df = df.dropna(subset=["skill_id", "user_id", "correct"]).copy()

    # 2. Keep only original problems (avoids multi-skill duplication)
    if "original" in df.columns:
        df = df[df["original"] == 1].copy()

    # 3. Cast types
    df["user_id"] = df["user_id"].astype(int)
    df["skill_id"] = df["skill_id"].astype(int)
    df["correct"] = df["correct"].astype(int)
    df["hint_count"] = df["hint_count"].astype(int)
    df["attempt_count"] = df["attempt_count"].astype(int)
    df["order_id"] = df["order_id"].astype(int)
    df["ms_first_response"] = df["ms_first_response"].astype(int)

    # 4. Sort: per student, chronological order
    df = df.sort_values(["user_id", "order_id"]).reset_index(drop=True)

    # 5. Filter out students with too few interactions (KT models need history)
    counts = df.groupby("user_id").size()
    keep_users = counts[counts >= 10].index
    df = df[df["user_id"].isin(keep_users)].reset_index(drop=True)

    # 6. Final column subset
    keep = [
        "user_id", "order_id", "skill_id", "skill_name",
        "correct", "hint_count", "attempt_count",
        "ms_first_response", "problem_id"
    ]
    keep = [c for c in keep if c in df.columns]
    df = df[keep]

    return df


def split_users(df: pd.DataFrame, test_frac: float = 0.2, seed: int = 42):
    """
    Student-level split. Returns (train_df, test_df, test_user_ids).
    The test users will also be used as the simulator on Day 3.
    """
    rng = np.random.default_rng(seed)
    users = df["user_id"].unique()
    rng.shuffle(users)
    n_test = int(len(users) * test_frac)
    test_users = set(users[:n_test])
    train_df = df[~df["user_id"].isin(test_users)].reset_index(drop=True)
    test_df = df[df["user_id"].isin(test_users)].reset_index(drop=True)
    return train_df, test_df, test_users


def main():
    print("Loading raw ASSISTments...")
    df = load_raw()
    print(f"  Raw shape: {df.shape}")

    print("Preprocessing...")
    df = preprocess(df)
    print(f"  After preprocess: {df.shape}")
    print(f"  Unique students: {df['user_id'].nunique()}")
    print(f"  Unique skills:   {df['skill_id'].nunique()}")
    print(f"  Mean correct:    {df['correct'].mean():.3f}")

    print("Splitting at student level (80/20)...")
    train_df, test_df, test_users = split_users(df)
    print(f"  Train: {len(train_df)} rows, {train_df['user_id'].nunique()} students")
    print(f"  Test:  {len(test_df)} rows, {test_df['user_id'].nunique()} students")

    # Save
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(PROCESSED_DIR / "assist_train.parquet", index=False)
    test_df.to_parquet(PROCESSED_DIR / "assist_test.parquet", index=False)

    # Save the test user IDs separately (Day 3 simulator)
    pd.Series(sorted(list(test_users)), name="user_id").to_csv(
        PROCESSED_DIR / "assist_test_users.csv", index=False
    )

    print("\nSaved:")
    print("  data/processed/assist_train.parquet")
    print("  data/processed/assist_test.parquet")
    print("  data/processed/assist_test_users.csv")


if __name__ == "__main__":
    main()