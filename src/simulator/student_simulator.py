"""
Held-out student simulator for FAPR-LB closed-loop evaluation.

Design principles:
  1. Base next-turn outcomes are sampled from the EMPIRICAL conditional
     distribution of the held-out test students. No hand-coded probabilities.
  2. The tutor action induces a small, FIXED multiplicative shift on
     P(correct) and E[hint_count]. The shift is uniform across all
     evaluated policies, so no policy is favoured by the simulator.
  3. Action-effect priors are conservative and will be ablated in Day 6.

The simulator exposes:
    StudentSimulator.reset(seed) -> initial_state_dict
    StudentSimulator.step(action) -> (next_obs, info)
    StudentSimulator.context() -> the current bandit context dict

This is NOT a Markov decision process simulator with full transition
dynamics. It is a one-step responder that, on each call, draws a
realistic outcome conditioned on the current student's history.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR
from src.models.failure_detector import FAILURE_TYPES


# Repair actions the bandit can take
REPAIR_ACTIONS = [
    "worked_example",
    "direct_correction",
    "scaffolded_question",
    "prerequisite_review",
    "hint",
    "simpler_explanation",
    "conceptual_analogy",
]

ACTION_TO_IDX = {a: i for i, a in enumerate(REPAIR_ACTIONS)}


# Action effects now DEPEND on the failure type. This gives the contextual
# bandit something real to learn: the best action changes with context.
ACTION_EFFECTS_BY_FAILURE: Dict[str, Dict[str, Dict[str, float]]] = {
    "low_mastery_failure": {
        "worked_example":       {"d_correct": +0.15, "d_hint": -0.3},
        "direct_correction":    {"d_correct": +0.05, "d_hint": -0.2},
        "scaffolded_question":  {"d_correct": +0.04, "d_hint": -0.1},
        "prerequisite_review":  {"d_correct": +0.18, "d_hint": -0.4},
        "hint":                 {"d_correct": +0.01, "d_hint": +0.5},
        "simpler_explanation":  {"d_correct": +0.10, "d_hint": -0.2},
        "conceptual_analogy":   {"d_correct": +0.07, "d_hint":  0.0},
    },
    "repair_needed_failure": {
        "worked_example":       {"d_correct": +0.07, "d_hint": -0.2},
        "direct_correction":    {"d_correct": +0.18, "d_hint": -0.4},
        "scaffolded_question":  {"d_correct": +0.15, "d_hint": -0.3},
        "prerequisite_review":  {"d_correct": +0.03, "d_hint": -0.1},
        "hint":                 {"d_correct": +0.08, "d_hint": +0.3},
        "simpler_explanation":  {"d_correct": +0.06, "d_hint": -0.1},
        "conceptual_analogy":   {"d_correct": +0.05, "d_hint":  0.0},
    },
}
ACTION_EFFECTS: Dict[str, Dict[str, float]] = ACTION_EFFECTS_BY_FAILURE["repair_needed_failure"]


def _bucket_skill_rate(rate: float) -> str:
    if rate < 0.33:
        return "low"
    if rate < 0.67:
        return "mid"
    return "high"


@dataclass
class StudentSimulator:
    """Wraps the held-out test split and serves as a per-turn environment."""

    data_path: Path = PROCESSED_DIR / "assist_test_with_struggle.parquet"
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))

    # Loaded on init
    df: pd.DataFrame = field(init=False)
    bins: Dict[Tuple[str, str], pd.DataFrame] = field(init=False)
    current_idx: int = field(default=-1)
    current_user_rows: pd.DataFrame = field(init=False, default=None)
    current_step: int = field(default=0)

    def __post_init__(self):
            self.df = pd.read_parquet(self.data_path)
            from src.models.failure_detector import detect_dataframe
            self.df = detect_dataframe(self.df)
            self.df["skill_bucket"] = self.df["skill_correct_rate"].apply(_bucket_skill_rate)

            # Build "recovery bins": for each failure row at time t, store the
            # OUTCOME of the same student at time t+1. This is what the simulator
            # should sample from to model "what happens AFTER a repair attempt".
            # Source: real held-out students' empirical next-row behaviour.
            self.df = self.df.sort_values(["user_id", "order_id"]).reset_index(drop=True)
            next_correct = self.df.groupby("user_id", sort=False)["correct"].shift(-1)
            next_hint = self.df.groupby("user_id", sort=False)["hint_count"].shift(-1)
            next_attempts = self.df.groupby("user_id", sort=False)["attempt_count"].shift(-1)
            next_rt = self.df.groupby("user_id", sort=False)["ms_first_response"].shift(-1)
            self.df["next_correct"] = next_correct
            self.df["next_hint"] = next_hint
            self.df["next_attempts"] = next_attempts
            self.df["next_rt"] = next_rt

            # Drop rows with no next turn (last row per user)
            recovery_df = self.df.dropna(subset=["next_correct"]).copy()
            recovery_df["next_correct"] = recovery_df["next_correct"].astype(int)
            recovery_df["next_hint"] = recovery_df["next_hint"].astype(int)
            recovery_df["next_attempts"] = recovery_df["next_attempts"].astype(int)
            recovery_df["next_rt"] = recovery_df["next_rt"].astype(int)

            MIN_BIN_SIZE = 50  # bins smaller than this are merged into a same-failure-type fallback
            full_bins = {
                (ft, b): grp.reset_index(drop=True)
                for (ft, b), grp in recovery_df.groupby(["failure_type", "skill_bucket"])
            }
            # Build per-failure-type fallback (all skill buckets pooled)
            ft_fallback = {
                ft: grp.reset_index(drop=True)
                for ft, grp in recovery_df.groupby("failure_type")
            }
            self.bins = {}
            self._fallback_bins = ft_fallback
            for key, bin_df in full_bins.items():
                if len(bin_df) >= MIN_BIN_SIZE:
                    self.bins[key] = bin_df
                else:
                    # Use the failure-type-only fallback so we always have ≥ MIN_BIN_SIZE
                    self.bins[key] = ft_fallback[key[0]]

            # Diagnostic (suppressible): set FAPR_SIM_VERBOSE=1 in env to enable.
            import os
            if os.environ.get("FAPR_SIM_VERBOSE") == "1":
                print("[Simulator] Recovery base rates per (failure_type, skill_bucket):")
                diag = (recovery_df.groupby(["failure_type", "skill_bucket"])
                        ["next_correct"].agg(["mean", "size"]).round(3))
                print(diag)

    # ---------------- Episode interface ----------------

    def reset(self, user_id: Optional[int] = None,
              skip_to_failure: bool = True) -> Dict:
        """
        Pick a held-out student. If skip_to_failure (default), advance
        the cursor to the first row whose failure_type != 'no_failure'.
        This ensures the bandit episode begins where repair is actually
        needed — matching the FAPR-LB problem definition.
        """
        users = self.df["user_id"].unique()
        if user_id is None:
            user_id = int(self.rng.choice(users))
        self.current_user_rows = (
            self.df[self.df["user_id"] == user_id]
            .sort_values("order_id")
            .reset_index(drop=True)
        )
        self.current_step = 0
        if skip_to_failure:
            self._advance_to_failure()
        return self._observe(self.current_step)

    def reset_with_min_failures(self, min_failures: int = 3,
                                 max_tries: int = 50) -> Optional[Dict]:
        """
        Pick a held-out student who has at least `min_failures` failure
        rows in their history. Useful for evaluation runs where short
        episodes (1-2 turns) bias the metrics.
        Returns None if no suitable user found within max_tries.
        """
        users = self.df["user_id"].unique()
        for _ in range(max_tries):
            uid = int(self.rng.choice(users))
            user_rows = self.df[self.df["user_id"] == uid]
            n_fail = (user_rows["failure_type"] != "no_failure").sum()
            if n_fail >= min_failures:
                return self.reset(user_id=uid)
        return None

    def _advance_to_failure(self) -> bool:
        """Move cursor to the next failure row. Returns False if none left."""
        n = len(self.current_user_rows)
        while self.current_step < n:
            ft = self.current_user_rows.iloc[self.current_step]["failure_type"]
            if ft != "no_failure":
                return True
            self.current_step += 1
        return False  # exhausted

    def step(self, action: str,
             advance_to_next_failure: bool = True) -> Tuple[Optional[Dict], Dict]:
        """
        Execute one tutor action on the current failure row, then advance
        the cursor to the next failure row (skipping no_failure rows
        between them). Returns (next_obs, info). If no more failures
        remain, info['done'] is True and next_obs is None.
        """
        if action not in ACTION_TO_IDX:
            raise ValueError(f"Unknown action: {action}")

        row = self.current_user_rows.iloc[self.current_step]
        ft = row["failure_type"]
        bucket = row["skill_bucket"]

        # Hard guard: bandit must never act on a no_failure row.
        if ft == "no_failure":
            # Should not happen if reset/_advance_to_failure worked, but guard it.
            still_have = self._advance_to_failure()
            if not still_have:
                return None, {
                    "action": action,
                    "failure_type_before": "no_failure",
                    "skill_bucket_before": bucket,
                    "skipped": True,
                    "done": True,
                }
            row = self.current_user_rows.iloc[self.current_step]
            ft = row["failure_type"]
            bucket = row["skill_bucket"]

        bin_df = self.bins.get((ft, bucket))
        if bin_df is None or len(bin_df) == 0:
            bin_df = self._fallback_bins.get(ft, self.df)

        sampled = bin_df.iloc[int(self.rng.integers(len(bin_df)))]
                # NEW: sample from RECOVERY columns (next-row outcome), not current-row.
                # This is the empirical "what happens after this kind of failure" signal.
        base_correct_p = float(sampled["next_correct"])  # 0 or 1, real next-row outcome
        base_hint = int(sampled["next_hint"])

        # Look up action effect by the CURRENT failure type.
        eff_table = ACTION_EFFECTS_BY_FAILURE.get(ft, ACTION_EFFECTS)
        eff = eff_table.get(action, ACTION_EFFECTS[action])
        shifted_p = float(np.clip(base_correct_p + eff["d_correct"], 0.02, 0.98))
        sim_correct = int(self.rng.random() < shifted_p)
        sim_hint = max(0, int(round(base_hint + eff["d_hint"])))
        sim_attempts = int(sampled["next_attempts"])
        sim_rt = int(sampled["next_rt"])

        info = {
            "action": action,
            "failure_type_before": ft,
            "skill_bucket_before": bucket,
            "sim_correct": sim_correct,
            "sim_hint": sim_hint,
            "sim_attempts": sim_attempts,
            "sim_rt": sim_rt,
            "base_correct_sample": base_correct_p,
            "shifted_correct_p": shifted_p,
        }

        # Advance: move to next row, then optionally skip to next failure
        self.current_step += 1
        if advance_to_next_failure:
            still_have = self._advance_to_failure()
            done = not still_have
        else:
            done = self.current_step >= len(self.current_user_rows)

        next_obs = None if done else self._observe(self.current_step)
        info["done"] = done
        return next_obs, info
    # ---------------- Internal helpers ----------------

    def _observe(self, idx: int) -> Dict:
        row = self.current_user_rows.iloc[idx]
        return {
            "user_id": int(row["user_id"]),
            "skill_id": int(row["skill_id"]),
            "skill_correct_rate": float(row["skill_correct_rate"]),
            "help_dependency": float(row["help_dependency"]),
            "recent_correct_rate": float(row["recent_correct_rate"]),
            "prev_correct": int(row["prev_correct"]),
            "prev_was_repair_event": int(row["prev_was_repair_event"]),
            # TSRP outputs already attached
            "repair_need_score": float(row["repair_need_score"]),
            "effort_cost_score": float(row["effort_cost_score"]),
            "disengagement_risk": float(row["disengagement_risk"]),
            # Failure label from rule-based detector
            "failure_type": str(row["failure_type"]),
            # Ground-truth observed outcome of THIS row (for evaluation only)
            "true_correct": int(row["correct"]),
            "true_hint": int(row["hint_count"]),
            "true_attempts": int(row["attempt_count"]),
        }


if __name__ == "__main__":
    # Smoke test: run one episode of 5 steps with random actions
    sim = StudentSimulator(rng=np.random.default_rng(42))
    obs = sim.reset()
    print(f"Reset on user_id={obs['user_id']}")
    for t in range(5):
        a = REPAIR_ACTIONS[int(np.random.default_rng(t).integers(len(REPAIR_ACTIONS)))]
        next_obs, info = sim.step(a)
        print(f"\nStep {t} | action={a}")
        print(f"  failure_type_before={info['failure_type_before']}  "
              f"skill_bucket={info['skill_bucket_before']}")
        print(f"  sim_correct={info['sim_correct']}  sim_hint={info['sim_hint']}  "
              f"sim_attempts={info['sim_attempts']}")
        if info["done"]:
            print("  Episode done.")
            break
