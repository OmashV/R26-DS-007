"""
Build the 10-dim context vector for the LinTS bandit.

Layout:
  0: bias = 1.0
  1: repair_need_score        (TSRP)
  2: effort_cost_score        (TSRP, z-scored at runtime)
  3: disengagement_risk       (TSRP)
  4: skill_correct_rate       (rolling feature)
  5: failure_type one-hot[low_mastery_failure]
  6: failure_type one-hot[repair_needed_failure]
  7: prev_action_was_repair   (1 if previous action in repair set, else 0)
  8: prev_outcome_correct     (1 if previous action led to sim_correct=1)
  9: prev_outcome_used_hint   (1 if previous action led to sim_hint>0)
"""

import numpy as np
from typing import Dict, Optional

CONTEXT_DIM = 10

REPAIR_ACTION_SET = {
    "worked_example",
    "direct_correction",
    "prerequisite_review",
}

# Z-score stats for effort_cost_score (estimated from train; we use loose defaults)
EFFORT_MEAN = 1.5
EFFORT_STD = 1.5


def build_context(obs: Dict, prev_action: Optional[str] = None,
                  prev_outcome: Optional[Dict] = None) -> np.ndarray:
    """
    obs           : current StudentSimulator observation dict
    prev_action   : the action the bandit took in the previous turn (or None)
    prev_outcome  : the info dict returned by the previous step (or None)
    """
    x = np.zeros(CONTEXT_DIM, dtype=float)

    x[0] = 1.0
    x[1] = float(obs["repair_need_score"])
    x[2] = (float(obs["effort_cost_score"]) - EFFORT_MEAN) / EFFORT_STD
    x[3] = float(obs["disengagement_risk"])
    x[4] = float(obs["skill_correct_rate"])

    ft = obs.get("failure_type", "no_failure")
    if ft == "low_mastery_failure":
        x[5] = 1.0
    elif ft == "repair_needed_failure":
        x[6] = 1.0

    if prev_action is not None and prev_action in REPAIR_ACTION_SET:
        x[7] = 1.0
    if prev_outcome is not None:
        x[8] = float(prev_outcome.get("sim_correct", 0))
        x[9] = float(prev_outcome.get("sim_hint", 0) > 0)

    return x