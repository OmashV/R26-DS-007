"""
Build the context vector for the LinTS bandit.

Layout (13 dims):
   0: bias = 1.0
   1: repair_need_score        (TSRP)
   2: effort_cost_score        (TSRP, z-scored at runtime)
   3: disengagement_risk       (TSRP)
   4: skill_correct_rate       (rolling feature)
   5: failure one-hot[low_mastery_failure]
   6: failure one-hot[repair_needed_failure]
   7: failure one-hot[disengagement_failure]
   8: failure one-hot[skill_regression_failure]
   9: failure one-hot[compounding_struggle_failure]
  10: prev_action_was_repair
  11: prev_outcome_correct
  12: prev_outcome_used_hint
"""

import numpy as np
from typing import Dict, Optional

CONTEXT_DIM = 13

REPAIR_ACTION_SET = {
    "worked_example",
    "direct_correction",
    "prerequisite_review",
}

# Z-score stats for effort_cost_score (loose defaults, fitted from train)
EFFORT_MEAN = 1.5
EFFORT_STD = 1.5

# Maps failure type to its one-hot index in the context vector.
# Order matches the layout above; no_failure has no slot (treated as
# all-zeros across the failure block).
FAILURE_TO_CTX_IDX = {
    "low_mastery_failure":          5,
    "repair_needed_failure":        6,
    "disengagement_failure":        7,
    "skill_regression_failure":     8,
    "compounding_struggle_failure": 9,
}


def build_context(obs: Dict, prev_action: Optional[str] = None,
                  prev_outcome: Optional[Dict] = None) -> np.ndarray:
    x = np.zeros(CONTEXT_DIM, dtype=float)

    x[0] = 1.0
    x[1] = float(obs["repair_need_score"])
    x[2] = (float(obs["effort_cost_score"]) - EFFORT_MEAN) / EFFORT_STD
    x[3] = float(obs["disengagement_risk"])
    x[4] = float(obs["skill_correct_rate"])

    ft = obs.get("failure_type", "no_failure")
    idx = FAILURE_TO_CTX_IDX.get(ft)
    if idx is not None:
        x[idx] = 1.0
    # no_failure: all failure-block dims stay 0

    if prev_action is not None and prev_action in REPAIR_ACTION_SET:
        x[10] = 1.0
    if prev_outcome is not None:
        x[11] = float(prev_outcome.get("sim_correct", 0))
        x[12] = float(prev_outcome.get("sim_hint", 0) > 0)

    return x