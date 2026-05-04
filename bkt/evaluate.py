"""
Evaluation utilities for the trained BKT model.

Produces AUC and RMSE on a held-out test split (R2, R8).
These metrics are required evidence for Progress Presentation 1 (requirements.md §10).

Intended usage (once implemented):
    python -m bkt.evaluate --model models/bkt_model.pkl --data data/raw/assistments.csv
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def evaluate_model(model: Any, test_data: Any) -> dict[str, float]:
    """
    Compute AUC and RMSE of model predictions on test_data (R2, R8).

    test_data is a held-out DataFrame returned by bkt.train.split_train_test().
    Returns a metrics dict: {auc: float, rmse: float}.
    """
    raise NotImplementedError


def generate_report(metrics: dict[str, float], output_path: str) -> None:
    """
    Write a markdown evaluation report to output_path.

    Report includes: dataset info, model parameters, AUC, RMSE, comparison
    against published ASSISTments baselines.
    """
    raise NotImplementedError
