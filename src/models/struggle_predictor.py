"""
Turn-Level Struggle & Repair-Need Predictor (TSRP) — Model 1.

PURPOSE:
    Predict the struggle profile of the upcoming tutor turn from interaction
    history. This is a per-turn behavioural prediction, NOT a per-skill
    knowledge / mastery estimate. There is no mastery output here.
    This component is structurally distinct from BKT: different inputs
    (multi-channel: hints, attempts, response time, previous tutor signals),
    different outputs (failure-mode vector, not mastery), different time
    scale (per-turn, not per-skill long-term).

OUTPUTS (struggle vector consumed by the bandit):
    repair_need_score    : P(this turn requires pedagogical repair)
    effort_cost_score    : expected (hint_count + attempt_count) on this turn
    disengagement_risk   : z-scored expected log response time on the NEXT turn

Three independent LightGBM heads. Head 3 uses a leak-safe feature subset
(excludes log_response_time of the current row) and predicts the NEXT
row's disengagement — a strict forward-looking task.
"""

from pathlib import Path
import json
import joblib

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import (
    roc_auc_score, log_loss, mean_squared_error, mean_absolute_error
)

from src.config import PROCESSED_DIR, ARTIFACTS_DIR, SEED
from src.models.features import build_features, FEATURE_COLS
from src.models.targets import (
    add_struggle_targets, fit_disengagement_zscore, apply_disengagement_zscore
)


MODEL_DIR = ARTIFACTS_DIR / "tsrp"
HEAD_REPAIR = MODEL_DIR / "head_repair_need.pkl"
HEAD_EFFORT = MODEL_DIR / "head_effort_cost.pkl"
HEAD_DISENG = MODEL_DIR / "head_disengagement.pkl"
META_PATH = MODEL_DIR / "meta.json"

# Heads 1 and 2 use the full feature set. Head 3 excludes the current-row
# response time to avoid leakage into a forward-looking RT target.
DISENG_FEATURE_COLS = [c for c in FEATURE_COLS if c != "log_response_time"]


def _train_classifier(X_tr, y_tr, X_te, y_te):
    model = lgb.LGBMClassifier(
        n_estimators=500, learning_rate=0.05, num_leaves=63,
        min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
        random_state=SEED, n_jobs=-1, verbosity=-1,
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_te, y_te)],
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )
    return model


def _train_regressor(X_tr, y_tr, X_te, y_te):
    model = lgb.LGBMRegressor(
        n_estimators=500, learning_rate=0.05, num_leaves=63,
        min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
        random_state=SEED, n_jobs=-1, verbosity=-1,
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_te, y_te)],
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )
    return model


def train():
    print("Loading processed splits...")
    train_df = pd.read_parquet(PROCESSED_DIR / "assist_train.parquet")
    test_df = pd.read_parquet(PROCESSED_DIR / "assist_test.parquet")

    print("Building features...")
    train_feat = build_features(train_df)
    test_feat = build_features(test_df)

    print("Building targets...")
    train_feat = add_struggle_targets(train_feat)
    test_feat = add_struggle_targets(test_feat)

    mu, sd = fit_disengagement_zscore(train_feat)
    train_feat = apply_disengagement_zscore(train_feat, mu, sd)
    test_feat = apply_disengagement_zscore(test_feat, mu, sd)

    print(f"  Train rows: {len(train_feat):,}")
    print(f"  Test rows:  {len(test_feat):,}")

    # ---------- HEAD 1: repair-need ----------
    print("\nTraining HEAD 1: repair-need classifier...")
    X_tr = train_feat[FEATURE_COLS].values
    X_te = test_feat[FEATURE_COLS].values
    y_tr = train_feat["y_repair_need"].values
    y_te = test_feat["y_repair_need"].values
    print(f"  Positive rate (train): {y_tr.mean():.3f}")
    m_repair = _train_classifier(X_tr, y_tr, X_te, y_te)
    p_te = m_repair.predict_proba(X_te)[:, 1]
    auc_repair = float(roc_auc_score(y_te, p_te))
    ll_repair = float(log_loss(y_te, p_te))
    print(f"  Test AUC: {auc_repair:.4f}  LogLoss: {ll_repair:.4f}")

    # ---------- HEAD 2: effort-cost ----------
    print("\nTraining HEAD 2: effort-cost regressor...")
    y_tr = train_feat["y_effort_cost"].values
    y_te = test_feat["y_effort_cost"].values
    m_effort = _train_regressor(X_tr, y_tr, X_te, y_te)
    p_te = m_effort.predict(X_te)
    rmse_effort = float(np.sqrt(mean_squared_error(y_te, p_te)))
    mae_effort = float(mean_absolute_error(y_te, p_te))
    print(f"  Test RMSE: {rmse_effort:.4f}  MAE: {mae_effort:.4f}")

    # ---------- HEAD 3: next-turn disengagement (leak-safe) ----------
    print("\nTraining HEAD 3: disengagement regressor (next-turn, leak-safe)...")
    # Drop rows where the next-row target is NaN (last row per user)
    tr3 = train_feat.dropna(subset=["y_next_disengagement"]).copy()
    te3 = test_feat.dropna(subset=["y_next_disengagement"]).copy()
    print(f"  Head 3 train rows after dropping last-row-per-user: {len(tr3):,}")
    print(f"  Head 3 test rows after dropping last-row-per-user:  {len(te3):,}")

    X_tr3 = tr3[DISENG_FEATURE_COLS].values
    X_te3 = te3[DISENG_FEATURE_COLS].values
    y_tr3 = tr3["y_next_disengagement"].values
    y_te3 = te3["y_next_disengagement"].values

    # Baseline: a constant predictor (mean of training y) for sanity comparison
    baseline_pred = np.full_like(y_te3, fill_value=float(y_tr3.mean()))
    rmse_baseline = float(np.sqrt(mean_squared_error(y_te3, baseline_pred)))
    print(f"  Constant-mean baseline RMSE on test: {rmse_baseline:.4f}")

    m_diseng = _train_regressor(X_tr3, y_tr3, X_te3, y_te3)
    p_te3 = m_diseng.predict(X_te3)
    rmse_diseng = float(np.sqrt(mean_squared_error(y_te3, p_te3)))
    mae_diseng = float(mean_absolute_error(y_te3, p_te3))
    print(f"  Test RMSE: {rmse_diseng:.4f}  MAE: {mae_diseng:.4f}")
    if rmse_diseng < rmse_baseline:
        improvement = (rmse_baseline - rmse_diseng) / rmse_baseline * 100
        print(f"  Beats constant-mean baseline by {improvement:.1f}%")
    else:
        print("  WARNING: Does not beat constant-mean baseline. "
              "Disengagement signal may be weak from history alone.")

    # ---------- Save ----------
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(m_repair, HEAD_REPAIR)
    joblib.dump(m_effort, HEAD_EFFORT)
    joblib.dump(m_diseng, HEAD_DISENG)

    meta = {
        "feature_cols_full": FEATURE_COLS,
        "feature_cols_diseng": DISENG_FEATURE_COLS,
        "disengagement_zscore_mu": mu,
        "disengagement_zscore_sd": sd,
        "metrics": {
            "repair_need_auc": auc_repair,
            "repair_need_logloss": ll_repair,
            "effort_cost_rmse": rmse_effort,
            "effort_cost_mae": mae_effort,
            "disengagement_rmse": rmse_diseng,
            "disengagement_rmse_baseline": rmse_baseline,
            "disengagement_mae": mae_diseng,
        },
        "n_train": int(len(train_feat)),
        "n_test": int(len(test_feat)),
    }
    META_PATH.write_text(json.dumps(meta, indent=2))
    print(f"\nSaved 3 heads + meta to {MODEL_DIR}")


class StrugglePredictor:
    """Inference wrapper. Outputs a 3-dim struggle vector. No mastery output."""

    def __init__(self, model_dir: Path = MODEL_DIR):
        self.m_repair = joblib.load(model_dir / "head_repair_need.pkl")
        self.m_effort = joblib.load(model_dir / "head_effort_cost.pkl")
        self.m_diseng = joblib.load(model_dir / "head_disengagement.pkl")
        meta = json.loads((model_dir / "meta.json").read_text())
        self.feature_cols_full = meta["feature_cols_full"]
        self.feature_cols_diseng = meta["feature_cols_diseng"]

    def predict(self, feature_row: dict) -> dict:
        x_full = np.array([[feature_row[c] for c in self.feature_cols_full]])
        x_diseng = np.array([[feature_row[c] for c in self.feature_cols_diseng]])
        return {
            "repair_need_score": float(self.m_repair.predict_proba(x_full)[0, 1]),
            "effort_cost_score": float(self.m_effort.predict(x_full)[0]),
            "disengagement_risk": float(self.m_diseng.predict(x_diseng)[0]),
        }


if __name__ == "__main__":
    train()