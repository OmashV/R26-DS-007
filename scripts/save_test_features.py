"""
Save the test split with engineered features and TSRP struggle vectors attached.
This file is the input to the Day 3 simulator and the bandit.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import json
import pandas as pd

from src.config import PROCESSED_DIR
from src.models.features import build_features
from src.models.targets import add_struggle_targets, apply_disengagement_zscore
from src.models.struggle_predictor import StrugglePredictor, META_PATH


def main():
    print("Loading test split...")
    test_df = pd.read_parquet(PROCESSED_DIR / "assist_test.parquet")

    print("Building features and targets...")
    feat = build_features(test_df)
    feat = add_struggle_targets(feat)

    meta = json.loads(META_PATH.read_text())
    mu, sd = meta["disengagement_zscore_mu"], meta["disengagement_zscore_sd"]
    feat = apply_disengagement_zscore(feat, mu, sd)

    print("Loading TSRP and predicting struggle vectors...")
    tsrp = StrugglePredictor()

    X_full = feat[tsrp.feature_cols_full].values
    X_diseng = feat[tsrp.feature_cols_diseng].values

    feat["repair_need_score"] = tsrp.m_repair.predict_proba(X_full)[:, 1]
    feat["effort_cost_score"] = tsrp.m_effort.predict(X_full)
    feat["disengagement_risk"] = tsrp.m_diseng.predict(X_diseng)

    out_path = PROCESSED_DIR / "assist_test_with_struggle.parquet"
    feat.to_parquet(out_path, index=False)
    print(f"Saved: {out_path}  rows={len(feat):,}")
    print("Columns:", list(feat.columns))


if __name__ == "__main__":
    main()