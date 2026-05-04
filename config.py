"""Configuration constants for the Meta-Agent."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# Paths
BKT_PARAMS_PATH = PROJECT_ROOT / "models" / "bkt_params.json"
DB_PATH = PROJECT_ROOT / "data" / "meta_agent.db"

# Mastery thresholds
MASTERY_STRONG_THRESHOLD = 0.7
MASTERY_WEAK_THRESHOLD = 0.3

# Regression detection
REGRESSION_DROP_THRESHOLD = 0.2  # mastery drop > this triggers regression flag