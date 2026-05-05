"""
Central configuration for the FAPR-LB project.
All paths are relative to the project root.
"""

from pathlib import Path

# Project root = parent of src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

ASSISTMENTS_RAW = RAW_DIR / "assistments"
MATHDIAL_RAW = RAW_DIR / "mathdial"

# Artifacts (saved models, encoders)
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

# Reports (figures, tables)
REPORTS_DIR = PROJECT_ROOT / "reports"

# Random seed used everywhere
SEED = 42

# ASSISTments 2009 skill-builder file we expect (we download below)
ASSISTMENTS_FILE = ASSISTMENTS_RAW / "skill_builder_data.csv"

# MathDial files we expect
MATHDIAL_TRAIN = MATHDIAL_RAW / "train.csv"
MATHDIAL_VAL = MATHDIAL_RAW / "validation.csv"


def ensure_dirs():
    """Create any missing folders. Safe to call multiple times."""
    for d in [RAW_DIR, PROCESSED_DIR, ASSISTMENTS_RAW, MATHDIAL_RAW,
              ARTIFACTS_DIR, REPORTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    ensure_dirs()
    print("Project root:", PROJECT_ROOT)
    print("Folders ready.")