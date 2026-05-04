"""
BKT model training pipeline.

Trains a Bayesian Knowledge Tracing model on the cleaned ASSISTments dataset
and saves it for inference.

Implements: FR8, FR9 (knowledge tracing requirements from requirements.md)
"""

import logging
import pickle
from pathlib import Path

import pandas as pd
from pyBKT.models import Model

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "bkt_training_data.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "bkt_model.pkl"

# Training config
RANDOM_SEED = 42
NUM_FITS = 5  # pyBKT runs fitting multiple times to avoid local optima

def load_training_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """
    Load the cleaned BKT training data.

    Args:
        path: Path to the processed CSV.

    Returns:
        DataFrame with columns: user_id, skill_name, correct, order_id

    Raises:
        FileNotFoundError: If the data file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Training data not found at {path}. "
            f"Run notebooks/bkt_exploration.ipynb first to generate it."
        )

    df = pd.read_csv(path)
    logger.info(
        f"Loaded {len(df):,} interactions, "
        f"{df['user_id'].nunique():,} students, "
        f"{df['skill_name'].nunique()} skills"
    )
    return df

def train_bkt(df: pd.DataFrame, num_fits: int = NUM_FITS, seed: int = RANDOM_SEED) -> Model:
    """
    Train a BKT model on the provided data.

    Args:
        df: Training data with user_id, skill_name, correct columns.
        num_fits: Number of fitting attempts (best result kept).
        seed: Random seed for reproducibility.

    Returns:
        Trained pyBKT Model.
    """
    logger.info(f"Training BKT model (num_fits={num_fits}, seed={seed})...")

    model = Model(seed=seed, num_fits=num_fits)
    model.fit(data=df)

    logger.info("Training complete.")
    return model

def save_model(model: Model, path: Path = MODEL_PATH) -> None:
    """Save the trained model to disk using pickle."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Model saved to {path}")


def summarise_parameters(model: Model) -> None:
    """Log a summary of the learned BKT parameters."""
    params = model.params()
    logger.info(f"Learned parameters for {len(params.index.get_level_values('skill').unique())} skills")
    logger.info("\nSample parameters (first 5 skills):")
    logger.info(f"\n{params.head(20)}")

def main() -> None:
    """Run the full training pipeline."""
    df = load_training_data()
    model = train_bkt(df)
    save_model(model)
    summarise_parameters(model)


if __name__ == "__main__":
    main()