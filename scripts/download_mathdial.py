"""
Download the MathDial dataset from Hugging Face Hub and save as CSV
to data/raw/mathdial/. Run once.
"""

import sys
from pathlib import Path

# Make src/ importable
sys.path.append(str(Path(__file__).resolve().parent.parent))

from datasets import load_dataset
from src.config import MATHDIAL_RAW, ensure_dirs


def main():
    ensure_dirs()
    print("Downloading eth-nlped/mathdial from Hugging Face...")
    ds = load_dataset("eth-nlped/mathdial")
    print("Splits found:", list(ds.keys()))

    for split_name, split_data in ds.items():
        out_path = MATHDIAL_RAW / f"{split_name}.csv"
        split_data.to_csv(out_path, index=False)
        print(f"  Saved {split_name}: {len(split_data)} rows -> {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()