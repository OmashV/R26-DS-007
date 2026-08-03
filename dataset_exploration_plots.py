"""
Dataset exploration visualisations for the Meta-Agent report (Objective 3).

Generates four plots from the cleaned ASSISTments 2009-2010 training data:
  1. Histogram of attempts per student
  2. Histogram of attempts per skill
  3. Bar chart of correctness rate per skill (top 10 easiest vs hardest)
  4. Heatmap of student-by-skill engagement density

Can be run from the project root directory.
"""

import sys
from pathlib import Path

# Get the directory of this script
script_dir = Path(__file__).parent

# Add the script directory to sys.path for imports
sys.path.append(str(script_dir))

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ─── Consistent style across all four plots ────────────────────────────────
sns.set_style("whitegrid")
plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 200,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "sans-serif",
})

# ─── Load the cleaned training data ────────────────────────────────────────
data_path = script_dir / "data" / "processed" / "bkt_training_data.csv"
df = pd.read_csv(data_path)

print(f"Loaded {len(df):,} interactions, "
      f"{df['user_id'].nunique():,} students, "
      f"{df['skill_name'].nunique()} skills")
print(f"Overall correctness rate: {df['correct'].mean():.1%}")

# Output directory for saved figures
OUT_DIR = script_dir / "docs" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ─── Plot 1: Histogram of attempts per student ─────────────────────────────
attempts_per_student = df.groupby("user_id").size()

fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(
    attempts_per_student,
    bins=60,
    color="#3B7DB8",
    edgecolor="white",
    linewidth=0.5,
)
ax.axvline(
    attempts_per_student.median(),
    color="#D85A30",
    linestyle="--",
    linewidth=1.5,
    label=f"Median = {int(attempts_per_student.median()):,}",
)
ax.axvline(
    attempts_per_student.mean(),
    color="#1D9E75",
    linestyle="--",
    linewidth=1.5,
    label=f"Mean = {int(attempts_per_student.mean()):,}",
)
ax.set_title("Distribution of attempts per student")
ax.set_xlabel("Number of attempts")
ax.set_ylabel("Number of students")
ax.set_xlim(0, attempts_per_student.quantile(0.99))  # clip the long tail for readability
ax.legend(frameon=False, loc="upper right")
plt.tight_layout()
plt.savefig(OUT_DIR / "01_attempts_per_student.png", bbox_inches="tight")
plt.show()


# ─── Plot 2: Histogram of attempts per skill ───────────────────────────────
attempts_per_skill = df.groupby("skill_name").size()

fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(
    attempts_per_skill,
    bins=40,
    color="#7F77DD",
    edgecolor="white",
    linewidth=0.5,
)
ax.axvline(
    attempts_per_skill.median(),
    color="#D85A30",
    linestyle="--",
    linewidth=1.5,
    label=f"Median = {int(attempts_per_skill.median()):,}",
)
ax.axvline(
    attempts_per_skill.mean(),
    color="#1D9E75",
    linestyle="--",
    linewidth=1.5,
    label=f"Mean = {int(attempts_per_skill.mean()):,}",
)
ax.set_title("Distribution of attempts per skill")
ax.set_xlabel("Number of attempts")
ax.set_ylabel("Number of skills")
ax.legend(frameon=False, loc="upper right")
plt.tight_layout()
plt.savefig(OUT_DIR / "02_attempts_per_skill.png", bbox_inches="tight")
plt.show()


# ─── Plot 3: Correctness rate per skill — top 10 easiest vs hardest ────────
correctness = (
    df.groupby("skill_name")
      .agg(correct_rate=("correct", "mean"), n=("correct", "size"))
      .sort_values("correct_rate")
)

# Only include skills with enough data to be meaningful
correctness = correctness[correctness["n"] >= 200]

easiest = correctness.tail(10).iloc[::-1]   # highest first
hardest = correctness.head(10)              # lowest first

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharex=True)

# Left — hardest skills
axes[0].barh(
    hardest.index,
    hardest["correct_rate"],
    color="#D85A30",
    edgecolor="white",
    linewidth=0.5,
)
axes[0].set_title("10 hardest skills")
axes[0].set_xlabel("Correctness rate")
axes[0].invert_yaxis()
axes[0].set_xlim(0, 1)
for i, (rate, n) in enumerate(zip(hardest["correct_rate"], hardest["n"])):
    axes[0].text(rate + 0.01, i, f"{rate:.1%}  (n={n:,})",
                 va="center", fontsize=9, color="#444")

# Right — easiest skills
axes[1].barh(
    easiest.index,
    easiest["correct_rate"],
    color="#1D9E75",
    edgecolor="white",
    linewidth=0.5,
)
axes[1].set_title("10 easiest skills")
axes[1].set_xlabel("Correctness rate")
axes[1].invert_yaxis()
axes[1].set_xlim(0, 1)
for i, (rate, n) in enumerate(zip(easiest["correct_rate"], easiest["n"])):
    axes[1].text(rate + 0.01, i, f"{rate:.1%}  (n={n:,})",
                 va="center", fontsize=9, color="#444")

fig.suptitle("Skill-level difficulty: easiest vs hardest skills",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(OUT_DIR / "03_skill_correctness_easy_vs_hard.png", bbox_inches="tight")
plt.show()


# ─── Plot 4: Heatmap of student × skill engagement density ─────────────────
# Build a binary attempted matrix: rows = students, cols = skills
attempted = (
    df.groupby(["user_id", "skill_name"])
      .size()
      .unstack(fill_value=0)
)

# Convert to binary "did student attempt this skill"
attempted_binary = (attempted > 0).astype(int)

# Order rows by total attempts (most engaged students at top)
attempted_binary = attempted_binary.loc[
    attempted_binary.sum(axis=1).sort_values(ascending=False).index
]
# Order columns by total attempts (most-attempted skills at left)
attempted_binary = attempted_binary[
    attempted_binary.sum(axis=0).sort_values(ascending=False).index
]

density = attempted_binary.values.mean()
print(f"\nStudent × skill density: {density:.1%} "
      f"({attempted_binary.values.sum():,} populated cells "
      f"out of {attempted_binary.size:,})")

fig, ax = plt.subplots(figsize=(11, 6.5))
ax.imshow(
    attempted_binary.values,
    aspect="auto",
    cmap="Blues",
    interpolation="nearest",
)
ax.set_title(
    f"Student × skill engagement matrix  "
    f"(density {density:.1%}; ordered by attempts)"
)
ax.set_xlabel("Skill (sorted by total attempts)")
ax.set_ylabel("Student (sorted by total attempts)")
ax.set_xticks([])
ax.set_yticks([])

# Subtle legend explaining the colour
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="#08306B", label="Attempted"),
    Patch(facecolor="#F7FBFF", edgecolor="#cccccc", label="Not attempted"),
]
ax.legend(
    handles=legend_elements,
    loc="upper right",
    frameon=True,
    facecolor="white",
    edgecolor="#cccccc",
)
plt.tight_layout()
plt.savefig(OUT_DIR / "04_student_skill_density.png", bbox_inches="tight")
plt.show()


print(f"\nAll four plots saved to {OUT_DIR.resolve()}")
