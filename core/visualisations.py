"""
Presentation-quality visualisations for the Meta-Agent demo.

Provides plot functions for:
- Mastery trajectory (P(mastery) evolution over a sequence of attempts)
- Knowledge graph snapshot (current mastery across all skills)
- Multi-session evolution (mastery change session-over-session)
- Student comparison (two students on the same skill)

All functions return matplotlib Figure objects so they can be displayed
inline in notebooks or saved to disk for the slide deck.
"""

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# Consistent visual style for all plots
COLOR_STRONG = "#2E7D32"   # green
COLOR_PARTIAL = "#F9A825"  # amber
COLOR_WEAK = "#C62828"     # red
COLOR_NEUTRAL = "#546E7A"  # slate
COLOR_REGRESSION = "#8E24AA"  # purple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / "docs" / "figures"


def ensure_figures_dir() -> None:
    """Make sure the figures output folder exists."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def plot_mastery_trajectory(
    skill: str,
    attempts: list[int],
    trajectory: list[float],
    student_id: Optional[str] = None,
    strong_threshold: float = 0.7,
    weak_threshold: float = 0.3,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot how mastery probability evolves across a sequence of attempts.

    Args:
        skill: Skill name (used in title).
        attempts: List of 0/1 values — what the student answered.
        trajectory: List of mastery probabilities, length = len(attempts) + 1
                    (first entry is the prior before any attempt).
        student_id: Optional student identifier for the title.
        strong_threshold: Mastery threshold for "strong" label (default 0.7).
        weak_threshold: Mastery threshold for "weak" label (default 0.3).
        save_path: If provided, save the figure to this path.

    Returns:
        matplotlib Figure object.
    """
    if len(trajectory) != len(attempts) + 1:
        raise ValueError(
            f"trajectory length must be len(attempts) + 1, "
            f"got {len(trajectory)} vs {len(attempts) + 1}"
        )

    fig, ax = plt.subplots(figsize=(10, 5.5))

    # X axis: attempt number (0 = prior, 1..N = after each attempt)
    x = list(range(len(trajectory)))

    # Shade the threshold bands
    ax.axhspan(strong_threshold, 1.0, alpha=0.08, color=COLOR_STRONG, zorder=0)
    ax.axhspan(weak_threshold, strong_threshold, alpha=0.08, color=COLOR_PARTIAL, zorder=0)
    ax.axhspan(0, weak_threshold, alpha=0.08, color=COLOR_WEAK, zorder=0)

    # Threshold lines
    ax.axhline(strong_threshold, color=COLOR_STRONG, linestyle="--", linewidth=1, alpha=0.6)
    ax.axhline(weak_threshold, color=COLOR_WEAK, linestyle="--", linewidth=1, alpha=0.6)

    # Trajectory line
    ax.plot(x, trajectory, color=COLOR_NEUTRAL, linewidth=2.5, zorder=2)

    # Markers: green up-triangle for correct, red down-triangle for wrong
    # The marker is placed at the trajectory point AFTER the attempt
    for i, a in enumerate(attempts):
        marker_x = i + 1
        marker_y = trajectory[i + 1]
        if a == 1:
            ax.scatter(marker_x, marker_y, marker="^", s=140,
                       color=COLOR_STRONG, edgecolor="white", linewidth=1.5,
                       zorder=3, label="Correct" if i == 0 or attempts[:i].count(1) == 0 else "")
        else:
            ax.scatter(marker_x, marker_y, marker="v", s=140,
                       color=COLOR_WEAK, edgecolor="white", linewidth=1.5,
                       zorder=3, label="Wrong" if i == 0 or attempts[:i].count(0) == 0 else "")

    # Prior marker (open circle at x=0)
    ax.scatter(0, trajectory[0], marker="o", s=100,
               facecolor="white", edgecolor=COLOR_NEUTRAL, linewidth=2,
               zorder=3, label="Prior")

    # Annotate the final mastery probability
    final_p = trajectory[-1]
    ax.annotate(
        f"P(mastery) = {final_p:.3f}",
        xy=(len(trajectory) - 1, final_p),
        xytext=(10, 0),
        textcoords="offset points",
        fontsize=11,
        fontweight="bold",
        va="center",
    )

    # Threshold labels on the right
    ax.text(len(trajectory) - 0.3, strong_threshold + 0.02, "Strong",
            fontsize=9, color=COLOR_STRONG, ha="right", alpha=0.8)
    ax.text(len(trajectory) - 0.3, weak_threshold + 0.02, "Partial",
            fontsize=9, color="#B8860B", ha="right", alpha=0.8)
    ax.text(len(trajectory) - 0.3, 0.02, "Weak",
            fontsize=9, color=COLOR_WEAK, ha="right", alpha=0.8)

    # Cosmetics
    title = f"Mastery trajectory — {skill}"
    if student_id:
        title += f" (student: {student_id})"
    ax.set_title(title, fontsize=13, pad=15)
    ax.set_xlabel("Attempt number", fontsize=11)
    ax.set_ylabel("P(mastery)", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(-0.3, len(trajectory) - 0.3 + 1.2)
    ax.set_xticks(x)
    ax.grid(True, axis="y", alpha=0.25, linestyle=":")
    ax.legend(loc="upper left", framealpha=0.95)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    if save_path:
        ensure_figures_dir()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved trajectory plot to {save_path}")

    return fig

def plot_knowledge_graph(
    graph: list[dict],
    student_id: Optional[str] = None,
    strong_threshold: float = 0.7,
    weak_threshold: float = 0.3,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Horizontal bar chart of a student's mastery across all attempted skills.

    Args:
        graph: List of mastery entries from KnowledgeGraph.get_student_graph().
        student_id: Optional student identifier for the title.
        strong_threshold: Threshold for "strong" colour band.
        weak_threshold: Threshold for "weak" colour band.
        save_path: If provided, save the figure to this path.

    Returns:
        matplotlib Figure object.
    """
    if not graph:
        raise ValueError("Cannot plot an empty knowledge graph")

    # Sort by mastery descending so the best skills are at the top
    sorted_graph = sorted(graph, key=lambda e: e["mastery_probability"], reverse=True)

    skills = [e["skill"] for e in sorted_graph]
    masteries = [e["mastery_probability"] for e in sorted_graph]

    # Colour each bar by category
    def colour_for(p: float) -> str:
        if p >= strong_threshold:
            return COLOR_STRONG
        if p < weak_threshold:
            return COLOR_WEAK
        return COLOR_PARTIAL

    colours = [colour_for(p) for p in masteries]

    # Mark regressions distinctly with a coloured edge
    edge_colours = []
    edge_widths = []
    for e in sorted_graph:
        prev = e.get("previous_mastery_probability")
        if prev is not None and prev - e["mastery_probability"] > 0.2 and prev >= weak_threshold:
            edge_colours.append(COLOR_REGRESSION)
            edge_widths.append(2.5)
        else:
            edge_colours.append("white")
            edge_widths.append(0.5)

    # Sizing scales with number of skills
    height = max(3.5, 0.45 * len(skills) + 1.5)
    fig, ax = plt.subplots(figsize=(10, height))

    bars = ax.barh(
        skills,
        masteries,
        color=colours,
        edgecolor=edge_colours,
        linewidth=edge_widths,
        height=0.7,
    )

    # Threshold lines
    ax.axvline(strong_threshold, color=COLOR_STRONG, linestyle="--", linewidth=1, alpha=0.5)
    ax.axvline(weak_threshold, color=COLOR_WEAK, linestyle="--", linewidth=1, alpha=0.5)

    # Annotate each bar with its probability
    for bar, p in zip(bars, masteries):
        ax.text(
            bar.get_width() + 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{p:.3f}",
            va="center",
            fontsize=9.5,
        )

    # Build a legend including category colours and the regression marker
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=COLOR_STRONG, label="Strong (≥ 0.7)"),
        Patch(facecolor=COLOR_PARTIAL, label="Partial (0.3 – 0.7)"),
        Patch(facecolor=COLOR_WEAK, label="Weak (< 0.3)"),
    ]
    if any(ec == COLOR_REGRESSION for ec in edge_colours):
        legend_handles.append(
            Patch(facecolor="white", edgecolor=COLOR_REGRESSION,
                  linewidth=2.5, label="↓ Regression detected")
        )
    ax.legend(handles=legend_handles, loc="lower right", framealpha=0.95)

    # Cosmetics
    title = "Knowledge graph"
    if student_id:
        title += f" — {student_id}"
    ax.set_title(title, fontsize=13, pad=15)
    ax.set_xlabel("P(mastery)", fontsize=11)
    ax.set_xlim(0, 1.15)
    ax.invert_yaxis()  # highest mastery at top
    ax.grid(True, axis="x", alpha=0.25, linestyle=":")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    if save_path:
        ensure_figures_dir()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved knowledge graph plot to {save_path}")

    return fig