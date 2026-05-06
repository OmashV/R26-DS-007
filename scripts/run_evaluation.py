"""
Day 6 main evaluation. Runs all policies + 3 LinTS ablations across
N_EPISODES per policy, computes cumulative regret + per-policy metrics,
saves plots and a results CSV.

Usage:
    python scripts/run_evaluation.py
"""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.config import REPORTS_DIR
from src.simulator.student_simulator import StudentSimulator, REPAIR_ACTIONS
from src.policy.lin_ts import LinTSBandit, RandomPolicy, FixedPolicy, RulePolicy
from src.policy.action_templates import template_for
from src.models.preference_scorer import PreferenceScorer
from src.pipeline.context import build_context, CONTEXT_DIM
from src.pipeline.closed_loop import (
    run_episode, run_many_episodes, reward_function, compose_student_context_text
)


# ---- Configuration ----
N_EPISODES = 200       # per policy per seed
MAX_TURNS = 20
MIN_FAILURES = 3
SEEDS = [1234, 2345, 3456]   # 3 seeds for averaged results


# Singleton sim â€” built once, reused for every episode. We only swap the
# RNG between episodes, which is cheap. This avoids re-loading the parquet
# and re-building the bins on every episode.
_GLOBAL_SIM = None

def get_sim(seed: int):
    global _GLOBAL_SIM
    if _GLOBAL_SIM is None:
        _GLOBAL_SIM = StudentSimulator(rng=np.random.default_rng(seed))
    else:
        _GLOBAL_SIM.rng = np.random.default_rng(seed)
    return _GLOBAL_SIM


# ---- Ablation variants of the closed loop ----
# We modify build_context / reward_function locally for ablations.

ABLATIONS = {
    # name -> (mask_indices_to_zero_in_context, drop_pps_in_reward)
    "linTS_full":         ([], False),
    "linTS_no_failure":   ([5, 6], False),                  # zero out failure-type one-hots
    "linTS_no_reflect":   ([7, 8, 9], False),               # zero out prev_action / prev_outcome
    "linTS_no_pps":       ([], True),                       # full context but reward ignores PPS
}


def run_episode_with_ablation(sim, policy, pps, episode_idx, max_turns,
                              context_mask, drop_pps_in_reward,
                              min_failures=MIN_FAILURES):
    """Like run_episode, but applies the ablation knobs."""
    obs = sim.reset_with_min_failures(min_failures=min_failures)
    if obs is None:
        return []

    logs = []
    prev_action = None
    prev_outcome = None

    for t in range(max_turns):
        if obs is None:
            break
        x = build_context(obs, prev_action, prev_outcome)
        # Apply context mask (zero out specified indices)
        if context_mask:
            x = x.copy()
            for idx in context_mask:
                x[idx] = 0.0

        student_text = compose_student_context_text(obs)
        cand_texts = [template_for(a) for a in REPAIR_ACTIONS]
        scores = pps.score_batch(student_text, cand_texts)
        pps_scores = dict(zip(REPAIR_ACTIONS, scores))
        action = policy.select(x)

        next_obs, info = sim.step(action)
        if info.get("skipped"):
            break

        pps_chosen = pps_scores[action]
        # Ablated reward: drop PPS contribution if specified
        if drop_pps_in_reward:
            r = float(np.clip(0.5 * info["sim_correct"]
                              - 0.2 * (info["sim_hint"] > 0), 0.0, 1.0))
        else:
            r = reward_function(info["sim_correct"], info["sim_hint"], pps_chosen)

        policy.update(action, x, r)

        logs.append({
            "episode": episode_idx, "turn": t,
            "user_id": int(obs["user_id"]),
            "failure_type": str(obs.get("failure_type", "")),
            "action": action,
            "sim_correct": info["sim_correct"],
            "sim_hint": info["sim_hint"],
            "pps_score": pps_chosen,
            "reward": r,
        })
        prev_action = action
        prev_outcome = info
        obs = next_obs
        if info["done"]:
            break
    return logs


def run_full_eval(policy_factory, pps, n_episodes, ablation=None,
                  base_seed=1234) -> pd.DataFrame:
    """Run n_episodes through a freshly created policy. Each episode uses a
    different sim seed for variety, but seeds are deterministic across runs."""
    policy = policy_factory()
    all_logs = []
    if ablation is None:
        # Baseline (non-ablated) path: use the standard run_episode
        for ep in range(n_episodes):
            sim = get_sim(seed=base_seed + ep)
            ep_logs = run_episode(sim, policy, pps, episode_idx=ep,
                                  max_turns=MAX_TURNS, min_failures=MIN_FAILURES)
            for l in ep_logs:
                all_logs.append({
                    "episode": l.episode, "turn": l.turn,
                    "user_id": l.user_id, "failure_type": l.failure_type,
                    "action": l.action,
                    "sim_correct": l.sim_correct, "sim_hint": l.sim_hint,
                    "pps_score": l.pps_score_chosen, "reward": l.reward,
                })
    else:
        context_mask, drop_pps = ablation
        for ep in range(n_episodes):
            sim = get_sim(seed=base_seed + ep)
            ep_logs = run_episode_with_ablation(
                sim, policy, pps, episode_idx=ep,
                max_turns=MAX_TURNS, context_mask=context_mask,
                drop_pps_in_reward=drop_pps,
            )
            all_logs.extend(ep_logs)
    return pd.DataFrame(all_logs)


def cumulative_regret(reward_series: pd.Series, optimal_per_turn: float = None) -> np.ndarray:
    """
    Cumulative regret relative to a reference 'optimal' per-turn reward.
    We use the best fixed policy's average reward as the reference,
    so regret(t) = sum_{i<=t} (optimal - r_i).
    """
    if optimal_per_turn is None:
        optimal_per_turn = reward_series.max()
    return np.cumsum(optimal_per_turn - reward_series.values)


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading PPS scorer (one-time cost)...")
    pps = PreferenceScorer()

    print(f"\n=== 3-seed evaluation ===")
    print(f"Seeds: {SEEDS}")
    print(f"Episodes per seed per policy: {N_EPISODES}")
    print(f"Total runs: {len(SEEDS)} seeds × 8 policies = {len(SEEDS)*8}")

    policies_factory = {
        "random":   lambda: RandomPolicy(seed=1),
        "fixed_we": lambda: FixedPolicy("worked_example"),
        "rule":     lambda: RulePolicy(),
        "lin_ts":   lambda: LinTSBandit(context_dim=CONTEXT_DIM, seed=0),
    }

    # Collect per-seed results
    main_results_per_seed = {name: [] for name in policies_factory}
    ablation_results_per_seed = {name: [] for name in ABLATIONS}

    for seed_idx, seed in enumerate(SEEDS):
        print(f"\n=== Seed {seed_idx+1}/{len(SEEDS)}: base_seed={seed} ===")

        # Stage 1: 4 policies
        for name, factory in policies_factory.items():
            print(f"  [policy: {name}] running {N_EPISODES} episodes...")
            df = run_full_eval(factory, pps, N_EPISODES, ablation=None,
                               base_seed=seed)
            main_results_per_seed[name].append(df)
            print(f"    turns={len(df)}  mean_reward={df['reward'].mean():.4f}  "
                  f"correct_rate={df['sim_correct'].mean():.4f}")

        # Stage 2: 4 ablations
        for name, ab in ABLATIONS.items():
            print(f"  [ablation: {name}] running {N_EPISODES} episodes...")
            df = run_full_eval(
                lambda: LinTSBandit(context_dim=CONTEXT_DIM, seed=0),
                pps, N_EPISODES, ablation=ab,
                base_seed=seed + 1000,
            )
            ablation_results_per_seed[name].append(df)
            print(f"    turns={len(df)}  mean_reward={df['reward'].mean():.4f}  "
                  f"correct_rate={df['sim_correct'].mean():.4f}")

    # ---- Aggregate across seeds ----
    print("\n=== Aggregating across seeds ===")
    summary_rows = []
    main_results = {}
    ablation_results = {}

    for name, dfs in {**main_results_per_seed, **ablation_results_per_seed}.items():
        if not dfs:
            continue
        rewards_per_seed = [df["reward"].mean() for df in dfs if not df.empty]
        correct_per_seed = [df["sim_correct"].mean() for df in dfs if not df.empty]
        hints_per_seed = [df["sim_hint"].mean() for df in dfs if not df.empty]
        # Pool the per-seed dataframes for plotting (use first seed)
        if name in policies_factory:
            main_results[name] = dfs[0]
        else:
            ablation_results[name] = dfs[0]
        summary_rows.append({
            "policy": name,
            "n_seeds": len(rewards_per_seed),
            "n_episodes": N_EPISODES,
            "mean_reward_avg": float(np.mean(rewards_per_seed)),
            "mean_reward_std": float(np.std(rewards_per_seed)),
            "correct_rate_avg": float(np.mean(correct_per_seed)),
            "correct_rate_std": float(np.std(correct_per_seed)),
            "mean_hint_avg": float(np.mean(hints_per_seed)),
        })

    summary_df = pd.DataFrame(summary_rows).round(4)
    summary_path = REPORTS_DIR / "evaluation_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved summary -> {summary_path}")
    print(summary_df.to_string(index=False))

    # ---------- Stage 4: Plots (uses first seed's data for trace plots) ----------
    print("\n=== Stage 4: Generating plots ===")

    # Plot 1: Cumulative reward curve (4 main policies, first seed)
    plt.figure(figsize=(8, 5))
    for name, df in main_results.items():
        if df.empty:
            continue
        cum_mean = df["reward"].cumsum() / (np.arange(len(df)) + 1)
        plt.plot(cum_mean.values, label=name, linewidth=1.6)
    plt.xlabel("Turn (across episodes)")
    plt.ylabel("Cumulative mean reward")
    plt.title(f"Closed-loop performance: 4 policies × {N_EPISODES} episodes (seed 1)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    p1 = REPORTS_DIR / "eval_cumreward_main.png"
    plt.savefig(p1, dpi=140); plt.close()
    print(f"  Saved {p1}")

    # Plot 2: Cumulative regret
    best_mean = max(df["reward"].mean() for df in main_results.values() if not df.empty)
    plt.figure(figsize=(8, 5))
    for name, df in main_results.items():
        if df.empty:
            continue
        regret = cumulative_regret(df["reward"], optimal_per_turn=best_mean)
        plt.plot(regret, label=name, linewidth=1.6)
    plt.xlabel("Turn")
    plt.ylabel("Cumulative regret")
    plt.title("Cumulative regret across policies")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    p2 = REPORTS_DIR / "eval_cumregret_main.png"
    plt.savefig(p2, dpi=140); plt.close()
    print(f"  Saved {p2}")

    # Plot 3: LinTS arm-pull distribution (first seed)
    lin_df = main_results["lin_ts"]
    arm_counts = lin_df["action"].value_counts().reindex(REPAIR_ACTIONS, fill_value=0)
    plt.figure(figsize=(8, 4.5))
    arm_counts.plot(kind="barh")
    plt.xlabel("Number of pulls")
    plt.title(f"LinTS arm-pull distribution after {N_EPISODES} episodes")
    plt.tight_layout()
    p3 = REPORTS_DIR / "eval_lin_ts_arm_pulls.png"
    plt.savefig(p3, dpi=140); plt.close()
    print(f"  Saved {p3}")

    # Plot 4: Ablation bar chart with error bars
    abl_means = {row["policy"]: (row["mean_reward_avg"], row["mean_reward_std"])
                 for _, row in summary_df.iterrows()
                 if row["policy"] in ABLATIONS}
    plt.figure(figsize=(8, 4.5))
    names = list(abl_means.keys())
    vals = [abl_means[n][0] for n in names]
    errs = [abl_means[n][1] for n in names]
    bars = plt.bar(names, vals, yerr=errs, capsize=5,
                   color=["#4c72b0", "#dd8452", "#55a467", "#c44e52"])
    for b, v in zip(bars, vals):
        plt.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}",
                 ha="center", fontsize=9)
    plt.ylabel("Mean reward per turn (±1 SD across seeds)")
    plt.title(f"Ablation: contribution of each component (avg over {len(SEEDS)} seeds)")
    plt.xticks(rotation=15)
    plt.tight_layout()
    p4 = REPORTS_DIR / "eval_ablations.png"
    plt.savefig(p4, dpi=140); plt.close()
    print(f"  Saved {p4}")

    # Plot 5: Per-failure-type action distribution (first seed)
    pivot = (lin_df.groupby(["failure_type", "action"]).size()
             .unstack(fill_value=0))
    pivot = pivot.div(pivot.sum(axis=1), axis=0)
    pivot = pivot.reindex(columns=REPAIR_ACTIONS, fill_value=0)
    pivot.plot(kind="bar", stacked=True, figsize=(9, 5),
               colormap="tab10")
    plt.ylabel("Fraction of pulls")
    plt.title("LinTS: action distribution per failure type")
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    plt.xticks(rotation=0)
    plt.tight_layout()
    p5 = REPORTS_DIR / "eval_action_per_failure.png"
    plt.savefig(p5, dpi=140); plt.close()
    print(f"  Saved {p5}")

    print(f"\n=== Done. {len(SEEDS)}-seed evaluation complete. ===")
    print(f"Plots in {REPORTS_DIR}")
if __name__ == "__main__":
    main()
