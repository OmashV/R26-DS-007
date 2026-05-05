"""
Closed-loop orchestrator for FAPR-LB.

Per turn:
  1. Get observation from simulator.
  2. Build context vector from obs + previous action/outcome.
  3. Score each candidate action's canned template via PPS.
  4. Bandit selects action via Thompson sampling.
  5. Simulator steps with the chosen action.
  6. Compute reward = 0.5 * sim_correct + 0.3 * pps_score - 0.2 * sim_hint_used.
  7. Bandit posterior update.
  8. Log everything.

The PPS scoring is computed for ALL candidate actions on each turn so we
can record per-arm PPS scores for the eventual reward and for analysis.
However, only the chosen action's PPS contributes to the reward.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.simulator.student_simulator import StudentSimulator, REPAIR_ACTIONS
from src.policy.lin_ts import LinTSBandit, RandomPolicy, FixedPolicy, RulePolicy
from src.policy.action_templates import template_for
from src.pipeline.context import build_context, CONTEXT_DIM


def compose_student_context_text(obs: Dict) -> str:
    """A short textual snippet describing the student's current situation,
    used as the 'context' passed to the PPS scorer."""
    ft = obs.get("failure_type", "unknown")
    sk = obs.get("skill_correct_rate", 0.5)
    return (f"Student is on a problem; their rolling correct rate on this "
            f"skill is {sk:.2f}. Detected failure type: {ft}. "
            f"Help them pedagogically.")


def reward_function(sim_correct: int, sim_hint: int, pps_score: float) -> float:
    r = 0.5 * sim_correct + 0.3 * pps_score - 0.2 * (sim_hint > 0)
    return float(np.clip(r, 0.0, 1.0))


@dataclass
class TurnLog:
    episode: int
    turn: int
    user_id: int
    failure_type: str
    action: str
    sim_correct: int
    sim_hint: int
    pps_score_chosen: float
    reward: float
    pps_scores_all: Dict[str, float] = field(default_factory=dict)


def run_episode(sim: StudentSimulator, policy, pps,
                episode_idx: int = 0, max_turns: int = 20,
                user_id: Optional[int] = None,
                min_failures: int = 3) -> List[TurnLog]:
    """Run one episode through the closed loop. Returns the turn-by-turn log."""
    if user_id is None:
        obs = sim.reset_with_min_failures(min_failures=min_failures)
        if obs is None:
            return []
    else:
        obs = sim.reset(user_id=user_id)

    logs: List[TurnLog] = []
    prev_action = None
    prev_outcome = None

    for t in range(max_turns):
        if obs is None:
            break

        # Build context
        x = build_context(obs, prev_action, prev_outcome)

        # Score every action's template once (for logging)
        student_text = compose_student_context_text(obs)
        cand_texts = [template_for(a) for a in REPAIR_ACTIONS]
        scores = pps.score_batch(student_text, cand_texts)
        pps_scores = dict(zip(REPAIR_ACTIONS, scores))

        # Policy selects action
        action = policy.select(x)

        # Step simulator
        next_obs, info = sim.step(action)
        if info.get("skipped"):
            break

        # Reward
        pps_chosen = pps_scores[action]
        r = reward_function(info["sim_correct"], info["sim_hint"], pps_chosen)

        # Posterior update
        policy.update(action, x, r)

        # Log
        logs.append(TurnLog(
            episode=episode_idx, turn=t,
            user_id=int(obs["user_id"]),
            failure_type=str(obs.get("failure_type", "")),
            action=action,
            sim_correct=info["sim_correct"],
            sim_hint=info["sim_hint"],
            pps_score_chosen=pps_chosen,
            reward=r,
            pps_scores_all=pps_scores,
        ))

        # Roll over
        prev_action = action
        prev_outcome = info
        obs = next_obs
        if info["done"]:
            break

    return logs


def run_many_episodes(sim_factory, policy, pps,
                      n_episodes: int = 30,
                      max_turns: int = 20,
                      min_failures: int = 3) -> pd.DataFrame:
    """Run n_episodes through `policy`. sim_factory() returns a fresh sim."""
    all_logs = []
    sim = sim_factory()
    for ep in range(n_episodes):
        print(f"    episode {ep + 1}/{n_episodes}", flush=True)
        ep_logs = run_episode(sim, policy, pps, episode_idx=ep,
                            max_turns=max_turns, min_failures=min_failures)
        all_logs.extend(ep_logs)
    if not all_logs:
        return pd.DataFrame()
    df = pd.DataFrame([{
        "episode": l.episode, "turn": l.turn,
        "user_id": l.user_id, "failure_type": l.failure_type,
        "action": l.action,
        "sim_correct": l.sim_correct, "sim_hint": l.sim_hint,
        "pps_score": l.pps_score_chosen, "reward": l.reward,
    } for l in all_logs])
    return df


if __name__ == "__main__":
    # Day-5 smoke: 30 episodes per policy, compare cumulative reward.
    from src.models.preference_scorer import PreferenceScorer

    print("Loading PPS...")
    pps = PreferenceScorer()
    print("Loading simulator...")

    def fresh_sim(seed=0):
        return StudentSimulator(rng=np.random.default_rng(seed))

    policies = {
        "random":   RandomPolicy(seed=1),
        "fixed_we": FixedPolicy("worked_example"),
        "rule":     RulePolicy(),
        "lin_ts":   LinTSBandit(context_dim=CONTEXT_DIM, seed=0),
    }

    results = {}
    for name, policy in policies.items():
        print(f"\nRunning policy: {name}")
        df = run_many_episodes(lambda s=name: fresh_sim(seed=hash(s) % 10000),
                               policy, pps, n_episodes=30, max_turns=20)
        results[name] = df
        if df.empty:
            print("  (no logs)")
            continue
        mean_r = df["reward"].mean()
        mean_correct = df["sim_correct"].mean()
        mean_hint = df["sim_hint"].mean()
        n_turns = len(df)
        print(f"  turns={n_turns}  mean_reward={mean_r:.3f}  "
              f"correct_rate={mean_correct:.3f}  mean_hint={mean_hint:.2f}")

    print("\n--- Summary ---")
    summary = pd.DataFrame({
        name: {
            "mean_reward": df["reward"].mean() if not df.empty else np.nan,
            "correct_rate": df["sim_correct"].mean() if not df.empty else np.nan,
            "n_turns": len(df),
        } for name, df in results.items()
    }).T
    print(summary.round(3))
