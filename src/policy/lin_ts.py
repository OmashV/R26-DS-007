"""
Linear Thompson Sampling bandit for FAPR-LB.

One Bayesian linear regression model per arm.
Per arm a: r = theta_a^T x + epsilon, epsilon ~ N(0, sigma^2)
Prior: theta_a ~ N(0, lambda^-1 * I)
Posterior after seeing data D_a:
  Sigma_a = (lambda * I + (1/sigma^2) * X_a^T X_a)^-1
  mu_a    = Sigma_a * (1/sigma^2) * X_a^T y_a
At decision time, sample theta_tilde_a ~ N(mu_a, Sigma_a) for each arm,
predict r_a = theta_tilde_a^T x, choose argmax.

This file is pure numpy. No torch, no sklearn.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import json

from src.config import ARTIFACTS_DIR


# Repair action arms — must match simulator REPAIR_ACTIONS list
REPAIR_ACTIONS = [
    "worked_example",
    "direct_correction",
    "scaffolded_question",
    "prerequisite_review",
    "hint",
    "simpler_explanation",
    "conceptual_analogy",
]


@dataclass
class LinTSArm:
    """One Bayesian linear regression model for one action arm."""
    d: int                          # context dimension
    lam: float = 1.0                # prior precision
    sigma2: float = 0.25            # noise variance (assumed)
    A: np.ndarray = field(init=False)   # = lam*I + (1/sigma2) * X^T X
    b: np.ndarray = field(init=False)   # = (1/sigma2) * X^T y
    n_pulls: int = 0

    def __post_init__(self):
        self.A = self.lam * np.eye(self.d)
        self.b = np.zeros(self.d)

    @property
    def Sigma(self) -> np.ndarray:
        return np.linalg.inv(self.A)

    @property
    def mu(self) -> np.ndarray:
        return self.Sigma @ self.b

    def sample_theta(self, rng: np.random.Generator) -> np.ndarray:
        """Draw theta from current posterior."""
        mu = self.mu
        Sigma = self.Sigma
        # Add a tiny jitter for numerical stability
        Sigma = Sigma + 1e-8 * np.eye(self.d)
        return rng.multivariate_normal(mu, Sigma)

    def update(self, x: np.ndarray, r: float) -> None:
        """Bayesian update with a single observation (x, r)."""
        x = x.reshape(-1)
        self.A += (1.0 / self.sigma2) * np.outer(x, x)
        self.b += (1.0 / self.sigma2) * r * x
        self.n_pulls += 1


class LinTSBandit:
    """LinTS over the repair-action arms."""

    def __init__(self, context_dim: int, actions: List[str] = None,
                 lam: float = 1.0, sigma2: float = 0.25,
                 seed: int = 0):
        self.actions = actions or REPAIR_ACTIONS
        self.d = context_dim
        self.arms: Dict[str, LinTSArm] = {
            a: LinTSArm(d=context_dim, lam=lam, sigma2=sigma2)
            for a in self.actions
        }
        self.rng = np.random.default_rng(seed)

    def select(self, x: np.ndarray) -> str:
        """Thompson sample: sample theta per arm, predict, argmax."""
        x = x.reshape(-1)
        best_a, best_r = None, -np.inf
        for a, arm in self.arms.items():
            theta = arm.sample_theta(self.rng)
            r_hat = float(theta @ x)
            if r_hat > best_r:
                best_r = r_hat
                best_a = a
        return best_a

    def update(self, action: str, x: np.ndarray, r: float) -> None:
        self.arms[action].update(x, r)

    def stats(self) -> Dict[str, Dict]:
        out = {}
        for a, arm in self.arms.items():
            out[a] = {
                "n_pulls": arm.n_pulls,
                "mu_norm": float(np.linalg.norm(arm.mu)),
            }
        return out

    def save(self, path: Path) -> None:
        payload = {
            "context_dim": self.d,
            "actions": self.actions,
            "arms": {
                a: {"A": arm.A.tolist(), "b": arm.b.tolist(),
                    "n_pulls": arm.n_pulls,
                    "lam": arm.lam, "sigma2": arm.sigma2}
                for a, arm in self.arms.items()
            },
        }
        path.write_text(json.dumps(payload))

    @classmethod
    def load(cls, path: Path, seed: int = 0) -> "LinTSBandit":
        payload = json.loads(path.read_text())
        b = cls(context_dim=payload["context_dim"],
                actions=payload["actions"], seed=seed)
        for a, blob in payload["arms"].items():
            arm = b.arms[a]
            arm.A = np.array(blob["A"])
            arm.b = np.array(blob["b"])
            arm.n_pulls = blob["n_pulls"]
            arm.lam = blob["lam"]
            arm.sigma2 = blob["sigma2"]
        return b


# ---- Baselines for comparison (Day 6 will use them) ----

class RandomPolicy:
    def __init__(self, actions=None, seed=0):
        self.actions = actions or REPAIR_ACTIONS
        self.rng = np.random.default_rng(seed)
    def select(self, x): return self.actions[int(self.rng.integers(len(self.actions)))]
    def update(self, action, x, r): pass


class FixedPolicy:
    def __init__(self, action, actions=None):
        self.action = action
        self.actions = actions or REPAIR_ACTIONS
    def select(self, x): return self.action
    def update(self, action, x, r): pass


class RulePolicy:
    """Hand-coded rule-based policy for comparison."""
    def __init__(self, actions=None):
        self.actions = actions or REPAIR_ACTIONS
    def select(self, x):
        # x layout: [bias, repair_need, effort, diseng, skill_rate,
        #            f_low_mast, f_repair, prev_repair_act, prev_correct, prev_used_hint]
        repair_need = x[1]
        skill_rate = x[4]
        f_low_mast = x[5]
        prev_correct = x[8]
        if f_low_mast > 0.5:
            return "prerequisite_review"   # low mastery → review prerequisites
        if prev_correct < 0.5:
            return "worked_example"        # we just failed → show example
        if repair_need > 0.7:
            return "direct_correction"
        return "scaffolded_question"
    def update(self, action, x, r): pass


if __name__ == "__main__":
    # Smoke test: random context, random rewards. Bandit should run without error.
    b = LinTSBandit(context_dim=10, seed=0)
    rng = np.random.default_rng(0)
    for t in range(50):
        x = rng.normal(size=10)
        a = b.select(x)
        r = float(rng.uniform())
        b.update(a, x, r)
    print("Bandit smoke OK.")
    print("Per-arm stats after 50 pulls:")
    for a, s in b.stats().items():
        print(f"  {a}: n_pulls={s['n_pulls']}, mu_norm={s['mu_norm']:.3f}")