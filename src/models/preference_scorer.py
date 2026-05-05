"""
Pedagogical Preference Scorer (PPS) — Model 3.

Loads the DeBERTa-v3 cross-encoder fine-tuned on MathDial preference pairs.
Outputs a scalar score per (context, candidate_turn) pair.

Higher score = pedagogically better turn given the student context.
The bandit reward uses these scores in normalised [0, 1] form.
"""

from pathlib import Path
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel

from src.config import ARTIFACTS_DIR

MODEL_NAME = "microsoft/deberta-v3-base"
MAX_LEN = 256
PPS_DIR = ARTIFACTS_DIR / "pps_deberta"


class PreferenceScorerNet(nn.Module):
    def __init__(self, model_name: str = MODEL_NAME):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.head = nn.Linear(hidden, 1)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]
        return self.head(cls).squeeze(-1)


class PreferenceScorer:
    """Inference wrapper. Loads the fine-tuned model and produces scores."""

    def __init__(self, model_dir: Path = PPS_DIR):
        # Tokenizer comes from local save (SPM file etc.)
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self.net = PreferenceScorerNet()
        state = torch.load(model_dir / "pps_state.pt", map_location="cpu", weights_only=True)
        self.net.load_state_dict(state)
        self.net.eval()
        # Calibration: store min/max we'll see in practice for [0,1] scaling.
        # We default to a sensible range based on margin loss output.
        self._lo = -4.0
        self._hi = -1.0

    @torch.no_grad()
    def raw_score(self, context: str, candidate_turn: str) -> float:
        enc = self.tokenizer(
            context, candidate_turn,
            max_length=MAX_LEN, padding="max_length",
            truncation=True, return_tensors="pt"
        )
        s = self.net(enc["input_ids"], enc["attention_mask"]).item()
        return float(s)

    def score(self, context: str, candidate_turn: str) -> float:
        """Return a [0, 1] normalised score for the bandit reward."""
        s = self.raw_score(context, candidate_turn)
        s_norm = (s - self._lo) / (self._hi - self._lo)
        return float(min(1.0, max(0.0, s_norm)))

    @torch.no_grad()
    def score_many(self, context: str, candidates: list) -> list:
        """Score multiple candidate turns against one context."""
        return [self.score(context, c) for c in candidates]

    @torch.no_grad()
    def score_batch(self, context: str, candidates: list) -> list:
        """Score multiple candidates against one context in a single batch."""
        enc = self.tokenizer(
            [context] * len(candidates),
            candidates,
            max_length=MAX_LEN, padding="max_length",
            truncation=True, return_tensors="pt"
        )
        s = self.net(enc["input_ids"], enc["attention_mask"]).cpu().numpy()
        s_norm = (s - self._lo) / (self._hi - self._lo)
        s_norm = s_norm.clip(0.0, 1.0)
        return [float(v) for v in s_norm]


if __name__ == "__main__":
    pps = PreferenceScorer()
    ctx = "I think the answer is 12 because I added 4 and 8."
    cands = [
        "Let's check that. What's 4 + 8 step by step? Show me each step.",  # scaffolded
        "Wrong, the answer is 16.",                                          # bad: reveal
        "Can you walk me through how you got 12? I want to understand your thinking.",  # probing
    ]
    for c in cands:
        print(f"{pps.score(ctx, c):.3f}  |  {c[:60]}")
