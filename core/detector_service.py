from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import joblib
import numpy as np
import pandas as pd


ReasoningFormatter = Callable[[str, Optional[str]], Any]
TextFormatter = Callable[[str], Any]


def format_talkmoves_reasoning(
    current_text: str,
    previous_student_text: Optional[str],
) -> str:
    """
    Match the frozen TalkMoves reasoning training representation exactly:

        "[PREVIOUS_STUDENT] " + previous
        + " [CURRENT_STUDENT] " + current

    The previous utterance is allowed to be empty for first-turn cases.
    """
    previous = (previous_student_text or "").strip()
    current = current_text.strip()

    return (
        "[PREVIOUS_STUDENT] "
        + previous
        + " [CURRENT_STUDENT] "
        + current
    )


def normalize_itspoke_text(text: str) -> str:
    if pd.isna(text):
        return ""

    text = str(text)

    # --------------------------------------------
    # Remove transcription convention tags
    # <nspn>, <spn>, <X>, etc.
    # --------------------------------------------
    text = re.sub(
        r"<[^>]+>",
        " ",
        text,
    )

    # --------------------------------------------
    # Remove overlap markers
    # ***word*** -> word
    # --------------------------------------------
    text = text.replace(
        "***",
        ""
    )

    # --------------------------------------------
    # False-start notation:
    # grav+gravity -> grav gravity
    # --------------------------------------------
    text = text.replace(
        "+",
        " "
    )

    # --------------------------------------------
    # Remove punctuation supplied by transcriber.
    #
    # This deliberately removes:
    # ?, ??, ..., commas, etc.
    # --------------------------------------------
    text = re.sub(
        r"[^\w\s']",
        " ",
        text,
    )

    # --------------------------------------------
    # Normalize whitespace
    # --------------------------------------------
    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    return text


def format_cima_clarification(text: str) -> str:
    """
    Frozen CIMA target-only model input: raw target student utterance.

    Lowercasing and TF-IDF are handled inside the saved sklearn Pipeline.
    """
    return str(text).strip()


def _positive_probability(
    model: Any,
    features: list[Any],
    *,
    positive_label: Any = 1,
) -> float:
    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(features))

        if probabilities.ndim != 2 or probabilities.shape[0] != 1:
            raise ValueError(
                "predict_proba() must return shape (1, n_classes)"
            )

        classes = getattr(model, "classes_", None)

        if classes is None:
            if probabilities.shape[1] != 2:
                raise ValueError(
                    "Model has no classes_ attribute and output is not binary"
                )
            probability = float(probabilities[0, 1])
        else:
            classes = list(classes)

            if positive_label not in classes:
                raise ValueError(
                    f"Positive label {positive_label!r} not found in "
                    f"model classes {classes!r}"
                )

            probability = float(
                probabilities[0, classes.index(positive_label)]
            )

    elif hasattr(model, "decision_function"):
        score = np.asarray(
            model.decision_function(features)
        ).reshape(-1)

        if score.size != 1:
            raise ValueError(
                "decision_function() must return one score for one input"
            )

        probability = float(
            1.0 / (1.0 + np.exp(-score[0]))
        )

    else:
        raise TypeError(
            "Detector model must implement predict_proba() "
            "or decision_function()"
        )

    return min(1.0, max(0.0, probability))


@dataclass(frozen=True)
class DetectorThresholds:
    reasoning: float = 0.30
    uncertainty: float = 0.25
    clarification: float = 0.40

    def __post_init__(self) -> None:
        for name, value in (
            ("reasoning", self.reasoning),
            ("uncertainty", self.uncertainty),
            ("clarification", self.clarification),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{name} threshold must be in [0, 1], got {value}"
                )


class DetectorService:
    """
    Production inference wrapper for the three frozen behavioural detectors.

    Frozen contracts:
    - TalkMoves reasoning: threshold 0.30, positive class 1
    - ITSPOKE uncertainty: threshold 0.25, positive class 1
    - CIMA clarification: threshold 0.40, positive class 1

    The service returns continuous probabilities. The Signal Resolver
    remains the authoritative place for thresholding.
    """

    def __init__(
        self,
        *,
        reasoning_model: Any,
        uncertainty_model: Any,
        clarification_model: Any,
        reasoning_formatter: ReasoningFormatter = format_talkmoves_reasoning,
        uncertainty_formatter: TextFormatter = normalize_itspoke_text,
        clarification_formatter: TextFormatter = format_cima_clarification,
        positive_label: Any = 1,
        thresholds: Optional[DetectorThresholds] = None,
    ) -> None:
        self.reasoning_model = reasoning_model
        self.uncertainty_model = uncertainty_model
        self.clarification_model = clarification_model

        self.reasoning_formatter = reasoning_formatter
        self.uncertainty_formatter = uncertainty_formatter
        self.clarification_formatter = clarification_formatter

        self.positive_label = positive_label
        self.thresholds = thresholds or DetectorThresholds()

    @classmethod
    def from_paths(
        cls,
        *,
        reasoning_model_path: str | Path,
        uncertainty_model_path: str | Path,
        clarification_model_path: str | Path,
        reasoning_formatter: ReasoningFormatter = format_talkmoves_reasoning,
        uncertainty_formatter: TextFormatter = normalize_itspoke_text,
        clarification_formatter: TextFormatter = format_cima_clarification,
        positive_label: Any = 1,
        thresholds: Optional[DetectorThresholds] = None,
    ) -> "DetectorService":
        paths = {
            "reasoning": Path(reasoning_model_path),
            "uncertainty": Path(uncertainty_model_path),
            "clarification": Path(clarification_model_path),
        }

        missing = [
            f"{name}: {path}"
            for name, path in paths.items()
            if not path.exists()
        ]

        if missing:
            raise FileNotFoundError(
                "Missing detector artifacts:\n- "
                + "\n- ".join(missing)
            )

        return cls(
            reasoning_model=joblib.load(paths["reasoning"]),
            uncertainty_model=joblib.load(paths["uncertainty"]),
            clarification_model=joblib.load(paths["clarification"]),
            reasoning_formatter=reasoning_formatter,
            uncertainty_formatter=uncertainty_formatter,
            clarification_formatter=clarification_formatter,
            positive_label=positive_label,
            thresholds=thresholds,
        )

    @classmethod
    def from_project_defaults(
        cls,
        project_root: str | Path,
    ) -> "DetectorService":
        root = Path(project_root)

        detector_dir = root / "models" / "detectors"

        return cls.from_paths(
            reasoning_model_path=detector_dir / "reasoning.joblib",
            uncertainty_model_path=detector_dir / "uncertainty.joblib",
            clarification_model_path=detector_dir / "clarification.joblib",
        )

    def reasoning_predictor(
        self,
        current_text: str,
        previous_student_text: Optional[str] = None,
    ) -> float:
        features = self.reasoning_formatter(
            current_text,
            previous_student_text,
        )

        return _positive_probability(
            self.reasoning_model,
            [features],
            positive_label=self.positive_label,
        )

    def uncertainty_predictor(
        self,
        current_text: str,
    ) -> float:
        features = self.uncertainty_formatter(current_text)

        return _positive_probability(
            self.uncertainty_model,
            [features],
            positive_label=self.positive_label,
        )

    def clarification_predictor(
        self,
        current_text: str,
    ) -> float:
        features = self.clarification_formatter(current_text)

        return _positive_probability(
            self.clarification_model,
            [features],
            positive_label=self.positive_label,
        )

    def predict_all(
        self,
        *,
        current_text: str,
        previous_student_text: Optional[str] = None,
    ) -> dict:
        reasoning = self.reasoning_predictor(
            current_text,
            previous_student_text,
        )
        uncertainty = self.uncertainty_predictor(current_text)
        clarification = self.clarification_predictor(current_text)

        return {
            "reasoning_probability": reasoning,
            "uncertainty_probability": uncertainty,
            "clarification_probability": clarification,
            "reasoning_present": (
                reasoning >= self.thresholds.reasoning
            ),
            "uncertainty_present": (
                uncertainty >= self.thresholds.uncertainty
            ),
            "clarification_present": (
                clarification >= self.thresholds.clarification
            ),
        }
