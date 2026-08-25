from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Correctness(str, Enum):
    CORRECT = "correct"
    PARTIAL = "partial"
    INCORRECT = "incorrect"
    UNKNOWN = "unknown"


class PrimarySignal(str, Enum):
    CORRECT_ANSWER = "correct_answer"
    CORRECT_EXPLANATION = "correct_explanation"
    PARTIAL_CORRECT = "partial_correct"
    INCORRECT_ANSWER = "incorrect_answer"

    # Conversational proxy evidence. This is deliberately distinct from an
    # evaluator-confirmed incorrect answer.
    BEHAVIOURAL_DIFFICULTY = "behavioural_difficulty"

    NO_UPDATE = "no_update"


class ObservationSource(str, Enum):
    EVALUATOR = "evaluator"
    BEHAVIOURAL_PROXY = "behavioural_proxy"
    NONE = "none"


class ResolverConfig(BaseModel):
    # Frozen detector thresholds.
    reasoning_threshold: float = Field(default=0.30, ge=0.0, le=1.0)
    uncertainty_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    clarification_threshold: float = Field(default=0.40, ge=0.0, le=1.0)

    # Existing evaluator evidence weights.
    partial_correct_weight: float = Field(default=0.40, ge=0.0, le=1.0)
    correct_weight: float = Field(default=1.00, ge=0.0, le=1.0)
    incorrect_weight: float = Field(default=1.00, ge=0.0, le=1.0)

    # Current incorrect + enough trailing same-skill incorrect observations.
    repeated_incorrect_count: int = Field(default=2, ge=1)

    # Behavioural BKT fusion.
    #
    # IMPORTANT:
    # These are configurable research parameters, not calibrated probabilities.
    # Keep them frozen during a final evaluation. Tune/justify them only on an
    # appropriate development/validation protocol.
    enable_behavioural_bkt: bool = True

    # Maximum confidence of an UNKNOWN-correctness behavioural proxy.
    behavioural_proxy_cap: float = Field(default=0.25, ge=0.0, le=1.0)

    # Modifiers for evaluator-backed positive observations.
    reasoning_positive_bonus_cap: float = Field(
        default=0.10, ge=0.0, le=1.0
    )
    uncertainty_positive_penalty_cap: float = Field(
        default=0.20, ge=0.0, le=1.0
    )
    clarification_positive_penalty_cap: float = Field(
        default=0.10, ge=0.0, le=1.0
    )

    # Modifiers for evaluator-backed negative observations.
    uncertainty_negative_bonus_cap: float = Field(
        default=0.10, ge=0.0, le=1.0
    )
    clarification_negative_bonus_cap: float = Field(
        default=0.05, ge=0.0, le=1.0
    )
    repeated_misunderstanding_bonus_cap: float = Field(
        default=0.10, ge=0.0, le=1.0
    )


class ResolverInput(BaseModel):
    event_id: str
    student_id: str
    skill_id: str
    correctness: Correctness
    evaluator_confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    allow_behavioural_proxy: bool = True

    reasoning_probability: Optional[float] = Field(
        default=None, ge=0.0, le=1.0
    )
    uncertainty_probability: Optional[float] = Field(
        default=None, ge=0.0, le=1.0
    )
    clarification_probability: Optional[float] = Field(
        default=None, ge=0.0, le=1.0
    )

    recent_same_skill_outcomes: list[Correctness] = Field(
        default_factory=list
    )


class BehaviourEvidence(BaseModel):
    reasoning_probability: Optional[float]
    reasoning_present: bool

    uncertainty_probability: Optional[float]
    uncertainty_present: bool

    clarification_probability: Optional[float]
    clarification_present: bool


class HistoryEvidence(BaseModel):
    repeated_misunderstanding: bool


class BKTUpdate(BaseModel):
    should_update: bool
    outcome: Optional[int]

    # Nominal strength assigned to the signal type. For evaluator-backed
    # observations this preserves the existing correct/partial/incorrect weight.
    # For behavioural proxies it is the configured behavioural_proxy_cap.
    evidence_weight: float = Field(ge=0.0, le=1.0)

    # Confidence from the authoritative answer evaluator. Behavioural-only
    # observations have no evaluator verdict and therefore store 0.0 here.
    evaluator_confidence: float = Field(ge=0.0, le=1.0)

    # Fused detector strength for behavioural-only proxy observations.
    # It remains 0.0 for evaluator-backed observations.
    behavioural_confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    # Multiplicative modifier applied to evaluator-backed observations.
    # 1.0 means behaviour had no net effect.
    behaviour_factor: float = Field(default=1.0, ge=0.0)

    # This is the only confidence that KnowledgeGraph/BKT should consume.
    update_confidence: float = Field(ge=0.0, le=1.0)

    observation_source: ObservationSource = ObservationSource.NONE

    # Behaviour/history features that actually contributed to BKT strength.
    # Raw probabilities are still retained in BehaviourEvidence.
    contributors: list[str] = Field(default_factory=list)


class ResolvedSignal(BaseModel):
    event_id: str
    student_id: str
    skill_id: str
    primary_signal: PrimarySignal
    bkt_update: BKTUpdate
    behaviour: BehaviourEvidence
    history: HistoryEvidence
    resolver_version: str = "2.0"


DEFAULT_RESOLVER_CONFIG = ResolverConfig()


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _present(
    probability: Optional[float],
    threshold: float,
) -> bool:
    return (
        probability is not None
        and float(probability) >= float(threshold)
    )


def _above_threshold_strength(
    probability: Optional[float],
    threshold: float,
) -> float:
    """
    Map probability above its frozen detector threshold onto [0, 1].

    Below/equal threshold -> 0.
    At probability 1.0 -> 1.

    The detector probability is NOT treated as BKT confidence directly.
    """
    if probability is None:
        return 0.0

    p = _clamp01(probability)
    t = _clamp01(threshold)

    if p <= t:
        return 0.0

    if t >= 1.0:
        return 0.0

    return _clamp01((p - t) / (1.0 - t))


def _is_repeated_misunderstanding(
    current: Correctness,
    history: list[Correctness],
    required_count: int,
) -> bool:
    """
    `required_count` includes the current incorrect event.

    With the default 2, one immediately previous same-skill incorrect event
    plus the current incorrect event is enough.
    """
    if current != Correctness.INCORRECT:
        return False

    trailing_previous_incorrect = 0

    for outcome in reversed(history):
        if outcome != Correctness.INCORRECT:
            break
        trailing_previous_incorrect += 1

    return (
        1 + trailing_previous_incorrect
        >= required_count
    )


def _behaviour_evidence(
    event: ResolverInput,
    config: ResolverConfig,
) -> BehaviourEvidence:
    return BehaviourEvidence(
        reasoning_probability=event.reasoning_probability,
        reasoning_present=_present(
            event.reasoning_probability,
            config.reasoning_threshold,
        ),
        uncertainty_probability=event.uncertainty_probability,
        uncertainty_present=_present(
            event.uncertainty_probability,
            config.uncertainty_threshold,
        ),
        clarification_probability=event.clarification_probability,
        clarification_present=_present(
            event.clarification_probability,
            config.clarification_threshold,
        ),
    )


def _no_update(
    event: ResolverInput,
    behaviour: BehaviourEvidence,
    history: HistoryEvidence,
) -> ResolvedSignal:
    return ResolvedSignal(
        event_id=event.event_id,
        student_id=event.student_id,
        skill_id=event.skill_id,
        primary_signal=PrimarySignal.NO_UPDATE,
        bkt_update=BKTUpdate(
            should_update=False,
            outcome=None,
            evidence_weight=0.0,
            evaluator_confidence=event.evaluator_confidence,
            behavioural_confidence=0.0,
            behaviour_factor=1.0,
            update_confidence=0.0,
            observation_source=ObservationSource.NONE,
            contributors=[],
        ),
        behaviour=behaviour,
        history=history,
    )


def _positive_update(
    *,
    event: ResolverInput,
    primary_signal: PrimarySignal,
    evidence_weight: float,
    behaviour: BehaviourEvidence,
    history: HistoryEvidence,
    config: ResolverConfig,
) -> ResolvedSignal:
    base_confidence = _clamp01(
        evidence_weight * event.evaluator_confidence
    )

    reasoning_strength = _above_threshold_strength(
        event.reasoning_probability,
        config.reasoning_threshold,
    )
    uncertainty_strength = _above_threshold_strength(
        event.uncertainty_probability,
        config.uncertainty_threshold,
    )
    clarification_strength = _above_threshold_strength(
        event.clarification_probability,
        config.clarification_threshold,
    )

    contributors: list[str] = []

    bonus = 0.0
    penalty = 0.0

    if reasoning_strength > 0.0:
        bonus += (
            config.reasoning_positive_bonus_cap
            * reasoning_strength
        )
        contributors.append("reasoning")

    if uncertainty_strength > 0.0:
        penalty += (
            config.uncertainty_positive_penalty_cap
            * uncertainty_strength
        )
        contributors.append("uncertainty")

    if clarification_strength > 0.0:
        penalty += (
            config.clarification_positive_penalty_cap
            * clarification_strength
        )
        contributors.append("clarification")

    behaviour_factor = max(0.0, 1.0 + bonus - penalty)

    update_confidence = _clamp01(
        base_confidence * behaviour_factor
    )

    return ResolvedSignal(
        event_id=event.event_id,
        student_id=event.student_id,
        skill_id=event.skill_id,
        primary_signal=primary_signal,
        bkt_update=BKTUpdate(
            should_update=True,
            outcome=1,
            evidence_weight=evidence_weight,
            evaluator_confidence=event.evaluator_confidence,
            behavioural_confidence=0.0,
            behaviour_factor=behaviour_factor,
            update_confidence=update_confidence,
            observation_source=ObservationSource.EVALUATOR,
            contributors=contributors,
        ),
        behaviour=behaviour,
        history=history,
    )


def _negative_update(
    *,
    event: ResolverInput,
    behaviour: BehaviourEvidence,
    history: HistoryEvidence,
    config: ResolverConfig,
) -> ResolvedSignal:
    base_confidence = _clamp01(
        config.incorrect_weight
        * event.evaluator_confidence
    )

    uncertainty_strength = _above_threshold_strength(
        event.uncertainty_probability,
        config.uncertainty_threshold,
    )
    clarification_strength = _above_threshold_strength(
        event.clarification_probability,
        config.clarification_threshold,
    )

    contributors: list[str] = []
    bonus = 0.0

    if uncertainty_strength > 0.0:
        bonus += (
            config.uncertainty_negative_bonus_cap
            * uncertainty_strength
        )
        contributors.append("uncertainty")

    if clarification_strength > 0.0:
        bonus += (
            config.clarification_negative_bonus_cap
            * clarification_strength
        )
        contributors.append("clarification")

    if history.repeated_misunderstanding:
        bonus += config.repeated_misunderstanding_bonus_cap
        contributors.append("repeated_misunderstanding")

    behaviour_factor = 1.0 + bonus

    update_confidence = _clamp01(
        base_confidence * behaviour_factor
    )

    return ResolvedSignal(
        event_id=event.event_id,
        student_id=event.student_id,
        skill_id=event.skill_id,
        primary_signal=PrimarySignal.INCORRECT_ANSWER,
        bkt_update=BKTUpdate(
            should_update=True,
            outcome=0,
            evidence_weight=config.incorrect_weight,
            evaluator_confidence=event.evaluator_confidence,
            behavioural_confidence=0.0,
            behaviour_factor=behaviour_factor,
            update_confidence=update_confidence,
            observation_source=ObservationSource.EVALUATOR,
            contributors=contributors,
        ),
        behaviour=behaviour,
        history=history,
    )


def _behavioural_proxy_update(
    *,
    event: ResolverInput,
    behaviour: BehaviourEvidence,
    history: HistoryEvidence,
    config: ResolverConfig,
) -> Optional[ResolvedSignal]:
    """
    Fuse uncertainty + clarification into ONE weak negative observation.

    Reasoning alone cannot establish correctness and therefore does not create
    a positive proxy observation.

    Noisy-OR fusion ensures overlapping uncertainty/clarification detectors do
    not create duplicate BKT observations or allow confidence to exceed the
    configured proxy cap.
    """
    if not event.allow_behavioural_proxy:
        return None

    if not config.enable_behavioural_bkt:
        return None

    uncertainty_strength = _above_threshold_strength(
        event.uncertainty_probability,
        config.uncertainty_threshold,
    )
    clarification_strength = _above_threshold_strength(
        event.clarification_probability,
        config.clarification_threshold,
    )

    if (
        uncertainty_strength <= 0.0
        and clarification_strength <= 0.0
    ):
        return None

    fused_strength = _clamp01(
        1.0
        - (1.0 - uncertainty_strength)
        * (1.0 - clarification_strength)
    )

    update_confidence = _clamp01(
        config.behavioural_proxy_cap * fused_strength
    )

    if update_confidence <= 0.0:
        return None

    contributors: list[str] = []

    if uncertainty_strength > 0.0:
        contributors.append("uncertainty")

    if clarification_strength > 0.0:
        contributors.append("clarification")

    return ResolvedSignal(
        event_id=event.event_id,
        student_id=event.student_id,
        skill_id=event.skill_id,
        primary_signal=PrimarySignal.BEHAVIOURAL_DIFFICULTY,
        bkt_update=BKTUpdate(
            should_update=True,
            outcome=0,
            evidence_weight=config.behavioural_proxy_cap,
            # There is deliberately no evaluator verdict for this update.
            evaluator_confidence=0.0,
            behavioural_confidence=fused_strength,
            behaviour_factor=1.0,
            update_confidence=update_confidence,
            observation_source=ObservationSource.BEHAVIOURAL_PROXY,
            contributors=contributors,
        ),
        behaviour=behaviour,
        history=history,
    )


def resolve_signal(
    event: ResolverInput,
    config: ResolverConfig = DEFAULT_RESOLVER_CONFIG,
) -> ResolvedSignal:
    """
    Resolve one genuine student event into at most one BKT observation.

    Priority:
    1. Evaluator-backed CORRECT / PARTIAL / INCORRECT evidence.
    2. If correctness is UNKNOWN, fused behavioural difficulty from
       uncertainty/clarification may create ONE weak negative proxy.
    3. Otherwise NO_UPDATE.

    Reasoning is never a standalone positive mastery signal.
    """
    behaviour = _behaviour_evidence(event, config)

    repeated_misunderstanding = _is_repeated_misunderstanding(
        event.correctness,
        event.recent_same_skill_outcomes,
        config.repeated_incorrect_count,
    )

    history = HistoryEvidence(
        repeated_misunderstanding=repeated_misunderstanding
    )

    if event.correctness == Correctness.CORRECT:
        primary = (
            PrimarySignal.CORRECT_EXPLANATION
            if behaviour.reasoning_present
            else PrimarySignal.CORRECT_ANSWER
        )

        return _positive_update(
            event=event,
            primary_signal=primary,
            evidence_weight=config.correct_weight,
            behaviour=behaviour,
            history=history,
            config=config,
        )

    if event.correctness == Correctness.PARTIAL:
        return _positive_update(
            event=event,
            primary_signal=PrimarySignal.PARTIAL_CORRECT,
            evidence_weight=config.partial_correct_weight,
            behaviour=behaviour,
            history=history,
            config=config,
        )

    if event.correctness == Correctness.INCORRECT:
        return _negative_update(
            event=event,
            behaviour=behaviour,
            history=history,
            config=config,
        )

    proxy = _behavioural_proxy_update(
        event=event,
        behaviour=behaviour,
        history=history,
        config=config,
    )

    if proxy is not None:
        return proxy

    return _no_update(
        event,
        behaviour,
        history,
    )
