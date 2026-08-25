from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import core.knowledge_graph as knowledge_graph_module
from core.cross_session_pipeline import CrossSessionStudentModelPipeline
from core.curriculum import Curriculum
from core.evaluation_contract import apply_evaluator_to_extraction
from core.knowledge_graph import KnowledgeGraph
from core.learning_path import generate_learning_path
from core.pipeline_adapter import resolve_extracted_events
from core.signal_resolver import PrimarySignal
from core.student_answer_evaluator import (
    EvaluatorContext,
    StudentAnswerEvaluator,
)


def test_global_reference_answer_must_not_credit_every_extracted_skill():
    transcript = [
        {
            "role": "tutor",
            "text": "What is 20 percent of 50? Explain your reasoning.",
        },
        {
            "role": "student",
            "text": "Because 20 percent is 0.2, I multiply 0.2 by 50 and get 10.",
        },
    ]

    raw_extraction = {
        "events": [
            {
                "turn_index": 1,
                "skill": "Percent Of",
                "evidence_span": (
                    "Because 20 percent is 0.2, "
                    "I multiply 0.2 by 50 and get 10."
                ),
                "student_text": transcript[1]["text"],
            },
            {
                "turn_index": 1,
                "skill": "Conversion of Fraction Decimals Percents",
                "evidence_span": "20 percent is 0.2",
                "student_text": transcript[1]["text"],
            },
            {
                "turn_index": 1,
                "skill": "Multiplication and Division Positive Decimals",
                "evidence_span": "I multiply 0.2 by 50 and get 10.",
                "student_text": transcript[1]["text"],
            },
        ]
    }

    evaluator = StudentAnswerEvaluator(
        EvaluatorContext(
            problem="What is 20 percent of 50?",
            reference_answer="10",
            assessed_skills=("Percent Of",),
        )
    )

    evaluated = apply_evaluator_to_extraction(
        extraction=raw_extraction,
        transcript=transcript,
        evaluator=evaluator,
    )

    assert [
        event["correctness"]
        for event in evaluated["events"]
    ] == [
        "correct",
        "unknown",
        "unknown",
    ]

    resolved = resolve_extracted_events(
        extraction=evaluated,
        transcript=transcript,
        student_id="guard_student",
        session_id="guard_session",
    )

    updates = [
        event for event in resolved
        if event.bkt_update.should_update
    ]

    assert len(updates) == 1
    assert updates[0].skill_id == "Percent Of"
    assert updates[0].bkt_update.outcome == 1


class FakeModels:
    def generate_content(self, **kwargs):
        return SimpleNamespace(
            text=(
                '{"response_type":"non_answer",'
                '"correctness":"unknown","confidence":0.0}'
            )
        )


class FakeClient:
    def __init__(self):
        self.models = FakeModels()


class ThreeSkillExtractor:
    def extract(self, transcript):
        student_text = transcript[1]["text"]
        return {
            "events": [
                {
                    "turn_index": 1,
                    "skill": skill,
                    "student_text": student_text,
                    "evidence_span": student_text,
                }
                for skill in ("Percent Of", "Ratio", "Fractions")
            ],
            "misconceptions": [],
        }


class HighDifficultyDetectors:
    def reasoning_predictor(self, current, previous=None):
        return 0.0

    def uncertainty_predictor(self, text):
        return 0.95

    def clarification_predictor(self, text):
        return 0.95


class FakeConnection:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def execute(self, *args, **kwargs):
        return None


def test_full_pipeline_blocks_cross_skill_behavioural_proxy(
    monkeypatch,
):
    transcript = [
        {
            "role": "tutor",
            "text": "What is 20 percent of 50?",
        },
        {
            "role": "student",
            "text": "I don't understand. Can you explain it?",
        },
    ]
    evaluator = StudentAnswerEvaluator(
        EvaluatorContext(
            problem="What is 20 percent of 50?",
            reference_answer="10",
            assessed_skills=("Percent Of",),
        ),
        objective_comparator=lambda student, reference: None,
        client=FakeClient(),
    )

    graph = object.__new__(KnowledgeGraph)
    graph.ensure_student = MagicMock()
    graph.get_attempts = MagicMock(return_value=[])
    graph.record_attempt = MagicMock()
    graph.update_mastery = MagicMock(
        return_value={
            "probability": 0.20,
            "label": "weak",
            "cold_start_used": False,
            "cold_start_details": None,
        }
    )
    stored_graph = [
        {
            "skill": "Percent Of",
            "mastery_probability": 0.20,
            "mastery_label": "weak",
            "previous_mastery_probability": None,
        }
    ]
    graph.get_student_graph = MagicMock(return_value=stored_graph)
    monkeypatch.setattr(
        knowledge_graph_module,
        "get_connection",
        lambda: FakeConnection(),
    )

    curriculum = Curriculum.from_dict(
        {
            "version": 1,
            "skills": [
                {"skill": "Percent Of", "prerequisites": []},
                {"skill": "Ratio", "prerequisites": []},
                {"skill": "Fractions", "prerequisites": []},
            ],
        },
        bkt_skills=["Percent Of", "Ratio", "Fractions"],
    )
    pipeline = CrossSessionStudentModelPipeline(
        concept_extractor=ThreeSkillExtractor(),
        evaluator=evaluator,
        detectors=HighDifficultyDetectors(),
        knowledge_graph=graph,
        curriculum=curriculum,
    )

    result = pipeline.process_transcript(
        transcript=transcript,
        student_id="student-1",
        session_id="session-1",
    )

    evaluated_by_skill = {
        event["skill"]: event
        for event in result["evaluated_extraction"]["events"]
    }
    assert evaluated_by_skill["Percent Of"]["evaluator_source"] == (
        "gemini_structured_non_answer"
    )
    assert evaluated_by_skill["Ratio"]["evaluator_source"] == (
        "skill_not_authorized_for_assessment"
    )
    assert evaluated_by_skill["Fractions"]["evaluator_source"] == (
        "skill_not_authorized_for_assessment"
    )

    resolved_by_skill = {
        event.skill_id: event
        for event in result["resolved_events"]
    }
    assert resolved_by_skill["Percent Of"].primary_signal == (
        PrimarySignal.BEHAVIOURAL_DIFFICULTY
    )
    assert resolved_by_skill["Percent Of"].bkt_update.should_update is True

    for unauthorized_skill in ("Ratio", "Fractions"):
        resolved = resolved_by_skill[unauthorized_skill]
        assert resolved.primary_signal == PrimarySignal.NO_UPDATE
        assert resolved.bkt_update.should_update is False
        assert resolved.behaviour.uncertainty_present is True
        assert resolved.behaviour.clarification_present is True

    # The real KnowledgeGraph orchestration receives all three resolved events,
    # but writes and recalculates mastery for the one authorized proxy only.
    graph.record_attempt.assert_called_once()
    assert graph.record_attempt.call_args.kwargs["skill"] == "Percent Of"
    graph.update_mastery.assert_called_once_with("student-1", "Percent Of")
    assert result["knowledge_graph_result"]["graph"] == stored_graph

    path = result["learning_path"]
    unseen_by_skill = {entry["skill"]: entry for entry in path["unseen"]}
    assert set(unseen_by_skill) == {"Ratio", "Fractions"}
    assert unseen_by_skill["Ratio"]["mastery_probability"] is None
    assert unseen_by_skill["Fractions"]["mastery_probability"] is None
    assert path == generate_learning_path(
        stored_graph,
        curriculum=curriculum,
    )
