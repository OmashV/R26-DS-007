from core.cross_session_pipeline import (
    CrossSessionStudentModelPipeline,
)
from core.signal_resolver import (
    ObservationSource,
    PrimarySignal,
)


class EmptyExtractor:
    def extract(self, transcript):
        return {
            "events": [],
            "misconceptions": [],
        }


class NeverCalledEvaluator:
    def __call__(self, event, transcript):
        raise AssertionError(
            "No extractor event exists, so evaluator should not be called."
        )


class FakeDetectors:
    def reasoning_predictor(self, current, previous=None):
        return 0.05

    def uncertainty_predictor(self, text):
        return 0.90

    def clarification_predictor(self, text):
        return 0.80


class CapturingKnowledgeGraph:
    def __init__(self):
        self.received = None

    def get_attempts(self, student_id, skill):
        return []

    def process_resolved_events(
        self,
        *,
        student_id,
        resolved_events,
        session_id,
    ):
        self.received = list(resolved_events)
        return {
            "session_id": session_id,
            "skills_updated": [],
            "behaviour_events": [],
            "graph": [],
        }


def test_non_answer_conversation_turn_can_reach_bkt_as_weak_proxy():
    transcript = [
        {
            "role": "tutor",
            "text": "Let's work on percentages. What part is confusing?",
        },
        {
            "role": "student",
            "text": "I don't understand.",
        },
    ]

    kg = CapturingKnowledgeGraph()

    pipeline = CrossSessionStudentModelPipeline(
        concept_extractor=EmptyExtractor(),
        evaluator=NeverCalledEvaluator(),
        detectors=FakeDetectors(),
        knowledge_graph=kg,
    )

    result = pipeline.process_transcript(
        transcript=transcript,
        student_id="student-1",
        session_id="session-1",
        behavioural_skill="Percent Of",
    )

    evaluated = result["evaluated_extraction"]

    assert len(evaluated["events"]) == 1
    assert evaluated["events"][0]["correctness"] == "unknown"
    assert evaluated["events"][0]["evaluator_confidence"] == 0.0
    assert evaluated["events"][0]["evaluator_source"] == (
        "behaviour_only_turn"
    )

    assert len(result["resolved_events"]) == 1

    resolved = result["resolved_events"][0]

    assert resolved.skill_id == "Percent Of"
    assert resolved.primary_signal == (
        PrimarySignal.BEHAVIOURAL_DIFFICULTY
    )
    assert resolved.bkt_update.outcome == 0
    assert resolved.bkt_update.should_update is True
    assert resolved.bkt_update.observation_source == (
        ObservationSource.BEHAVIOURAL_PROXY
    )
    assert 0.0 < resolved.bkt_update.update_confidence <= 0.25

    assert kg.received is not None
    assert len(kg.received) == 1


def test_existing_same_turn_skill_event_is_not_duplicated():
    class ExistingExtractor:
        def extract(self, transcript):
            return {
                "events": [
                    {
                        "turn_index": 1,
                        "skill": "Percent Of",
                        "evidence_span": "10",
                        "student_text": "10",
                    }
                ],
                "misconceptions": [],
            }

    class CorrectEvaluator:
        def __call__(self, event, transcript):
            return {
                "correctness": "correct",
                "confidence": 1.0,
                "source": "test",
            }

    transcript = [
        {
            "role": "tutor",
            "text": "What is 20 percent of 50?",
        },
        {
            "role": "student",
            "text": "10",
        },
    ]

    kg = CapturingKnowledgeGraph()

    pipeline = CrossSessionStudentModelPipeline(
        concept_extractor=ExistingExtractor(),
        evaluator=CorrectEvaluator(),
        detectors=FakeDetectors(),
        knowledge_graph=kg,
    )

    result = pipeline.process_transcript(
        transcript=transcript,
        student_id="student-1",
        session_id="session-1",
        behavioural_skill="Percent Of",
    )

    same_skill_events = [
        event
        for event in result["evaluated_extraction"]["events"]
        if event["turn_index"] == 1
        and event["skill"] == "Percent Of"
    ]

    assert len(same_skill_events) == 1
    assert len(result["resolved_events"]) == 1
    assert result["resolved_events"][0].bkt_update.outcome == 1
