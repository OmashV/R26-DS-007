"""Phase 2 integration tests for cross-session learning-path regeneration."""

from copy import deepcopy

import core.cross_session_pipeline as pipeline_module
from core.cross_session_pipeline import CrossSessionStudentModelPipeline
from core.curriculum import Curriculum
from core.learning_path import generate_learning_path as real_generate_learning_path


class EmptyExtractor:
    def extract(self, transcript):
        return {"events": [], "misconceptions": []}


class NeverCalledEvaluator:
    def __call__(self, event, transcript):
        raise AssertionError("The empty extraction must not invoke the evaluator")


class NoopDetectors:
    def reasoning_predictor(self, current, previous=None):
        raise AssertionError("The resolver is isolated in these integration tests")

    def uncertainty_predictor(self, text):
        raise AssertionError("The resolver is isolated in these integration tests")

    def clarification_predictor(self, text):
        raise AssertionError("The resolver is isolated in these integration tests")


class SequencedKnowledgeGraph:
    """Models mastery changing atomically inside process_resolved_events()."""

    def __init__(self, graphs_after_update: list[list[dict]]):
        self.graphs_after_update = [deepcopy(graph) for graph in graphs_after_update]
        self.current_graph: list[dict] = []
        self.process_calls = 0
        self.attempt_writes = 0
        self.mastery_updates = 0
        self.get_student_graph_calls = 0
        self.timeline: list[str] = []

    def get_attempts(self, student_id, skill):
        return []

    def get_student_graph(self, student_id):
        self.get_student_graph_calls += 1
        return deepcopy(self.current_graph)

    def process_resolved_events(
        self,
        *,
        student_id,
        resolved_events,
        session_id,
    ):
        update_index = min(
            self.process_calls,
            len(self.graphs_after_update) - 1,
        )
        self.current_graph = deepcopy(self.graphs_after_update[update_index])
        self.process_calls += 1
        self.attempt_writes += len(resolved_events)
        self.mastery_updates += len(resolved_events)
        self.timeline.append("mastery_updated")
        return {
            "session_id": session_id,
            "skills_updated": [],
            "behaviour_events": [],
            "graph": deepcopy(self.current_graph),
        }


def make_curriculum(records: list[tuple[str, list[str]]]) -> Curriculum:
    return Curriculum.from_dict(
        {
            "version": 1,
            "skills": [
                {"skill": skill, "prerequisites": prerequisites}
                for skill, prerequisites in records
            ],
        },
        bkt_skills=[skill for skill, _ in records],
    )


def mastery(skill: str, probability: float, previous=None) -> dict:
    return {
        "skill": skill,
        "mastery_probability": probability,
        "previous_mastery_probability": previous,
    }


def skill_names(entries: list[dict]) -> list[str]:
    return [entry["skill"] for entry in entries]


def build_pipeline(
    knowledge_graph: SequencedKnowledgeGraph,
    curriculum: Curriculum | None = None,
) -> CrossSessionStudentModelPipeline:
    return CrossSessionStudentModelPipeline(
        concept_extractor=EmptyExtractor(),
        evaluator=NeverCalledEvaluator(),
        detectors=NoopDetectors(),
        knowledge_graph=knowledge_graph,
        curriculum=curriculum,
    )


def process(pipeline, session_id="session-1") -> dict:
    return pipeline.process_transcript(
        transcript=[],
        student_id="student-1",
        session_id=session_id,
    )


def isolate_resolver(monkeypatch):
    resolved_event = object()
    monkeypatch.setattr(
        pipeline_module,
        "resolve_extracted_events",
        lambda **kwargs: [resolved_event],
    )


def test_mastery_update_completes_before_path_generation(monkeypatch):
    isolate_resolver(monkeypatch)
    curriculum = make_curriculum([("Foundation", [])])
    graph = SequencedKnowledgeGraph(
        [[mastery("Foundation", 0.82)]]
    )

    def tracking_generate_learning_path(updated_graph, *, curriculum):
        assert graph.timeline == ["mastery_updated"]
        graph.timeline.append("path_generated")
        return real_generate_learning_path(
            updated_graph,
            curriculum=curriculum,
        )

    monkeypatch.setattr(
        pipeline_module,
        "generate_learning_path",
        tracking_generate_learning_path,
    )

    result = process(build_pipeline(graph, curriculum))

    assert graph.timeline == ["mastery_updated", "path_generated"]
    assert skill_names(result["learning_path"]["already_strong"]) == [
        "Foundation"
    ]
    assert result["learning_path"]["already_strong"][0][
        "mastery_probability"
    ] == 0.82


def test_weak_prerequisite_is_recommended_while_downstream_is_blocked(
    monkeypatch,
):
    isolate_resolver(monkeypatch)
    curriculum = make_curriculum(
        [
            ("Foundation", []),
            ("Target", ["Foundation"]),
        ]
    )
    graph = SequencedKnowledgeGraph(
        [[mastery("Foundation", 0.20), mastery("Target", 0.50)]]
    )

    path = process(build_pipeline(graph, curriculum))["learning_path"]

    assert skill_names(path["recommended_order"]) == ["Foundation"]
    assert skill_names(path["blocked"]) == ["Target"]
    assert path["blocked"][0]["unmet_prerequisites"] == ["Foundation"]
    assert "Earliest actionable prerequisite(s): Foundation" in path["blocked"][
        0
    ]["priority_reason"]


def test_downstream_becomes_ready_after_prerequisite_becomes_strong(monkeypatch):
    isolate_resolver(monkeypatch)
    curriculum = make_curriculum(
        [
            ("Foundation", []),
            ("Target", ["Foundation"]),
        ]
    )
    graph = SequencedKnowledgeGraph(
        [
            [mastery("Foundation", 0.20), mastery("Target", 0.50)],
            [mastery("Foundation", 0.80), mastery("Target", 0.50)],
        ]
    )
    pipeline = build_pipeline(graph, curriculum)

    first_path = process(pipeline, "session-1")["learning_path"]
    second_path = process(pipeline, "session-2")["learning_path"]

    assert skill_names(first_path["blocked"]) == ["Target"]
    assert skill_names(second_path["blocked"]) == []
    assert skill_names(second_path["learn_next"]) == ["Target"]
    assert second_path["learn_next"][0]["planning_status"] == "ready_to_learn"


def test_unseen_curriculum_skill_is_returned_with_none_mastery(monkeypatch):
    isolate_resolver(monkeypatch)
    curriculum = make_curriculum(
        [
            ("Foundation", []),
            ("Unseen Target", ["Foundation"]),
        ]
    )
    graph = SequencedKnowledgeGraph([[mastery("Foundation", 0.90)]])

    path = process(build_pipeline(graph, curriculum))["learning_path"]

    assert skill_names(path["unseen"]) == ["Unseen Target"]
    unseen = path["unseen"][0]
    assert unseen["mastery_probability"] is None
    assert unseen["mastery_status"] == "unseen"
    assert unseen["planning_status"] == "ready_to_learn"


def test_no_curriculum_uses_backward_compatible_mastery_buckets(monkeypatch):
    isolate_resolver(monkeypatch)
    graph = SequencedKnowledgeGraph(
        [[
            mastery("Weak", 0.10),
            mastery("Partial", 0.50),
            mastery("Strong", 0.80),
        ]]
    )

    result = process(build_pipeline(graph))
    path = result["learning_path"]

    assert set(path) == {
        "revise_urgently",
        "learn_next",
        "already_strong",
        "regressions",
        "summary",
    }
    assert skill_names(path["revise_urgently"]) == ["Weak"]
    assert skill_names(path["learn_next"]) == ["Partial"]
    assert skill_names(path["already_strong"]) == ["Strong"]
    assert result["knowledge_graph_result"]["graph"] == graph.current_graph


def test_path_derivation_creates_no_extra_attempts_or_mastery_updates(monkeypatch):
    isolate_resolver(monkeypatch)
    curriculum = make_curriculum([("Skill", [])])
    graph = SequencedKnowledgeGraph([[mastery("Skill", 0.40)]])

    process(build_pipeline(graph, curriculum))

    assert graph.process_calls == 1
    assert graph.attempt_writes == 1
    assert graph.mastery_updates == 1
    assert graph.get_student_graph_calls == 0


def test_identical_mastery_and_curriculum_produce_identical_paths(monkeypatch):
    isolate_resolver(monkeypatch)
    curriculum = make_curriculum(
        [
            ("A", []),
            ("B", ["A"]),
            ("C", []),
        ]
    )
    current_graph = [mastery("A", 0.20), mastery("B", 0.50)]
    graph = SequencedKnowledgeGraph([current_graph, current_graph])
    pipeline = build_pipeline(graph, curriculum)

    first = process(pipeline, "session-1")["learning_path"]
    second = process(pipeline, "session-2")["learning_path"]

    assert first == second
