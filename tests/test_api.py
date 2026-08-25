"""
Integration tests for the FastAPI endpoints.
The learning-path route uses real deterministic planner output; the remaining
routes retain their existing mock contracts.
Satisfies the Progress Presentation 1 acceptance criterion: endpoints reachable
and returning valid responses.
"""

import pytest
from fastapi.testclient import TestClient

from api.routes import get_path_curriculum, get_path_knowledge_graph
from core.curriculum import Curriculum
from core.learning_path import generate_learning_path
from main import app

client = TestClient(app)


class ReadOnlyKnowledgeGraph:
    def __init__(self, graph):
        self.graph = graph
        self.graph_reads = 0
        self.attempt_writes = 0
        self.mastery_updates = 0

    def get_student_graph(self, student_id):
        self.graph_reads += 1
        return [dict(entry) for entry in self.graph]

    def record_attempt(self, *args, **kwargs):
        self.attempt_writes += 1
        raise AssertionError("The path endpoint must not write attempts")

    def update_mastery(self, *args, **kwargs):
        self.mastery_updates += 1
        raise AssertionError("The path endpoint must not update mastery")


@pytest.fixture
def path_dependencies():
    graph = [
        {
            "skill": "Foundation",
            "mastery_probability": 0.20,
            "previous_mastery_probability": None,
        },
        {
            "skill": "Target",
            "mastery_probability": 0.50,
            "previous_mastery_probability": None,
        },
        {
            "skill": "Percent Of",
            "mastery_probability": 0.40,
            "previous_mastery_probability": None,
        },
    ]
    curriculum = Curriculum.from_dict(
        {
            "version": 1,
            "skills": [
                {"skill": "Foundation", "prerequisites": []},
                {"skill": "Target", "prerequisites": ["Foundation"]},
                {"skill": "Percent Of", "prerequisites": []},
                {"skill": "Ratio", "prerequisites": []},
                {"skill": "Fractions", "prerequisites": []},
                {
                    "skill": "Unseen Descendant",
                    "prerequisites": ["Target"],
                },
            ],
        },
        bkt_skills=[
            "Foundation",
            "Target",
            "Percent Of",
            "Ratio",
            "Fractions",
            "Unseen Descendant",
        ],
    )
    knowledge_graph = ReadOnlyKnowledgeGraph(graph)
    app.dependency_overrides[get_path_knowledge_graph] = lambda: knowledge_graph
    app.dependency_overrides[get_path_curriculum] = lambda: curriculum

    yield knowledge_graph, curriculum, graph

    app.dependency_overrides.pop(get_path_knowledge_graph, None)
    app.dependency_overrides.pop(get_path_curriculum, None)


def test_process_session_returns_200():
    response = client.post(
        "/session/process",
        json={"student_id": "s_test", "transcript": {"turns": []}},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["student_id"] == "s_test"
    assert "concepts_extracted" in body
    assert body["graph_updated"] is True


def test_get_student_profile_returns_200():
    response = client.get("/student/s_test/profile")
    assert response.status_code == 200
    body = response.json()
    assert body["student_id"] == "s_test"
    assert isinstance(body["concepts"], list)


def test_get_student_path_returns_real_prerequisite_aware_path(
    path_dependencies,
):
    knowledge_graph, curriculum, graph = path_dependencies
    response = client.get("/student/s_test/path")

    assert response.status_code == 200
    body = response.json()
    expected = generate_learning_path(graph, curriculum=curriculum)

    assert body["student_id"] == "s_test"
    assert "revise_urgently" in body
    assert "learn_next" in body
    assert "already_strong" in body
    assert "_mock" not in body
    assert body["recommended_order"] == expected["recommended_order"]
    assert body["blocked"] == expected["blocked"]
    assert body["unseen"] == expected["unseen"]

    target = next(entry for entry in body["blocked"] if entry["skill"] == "Target")
    assert target["unmet_prerequisites"] == ["Foundation"]

    # API rendering is a read-only derived view.
    assert knowledge_graph.graph_reads == 1
    assert knowledge_graph.attempt_writes == 0
    assert knowledge_graph.mastery_updates == 0


def test_get_student_path_is_deterministic_for_unchanged_mastery(
    path_dependencies,
):
    knowledge_graph, _curriculum, _graph = path_dependencies

    first = client.get("/student/s_test/path")
    second = client.get("/student/s_test/path")

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json() == second.json()
    assert knowledge_graph.graph_reads == 2
    assert knowledge_graph.attempt_writes == 0
    assert knowledge_graph.mastery_updates == 0


def test_proxy_blocked_unauthorized_skills_remain_unseen_in_api_path(
    path_dependencies,
):
    _knowledge_graph, _curriculum, graph = path_dependencies

    # Ratio and Fractions have no mastery rows: their only candidate evidence
    # was rejected by the behavioural-proxy authorization guard.
    assert {entry["skill"] for entry in graph}.isdisjoint({"Ratio", "Fractions"})

    body = client.get("/student/s_test/path").json()
    unseen = {entry["skill"]: entry for entry in body["unseen"]}

    assert unseen["Ratio"]["mastery_probability"] is None
    assert unseen["Fractions"]["mastery_probability"] is None
    assert unseen["Ratio"]["mastery_status"] == "unseen"
    assert unseen["Fractions"]["mastery_status"] == "unseen"


def test_create_new_student_returns_200():
    response = client.post(
        "/student/new",
        json={"first_message": "Hi, I need help with fractions."},
    )
    assert response.status_code == 200
    body = response.json()
    assert "student_id" in body
    assert body["profile"]["is_cold_start"] is True
