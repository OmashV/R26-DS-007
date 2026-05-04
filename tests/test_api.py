"""
Integration smoke tests for the FastAPI endpoints.
Verifies that all four routes return HTTP 200 with the expected mock shape.
Satisfies the Progress Presentation 1 acceptance criterion: endpoints reachable
and returning valid responses.
"""

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


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


def test_get_student_path_returns_200():
    response = client.get("/student/s_test/path")
    assert response.status_code == 200
    body = response.json()
    assert body["student_id"] == "s_test"
    assert "revise_urgently" in body
    assert "learn_next" in body
    assert "already_strong" in body


def test_create_new_student_returns_200():
    response = client.post(
        "/student/new",
        json={"first_message": "Hi, I need help with fractions."},
    )
    assert response.status_code == 200
    body = response.json()
    assert "student_id" in body
    assert body["profile"]["is_cold_start"] is True
