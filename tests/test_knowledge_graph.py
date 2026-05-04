"""
Smoke tests for core.knowledge_graph.
NFR6 — each core module shall have unit test coverage for its primary code path.
"""

import pytest
from core.knowledge_graph import load_graph, update_graph, detect_regressions, save_graph


def test_load_graph_is_stubbed():
    with pytest.raises(NotImplementedError):
        load_graph("s1")


def test_update_graph_is_stubbed():
    with pytest.raises(NotImplementedError):
        update_graph("s1", [], [])


def test_detect_regressions_is_stubbed():
    with pytest.raises(NotImplementedError):
        detect_regressions({"student_id": "s1", "concepts": []})


def test_save_graph_is_stubbed():
    with pytest.raises(NotImplementedError):
        save_graph("s1", {"student_id": "s1", "concepts": []})
