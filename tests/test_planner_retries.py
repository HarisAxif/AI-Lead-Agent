from types import SimpleNamespace

import pytest
import planner
import gemini_llm as backend
from schemas import SearchRequest, EvidencePlan


def setup_calls(monkeypatch, generate):
    monkeypatch.setattr(backend, "ensure_ready", lambda: SimpleNamespace(
        models=SimpleNamespace(generate_content=generate)))
    monkeypatch.setattr(backend, "limiter", SimpleNamespace(acquire=lambda: None))
    monkeypatch.setattr(backend, "handle_quota_error", lambda error: False)


def test_planning_recovers_using_fallback(monkeypatch):
    calls, sleeps = [], []
    expected = EvidencePlan(ideal_buyer="AR-menu prospect")
    def generate(**kwargs):
        calls.append(kwargs)
        if len(calls) < 3:
            raise RuntimeError("503 UNAVAILABLE high demand")
        return SimpleNamespace(parsed=expected)
    setup_calls(monkeypatch, generate)
    monkeypatch.setattr(backend, "GEMINI_FALLBACK_MODEL", "fallback")
    monkeypatch.setattr(backend, "GEMINI_THIRD_MODEL", None)
    monkeypatch.setattr(planner.time, "sleep", sleeps.append)
    assert planner._generate_research_plan(SearchRequest(service="virtual menu")) is expected
    assert [c["model"] for c in calls] == [backend.GEMINI_MODEL, "fallback", "fallback"]
    assert sleeps == [2, 4]
    assert "augmented reality (AR) menu" in calls[0]["contents"][0]


@pytest.mark.parametrize("code", [503, 504])
def test_exhausted_planning_is_not_cached(monkeypatch, tmp_path, code):
    calls = []
    def generate(**kwargs):
        calls.append(kwargs)
        error = RuntimeError("Temporary failure")
        error.code = code
        raise error
    setup_calls(monkeypatch, generate)
    monkeypatch.setattr(planner.time, "sleep", lambda seconds: None)
    monkeypatch.delenv("NO_PLAN_CACHE", raising=False)
    monkeypatch.setattr(planner, "PLAN_CACHE_DIR", tmp_path)
    with pytest.raises(backend.GeminiModelError):
        planner.make_research_plan(SearchRequest(service="virtual menu"))
    assert len(calls) == 3
    assert not list(tmp_path.iterdir())


def test_permanent_planning_error_is_not_retried(monkeypatch):
    calls = []
    def generate(**kwargs):
        calls.append(kwargs)
        raise ValueError("400 INVALID_ARGUMENT")
    setup_calls(monkeypatch, generate)
    with pytest.raises(backend.GeminiModelError):
        planner._generate_research_plan(SearchRequest())
    assert len(calls) == 1
