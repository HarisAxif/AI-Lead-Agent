import ast
from pathlib import Path
import pytest
import discovery
import gemini_limits
from schemas import SearchResultClassification


class Search:
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def text(self, *args, **kwargs):
        return [{"href": "https://example.com", "title": "Studio", "body": "Music"}]


def test_rejected_domain_reuses_classification_across_queries(monkeypatch):
    calls = []
    def classify(candidates, *args):
        calls.extend(candidates)
        return [{"candidate": c, "error": None, "classification":
                 SearchResultClassification(result_type="directory",
                    matches_target_industry=True, confidence=1)} for c in candidates]
    monkeypatch.setattr(discovery, "DDGS", Search)
    monkeypatch.setattr(discovery, "classify_candidates_parallel", classify)
    cache = {}
    for query in ("first", "second"):
        result = discovery.search_one_query(query, set(), set(), 5, "studio", "city", cache)
        assert result["candidates"] == []
    assert len(calls) == 1


def test_api_failure_never_admits_an_unclassified_domain(monkeypatch):
    monkeypatch.setattr(discovery, "DDGS", Search)
    monkeypatch.setattr(discovery, "classify_candidates_parallel", lambda candidates, *args:
        [{"candidate": c, "classification": None, "error": "429 RESOURCE_EXHAUSTED"}
         for c in candidates])
    cache = {}
    result = discovery.search_one_query("query", set(), set(), 5, "studio", "city", cache)
    assert result["candidates"] == []
    assert result["classifier_failed_count"] == 1
    assert cache == {}


def test_exhaustion_preserves_actual_round_count():
    state = dict(round=5, number_of_leads=10, qualified_leads=[],
                 location="city", industry="studios", plan={}, geo_terms=[])
    state["seen_queries"] = [q.lower() for q in discovery.build_query_pool(state)]
    result = discovery.search_companies(state)
    assert result["round"] == 5
    assert result["search_exhausted"] is True
    # Exercise the production router without importing the interactive CLI.
    tree = ast.parse(Path("main.py").read_text(encoding="utf-8"))
    router = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                  and n.name == "route_after_research")
    namespace = {"AgentState": dict, "MAX_ROUNDS": 12}
    exec(compile(ast.Module(body=[router], type_ignores=[]), "main.py", "exec"), namespace)
    assert namespace["route_after_research"]({**state, **result}) == "show_results"


@pytest.mark.parametrize("message,expected", [
    ("Please retry in 20.913894138s. retryDelay: '20s'", 21.913894138),
    ("429 RESOURCE_EXHAUSTED", 31),
    ("retryDelay: '65s'", 66),
])
def test_retry_delay_honors_server(message, expected):
    assert gemini_limits.retry_delay(RuntimeError(message)) == pytest.approx(expected)


def test_pacing_and_cooldown_use_shared_clock(monkeypatch):
    clock = [0.0]
    sleeps = []
    monkeypatch.setattr(gemini_limits.time, "monotonic", lambda: clock[0])
    def sleep(seconds):
        sleeps.append(seconds)
        clock[0] += seconds
    monkeypatch.setattr(gemini_limits.time, "sleep", sleep)
    limiter = gemini_limits.RequestLimiter(2)
    limiter.acquire()
    limiter.acquire()
    limiter.acquire()
    assert clock[0] >= 60
    limiter.defer(21)
    limiter.acquire()
    assert clock[0] >= 81
    assert max(sleeps) <= 60
