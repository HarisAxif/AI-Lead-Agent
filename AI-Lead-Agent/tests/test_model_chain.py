from types import SimpleNamespace

import pytest
import clarify
import planner
import discovery
import qualify
from gemini_models import model_for_attempt
from schemas import SearchRequest


def test_model_order_and_missing_slots():
    assert [model_for_attempt(i, "one", "two", "three") for i in (1, 2, 3)] == ["one", "two", "three"]
    assert model_for_attempt(3, "one", "two") == "two"
    assert model_for_attempt(2, "one", None, "three") == "three"
    assert model_for_attempt(2, "one", "one", "three") == "three"


@pytest.mark.parametrize("module", [clarify, planner, discovery, qualify])
def test_third_model_recovers_after_two_failures(monkeypatch, module):
    import gemini_llm as backend
    from schemas import EvidencePlan, SearchResultClassification, LeadAnalysis
    if module is clarify:
        expected = SearchRequest(location="Lahore")
    elif module is planner:
        expected = EvidencePlan(ideal_buyer="Restaurants")
    elif module is discovery:
        expected = SearchResultClassification(result_type="single_business",
            matches_target_industry=True, confidence=0.9)
    else:
        expected = LeadAnalysis(company_name="Example", is_single_business_site=True,
            in_target_location=True, matches_ideal_buyer=True, hit_disqualifier=False,
            reason="Evidence", opportunity="Discuss service")
    calls = []
    def generate(**kwargs):
        calls.append(kwargs["model"])
        if len(calls) < 3:
            error = RuntimeError("503 UNAVAILABLE")
            error.code = 503
            raise error
        return SimpleNamespace(parsed=expected)
    monkeypatch.setattr(backend, "ensure_ready", lambda: SimpleNamespace(
        models=SimpleNamespace(generate_content=generate)))
    for name, value in (("GEMINI_MODEL", "one"), ("GEMINI_FALLBACK_MODEL", "two"),
                        ("GEMINI_THIRD_MODEL", "three")):
        monkeypatch.setattr(backend, name, value)
    monkeypatch.setattr(backend, "limiter", SimpleNamespace(acquire=lambda: None))
    monkeypatch.setattr(backend, "handle_quota_error", lambda e: False)
    monkeypatch.setattr(backend.time, "sleep", lambda seconds: None)
    if module is clarify:
        assert module._structured_call("prompt", SearchRequest) is expected
    elif module is planner:
        assert module._generate_research_plan(SearchRequest()) is expected
    elif module is discovery:
        assert module.classify_search_result({}, "restaurants", "Pakistan") == (expected, None)
    else:
        company = {"domain":"example.com", "title":"Example", "url":"https://example.com", "snippet":""}
        site = {"text":"Restaurant in Pakistan", "footprint":""}
        state = {"industry":"restaurants", "location":"Pakistan", "service":"booking",
                 "plan":{"ideal_buyer":"restaurant", "need_indicators":[], "disqualifiers":[]}}
        assert module.judge_company_with_retry(0, company, site, state)["analysis"] is expected
    assert calls == ["one", "two", "three"]
