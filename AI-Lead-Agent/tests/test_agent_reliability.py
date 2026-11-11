"""Behavior regressions for free-provider operation and the reported failed run."""
from types import SimpleNamespace
import pytest

import adaptive_discovery
import discovery
import gemini_llm
import places
import trustpilot_reviews
import website
from web_search import search_public


def test_nonvisual_run_recovers_only_failed_sites(monkeypatch):
    companies = [{"domain": f"{name}.example", "url": f"https://{name}.example"}
                 for name in ("ok", "failed")]
    monkeypatch.setattr(website, "inspect_html",
                        lambda c, p: {"reachable": c["domain"].startswith("ok")})
    rendered = []
    def recover(items, plan, total):
        for _, company, site in items:
            rendered.append(company["domain"])
            site.update(reachable=True, text="Verified official business page")
    monkeypatch.setattr(website, "_render_partition", recover)
    sites = website.inspect_all(companies, {"visual_review_needed": False})
    assert rendered == ["failed.example"]
    assert all(site["reachable"] for site in sites)


def test_browser_fallback_cannot_use_dom_metadata_as_business_text(monkeypatch):
    company = {"domain": "example.com", "url": "https://example.com"}
    site = {"reachable": False, "footprint": "", "emails": [], "phones": []}
    captured = []
    def render(browser, company, screenshots):
        captured.append(screenshots)
        return {"rendered_text": "", "rendered_facts": "rendered_body_text_length: 0"}
    monkeypatch.setattr(website, "render_site", render)
    website._render_one(None, 1, company, site, {}, 1)
    assert not site["reachable"]
    assert captured == [False]


def test_missing_browser_does_not_crash_entire_batch(monkeypatch):
    def unavailable():
        raise RuntimeError("Chromium not installed")
    monkeypatch.setattr(website, "sync_playwright", unavailable)
    site = {"reachable": False}
    website._render_partition([(1, {}, site)], {})
    assert not site["reachable"]
    assert "Chromium" in site["browser_error"]


def test_empty_search_result_tries_second_backend(monkeypatch):
    monkeypatch.setenv("DDGS_BACKEND", "duckduckgo")
    monkeypatch.setenv("DDGS_FALLBACK_BACKEND", "yahoo")
    calls = []
    class Search:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def text(self, query, **kwargs):
            calls.append(kwargs["backend"])
            return [] if len(calls) == 1 else [{"href": "https://example.org"}]
    assert search_public("plumbers reviews UK", Search)
    assert calls == ["duckduckgo", "yahoo"]


def test_empty_planner_response_keeps_searching(monkeypatch):
    state = {"industry": "plumbers", "location": "UK", "number_of_leads": 3,
             "qualified_leads": [], "plan": {}, "round": 1,
             "research_feedback": [{"outcome": "no_candidates"}]}
    update = adaptive_discovery.revise_discovery(state, lambda p:
        adaptive_discovery.DiscoveryRevision(queries=[], stop=False, reason="Try another source"))
    assert not update.get("query_pool_exhausted")
    assert discovery.build_query_pool({**state, **update})


def test_paid_credentials_are_ignored_without_explicit_opt_in(monkeypatch):
    monkeypatch.delenv("ALLOW_PAID_PROVIDERS", raising=False)
    monkeypatch.setenv("GOOGLE_PLACES_API_KEY", "unused-test-key")
    monkeypatch.setenv("TRUSTPILOT_API_KEY", "unused-test-key")
    assert not places.is_configured()
    assert not trustpilot_reviews.configured()


def test_daily_quota_stops_subsequent_calls(monkeypatch):
    from pydantic import BaseModel
    class Answer(BaseModel):
        ok: bool
    class Quota(Exception):
        code = 429
    calls = []
    def generate(**kwargs):
        calls.append(kwargs)
        raise Quota("RequestsPerDay exceeded")
    monkeypatch.setattr(gemini_llm, "ensure_ready", lambda: SimpleNamespace(
        models=SimpleNamespace(generate_content=generate)))
    monkeypatch.setattr(gemini_llm.limiter, "acquire", lambda: None)
    for _ in range(2):
        with pytest.raises(gemini_llm.GeminiModelError):
            gemini_llm.structured_call("test", Answer)
    assert len(calls) == 1


def test_main_import_does_not_prompt_or_call_api(monkeypatch):
    import importlib
    monkeypatch.setattr("builtins.input", lambda *a: pytest.fail("Prompt during import"))
    monkeypatch.setattr(gemini_llm, "ensure_ready", lambda: pytest.fail("API during import"))
    import main
    importlib.reload(main)
    assert callable(main.main)


def test_browser_recovery_preserves_identity_and_source(monkeypatch):
    company = {"domain": "example.com", "url": "https://example.com"}
    site = {"reachable": False, "footprint": "", "emails": [], "phones": []}
    monkeypatch.setattr(website, "render_site", lambda *a, **k: {
        "rendered_text": "Example Plumbing serves Leeds, United Kingdom.",
        "rendered_facts": "", "business_name": "Example Plumbing",
        "rendered_url": "https://www.example.com/contact",
        "discovered_links": ["https://www.example.com/about"],
        "verified_review_profiles": ["https://reviews.example.org/example"],
    })
    website._render_one(None, 1, company, site, {}, 1)
    assert site["reachable"] and site["business_name"] == "Example Plumbing"
    assert site["pages_read"] == ["https://www.example.com/contact"]
    assert site["fallback_source"] == "browser_render"


@pytest.mark.parametrize("failure,exit_code", [(KeyboardInterrupt, 130), (RuntimeError, 1)])
def test_interrupted_workflow_exports_completed_stages(monkeypatch, tmp_path, failure, exit_code):
    import main
    from schemas import SearchRequest, EvidencePlan
    request = SearchRequest(location="UK", industry="plumbers", service="booking reminders",
                            number_of_leads=2)
    monkeypatch.setattr(main, "ensure_ready", lambda: None)
    monkeypatch.setattr(main, "prepare_request", lambda *a: request)
    monkeypatch.setattr(main, "make_research_plan", lambda *a: EvidencePlan(ideal_buyer="plumbers"))
    monkeypatch.setattr(main.result_cache, "load", lambda *a, **kw: None)
    monkeypatch.setattr(main.result_cache, "save", lambda *a: False)
    monkeypatch.setattr(main, "show_results", lambda *a: {})
    saved = []
    def export(request, plan, state, timings):
        saved.append(state)
        return tmp_path
    monkeypatch.setattr(main, "export_run", export)
    def stream(state, config):
        assert config["recursion_limit"] > 2 * discovery.MAX_ROUNDS
        yield {"research_companies": {"qualified_leads": [{"company": "Verified business"}]}}
        raise failure()
    monkeypatch.setattr(main, "graph", SimpleNamespace(stream=stream))
    assert main.main(["--request", "Find two plumbers"]) == exit_code
    assert saved[0]["qualified_leads"] == [{"company": "Verified business"}]
    assert saved[0]["discovery_stop_reason"]


def test_compound_criterion_rejects_phone_only_evidence_even_if_first_audit_accepts(monkeypatch):
    import evidence_gate as gate
    criterion = "Reviews mention unanswered calls and missed appointments without notification"
    evidence = [{"indicator": criterion, "evidence": "It just goes to answer phone.",
                 "source": "external_quote", "source_url": "https://reviews.example.org/company"}]
    calls = []
    def audit(prompt, schema):
        calls.append(prompt)
        if len(calls) == 1:
            return gate.EvidenceAudit(accepted_evidence_indices=[0], criteria_checks=[
                gate.CriterionCheck(criterion_index=0, supported=True, evidence_indices=[0],
                                    reason="Overbroad first-stage claim")])
        # The independent check sees raw observations, not the misleading label.
        observations = prompt.split("OBSERVATIONS: ", 1)[1]
        assert "missed appointments" not in observations and "indicator" not in observations
        return gate.EvidenceAudit(accepted_evidence_indices=[0], criteria_checks=[
            gate.CriterionCheck(criterion_index=0, supported=True, evidence_indices=[0], reason="Phone complaint"),
            gate.CriterionCheck(criterion_index=1, supported=False, evidence_indices=[],
                                reason="No missed visit", missing_facts=["Missed appointment without notification"])])
    monkeypatch.setattr(gate, "structured_call", audit)
    site = {"_external_criteria_required": [criterion]}
    _, ok, _ = gate.audit_evidence(evidence, [criterion], site)
    assert not ok and not site["_external_criteria_support"][criterion.lower()]
    assert site["criteria_audit"][0]["missing_facts"] == ["Missed appointment without notification"]


def test_clause_audit_requires_all_clauses_and_valid_indices(monkeypatch):
    import evidence_gate as gate
    evidence = [{"evidence": "Only calls are unanswered", "source": "external_quote"}]
    monkeypatch.setattr(gate, "structured_call", lambda *a: gate.EvidenceAudit(
        accepted_evidence_indices=[0], criteria_checks=[
            gate.CriterionCheck(criterion_index=0, supported=True, evidence_indices=[0], reason="One clause")]))
    criteria = ["Unanswered calls and missed appointments"]
    audit, _, _ = gate.audit_clause_support(evidence, criteria, {0})
    assert not gate.apply_audit(evidence, criteria, audit)[1]


def test_or_criterion_is_not_accidentally_converted_to_and():
    from evidence_gate import criterion_clauses
    criterion = "Slow responses OR missed appointments and no notice"
    assert criterion_clauses(criterion) == [criterion]


def test_exported_audit_indices_follow_filtered_evidence(monkeypatch):
    import evidence_gate as gate
    evidence = [{"indicator": "irrelevant", "evidence": "Website has a phone", "source": "website_text"},
                {"indicator": "unanswered calls", "evidence": "Nobody answered my calls", "source": "external_quote"}]
    monkeypatch.setattr(gate, "structured_call", lambda *a: gate.EvidenceAudit(
        accepted_evidence_indices=[1], criteria_checks=[
            gate.CriterionCheck(criterion_index=0, supported=True, evidence_indices=[1], reason="Calls unanswered")]))
    site = {}
    filtered, ok, _ = gate.audit_evidence(evidence, ["Unanswered calls"], site)
    assert ok and len(filtered) == 1
    assert site["criteria_audit"][0]["evidence_indices"] == [0]
    assert site["clause_audit"][0]["evidence_indices"] == [0]
