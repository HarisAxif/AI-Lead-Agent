from types import SimpleNamespace
import httpx
import discovery
import website
from schemas import SearchResultClassification


def test_permanent_http_errors_are_not_retried():
    calls = []
    def handle(request):
        calls.append(str(request.url))
        return httpx.Response(404)
    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        assert website.fetch_page("https://example.com/missing", client) is None
    assert calls == ["https://example.com/missing", "http://example.com/missing"]


def test_transient_http_error_retries_with_same_client():
    calls = []
    def handle(request):
        calls.append(request)
        return httpx.Response(503 if len(calls) == 1 else 200,
                              headers={"content-type": "text/html"}, text="ok")
    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        assert website.fetch_page("https://example.com", client).text == "ok"
    assert len(calls) == 2


def test_duplicate_search_domains_are_classified_once(monkeypatch):
    class Search:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def text(self, *args, **kwargs):
            return [{"href": "https://example.com/a"},
                    {"href": "https://www.example.com/b"}]
    captured = []
    def classify(candidates, *args):
        captured.extend(candidates)
        return [{"candidate": c, "error": None, "classification":
                 SearchResultClassification(result_type="single_business",
                    matches_target_industry=True, confidence=1)} for c in candidates]
    monkeypatch.setattr(discovery, "DDGS", Search)
    monkeypatch.setattr(discovery, "classify_candidates_parallel", classify)
    result = discovery.search_one_query("query", set(), set(), 3, "industry", "city")
    assert len(captured) == len(result["candidates"]) == 1
    assert result["duplicate_count"] == 1


def test_visual_workers_keep_results_in_company_order(monkeypatch):
    monkeypatch.setenv("RENDER_WORKERS", "3")
    companies = [{"domain": str(i), "url": "https://example.com"} for i in range(7)]
    monkeypatch.setattr(website, "inspect_html", lambda c, p:
                        {"reachable": True, "domain": c["domain"]})
    batches = []
    def render(items, plan, total=None):
        batches.append(items)
        for index, company, site in items:
            site["rendered"] = company["domain"]
    monkeypatch.setattr(website, "_render_partition", render)
    results = website.inspect_all(companies, {"visual_review_needed": True})
    assert len(batches) == 3
    assert [s["rendered"] for s in results] == [c["domain"] for c in companies]


def test_last_lead_does_not_trigger_three_inspections(monkeypatch):
    budgets = []
    def search(**kwargs):
        budgets.append(kwargs["slots_remaining"])
        return {"candidates": [{"domain": "example.com", "url": "https://example.com"}],
                "raw_count": 1, "blocked_count": 0, "duplicate_count": 0,
                "classifier_rejected_count": 0, "classifier_failed_count": 0}
    monkeypatch.setattr(discovery, "search_one_query", search)
    state = {"round": 0, "number_of_leads": 5, "qualified_leads": [{}] * 4,
             "location": "city", "industry": "industry", "plan": {},
             "geo_terms": [], "seen_domains": []}
    result = discovery.search_companies(state)
    assert budgets == [1]
    assert len(result["companies"]) == 1


def test_content_wait_keeps_slow_site_fallback():
    class Page:
        def __init__(self): self.waits = []
        def wait_for_function(self, expression, timeout):
            assert timeout == 9000
            raise RuntimeError("content not ready")
        def wait_for_timeout(self, milliseconds): self.waits.append(milliseconds)
        def evaluate(self, expression): pass
    page = Page()
    website.wait_for_page_to_settle(page)
    assert sum(page.waits) == 1100
