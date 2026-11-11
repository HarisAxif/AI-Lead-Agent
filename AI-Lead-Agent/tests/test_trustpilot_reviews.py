import json
import httpx
import pytest
import trustpilot_reviews as provider
import external_research
from exporter import public_lead

UNIT = "507f191e810c19729de860ea"
REVIEW = "507f191e810c19729de860eb"
DOMAIN = "exampleplumbing.co.uk"


@pytest.fixture(autouse=True)
def reset(monkeypatch):
    provider.reset_run()
    monkeypatch.delenv("TRUSTPILOT_API_KEY", raising=False)
    monkeypatch.delenv("TRUSTPILOT_MAX_REQUESTS_PER_RUN", raising=False)
    yield
    provider.reset_run()


def client(monkeypatch, handler):
    monkeypatch.setenv("ALLOW_PAID_PROVIDERS", "1")
    original = httpx.Client
    monkeypatch.setenv("TRUSTPILOT_API_KEY", "test-secret")
    monkeypatch.setattr(provider.httpx, "Client", lambda **kw: original(
        transport=httpx.MockTransport(handler), **kw))


def unit(domain=DOMAIN):
    return {"businessUnits": [{"id": UNIT, "name": {
        "identifying": domain, "referring": ["www." + domain], "display": "Example Plumbing"}}]}


def test_unconfigured_provider_makes_no_requests(monkeypatch):
    monkeypatch.setattr(provider.httpx, "Client", lambda **kw: pytest.fail("No requests without a key"))
    assert provider.collect_reviews({"domain": DOMAIN}) == {"sources": [], "errors": []}


def test_exact_identity_and_review_citation(monkeypatch):
    calls = []
    def handle(request):
        calls.append(request)
        assert request.headers["apikey"] == "test-secret"
        assert "test-secret" not in str(request.url)
        if request.url.path.endswith("/business-units"):
            assert request.url.params["domain"] == DOMAIN
            return httpx.Response(200, json=unit())
        return httpx.Response(200, json={"reviews": [{"id": REVIEW,
            "text": "Nobody answered and no update when the engineer missed the appointment.",
            "createdAt": "2026-10-01T10:00:00Z"}]})
    client(monkeypatch, handle)
    result = provider.collect_reviews({"domain": DOMAIN})
    assert len(calls) == 2 and not result["errors"]
    assert result["sources"][0]["url"] == "https://www.trustpilot.com/reviews/" + REVIEW
    assert "Nobody answered" in result["sources"][0]["text"]
    assert result["sources"][0]["provider"] == "trustpilot_api"


def test_mismatching_business_is_rejected_without_fetching_reviews(monkeypatch):
    calls = []
    def handle(request):
        calls.append(request)
        return httpx.Response(200, json=unit("otherbusiness.co.uk"))
    client(monkeypatch, handle)
    result = provider.collect_reviews({"domain": DOMAIN})
    assert not result["sources"] and result["errors"][0]["kind"] == "provider_identity"
    assert len(calls) == 1


def test_entitlement_failure_is_not_retried_and_does_not_leak_key(monkeypatch):
    calls = []
    def handle(request):
        calls.append(request)
        return httpx.Response(403, text="secret echo: test-secret")
    client(monkeypatch, handle)
    first = provider.collect_reviews({"domain": DOMAIN})
    second = provider.collect_reviews({"domain": DOMAIN})
    assert len(calls) == 1
    assert "test-secret" not in json.dumps(first) + json.dumps(second)
    assert "Insights" in first["errors"][0]["error"]


def test_provider_budget_and_page_limit(monkeypatch):
    calls = []
    def handle(request):
        calls.append(request)
        if request.url.path.endswith("/business-units"):
            return httpx.Response(200, json=unit())
        return httpx.Response(200, json={"reviews": [], "nextToken": "another-page"})
    client(monkeypatch, handle)
    provider.collect_reviews({"domain": DOMAIN})
    assert len(calls) == 3  # lookup and at most two review pages
    provider.reset_run()
    calls.clear()
    monkeypatch.setenv("TRUSTPILOT_MAX_REQUESTS_PER_RUN", "1")
    result = provider.collect_reviews({"domain": DOMAIN})
    assert len(calls) == 1 and "budget" in result["errors"][0]["error"]


def test_api_success_avoids_html_scraping(monkeypatch):
    source = {"url": "https://www.trustpilot.com/reviews/" + REVIEW,
              "text": "Verified fetched API text", "provider": "trustpilot_api"}
    monkeypatch.setattr(provider, "collect_reviews", lambda c: {"sources": [source], "errors": []})
    monkeypatch.setattr(external_research, "DDGS", lambda *a: pytest.fail("API result already available"))
    result = external_research.collect_external_sources({"domain": DOMAIN}, "UK", {
        "external_search_queries": ["{company} reviews"]})
    assert result["sources"] == [source]


def test_saved_evidence_does_not_retain_api_review_text():
    original = {"evidence": [{"indicator": "Unanswered calls", "source": "external_quote",
        "evidence": "Private-to-this-run review text", "provider": "trustpilot_api",
        "source_url": "https://www.trustpilot.com/reviews/" + REVIEW}]}
    saved = public_lead(original)
    assert "Private-to-this-run" not in json.dumps(saved)
    assert saved["evidence"][0]["source_url"].endswith(REVIEW)
    assert original["evidence"][0]["evidence"] == "Private-to-this-run review text"
