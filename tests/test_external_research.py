from types import SimpleNamespace

import httpx
import external_research as external
from qualify import verify_findings, external_criteria_met, has_gap_evidence
from schemas import IndicatorFinding


INDICATOR = "Parents report missed school announcements"
QUOTE = "At Example School, we missed the term announcement because parents were not notified."
URL = "https://reviews.example.org/example-school"


def test_external_quote_requires_fetched_url_and_exact_text():
    site = {"text": "", "footprint": "", "external_sources": [
        {"url": URL, "text": QUOTE}]}
    good = IndicatorFinding(indicator=INDICATOR, present=True,
                            evidence_type="external_quote", quote=QUOTE, source_url=URL)
    verified = verify_findings([good], site, [INDICATOR])
    assert len(verified) == 1
    assert verified[0]["source_url"] == URL
    assert external_criteria_met(verified, [INDICATOR])
    assert has_gap_evidence(verified)
    for bad in [good.model_copy(update={"source_url": "https://fake.example.org"}),
                good.model_copy(update={"quote": "An invented complaint not present in the fetched page"})]:
        assert not verify_findings([bad], site, [INDICATOR])


def test_website_proxies_cannot_satisfy_required_complaints():
    assert not external_criteria_met([
        {"indicator": INDICATOR, "source": "website_text"}], [INDICATOR])
    assert not external_criteria_met([], [INDICATOR])


def test_collector_reads_source_instead_of_trusting_snippet(monkeypatch):
    company = {"title": "Example School", "domain": "example.edu", "url": "https://example.edu"}
    class Search:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def text(self, query, **kwargs):
            assert "Example School" in query
            return [{"href": URL, "title": "Reviews", "body": "Invented snippet"},
                    {"href": "https://blocked.example.org", "body": QUOTE}]
    def handle(request):
        if request.url.host == "blocked.example.org":
            return httpx.Response(403)
        return httpx.Response(200, headers={"content-type": "text/html"},
                              text="<html><p>" + QUOTE + " Further school communication discussion.</p></html>")
    original_client = httpx.Client
    monkeypatch.setattr(external, "DDGS", Search)
    monkeypatch.setattr(external.httpx, "Client", lambda **kwargs: original_client(
        transport=httpx.MockTransport(handle), **kwargs))
    result = external.collect_external_sources(company, "Pakistan", {
        "external_search_queries": ['"{company}" "{location}" parent missed announcements']})
    assert len(result["sources"]) == 1
    assert QUOTE in result["sources"][0]["text"]
    assert "Invented snippet" not in result["sources"][0]["text"]
    assert len(result["errors"]) == 1


def test_identity_and_private_url_guards():
    company = {"title": "Example School", "domain": "example.edu"}
    assert external.same_business_source(QUOTE, company)
    assert not external.same_business_source("Different School complaints", company)
    for url in ["http://localhost/", "http://127.0.0.1/", "http://10.0.0.1/", "file:///tmp/reviews"]:
        assert not external.public_url(url)


def test_website_only_plans_skip_external_search(monkeypatch):
    monkeypatch.setattr(external, "collect_external_sources", lambda *args: (_ for _ in ()).throw(AssertionError()))
    external.inspect_external_all([], [], "Pakistan", {})


def test_external_stage_attaches_fetched_pages(monkeypatch):
    calls = []
    def collect(company, location, plan):
        calls.append((company["domain"], location))
        return {"sources": [{"url": URL, "text": QUOTE}], "errors": []}
    monkeypatch.setattr(external, "collect_external_sources", collect)
    sites = [{"reachable": True}, {"reachable": False}]
    external.inspect_external_all(
        [{"domain": "example.edu"}, {"domain": "offline.edu"}], sites,
        "Pakistan", {"external_search_queries": ["reviews"]})
    assert calls == [("example.edu", "Pakistan")]
    assert sites[0]["external_sources"][0]["url"] == URL
    assert "external_sources" not in sites[1]


def test_planner_routes_explicit_reviews_without_places(monkeypatch):
    import planner
    from schemas import EvidencePlan, SearchRequest
    monkeypatch.delenv("GOOGLE_PLACES_API_KEY", raising=False)
    captured = []
    def generate(prompt, schema):
        captured.append(prompt)
        return EvidencePlan(ideal_buyer="schools")
    monkeypatch.setattr(planner, "structured_call", generate)
    criterion = "customer reviews complain about missed announcements"
    plan = planner._generate_research_plan(SearchRequest(
        industry="schools", location="Pakistan", service="parent communication software",
        number_of_leads=2, criteria=[criterion]))
    assert criterion in plan.external_criteria
    assert criterion in plan.need_indicators
    assert "{company}" in plan.external_search_queries[0]
    assert "Search snippets alone never establish evidence" in captured[0]


def test_validator_does_not_invent_private_complaint_requirement(monkeypatch):
    import clarify
    from schemas import SearchRequest, CriteriaValidation, CriterionAssessment
    criterion = "whose customers complain about missed announcements"
    monkeypatch.setattr(clarify, "_structured_call", lambda *args: CriteriaValidation(
        assessments=[CriterionAssessment(claim=criterion, classification="unverifiable",
            reason="Cannot access private customer complaints",
            suggested_replacement="no announcements page")]))
    result = clarify.validate_criteria(SearchRequest(industry="schools", location="Lahore",
        service="parent communication", number_of_leads=2, criteria=[criterion]))
    assert result.assessments[0].classification == "verifiable_public"
    assert result.assessments[0].claim == criterion
    assert result.assessments[0].suggested_replacement is None


def test_validator_preserves_private_and_service_rejections(monkeypatch):
    import clarify
    from schemas import SearchRequest, CriteriaValidation, CriterionAssessment
    cases = [
        ("private customer complaints", "unverifiable"),
        ("customers complain and will lose revenue next month", "unverifiable"),
        ("80 percent of customer reviews complain", "unverifiable"),
        ("customers complain about SEO", "mismatched_with_service"),
    ]
    for criterion, classification in cases:
        monkeypatch.setattr(clarify, "_structured_call", lambda *args: CriteriaValidation(
            assessments=[CriterionAssessment(claim=criterion,
                classification=classification, reason="Original reason")]))
        result = clarify.validate_criteria(SearchRequest(industry="schools", location="Lahore",
            service="parent communication", number_of_leads=2, criteria=[criterion]))
        assert result.assessments[0].classification == classification


def test_planner_deduplicates_paraphrased_external_criterion(monkeypatch):
    import planner
    from schemas import EvidencePlan, SearchRequest
    criterion = "because patients complain about being unreachable"
    monkeypatch.setattr(planner, "structured_call", lambda *a: EvidencePlan(
        ideal_buyer="clinics", external_criteria=["patients complain about being unreachable"],
        external_search_queries=["{company} reviews"]))
    plan = planner._generate_research_plan(SearchRequest(industry="dental clinics",
        location="Lahore", service="answering service", number_of_leads=10,
        criteria=[criterion]))
    assert plan.external_criteria == [criterion]
    assert criterion in plan.need_indicators


def test_failed_pages_do_not_consume_entire_source_budget(monkeypatch):
    company = {"title": "Example School", "domain": "example.edu"}
    class Search:
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def text(self, query, **kwargs):
            if " first " in query:
                return [{"href": f"https://blocked.example.org/{i}"} for i in range(4)]
            return [{"href": URL}]
    def handle(request):
        if request.url.host == "blocked.example.org":
            return httpx.Response(403)
        return httpx.Response(200, headers={"content-type": "text/html"},
            text="<p>" + QUOTE + " More discussion of school communication.</p>")
    original = httpx.Client
    monkeypatch.setattr(external, "DDGS", Search)
    monkeypatch.setattr(external.httpx, "Client", lambda **kw: original(
        transport=httpx.MockTransport(handle), **kw))
    result = external.collect_external_sources(company, "Lahore", {
        "external_search_queries": ["first", "second"]})
    assert len(result["sources"]) == 1
    assert len(result["errors"]) == 4
    assert len(result["diagnostics"]) == 2


def test_external_search_retries_another_backend(monkeypatch):
    class Search:
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def text(self, query, **kwargs):
            if kwargs.get("backend") != "yahoo":
                raise RuntimeError("DNS failure")
            return []
    monkeypatch.setattr(external, "DDGS", Search)
    result = external.collect_external_sources({"domain": "example.edu"}, "Lahore", {
        "external_search_queries": ["reviews"]})
    assert result["diagnostics"][0]["fallback"] is True
    assert result["diagnostics"][0]["results"] == 0
    assert len(result["errors"]) == 1


def test_business_queries_remove_tagline_and_keep_location():
    company = {"title": "St. Paul's Free Clinic | Quality health care is available. And it's free.",
               "domain": "stpaulsfreeclinic.org"}
    queries = external.external_queries(company, "Erie, Pennsylvania", {
        "external_search_queries": ["site:yelp.com {company} {location} reviews"]})
    assert queries == ['"St. Paul\'s Free Clinic" Erie, Pennsylvania reviews']
    assert external.same_business_source("Reviews of St. Paul's Free Clinic in Erie", company)
    assert not external.same_business_source("Other Free Clinic in York", company)


def test_rate_limited_host_is_not_fetched_repeatedly(monkeypatch):
    company = {"title": "Example School", "domain": "example.edu"}
    class Search:
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def text(self, *a, **kw):
            return [{"href": f"https://denied.example.org/{i}"} for i in range(4)]
    requests = []
    def handle(request):
        requests.append(str(request.url))
        return httpx.Response(429)
    original = httpx.Client
    monkeypatch.setattr(external, "DDGS", Search)
    monkeypatch.setattr(external.httpx, "Client", lambda **kw: original(
        transport=httpx.MockTransport(handle), **kw))
    result = external.collect_external_sources(company, "Erie", {
        "external_search_queries": ["{company} reviews"]})
    assert len(requests) == 1
    assert len(result["errors"]) == 4
    assert all(error["kind"] == "access_denied" for error in result["errors"])


def test_discovery_search_retries_after_primary_failure(monkeypatch):
    import discovery
    calls = []
    class Search:
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def text(self, query, **kw):
            calls.append(kw)
            if kw.get("backend") != "yahoo":
                raise RuntimeError("DNS refused")
            return []
    monkeypatch.setattr(discovery, "DDGS", Search)
    result = discovery.search_one_query("clinic Erie", set(), set(), 5, "clinics", "Erie")
    assert len(calls) == 2
    assert "backend" in calls[1]
    assert result["raw_count"] == 0


def test_generic_maps_page_is_not_fetched(monkeypatch):
    class Search:
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def text(self, *a, **kw):
            return [{"href": "https://maps.google.com/"}]
    monkeypatch.setattr(external, "DDGS", Search)
    result = external.collect_external_sources({"title": "Example School", "domain": "example.edu"},
        "Erie", {"external_search_queries": ["{company} reviews"]})
    assert result["errors"][0]["kind"] == "generic_listing"
    assert result["sources"] == []
