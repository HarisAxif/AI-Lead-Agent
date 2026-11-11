from datetime import timedelta
from types import SimpleNamespace

import httpx
import pytest
from bs4 import BeautifulSoup

import discovery
import external_research as external
import review_discovery
import website
from business_identity import extract_identity, search_name
from schemas import (SearchResultClassification, SearchClassificationBatch,
                     IndexedSearchClassification, IndicatorFinding)
from qualify import verify_findings


def test_home_title_uses_structured_business_name():
    soup = BeautifulSoup("""<title>Home</title><script type="application/ld+json">
    {"@type":"Plumber","name":"JS Gas and Plumbing","url":"https://jsgasandplumbing.co.uk"}
    </script>""", "html.parser")
    identity = extract_identity(soup, "jsgasandplumbing.co.uk")
    assert identity["business_name"] == "JS Gas and Plumbing"
    assert search_name({**identity, "domain": "jsgasandplumbing.co.uk"}) == "JS Gas and Plumbing"


@pytest.mark.parametrize("title", ["Home", "Professional Plumbing Services", "Boiler Repair Glasgow"])
def test_generic_titles_do_not_become_business_names(title):
    company = {"title": title, "domain": "heatingdynamics.co.uk"}
    assert search_name(company, "Glasgow") == "heatingdynamics.co.uk"


def test_brand_suffix_is_preferred_to_generic_prefix():
    assert search_name({"title": "Boiler Repair Glasgow | Heating Dynamics", "domain": "heatingdynamics.co.uk"}) == "Heating Dynamics"


def test_foreign_organization_schema_is_not_used():
    soup = BeautifulSoup("""<title>Home</title><script type="application/ld+json">
    {"@type":"Organization","name":"Website Agency","url":"https://agency.example"}
    </script>""", "html.parser")
    assert extract_identity(soup, "jsgasandplumbing.co.uk")["business_name"] == "jsgasandplumbing.co.uk"


def test_complaint_queries_are_first_in_pool():
    state = {"industry": "plumbers", "location": "UK", "plan": {
        "complaint_search_queries": ["plumbers UK unanswered calls reviews"],
        "search_queries": ["plumbers UK official website"]}}
    assert discovery.build_query_pool(state)[0] == "plumbers UK unanswered calls reviews"


def test_batch_classification_uses_one_call_and_rejects_missing_indices(monkeypatch):
    candidates = [{"url": f"https://company{i}.example/", "domain": f"company{i}.example"} for i in range(4)]
    calls = []
    def classify(prompt, schema):
        calls.append(prompt)
        return SearchClassificationBatch(items=[IndexedSearchClassification(candidate_index=i,
            classification=SearchResultClassification(result_type="single_business",
                matches_target_industry=True, confidence=0.9)) for i in range(4)])
    monkeypatch.setattr(discovery, "structured_call", classify)
    results = discovery.classify_candidates_parallel(candidates, "plumbers", "UK")
    assert len(calls) == 1
    assert all(item["classification"] for item in results)
    monkeypatch.setattr(discovery, "structured_call", lambda *a: SearchClassificationBatch(items=[]))
    assert all(item["classification"] is None for item in discovery.classify_candidates_parallel(candidates, "plumbers", "UK"))


def test_wrong_trustpilot_profile_cannot_match_generic_name():
    company = {"title": "Glasgow Boiler Repair", "domain": "heatingdynamics.co.uk"}
    assert not external.same_business_source("Glasgow Boiler Repair heatingdynamics.co.uk", company,
        "https://uk.trustpilot.com/review/differentcompany.co.uk")


def test_review_jsonld_is_available_as_fetched_evidence():
    soup = BeautifulSoup("""<h1>Example Plumbing</h1><script type="application/ld+json">
    {"@type":"Review","reviewBody":"They missed the appointment without updates and never answered my calls."}
    </script>""", "html.parser")
    assert "never answered my calls" in external.readable_external_text(soup)


def test_refused_profile_does_not_block_another_profile(monkeypatch):
    company = {"title": "Example Plumbing", "domain": "exampleplumbing.co.uk"}
    urls = ["https://reviews.example.org/denied", "https://reviews.example.org/readable"]
    class Search:
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def text(self, *a, **kw): return [{"href": url} for url in urls]
    requests = []
    def handle(request):
        requests.append(str(request.url))
        if request.url.path == "/denied": return httpx.Response(403)
        return httpx.Response(200, headers={"content-type":"text/html"}, text=
            "Example Plumbing at exampleplumbing.co.uk missed my appointment without calling me. I tried repeatedly and nobody answered my calls.")
    original = httpx.Client
    monkeypatch.setattr(external, "DDGS", Search)
    monkeypatch.setattr(external.httpx, "Client", lambda **kw: original(transport=httpx.MockTransport(handle), **kw))
    result = external.collect_external_sources(company, "UK", {"external_search_queries":["{company} reviews"]})
    assert len(requests) == 2
    assert len(result["sources"]) == 1


def test_review_discovery_to_verified_quote_pipeline(monkeypatch):
    review_url = "https://uk.trustpilot.com/review/johnspipe.co.uk"
    criterion = "Customers complain of unanswered calls and no updates on missed appointments"
    quote = "John's Pipe Co missed my appointment without an update and did not answer my calls."
    plan = {"complaint_search_queries": ["plumbers UK missed appointments reviews"],
            "external_search_queries": ["{company} reviews"], "pages_to_check": [], "country_code":"GB"}
    class Search:
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def text(self, *a, **kw):
            return [{"href":review_url,"title":"John's Pipe Co Reviews","body":"Snippet is not evidence"}]
    monkeypatch.setattr(discovery, "DDGS", Search)
    monkeypatch.setattr(external, "DDGS", Search)
    monkeypatch.setattr(discovery, "structured_call", lambda *a: SearchClassificationBatch(items=[
        IndexedSearchClassification(candidate_index=0, classification=SearchResultClassification(
            result_type="single_business", matches_target_industry=True, confidence=0.9))]))
    found = discovery.run_one_query(plan["complaint_search_queries"][0], {
        "plan":plan,"industry":"plumbers","location":"United Kingdom"}, set(), set(), 3)
    company = found["candidates"][0]
    assert company["seed_external_urls"] == [review_url]
    response = httpx.Response(200, request=httpx.Request("GET",company["url"]),
        headers={"content-type":"text/html"}, text="""<title>Home</title><h1>John's Pipe Co</h1>
        <p>Plumbing and heating services in London, United Kingdom.</p>
        <script type="application/ld+json">{"@type":"Plumber","name":"John's Pipe Co","url":"https://johnspipe.co.uk"}</script>""")
    response.elapsed = timedelta(seconds=0.01)
    monkeypatch.setattr(website, "fetch_page", lambda *a: response)
    site = website.inspect_html(company, plan)
    assert site["business_name"] == "John's Pipe Co"
    original = httpx.Client
    monkeypatch.setattr(external.httpx, "Client", lambda **kw: original(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, headers={"content-type":"text/html"},
            text="<h1>John's Pipe Co ? johnspipe.co.uk</h1><p>"+quote+"</p>")), **kw))
    external.inspect_external_all([company], [site], "United Kingdom", plan)
    finding = IndicatorFinding(indicator=criterion, present=True, evidence_type="external_quote",
        quote=quote, source_url=review_url)
    verified = verify_findings([finding], site, [criterion])
    assert len(verified) == 1 and verified[0]["source_url"] == review_url
    assert "Snippet is not evidence" not in site["external_sources"][0]["text"]


def test_main_website_fetch_blocks_localhost_redirect():
    requests = []
    def handle(request):
        requests.append(str(request.url))
        return httpx.Response(307, headers={"location":"http://localhost/"})
    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        assert website.fetch_page("https://public.example/", client) is None
    assert requests == ["https://public.example/"]
    assert "non-public" in website.FETCH_ERRORS["https://public.example/"]


def test_main_website_fetch_does_not_retry_access_denied():
    requests = []
    def handle(request):
        requests.append(str(request.url))
        return httpx.Response(403)
    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        assert website.fetch_page("https://denied.example/", client) is None
    assert requests == ["https://denied.example/"]


@pytest.mark.parametrize("audit_supports_both", [True, False])
def test_single_complaint_passage_reaches_audit(monkeypatch, audit_supports_both):
    import qualify
    from schemas import LeadAnalysis
    criterion = "Customer reviews mention unanswered calls and missed appointments without notification"
    quote = "Example Plumbing missed my appointment without any notification and never answered my calls."
    url = "https://reviews.example.org/example-plumbing"
    company = {"domain":"exampleplumbing.co.uk", "title":"Example Plumbing",
               "url":"https://exampleplumbing.co.uk/", "snippet":"UK plumber"}
    site = {"reachable":True, "text":"Example Plumbing in the United Kingdom", "footprint":"",
            "emails":[], "phones":[], "pages_read":[company["url"]],
            "external_sources":[{"url":url,"text":quote}]}
    analysis = LeadAnalysis(company_name="Example Plumbing", is_single_business_site=True,
        in_target_location=True, matches_ideal_buyer=True, hit_disqualifier=False,
        reason="Complaint", opportunity="Discuss call handling", findings=[IndicatorFinding(
            indicator=criterion, present=True, evidence_type="external_quote", quote=quote, source_url=url)])
    monkeypatch.setattr(qualify, "obvious_non_business_reason", lambda *a: None)
    monkeypatch.setattr(qualify, "inspect_all", lambda *a: [site])
    monkeypatch.setattr(qualify, "inspect_external_all", lambda *a: None)
    monkeypatch.setattr(qualify, "run_parallel_gemini", lambda *a: {0:{"analysis":analysis,"error":None}})
    audits = []
    def audit(evidence, criteria, context):
        audits.append(evidence)
        return evidence, audit_supports_both, "Only one clause supported" if not audit_supports_both else ""
    monkeypatch.setattr(qualify, "audit_evidence", audit)
    state = {"companies":[company], "number_of_leads":1, "qualified_leads":[], "rejected_leads":[],
        "research_stats":{}, "service":"call handling", "industry":"plumbers", "location":"United Kingdom",
        "plan":{"need_indicators":[criterion], "user_criteria":[criterion],
                "external_criteria":[criterion], "evidence_observable":True, "adaptive_research":False}}
    result = qualify.research_companies(state)
    assert len(audits) == 1 and len(audits[0]) == 1
    assert len(result["qualified_leads"]) == int(audit_supports_both)


def test_distinct_complaints_are_not_collapsed_to_one_category():
    from qualify import dedupe_evidence, external_criteria_met
    evidence = [{"source":"external_quote", "indicator":"Unanswered calls", "evidence":"Calls went unanswered."},
                {"source":"external_quote", "indicator":"Missed visits", "evidence":"They missed two appointments without notification."},
                {"source":"external_quote", "indicator":"Required compound complaint", "evidence":"Calls went unanswered."}]
    result = dedupe_evidence(evidence)
    assert len(result) == 2
    assert external_criteria_met(result, ["Required compound complaint"])


def test_html_entity_in_business_name_is_decoded():
    assert search_name({"domain":"leesplumbing.co.uk", "business_name":"Lees Plumbing &amp; Heating"}) == "Lees Plumbing & Heating"
