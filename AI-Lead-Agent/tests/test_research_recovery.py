"""Regressions for worker budgets, source identity and research recovery."""
import time

import httpx
import pytest
from bs4 import BeautifulSoup

import adaptive_research as loop
import external_research as external
import research_budget as budget
from tests.test_adaptive_research import fixtures, evidence_result


def test_html_workers_inherit_run_deadline(monkeypatch):
    import website
    company, _, _ = fixtures()
    remaining = []
    def inspect(c, p):
        remaining.append(budget.remaining())
        return {"reachable":False}
    monkeypatch.setattr(website, "inspect_html", inspect)
    monkeypatch.setattr(website, "_render_partition", lambda *a: None)
    website.inspect_all([company, company], {"_deadline":time.time()+5})
    assert len(remaining) == 2 and all(0 < left <= 5 for left in remaining)


def test_expired_html_budget_starts_no_http_requests(monkeypatch):
    import website
    company, _, _ = fixtures()
    monkeypatch.setattr(website.httpx, "get", lambda *a, **k:pytest.fail("Expired worker made HTTP request"))
    sites = website.inspect_all([company], {"_deadline":time.time()-1})
    assert not sites[0]["reachable"]


def test_external_workers_inherit_run_deadline(monkeypatch):
    company, site, _ = fixtures()
    remaining = []
    def collect(*args):
        remaining.append(budget.remaining())
        return {"sources":[],"errors":[]}
    monkeypatch.setattr(external, "collect_external_sources", collect)
    external.inspect_external_all([company], [site], "UK", {
        "external_search_queries":["reviews"],"_deadline":time.time()+5})
    assert 0 < remaining[0] <= 5


@pytest.mark.parametrize("target", ["http://localhost/", "https://uk.trustpilot.com/review/example.org",
                                    "https://exampleplumbing.co.uk/reviews"])
def test_redirects_cannot_bypass_source_exclusions(target):
    company, _, _ = fixtures()
    requests = []
    def handle(request):
        requests.append(str(request.url))
        return httpx.Response(302, headers={"location":target})
    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(ValueError):
            external.fetch_external_page(client, "https://reviews.example.org/old", company,
                {"_excluded_host_families":["trustpilot.com"]})
    assert requests == ["https://reviews.example.org/old"]


def test_redirect_hops_use_remaining_deadline(monkeypatch):
    company, _, _ = fixtures()
    clock = [100.0]
    monkeypatch.setattr(budget.time, "time", lambda:clock[0])
    timeouts = []
    def handle(request):
        timeouts.append(request.extensions["timeout"]["read"])
        if len(timeouts) == 1:
            clock[0] += 4
            return httpx.Response(302, headers={"location":"/new"})
        return httpx.Response(200)
    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        response = external.fetch_external_page(client, "https://reviews.example.org/old", company,{"_deadline":105})
    assert str(response.url).endswith("/new") and timeouts == [5, 1]


def test_planner_failure_recovers_without_approving_a_lead():
    company, site, state = fixtures()
    calls = []
    def decide(*args):
        raise RuntimeError("503 planner unavailable")
    def collect(c, location, plan):
        calls.extend(plan["external_search_queries"])
        return evidence_result("recovery", "A fetched customer report")
    _, assessment, ledger = loop.run_candidate_loop(company, site, state, False,
        lambda a:{"qualified":a,"missing":[] if a else ["Public reports of unanswered calls"]},
        lambda *a:True, decide=decide, collect=collect)
    assert calls and assessment["qualified"]
    assert ledger["steps"][0]["recovery"]
    assert "503" in ledger["steps"][0]["planner_error"]


def test_short_planning_timeout_leaves_time_for_recovery():
    from gemini_llm import GeminiBudgetError
    company, site, state = fixtures()
    def decide(*args):
        raise GeminiBudgetError("Planning request timed out")
    _, assessment, ledger = loop.run_candidate_loop(company, site, state, False,
        lambda a:{"qualified":a,"missing":[] if a else ["Public complaints"]}, lambda *a:True,
        decide=decide, collect=lambda *a:evidence_result("recovery", "Customer report"))
    assert assessment["qualified"] and ledger["stop_reason"] == "qualified"


def test_identity_search_inspects_owned_result_in_same_step():
    company, site, state = fixtures()
    contact = company["url"] + "contact"
    def collect(*args):
        return {"sources":[],"errors":[],"fetch_attempts":0,"search_results":[{"url":contact}]}
    inspected = []
    def inspect(urls, c, s, deadline):
        inspected.extend(urls)
        s["business_address"] = "Leeds, UK"
        return len(urls)
    _, assessment, ledger = loop.run_candidate_loop(company, site, state, False,
        lambda a:{"qualified":a,"missing":[] if a else ["Branch identity and location"]},
        lambda c,s,st:bool(s.get("business_address")),
        decide=lambda *a:loop.ResearchAction(action="resolve_identity",reason="Verify branch",queries=["Example contact"]),
        collect=collect,inspect=inspect)
    assert inspected == [contact] and assessment["qualified"]
    assert len(ledger["steps"]) == 1


def test_empty_round_reports_query_outcomes_to_next_discovery():
    import qualify
    from adaptive_discovery import revise_discovery, DiscoveryRevision
    _, _, state = fixtures()
    state.update(companies=[],round=1,qualified_leads=[],rejected_leads=[],research_stats={},
        discovery_outcomes=[{"query":"original search","raw_results":0,"failed":True}])
    feedback = qualify.research_companies(state)
    assert feedback["research_feedback"][0]["outcome"] == "no_candidates"
    prompts = []
    def decide(prompt):
        prompts.append(prompt)
        return DiscoveryRevision(stop=True,reason="Sources unavailable")
    revise_discovery({**state,**feedback}, decide)
    assert "original search" in prompts[0] and "raw_results" in prompts[0]


def test_discovery_planner_failure_preserves_unused_query_pool():
    from adaptive_discovery import revise_discovery
    _, _, state = fixtures()
    state["research_feedback"] = [{"missing":["Public complaints"]}]
    def decide(prompt):
        raise RuntimeError("503")
    update = revise_discovery(state, decide)
    assert not update.get("query_pool_exhausted")
    assert update["adaptive_queries"] == []
    assert update["discovery_trace"][0]["outcome"] == "discovery_planner_unavailable"


def test_structured_review_is_not_lost_on_long_pages():
    quote = "Example School never told parents about the cancelled lessons."
    soup = BeautifulSoup('<p>Example School ' + 'boilerplate '*4000 + '</p>'
        '<script type="application/ld+json">{"reviewBody":"'+quote+'"}</script>', "html.parser")
    text = external.readable_external_text(soup)
    assert quote in text and len(text) <= 24000


def test_business_identity_requires_complete_name_tokens():
    company = {"title":"Example School","domain":"example.edu"}
    assert not external.same_business_source("NotExample School complaints", company)
    assert external.same_business_source("At Example School, nobody replied", company)


def test_shortfall_from_api_failure_is_not_cached_for_a_week(tmp_path, monkeypatch):
    import result_cache
    monkeypatch.setattr(result_cache,"CACHE_DIR",tmp_path)
    monkeypatch.setattr(result_cache,"public_lead",lambda lead:lead)
    state = {"qualified_leads":[{"company":"Verified business"}],"research_stats":{"api_failures":1}}
    assert not result_cache.save(["partial"],5,state,False)
    assert not list(tmp_path.glob("*.json"))


def test_quota_wait_reports_research_budget_instead_of_provider_failure(monkeypatch):
    from gemini_limits import RequestLimiter
    monkeypatch.setenv("GEMINI_MAX_WAIT_SECONDS", "90")
    limiter = RequestLimiter(1)
    limiter.defer(60)
    with budget.bounded(time.time()+2), pytest.raises(budget.ResearchBudgetExceeded):
        limiter.acquire()


def test_initial_research_switches_away_from_refused_review_provider(monkeypatch):
    company, _, _ = fixtures()
    urls = ["https://uk.trustpilot.com/review/exampleplumbing.co.uk",
            "https://www.trustpilot.com/review/exampleplumbing.co.uk?page=2",
            "https://reviews.example.org/example-plumbing"]
    class Search:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def text(self, *args, **kwargs): return [{"href":url} for url in urls]
    requests = []
    def handle(request):
        requests.append(str(request.url))
        if external.source_family(str(request.url)) == "trustpilot.com":
            return httpx.Response(403)
        return httpx.Response(200,headers={"content-type":"text/html"}, text=
            "Example Plumbing at exampleplumbing.co.uk missed my booked appointment and never called me. "
            "I phoned several times but nobody answered. This is my experience as their customer.")
    original = httpx.Client
    monkeypatch.setattr(external,"DDGS",Search)
    monkeypatch.setattr(external.httpx,"Client",lambda **kw:original(transport=httpx.MockTransport(handle),**kw))
    result = external.collect_external_sources(company,"UK",{"external_search_queries":["{company} reviews"],"_skip_provider":True})
    assert requests == [urls[0],urls[2]]
    assert len(result["sources"]) == 1


def test_source_exclusion_handles_profile_path_and_preserves_negative_site_filter():
    query = external.exclude_sources('site:uk.trustpilot.com/review/ "Example" -site:yelp.com', {"trustpilot.com"})
    assert "site:uk.trustpilot.com" not in query
    assert "-site:trustpilot.com" in query and "-site:yelp.com" in query
    company, _, _ = fixtures()
    assert "-site:yelp.com" in external.external_queries(company,"UK",{"external_search_queries":[query]})[0]


def test_branch_identity_takes_precedence_over_network_metadata():
    from business_identity import extract_identity
    soup = BeautifulSoup('''<script type="application/ld+json">{"@graph":[
        {"@type":"Organization","name":"Example Health Network","url":"https://example.org/"},
        {"@type":"MedicalClinic","name":"Example Erie Practice","url":"https://example.org/clinics/erie",
         "address":{"addressLocality":"Erie","addressRegion":"Pennsylvania"}}
    ]}</script>''',"html.parser")
    identity = extract_identity(soup,"example.org","https://example.org/clinics/erie")
    assert identity["business_name"] == "Example Erie Practice"
    assert identity["business_address"] == "Erie, Pennsylvania"


def test_network_contact_page_cannot_replace_a_verified_branch(monkeypatch):
    import website
    company, site, _ = fixtures()
    site.update(identity_url=company["url"]+"clinics/leeds", business_name="Example Leeds Practice",
        business_address="Leeds, UK", verified_aliases=["Example Leeds Practice"])
    contact = company["url"]+"contact"
    response = httpx.Response(200,request=httpx.Request("GET",contact),text='''
        <script type="application/ld+json">{"@type":"Organization","name":"Example National Network",
        "url":"https://exampleplumbing.co.uk/"}</script>''')
    monkeypatch.setattr(website,"fetch_page",lambda url:response)
    loop._inspect_owned([contact],company,site,time.time()+5)
    assert site["business_name"] == "Example Leeds Practice" and site["business_address"] == "Leeds, UK"
    assert "Example National Network" not in site["verified_aliases"]


def test_malformed_and_imitation_review_links_do_not_corrupt_identity():
    soup = BeautifulSoup('''<a href="https://[broken">Reviews</a>
        <a href="https://nottrustpilot.com/review/wrong.org">Reviews</a>
        <a href="//uk.trustpilot.com/review/example.org">Trustpilot reviews</a>''',"html.parser")
    assert external.owned_review_profiles(soup,"https://example.org/") == ["https://uk.trustpilot.com/review/example.org"]


@pytest.mark.parametrize("url,expected", [("https://example.org/", "continue"),
    ("http://localhost/", "abort"), ("http://127.0.0.1/", "abort"), ("http://192.168.1.2/", "abort")])
def test_browser_requests_cannot_follow_private_redirects(url, expected):
    from types import SimpleNamespace
    from website import route_public_requests
    calls = []
    route = SimpleNamespace(request=SimpleNamespace(url=url),abort=lambda:calls.append("abort"),
                            continue_=lambda:calls.append("continue"))
    route_public_requests(route)
    assert calls == [expected]


def test_planner_anchors_complaint_discovery_to_requested_location(monkeypatch):
    import planner
    from schemas import EvidencePlan, SearchRequest
    monkeypatch.setattr(planner,"structured_call",lambda *a:EvidencePlan(ideal_buyer="School",
        complaint_search_queries=["schools missed announcements reviews"]))
    plan = planner._generate_research_plan(SearchRequest(industry="schools",location="Lahore, Pakistan",
        service="Parent communications",number_of_leads=1,criteria=[]))
    assert all("Lahore, Pakistan" in query for query in plan.complaint_search_queries)


def test_model_failure_stays_visible_in_candidate_audit_and_discovery_feedback(monkeypatch):
    import qualify
    company, site, state = fixtures()
    site.update(footprint="",emails=[],phones=[])
    state.update(companies=[company],qualified_leads=[],rejected_leads=[],research_stats={})
    state["plan"].update(need_indicators=[],disqualifiers=[],evidence_observable=True)
    monkeypatch.setattr(qualify,"inspect_all",lambda *a:[site])
    monkeypatch.setattr(qualify,"inspect_external_all",lambda *a:None)
    monkeypatch.setattr(qualify,"run_parallel_gemini",lambda *a:{0:{"error":"503 model unavailable"}})
    result = qualify.research_companies(state)
    assert result["rejected_leads"][0]["rejection_type"] == "api_unavailable"
    assert result["research_feedback"][0]["research_loop"]["stop_reason"] == "model_unavailable"
    assert result["research_stats"]["api_failures"] == 1
