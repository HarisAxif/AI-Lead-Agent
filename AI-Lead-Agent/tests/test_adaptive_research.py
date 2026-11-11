import copy
import json
import time
from types import SimpleNamespace

import pytest

from adaptive_research import run_candidate_loop, ResearchAction, _inspect_owned
from adaptive_discovery import revise_discovery, DiscoveryRevision, DiscoveryQuery


def fixtures():
    company = {"title":"Example Plumbing", "domain":"exampleplumbing.co.uk", "url":"https://exampleplumbing.co.uk/","snippet":"UK plumbing services"}
    site = {"reachable":True,"text":"Example Plumbing in Leeds", "business_name":"Example Plumbing",
            "pages_read":[company["url"]], "external_sources":[], "external_errors":[]}
    state = {"industry":"plumbing companies","location":"United Kingdom","service":"call handling",
             "number_of_leads":2,"plan":{"user_criteria":["Unanswered calls AND missed appointments without updates"],
                                          "adaptive_research":True}}
    return company, site, state


def evidence_result(label, quote):
    return {"sources":[{"url":f"https://reviews.example.org/{label}","text":quote}],
            "errors":[],"fetch_attempts":1,"attempted_urls":[f"https://reviews.example.org/{label}"]}


def test_missing_clause_drives_second_search_and_verification():
    company, site, state = fixtures()
    observed = []
    def evaluate(analysis):
        missing = (["Unanswered calls", "Missed-appointment updates"] if analysis == 0 else
                   ["Missed-appointment updates"] if analysis == 1 else [])
        return {"qualified":not missing,"missing":missing}
    def decide(c, s, context, assessment, trace, remaining):
        observed.append(list(assessment["missing"]))
        return ResearchAction(action="search", reason="Resolve missing clause", queries=[
            '"Example Plumbing" unanswered calls' if not trace else '"Example Plumbing" missed appointment no notice'])
    calls = []
    def collect(c, location, plan):
        calls.append(plan["external_search_queries"])
        assert plan["_skip_provider"]
        return evidence_result(str(len(calls)), "Calls unanswered" if len(calls)==1 else "Missed appointment without notice")
    analysis, assessment, ledger = run_candidate_loop(company, site, state, 0, evaluate,
        lambda c,s,st:len(s["external_sources"]), decide=decide, collect=collect)
    assert analysis == 2 and assessment["qualified"]
    assert observed == [["Unanswered calls","Missed-appointment updates"],["Missed-appointment updates"]]
    assert ledger["stop_reason"] == "qualified" and len(ledger["steps"]) == 2
    assert calls[0] != calls[1]


def test_access_failure_is_observed_before_changing_source():
    company, site, state = fixtures()
    choices = []
    def decide(c, s, st, a, trace, budget):
        if not trace:
            return ResearchAction(action="search", reason="Try reviews", queries=['"Example Plumbing" reviews'])
        assert s["external_errors"][0]["status"] == 403
        choices.append("saw refusal")
        return ResearchAction(action="change_source", reason="Source refused; try a forum",
                              queries=['site:reddit.com "Example Plumbing" missed appointment'])
    def collect(c, location, plan):
        if "reddit" in plan["external_search_queries"][0]:
            return evidence_result("forum", "Complete complaint")
        return {"sources":[],"errors":[{"url":"https://blocked.example/review","status":403,
            "kind":"access_denied","error":"Refused"}],"attempted_urls":["https://blocked.example/review"],"fetch_attempts":1}
    _, assessment, ledger = run_candidate_loop(company, site, state, False,
        lambda a:{"qualified":a,"missing":[] if a else ["Public complaints"]},
        lambda *a:True, decide=decide, collect=collect)
    assert choices == ["saw refusal"] and assessment["qualified"]
    assert ledger["steps"][1]["action"] == "change_source"


def test_identity_investigation_uses_observed_owned_page_and_verified_alias(monkeypatch):
    import website
    import httpx
    company, site, state = fixtures()
    url = company["url"] + "about"
    site["discovered_links"] = [url]
    response = httpx.Response(200, request=httpx.Request("GET", url), text="""<h1>Example Plumbing</h1>
    <script type="application/ld+json">{"@type":"Plumber","name":"Example Plumbing",
      "alternateName":"Example Heating Ltd","url":"https://exampleplumbing.co.uk/",
      "address":{"addressLocality":"Leeds"}}</script>""")
    monkeypatch.setattr(website, "fetch_page", lambda target: response)
    def decide(*args):
        return ResearchAction(action="resolve_identity", reason="Check legal alias on owned page", urls=[url])
    _, assessment, ledger = run_candidate_loop(company, site, state, False,
        lambda a:{"qualified":a,"missing":[] if a else ["Verify alias"]},
        lambda c,s,st: "Example Heating Ltd" in s.get("verified_aliases", []), decide=decide)
    assert assessment["qualified"] and "Leeds" in site["business_address"]
    assert ledger["steps"][0]["executed_urls"] == [url]


def test_invented_and_private_urls_are_never_fetched():
    company, site, state = fixtures()
    def decide(*args):
        return ResearchAction(action="fetch", reason="Bad URLs", urls=["http://localhost/", "https://unobserved.example/"])
    def collect(*args): pytest.fail("No unobserved/private URL may be fetched")
    _, assessment, ledger = run_candidate_loop(company, site, state, False,
        lambda a:{"qualified":False,"missing":["Need evidence"]}, lambda *a:pytest.fail("No new data"),
        decide=decide, collect=collect)
    assert not assessment["qualified"] and ledger["stop_reason"] == "no_new_information"
    assert all(not step["executed_urls"] for step in ledger["steps"])


def test_owned_investigation_preserves_identity_and_discovers_next_page(monkeypatch):
    import httpx
    import website
    company, site, _ = fixtures()
    site["verified_aliases"] = ["Example Heating Ltd"]
    url = company["url"] + "about"
    contact = company["url"] + "contact"
    response = httpx.Response(200, request=httpx.Request("GET", url),
        text='<title>About Us</title><a href="/contact">Contact our Leeds office</a>')
    monkeypatch.setattr(website, "fetch_page", lambda target: response)
    assert _inspect_owned([url], company, site, time.time()+5) == 1
    assert site["business_name"] == "Example Plumbing"
    assert site["verified_aliases"] == ["Example Heating Ltd"]
    assert contact in site["discovered_links"]


def test_repeated_queries_do_not_repeat_network_actions():
    company, site, state = fixtures()
    calls = []
    def collect(*args):
        calls.extend(args[2]["external_search_queries"])
        return {"sources":[],"errors":[],"fetch_attempts":0}
    _, _, ledger = run_candidate_loop(company, site, state, False,
        lambda a:{"qualified":False,"missing":["Need evidence"]}, lambda *a:False,
        decide=lambda *a:ResearchAction(action="search",reason="Try",queries=["same query"]), collect=collect)
    assert calls.count("same query") == 1
    assert len(calls) == len(set(calls)) and ledger["stop_reason"] == "no_new_information"


def test_page_and_search_budgets_are_enforced(monkeypatch):
    monkeypatch.setenv("RESEARCH_MAX_PAGES_PER_COMPANY", "1")
    monkeypatch.setenv("RESEARCH_MAX_SEARCHES_PER_COMPANY", "1")
    company, site, state = fixtures()
    calls = []
    def collect(c,l,plan):
        calls.append(plan)
        assert len(plan["external_search_queries"]) == 1
        assert plan["_max_external_attempts"] == 1
        return evidence_result("first", "Not enough evidence yet")
    _, _, ledger = run_candidate_loop(company, site, state, False,
        lambda a:{"qualified":False,"missing":["Second clause"]}, lambda *a:False,
        decide=lambda *a:ResearchAction(action="search",reason="Try",queries=["query one","query two"]),collect=collect)
    assert len(calls) == 1 and ledger["stop_reason"] == "budget_exhausted"


def test_deadline_expiring_during_planning_prevents_action():
    company, site, state = fixtures()
    clock = [time.time()]
    state["deadline"] = clock[0]+5
    def decide(*args):
        clock[0] += 10
        return ResearchAction(action="search",reason="Late",queries=["query"])
    _, _, ledger = run_candidate_loop(company, site, state, False,
        lambda a:{"qualified":False,"missing":["Evidence"]},lambda *a:False,
        decide=decide,collect=lambda *a:pytest.fail("No action after deadline"),now=lambda:clock[0])
    assert ledger["stop_reason"] == "budget_exhausted"


def test_planner_cannot_approve_a_candidate_by_finishing():
    company, site, state = fixtures()
    _, assessment, ledger = run_candidate_loop(company, site, state, False,
        lambda a:{"qualified":False,"missing":["Unproven complaint"]},lambda *a:True,
        decide=lambda *a:ResearchAction(action="finish",reason="I think it qualifies"))
    assert not assessment["qualified"] and ledger["stop_reason"] == "insufficient_evidence"


def test_retry_that_cannot_fit_budget_is_reported_as_time_exhaustion():
    from gemini_llm import GeminiBudgetError
    company, site, state = fixtures()
    state["deadline"] = time.time()+5
    def decide(*args):
        raise GeminiBudgetError("Research time budget exhausted before retry")
    _, _, ledger = run_candidate_loop(company, site, state, False,
        lambda a:{"qualified":False,"missing":["Public complaint"]}, lambda *a:False, decide=decide)
    assert ledger["stop_reason"] == "budget_exhausted"
    assert ledger["exhausted_budget"] == "time"
    revision = revise_discovery({**state,"research_feedback":[{"missing_facts":["Public complaint"]}]}, decide=decide)
    assert revision["discovery_stop_reason"] == "time_budget_exhausted"


def test_qualified_candidate_uses_no_more_tools():
    company, site, state = fixtures()
    _, assessment, ledger = run_candidate_loop(company, site, state, True,
        lambda a:{"qualified":True,"missing":[]},lambda *a:pytest.fail("Already qualified"),
        decide=lambda *a:pytest.fail("Already qualified"))
    assert assessment["qualified"] and ledger["stop_reason"] == "qualified"


def feedback_state():
    _,_,state = fixtures()
    state.update(round=1, qualified_leads=[],seen_queries=["old query"],failed_queries=[],research_feedback=[{
        "website":"https://exampleplumbing.co.uk/","missing":["Missed-appointment notification"],
        "external_research":{"errors":[{"kind":"access_denied","status":403,"url":"https://uk.trustpilot.com/review/exampleplumbing.co.uk"}]}}])
    return state


def test_next_discovery_uses_actual_missing_clause_and_access_errors():
    state = feedback_state()
    original = copy.deepcopy(state["plan"])
    prompts = []
    def decide(prompt):
        prompts.append(prompt)
        return DiscoveryRevision(reason="Need another review source", queries=[DiscoveryQuery(
            query="site:reviews.io UK plumbers missed appointment no notice",mode="reviews",reason="Different source")])
    update = revise_discovery(state, decide)
    assert "Missed-appointment notification" in prompts[0] and "403" in prompts[0]
    assert update["adaptive_review_queries"] == update["adaptive_queries"]
    assert state["plan"] == original


def test_discovery_rejects_repeated_queries_and_preserves_search_pool():
    state = feedback_state()
    result = revise_discovery(state, lambda p:DiscoveryRevision(reason="Repeat", queries=[
        DiscoveryQuery(query="old query",mode="business",reason="Repeated")]))
    assert not result.get("query_pool_exhausted") and result["adaptive_queries"] == []


def test_discovery_replan_budget_stops_without_model_call(monkeypatch):
    state = feedback_state()
    state["discovery_trace"] = [{"round":1}]
    monkeypatch.setenv("DISCOVERY_MAX_REPLANS", "1")
    result = revise_discovery(state, lambda p:pytest.fail("No model call after budget"))
    assert not result.get("query_pool_exhausted") and result["adaptive_queries"] == []


def test_discovery_can_stop_for_source_access_without_claiming_no_leads_exist():
    state = feedback_state()
    result = revise_discovery(state, lambda p:DiscoveryRevision(stop=True,
        reason="Required review sources are inaccessible; cannot verify complaints"))
    assert not result.get("query_pool_exhausted")
    assert result["discovery_trace"][-1]["stop"]
    assert "inaccessible" in result["discovery_stop_reason"]


def test_full_qualification_loop_combines_two_fetched_passages(monkeypatch):
    import httpx
    import qualify
    import evidence_gate
    import external_research as external
    from schemas import LeadAnalysis, IndicatorFinding
    from evidence_gate import EvidenceAudit, CriterionCheck
    company, site, state = fixtures()
    site.update(footprint="", emails=[], phones=[], pages_read=[company["url"]])
    compound = "Reviews mention unanswered calls and missed appointments without notification"
    phone = "Example Plumbing never answered my phone calls and nobody called me back."
    appointment = "Example Plumbing missed the appointment without any notification or explanation."
    state.update(companies=[company],qualified_leads=[],rejected_leads=[],research_stats={})
    state["plan"].update(ideal_buyer="UK plumbers",disqualifiers=[],need_indicators=["Unanswered calls","Missed appointments without notification",compound],
        user_criteria=[compound],external_criteria=[compound],evidence_observable=True)
    def analysis():
        findings = []
        for source in site["external_sources"]:
            is_phone = "never answered" in source["text"]
            findings.append(IndicatorFinding(indicator="Unanswered calls" if is_phone else "Missed appointments without notification",
                present=True,evidence_type="external_quote",quote=phone if is_phone else appointment,source_url=source["url"]))
        return LeadAnalysis(company_name="Example Plumbing",is_single_business_site=True,in_target_location=True,
            matches_ideal_buyer=True,hit_disqualifier=False,reason="Evidence",opportunity="Discuss communication",findings=findings)
    monkeypatch.setattr(qualify,"obvious_non_business_reason",lambda *a:None)
    monkeypatch.setattr(qualify,"inspect_all",lambda *a:[site])
    monkeypatch.setattr(qualify,"inspect_external_all",lambda *a:None)
    monkeypatch.setattr(qualify,"run_parallel_gemini",lambda *a:{0:{"analysis":analysis(),"error":None}})
    monkeypatch.setattr(qualify,"structured_call",lambda *a,**kw:analysis())
    def decide(c,s,st,assessment,trace,remaining):
        if not trace:
            return ResearchAction(action="search",reason="Need unanswered-call evidence",queries=["Example Plumbing unanswered calls"])
        assert "Missed appointment" in " ".join(assessment["missing"])
        return ResearchAction(action="change_source",reason="Need remaining notification clause",
            queries=["Example Plumbing missed appointments without notification"])
    monkeypatch.setattr("adaptive_research.choose_action",decide)
    audits = []
    def audit(prompt,schema):
        complete = len(site["external_sources"]) == 2
        audits.append(complete)
        if "OBSERVATIONS:" in prompt:
            return EvidenceAudit(accepted_evidence_indices=[0,1], criteria_checks=[
                CriterionCheck(criterion_index=0,supported=True,evidence_indices=[0],reason="Calls unanswered"),
                CriterionCheck(criterion_index=1,supported=True,evidence_indices=[1],reason="Missed appointment without notice")])
        return EvidenceAudit(accepted_evidence_indices=list(range(len(site["external_sources"]))),
            criteria_checks=[CriterionCheck(criterion_index=0,supported=complete,
                evidence_indices=[0,1] if complete else [],reason="Both clauses established" if complete else "Missing appointment evidence",
                missing_facts=[] if complete else ["Missed appointment without notification"])])
    monkeypatch.setattr(evidence_gate,"structured_call",audit)
    class Search:
        def __enter__(self): return self
        def __exit__(self,*a): pass
        def text(self,query,**kw):
            return [{"href":"https://reviews.example.org/phone" if "unanswered" in query else "https://forum.example.org/appointment",
                     "title":"Example Plumbing customer complaint","body":"Snippet is not proof"}]
    monkeypatch.setattr(external,"DDGS",Search)
    original = httpx.Client
    def respond(request):
        quote = phone if request.url.path == "/phone" else appointment
        return httpx.Response(200,headers={"content-type":"text/html"},
            text="<h1>Example Plumbing exampleplumbing.co.uk</h1><p>"+quote+" Additional customer experience details.</p>")
    monkeypatch.setattr(external.httpx,"Client",lambda **kw:original(transport=httpx.MockTransport(respond),**kw))
    result = qualify.research_companies(state)
    assert len(result["qualified_leads"]) == 1 and audits == [False,True,True], result
    record = result["qualified_leads"][0]
    assert len(record["evidence"]) == 2
    assert record["research_loop"]["stop_reason"] == "qualified"
    assert len(record["research_loop"]["steps"]) == 2
    assert result["research_feedback"][0]["qualified"] is True


def test_discovery_node_executes_revised_queries(monkeypatch):
    import discovery
    import adaptive_discovery
    state = feedback_state()
    state.update(seen_domains=[], geo_terms=[],research_stats={})
    monkeypatch.setattr(adaptive_discovery,"structured_call",lambda *a:DiscoveryRevision(
        reason="Missed notification evidence requires another review source",
        queries=[DiscoveryQuery(query="site:reviews.io UK plumbers missed visits no notice",mode="reviews",reason="Alternative source")]))
    executed = []
    def query(text, context, *args):
        executed.append((text,context["adaptive_review_queries"]))
        return {"candidates":[{"domain":"newplumber.co.uk","url":"https://newplumber.co.uk/"}],
            "raw_count":1,"duplicate_count":0,"blocked_count":0,"classifier_rejected_count":0,"classifier_failed_count":0}
    monkeypatch.setattr(discovery,"run_one_query",query)
    result = discovery.search_companies(state)
    assert executed[0][0] in executed[0][1]
    assert len(executed) > 1  # unused searches fill the remaining round allowance
    assert result["companies"][0]["domain"] == "newplumber.co.uk"
    assert result["discovery_trace"][0]["reason"].startswith("Missed notification")


def test_named_business_in_forum_can_be_resolved_to_official_site(monkeypatch):
    import discovery
    source = "https://www.reddit.com/r/UK/comments/abc/"
    hits = [{"href":source,"title":"Example Plumbing missed my appointment","body":"Example Plumbing in Leeds never answered calls."}]
    monkeypatch.setattr(discovery,"structured_call",lambda *a:discovery.BusinessHints(businesses=[
        discovery.NamedBusinessHint(name="Example Plumbing",source_url=source)]))
    class Search:
        def __enter__(self): return self
        def __exit__(self,*a): pass
        def text(self,query,**kw):
            assert '"Example Plumbing"' in query
            return [{"href":"https://exampleplumbing.co.uk/leeds","title":"Example Plumbing Leeds","body":"UK plumbing services"}]
    monkeypatch.setattr(discovery,"DDGS",Search)
    _,_,state = fixtures()
    result = discovery.resolve_complaint_hints(hits,state,set(),2)
    assert result[0]["seed_external_urls"] == [source]
    assert result[0]["url"].endswith("/leeds")


def test_invented_business_hint_is_not_resolved(monkeypatch):
    import discovery
    monkeypatch.setattr(discovery,"structured_call",lambda *a:discovery.BusinessHints(businesses=[
        discovery.NamedBusinessHint(name="Invented Company",source_url="https://reviews.example.org/real")]))
    monkeypatch.setattr(discovery,"DDGS",lambda *a:pytest.fail("Invented business cannot trigger searches"))
    _,_,state = fixtures()
    assert discovery.resolve_complaint_hints([{"href":"https://reviews.example.org/real","title":"Different Company"}],state,set(),2) == []


def test_model_call_inherits_remaining_research_deadline(monkeypatch):
    import gemini_llm as backend
    from pydantic import BaseModel
    from research_budget import bounded
    class Answer(BaseModel):
        ok: bool
    captured = []
    def generate(**kwargs):
        captured.append(kwargs)
        return SimpleNamespace(parsed=Answer(ok=True))
    monkeypatch.setattr(backend,"ensure_ready",lambda:SimpleNamespace(models=SimpleNamespace(generate_content=generate)))
    monkeypatch.setattr(backend.limiter,"acquire",lambda:None)
    with bounded(time.time()+5):
        assert backend.structured_call("test",Answer).ok
    timeout = captured[0]["config"]["http_options"]["timeout"]
    assert 0 < timeout <= 5000
    with bounded(time.time()-1):
        with pytest.raises(backend.GeminiModelError,match="budget"):
            backend.structured_call("test",Answer)
    assert len(captured) == 1


def test_compound_criterion_does_not_accept_website_proxy(monkeypatch):
    import evidence_gate
    from evidence_gate import EvidenceAudit, CriterionCheck
    evidence = [{"indicator":"calls","source":"external_quote","evidence":"Calls unanswered"},
                {"indicator":"appointments","source":"website_text","evidence":"We offer appointment updates"}]
    monkeypatch.setattr(evidence_gate,"structured_call",lambda *a:EvidenceAudit(
        accepted_evidence_indices=[0,1], criteria_checks=[CriterionCheck(
            criterion_index=0,supported=True,evidence_indices=[0,1],reason="Incorrect source types")]))
    site = {"text":"We offer appointment updates", "_external_criteria_required":["Public complaints about calls AND visits"]}
    evidence_gate.audit_evidence(evidence,["Public complaints about calls AND visits"],site)
    assert not site["_external_criteria_support"]["public complaints about calls and visits"]


@pytest.mark.parametrize("action", ["change_source", "search"])
def test_change_source_excludes_all_blocked_provider_hostnames(action):
    company, site, state = fixtures()
    site["external_errors"] = [{"url":"https://uk.trustpilot.com/review/exampleplumbing.co.uk",
                               "kind":"access_denied","status":403}]
    plans = []
    def collect(c,l,plan):
        plans.append(plan)
        assert "trustpilot.com" in plan["_excluded_host_families"]
        query = plan["external_search_queries"][0]
        assert "site:uk.trustpilot.com" not in query and "-site:trustpilot.com" in query
        return evidence_result("other-source", "Complaint verified elsewhere")
    _, assessment, ledger = run_candidate_loop(company, site, state, False,
        lambda a:{"qualified":a,"missing":[] if a else ["Complaint"]},lambda *a:True,
        decide=lambda *a:ResearchAction(action=action,reason="Try another provider",
            queries=['site:uk.trustpilot.com "Example Plumbing" complaints']),collect=collect)
    assert assessment["qualified"] and len(plans) == 1
