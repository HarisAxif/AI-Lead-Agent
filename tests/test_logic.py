"""
Offline tests: no Gemini key, network or browser needed.
Run:  python -m pytest -q
"""

import json
from types import SimpleNamespace

import pytest
from bs4 import BeautifulSoup

import clarify
import discovery
import qualify
import website
from exporter import export_run
from planner import plan_cache_key
from schemas import (
    CriterionAssessment,
    IndicatorFinding,
    IntentResult,
    ReplyAction,
    SearchRequest,
)


def make_request(**kw):
    base = dict(
        number_of_leads=5,
        location="Dubai",
        industry="accounting firms",
        service="AI customer support automation",
        criteria=[],
        country_code="AE",
    )
    base.update(kw)
    return SearchRequest(**base)


# ---------------------------------------------------------------- volume rule

def test_volume_clause_becomes_proxy_and_other_clause_is_kept(capsys):
    req = make_request(criteria=[
        "because they have high client inquiry volume but weak online support channels"
    ])
    req = clarify.apply_volume_rule(req)
    assert clarify.VOLUME_PROXY_CRITERION in req.criteria
    assert any("weak online support channels" in c for c in req.criteria)
    assert not any("inquiry volume" in c for c in req.criteria)
    assert "can't verify volume" in capsys.readouterr().out


@pytest.mark.parametrize("text", [
    "because they have thousands of daily support tickets and a huge call center",
    "need better SEO rankings",
    "they will lose customers next month",
])
def test_non_volume_criteria_are_untouched(text):
    req = make_request(criteria=[text])
    assert clarify.apply_volume_rule(req).criteria == [text]


# ------------------------------------------------------------ reply handling

def problem(claim="200-agent call center", repl=None):
    return CriterionAssessment(
        claim=claim,
        classification="implausible_for_industry",
        reason="x",
        suggested_replacement=repl,
    )


def run_resolve(monkeypatch, replies, replacements=None, classify=None):
    """Drive resolve_criteria with scripted input and stubbed Gemini calls."""
    ok = CriterionAssessment(
        claim="solo practitioner", classification="verifiable_public", reason=""
    )
    bad = problem(repl=replacements)
    validation = SimpleNamespace(assessments=[ok, bad])

    calls = {"n": 0}

    def fake_validate(request):
        calls["n"] += 1
        # second validation (after revised criteria) is clean
        return SimpleNamespace(assessments=[ok]) if calls["n"] > 1 else validation

    monkeypatch.setattr(clarify, "validate_criteria", fake_validate)
    monkeypatch.setattr(clarify, "classify_reply", classify or (
        lambda a, p: ReplyAction(action="drop_criteria")))
    it = iter(replies)
    monkeypatch.setattr("builtins.input", lambda *_: next(it))
    req = make_request(criteria=["200-agent call center"])
    return clarify.resolve_criteria(req)


def test_insist_original_drops_claim_and_continues(monkeypatch, capsys):
    out = run_resolve(
        monkeypatch, ["no, search exactly as I said"],
        classify=lambda a, p: ReplyAction(action="insist_original"))
    assert out is not None
    assert out.criteria == ["solo practitioner"]
    assert "can't use that claim as evidence" in capsys.readouterr().out


def test_cancel_returns_none(monkeypatch):
    assert run_resolve(monkeypatch, ["cancel"],
                       classify=lambda a, p: ReplyAction(action="cancel")) is None


def test_two_unrelated_replies_stop(monkeypatch):
    out = run_resolve(monkeypatch, ["blah", "blah"],
                      classify=lambda a, p: ReplyAction(action="unrelated"))
    assert out is None


def test_revised_criteria_replaces_and_revalidates(monkeypatch):
    out = run_resolve(
        monkeypatch, ["phone-only booking"],
        classify=lambda a, p: ReplyAction(
            action="revised_criteria", revised_text="phone-only booking"))
    assert out.criteria == ["phone-only booking"]


def test_accept_alternatives_uses_replacements(monkeypatch):
    out = run_resolve(
        monkeypatch, ["yes"], replacements="phone or form contact only",
        classify=lambda a, p: ReplyAction(action="accept_alternatives"))
    assert out.criteria == ["solo practitioner", "phone or form contact only"]


def test_quick_replies_need_no_llm(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("LLM must not be called for yes/no/cancel")
    monkeypatch.setattr(clarify, "_structured_call", boom)
    assert clarify.classify_reply("Yes", []).action == "accept_alternatives"
    assert clarify.classify_reply("no", []).action == "drop_criteria"
    assert clarify.classify_reply("cancel", []).action == "cancel"


# ------------------------------------------------------------ prepare_request

def stub_prepare(monkeypatch, intent, understood=None):
    monkeypatch.setattr(clarify, "classify_intent", lambda *a, **k: intent)
    monkeypatch.setattr(clarify, "understand_request",
                        lambda *a, **k: understood or make_request())
    monkeypatch.setattr(clarify, "resolve_criteria", lambda r: r)
    monkeypatch.setattr(clarify, "clarify_request", lambda r: r)
    monkeypatch.setattr(clarify, "validate_service_fit",
                        lambda r: SimpleNamespace(valid=True, reason=""))


def test_out_of_scope_is_rejected(monkeypatch, capsys):
    stub_prepare(monkeypatch, IntentResult(intent="out_of_scope"))
    assert clarify.prepare_request("write cold emails") is None
    assert "only find and qualify leads" in capsys.readouterr().out


def test_mixed_request_skips_extra_part_and_continues(monkeypatch, capsys):
    stub_prepare(monkeypatch, IntentResult(
        intent="mixed_request",
        lead_part="Find 5 dentists in Lahore",
        unsupported_part="list the tools and APIs you use"))
    assert clarify.prepare_request("...") is not None
    assert "I skipped" in capsys.readouterr().out


def test_mixed_without_lead_part_is_rejected(monkeypatch):
    stub_prepare(monkeypatch, IntentResult(intent="mixed_request"))
    assert clarify.prepare_request("...") is None


def test_cancel_inside_criteria_stops_pipeline(monkeypatch):
    stub_prepare(monkeypatch, IntentResult(intent="lead_request"))
    monkeypatch.setattr(clarify, "resolve_criteria", lambda r: None)
    assert clarify.prepare_request("...") is None


# ------------------------------------------------------------------ evidence

@pytest.mark.parametrize("indicator,fact,expected", [
    # N D Savla: zero forms must not support "has a contact form"
    ("Presence of a generic contact form as the primary method for support",
     "rendered_visible_forms: 0", False),
    ("Absence of a Contact Us form", "rendered_visible_forms: 0", True),
    ("Absence of a Contact Us form", "rendered_visible_forms: 1", False),
    ("Presence of a contact form", "rendered_visible_forms: 3", True),
    # Montgomery-Stryker: a clickable count proves nothing
    ("Absence of any chat or automated response feature",
     "rendered_clickable_count: 23", False),
    ("Absence of interactive support elements",
     "rendered_visible_forms: 0", False),
    ("Phone number listed on the contact page",
     "phones_found: 2", True),
])
def test_technical_fact_supports(indicator, fact, expected):
    assert qualify.technical_fact_supports(indicator, fact) is expected


def test_dedupe_counts_same_fact_once():
    items = [
        {"indicator": "Absence of any chat or automated response feature",
         "evidence": "no bubble", "source": "visual_review"},
        {"indicator": "Lack of any interactive chat or messaging bubble",
         "evidence": "no bubble", "source": "visual_review"},
        {"indicator": "Contact only by phone", "evidence": "Tel: 1",
         "source": "website_text"},
    ]
    out = qualify.dedupe_evidence(items)
    assert len(out) == 2
    assert {i["category"] for i in out} == {"chat", "phone"}


def test_dedupe_prefers_hard_evidence_over_visual():
    items = [
        {"indicator": "No chat widget", "evidence": "screenshot",
         "source": "visual_review"},
        {"indicator": "No chat widget on site", "evidence": "text",
         "source": "website_text"},
    ]
    out = qualify.dedupe_evidence(items)
    assert len(out) == 1 and out[0]["source"] == "website_text"


def finding(indicator):
    return IndicatorFinding(indicator=indicator, present=True,
                            evidence_type="text_quote", quote="FAQ")


def site(text="", footprint=""):
    return {"text": text, "footprint": footprint,
            "desktop_screenshot": None, "mobile_screenshot": None}


def test_no_faq_claim_contradicted_by_faq_on_page():
    f = finding("Absence of a self-service portal or FAQ-based support")
    assert qualify.evidence_contradicts_measured_facts(
        f, site(text="Frequently asked questions FAQ about VAT"))


def test_no_chat_claim_contradicted_by_chat_vendor_script():
    f = finding("Website lacks a live chat widget")
    s = site(footprint="third_party_script_hosts: ['embed.tawk.to']")
    assert qualify.evidence_contradicts_measured_facts(f, s)


def test_chat_claim_ok_when_no_vendor():
    f = finding("Website lacks a live chat widget")
    assert not qualify.evidence_contradicts_measured_facts(
        f, site(footprint="third_party_script_hosts: []"))


def test_verify_findings_rejects_form_count_as_support_for_form():
    s = site(text="contact us", footprint="rendered_visible_forms: 0")
    f = IndicatorFinding(
        indicator="Presence of a generic contact form as the primary method",
        present=True, evidence_type="technical_fact",
        footprint_fact="rendered_visible_forms: 0")
    assert qualify.verify_findings([f], s) == []


# ------------------------------------------------------------------ contacts

HTML = """
<html><body>
<a href="tel:+971 4 570 5345">Call</a>
<a href="tel:+971 4 5705345">Call again</a>
<a href="mailto:info@firm.ae">Mail</a>
<footer>
  <div>Designed by DominantSite <a href="mailto:hola@dominantsite.com">hola@dominantsite.com</a>
  <a href="tel:+91 9819 000 445">+91 9819 000 445</a></div>
</footer>
</body></html>
"""


def test_contacts_validated_and_deduped():
    soup = BeautifulSoup(HTML, "html.parser")
    emails, phones = website.extract_contacts(
        [soup],
        "Reach us +971 56 164 3075 or +91 9819 000 445 or info@firm.ae",
        "firm.ae", "AE")
    assert emails == ["info@firm.ae"]                      # developer email dropped
    assert phones.count("+971 4 570 5345") == 1            # duplicate collapsed
    assert "+971 56 164 3075" in phones
    assert not any(p.startswith("+91") for p in phones)    # wrong country dropped


def test_free_mail_allowed_but_foreign_domain_not():
    assert website.email_belongs_to_site("firm@gmail.com", "firm.ae")
    assert website.email_belongs_to_site("a@mail.firm.ae", "firm.ae")
    assert not website.email_belongs_to_site("hola@dominantsite.com", "firm.ae")


def test_us_numbers_normalised():
    key, display = website.normalize_phone("(307) 742-2479", "US")
    assert key == "+13077422479"
    assert website.normalize_phone("+971 4 570 5345", "US") is None


def test_unknown_country_keeps_plus_numbers():
    assert website.normalize_phone("+44 161 496 0000", None) is not None


# ----------------------------------------------------------------- discovery

def base_state(**kw):
    s = dict(location="Wyoming", industry="funeral homes", number_of_leads=5,
             service="AI voice receptionist", plan={}, companies=[],
             qualified_leads=[], rejected_leads=[], seen_domains=[],
             seen_queries=[], failed_queries=[], geo_terms=None,
             research_stats={}, round=0)
    s.update(kw)
    return s


def test_geo_terms_become_queries():
    q = discovery.build_query_pool(base_state(geo_terms=["Casper", "Cody"]))
    assert "funeral homes in Casper, Wyoming" in q
    assert "funeral homes in Cody, Wyoming" in q


def fake_result(failed=False, cands=0):
    return {"failed": failed, "candidates": [
        {"domain": f"d{i}.com", "url": f"https://d{i}.com", "title": "t"}
        for i in range(cands)],
        "raw_count": 0, "blocked_count": 0, "duplicate_count": 0,
        "classifier_rejected_count": 0, "classifier_failed_count": 0}


def test_failed_query_is_retried_then_given_up(monkeypatch):
    monkeypatch.setattr(discovery, "expand_geo_terms", lambda *a: ["Casper"])
    monkeypatch.setattr(discovery, "search_one_query",
                        lambda **k: fake_result(failed=True))
    state = base_state()

    out1 = discovery.search_companies(state)
    first = discovery.build_query_pool({**state, "geo_terms": ["Casper"]})[0].lower()
    assert first not in out1["seen_queries"]          # not consumed yet
    assert first in out1["failed_queries"]

    state2 = {**state, **out1}
    out2 = discovery.search_companies(state2)
    assert first in out2["seen_queries"]              # given up after 2nd failure


def test_unused_queries_carry_over_when_round_ends_early(monkeypatch):
    monkeypatch.setenv("SEARCH_WORKERS", "1")
    monkeypatch.setattr(discovery, "expand_geo_terms", lambda *a: [])
    # first query alone fills the round -> the other 3 must stay unused
    monkeypatch.setattr(discovery, "search_one_query",
                        lambda **k: fake_result(cands=10))
    out = discovery.search_companies(base_state())
    assert len(out["seen_queries"]) == 1


def test_geo_expansion_runs_once(monkeypatch):
    calls = []
    monkeypatch.setattr(discovery, "expand_geo_terms",
                        lambda *a: calls.append(1) or [])
    monkeypatch.setattr(discovery, "search_one_query",
                        lambda **k: fake_result())
    out = discovery.search_companies(base_state())
    discovery.search_companies({**base_state(), **out})
    assert len(calls) == 1


def test_learned_blocklist_needs_two_rejections(tmp_path, monkeypatch):
    import blocklist
    monkeypatch.setattr(blocklist, "BLOCKLIST_PATH", tmp_path / "bl.json")
    blocklist.learn_rejected_domain("dir.example.com")
    assert "dir.example.com" not in blocklist.load_learned_blocklist()
    blocklist.learn_rejected_domain("dir.example.com")
    assert "dir.example.com" in blocklist.load_learned_blocklist()


def test_static_blocklist_covers_directories_from_logs():
    for d in ("clutch.co", "goodfirms.co", "bing.com", "legacy.com",
              "funeralfinder.us", "bbb.org"):
        assert discovery.is_blocked_domain(d), d
    assert not discovery.is_blocked_domain("rvguae.com")


# -------------------------------------------------------------- plan + export

def test_plan_cache_key_is_stable_and_order_insensitive():
    a = make_request(criteria=["x", "y"])
    b = make_request(criteria=["y", "x"], number_of_leads=99)
    assert plan_cache_key(a) == plan_cache_key(b)
    assert plan_cache_key(a) != plan_cache_key(make_request(location="Abu Dhabi"))


def test_export_run_writes_csv_json_report(tmp_path):
    state = {
        "qualified_leads": [{
            "company": "RVG", "website": "https://rvguae.com", "tier": "evidence-backed",
            "offices": "Dubai", "phones": ["+971 4 886 6844"], "emails": ["info@rvguae.com"],
            "reason": "r", "opportunity": "p",
            "evidence": [{"indicator": "i", "evidence": "e"}]}],
        "rejected_leads": [{"website": "https://x.com", "rejection_type": "disqualified",
                            "rejection_reason": "Not a single-business website."}],
        "research_stats": {"qualified": 1}, "seen_queries": ["a", "b"],
    }
    run_dir = export_run(make_request(), {"k": 1}, state, {"total": 1.0},
                         base_dir=str(tmp_path))
    assert (run_dir / "leads.csv").read_text(encoding="utf-8-sig").splitlines()[0].startswith("company,")
    assert json.loads((run_dir / "leads.json").read_text())[0]["company"] == "RVG"
    report = json.loads((run_dir / "report.json").read_text())
    assert report["result"]["rejected_by_type"] == {"disqualified": 1}
    assert report["result"]["queries_used"] == 2


# ============================================================ round 2 fixes

# ---- subpage crawl: "Contact Us page" never matched a link label ----------

NAV_HTML = """
<nav>
  <a href="/services/audit">Audit services</a>
  <a href="/services/tax">Tax services</a>
  <a href="/services/vat">VAT services</a>
  <a href="/services/payroll">Payroll services</a>
  <a href="/about-us">About Us</a>
  <a href="/contact-us">Contact Us</a>
  <a href="https://other.com/contact">Contact elsewhere</a>
  <a href="mailto:a@b.com">mail</a>
  <a href="#top">top</a>
</nav>
"""


def test_plan_phrases_become_keywords():
    kw = website.normalize_page_keywords(
        ["Contact Us page", "Home page", "Support or Help Center page"])
    assert kw[:2] == ["contact", "about"]
    assert "support" in kw and "help" in kw
    assert "page" not in kw and "home" not in kw


def test_contact_and_about_beat_long_service_menus():
    soup = BeautifulSoup(NAV_HTML, "html.parser")
    links = website.find_relevant_internal_links(
        soup, "https://firm.ae/", ["Contact Us page", "Services page"])
    assert links[0].endswith("/contact-us")
    assert links[1].endswith("/about-us")
    assert len(links) == website.MAX_INTERNAL_PAGES
    assert all("other.com" not in link for link in links)


# ---- contacts from WhatsApp links and rendered DOM -------------------------

def test_whatsapp_link_becomes_phone():
    soup = BeautifulSoup(
        '<a href="https://wa.me/971561643075">chat</a>', "html.parser")
    emails, phones = website.extract_contacts([soup], "", "firm.ae", "AE")
    assert phones == ["+971 56 164 3075"]


def test_rendered_contacts_are_merged_validated_and_deduped():
    site = {"emails": ["info@firm.ae"], "phones": ["+971 4 570 5345"]}
    facts = (
        "rendered_phone_links: ['tel:+971 4 5705345', 'tel:+91 98190 00445']\n"
        "rendered_email_links: ['mailto:sales@firm.ae', 'mailto:x@dev.com']\n"
        "rendered_whatsapp_links: ['https://wa.me/971561643075']\n"
    )
    website.merge_rendered_contacts(site, facts, "firm.ae", "AE")
    assert site["phones"] == ["+971 4 570 5345", "+971 56 164 3075"]
    assert site["emails"] == ["info@firm.ae", "sales@firm.ae"]


# ---- evidence must match a planned indicator with the same meaning ---------

PLANNED = [
    "Website lacks a live chat or chatbot feature",
    "Absence of an automated inquiry or booking system",
    "Public mention of multiple office locations in Dubai",
    "Contact page only provides a static email address or phone number",
]


def test_exact_planned_indicator_is_kept():
    assert qualify.canonical_indicator(PLANNED[0], PLANNED) == PLANNED[0]


def test_reversed_booking_indicator_is_dropped():
    # Bestax: "Presence of automated booking" used as evidence of its ABSENCE
    assert qualify.canonical_indicator(
        "Presence of automated inquiry or booking system", PLANNED) is None


def test_unrelated_paraphrase_is_dropped():
    assert qualify.canonical_indicator(
        "Presence of a chat widget", PLANNED) is None


def test_neutral_paraphrase_of_positive_indicator_is_kept():
    assert qualify.canonical_indicator(
        "Presence of multiple office locations", PLANNED) == PLANNED[2]


def test_verify_findings_drops_reversed_and_unplanned_items():
    s = site(text="Book Free 20-Min Call  Offices in Dubai and Canada",
             footprint="")
    s["desktop_screenshot"] = "x"
    findings = [
        IndicatorFinding(
            indicator="Presence of automated inquiry or booking system",
            present=True, evidence_type="text_quote",
            quote="Book Free 20-Min Call"),
        IndicatorFinding(
            indicator="Presence of multiple office locations",
            present=True, evidence_type="text_quote",
            quote="Offices in Dubai and Canada"),
        IndicatorFinding(
            indicator="Presence of a chat widget", present=True,
            evidence_type="visual_observation",
            visual_observation="floating chat widget visible"),
    ]
    out = qualify.verify_findings(findings, s, PLANNED)
    assert [item["indicator"] for item in out] == [PLANNED[2]]


# ---- company-size signals alone can never qualify a lead -------------------

def test_scale_and_credibility_only_is_not_enough():
    items = [
        {"indicator": "Website lists a broad range of services", "source": "website_text"},
        {"indicator": "Presence of client testimonials", "source": "website_text"},
    ]
    items = qualify.dedupe_evidence(items)
    assert len(items) == 2
    assert not qualify.has_gap_evidence(items)


def test_gap_plus_proxy_is_enough():
    items = qualify.dedupe_evidence([
        {"indicator": "Website lacks a live chat widget", "source": "website_text"},
        {"indicator": "Serving 12000+ clients", "source": "website_text"},
    ])
    assert qualify.has_gap_evidence(items)


# ---- outcome breakdown receives discovery counts ---------------------------

def test_discovery_reports_filtered_count_in_stats(monkeypatch):
    monkeypatch.setenv("SEARCH_WORKERS", "1")
    monkeypatch.setattr(discovery, "expand_geo_terms", lambda *a: [])
    res = fake_result(cands=10)
    res["blocked_count"] = 4
    res["classifier_rejected_count"] = 2
    monkeypatch.setattr(discovery, "search_one_query", lambda **k: res)
    out = discovery.search_companies(base_state())
    assert out["research_stats"]["directory_filtered"] == 6


# ============================================================ round 3 fixes

@pytest.mark.parametrize("before,after", [
    ("handle high-volume inquiries regarding Tax",
     "handle routine inquiries regarding Tax"),
    ("The firm has a large recurring client base and high inbound inquiry volume.",
     "The firm has a large recurring client base and inbound inquiries."),
    ("an established firm with a high volume of clients (4,900+)",
     "an established firm with a large number of clients (4,900+)"),
    ("to manage high volumes of calls", "to manage regular calls"),
])
def test_unverified_volume_claims_are_softened(before, after):
    assert qualify.soften_volume_claims(before) == after


def test_text_without_volume_claim_is_unchanged():
    text = "They rely on a phone number and a contact form."
    assert qualify.soften_volume_claims(text) == text


def ev(indicator, source):
    return {"indicator": indicator, "source": source}


def test_evidence_strength_levels():
    d = qualify.dedupe_evidence
    limited = d([ev("Serving 12000+ clients", "website_text"),
                 ev("Website lacks a live chat", "visual_review")])
    moderate = d([ev("Serving 12000+ clients", "website_text"),
                  ev("Absence of an automated booking system", "website_text")])
    strong = d([ev("Absence of an automated booking system", "website_text"),
                ev("Contact page only offers email or phone", "website_text")])
    assert qualify.evidence_strength(limited) == "limited"
    assert qualify.evidence_strength(moderate) == "moderate"
    assert qualify.evidence_strength(strong) == "strong"


def test_generic_mailboxes_preferred_over_named_staff():
    emails = ["arun.sankaran@claemirates.com", "contact@claemirates.com",
              "pradeep.sai@claemirates.com"]
    assert website.prefer_generic_emails(emails) == ["contact@claemirates.com"]


def test_named_emails_kept_when_nothing_generic():
    emails = ["jane.doe@firm.ae"]
    assert website.prefer_generic_emails(emails) == emails


def test_blog_post_slugs_are_not_crawled_as_about_pages():
    html = """
    <a href="/our-company/">About Us</a>
    <a href="/what-should-restaurant-owners-in-dubai-know-about-bookkeeping/">blog</a>
    <a href="/contact-us/">Contact Us</a>
    """
    links = website.find_relevant_internal_links(
        BeautifulSoup(html, "html.parser"), "https://ebs.ae/", ["Contact Us page"])
    assert not any("restaurant" in link for link in links)
    assert links[0].endswith("/contact-us/")


def test_export_includes_evidence_strength(tmp_path):
    state = {"qualified_leads": [{"company": "X", "evidence_strength": "limited"}],
             "rejected_leads": []}
    run_dir = export_run(make_request(), {}, state, {}, base_dir=str(tmp_path))
    header = (run_dir / "leads.csv").read_text(encoding="utf-8-sig").splitlines()
    assert "evidence_strength" in header[0] and ",limited," in header[1]


def test_validator_prompt_states_what_the_agent_can_check(monkeypatch):
    captured = {}

    def fake_call(prompt, schema):
        captured["prompt"] = prompt
        return SimpleNamespace(assessments=[])

    monkeypatch.setattr(clarify, "_structured_call", fake_call)
    clarify.validate_criteria(make_request(criteria=["customers complain about slow replies"]))
    prompt = captured["prompt"]
    assert "public external pages" in prompt
    assert "public customer reviews" in prompt
    assert 'Never say that public information is "not publicly accessible"' in prompt


# ============================================================ Google Places

import httpx
import places


@pytest.fixture
def places_on(monkeypatch):
    monkeypatch.setenv("ALLOW_PAID_PROVIDERS", "1")
    monkeypatch.setenv("GOOGLE_PLACES_API_KEY", "test-key")
    monkeypatch.delenv("PLACES_DISABLED", raising=False)
    monkeypatch.delenv("PLACES_MAX_REQUESTS_PER_RUN", raising=False)
    places.reset_run()
    yield
    places.reset_run()


@pytest.fixture
def places_off(monkeypatch):
    monkeypatch.delenv("GOOGLE_PLACES_API_KEY", raising=False)
    places.reset_run()


def raw_place(domain="agency.ae", reviews=None, status="OPERATIONAL", pid="abc"):
    return {
        "id": pid,
        "displayName": {"text": "Agency"},
        "websiteUri": f"https://www.{domain}/en",
        "googleMapsUri": f"https://maps.google.com/?cid={pid}",
        "businessStatus": status,
        "rating": 4.1,
        "userRatingCount": 120,
        "reviews": reviews or [],
    }


def raw_review(text, rating=2):
    return {
        "text": {"text": text},
        "rating": rating,
        "relativePublishTimeDescription": "a month ago",
        "authorAttribution": {"displayName": "A. User", "uri": "https://g.co/u"},
        "googleMapsUri": "https://maps.google.com/?review=1",
    }


def mock_client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_search_text_sends_key_fieldmask_and_parses(places_on):
    seen = {}

    def handler(request):
        seen["headers"] = request.headers
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"places": [raw_place(
            reviews=[raw_review("They never replied to my emails.")])]})

    out = places.search_text("real estate agencies in Dubai", "ae",
                             client=mock_client(handler))
    assert seen["headers"]["x-goog-api-key"] == "test-key"
    assert "places.reviews" in seen["headers"]["x-goog-fieldmask"]
    assert seen["body"]["textQuery"] == "real estate agencies in Dubai"
    assert seen["body"]["regionCode"] == "AE"
    assert out[0]["place_id"] == "abc"
    assert out[0]["reviews"][0]["maps_uri"].endswith("review=1")
    assert places.requests_made() == 1


def test_403_disables_places_for_the_run_without_leaking_key(places_on):
    def handler(request):
        return httpx.Response(403, json={"error": {"message": "API not enabled"}})

    with pytest.raises(places.PlacesError) as err:
        places.search_text("x", client=mock_client(handler))
    assert "test-key" not in str(err.value)
    assert not places.can_search()
    assert "API not enabled" in places.disabled_reason()


def test_429_and_network_errors_do_not_disable(places_on):
    with pytest.raises(places.PlacesError):
        places.search_text("x", client=mock_client(lambda r: httpx.Response(429)))

    def boom(request):
        raise httpx.ConnectTimeout("t")

    with pytest.raises(places.PlacesError):
        places.search_text("x", client=mock_client(boom))
    assert places.can_search()


def test_request_cap_stops_places(places_on, monkeypatch):
    monkeypatch.setenv("PLACES_MAX_REQUESTS_PER_RUN", "2")
    ok = lambda r: httpx.Response(200, json={"places": []})
    for _ in range(2):
        places.search_text("x", client=mock_client(ok))
    assert not places.can_search()
    with pytest.raises(places.PlacesError):
        places.search_text("x", client=mock_client(ok))


def test_places_disabled_switch(places_on, monkeypatch):
    monkeypatch.setenv("PLACES_DISABLED", "1")
    assert not places.is_configured()


@pytest.mark.parametrize("text", [
    "They never replied to my emails.",
    "It took them three days to get back to me.",
    "No response to my inquiry after a week.",
    "Terrible communication, slow to respond.",
    "The agent ignored my messages.",
    "I'm still waiting for a reply.",
    "They don't answer the phone.",
    "Unresponsive team.",
    "I couldn't get a reply for two weeks.",
    "Took forever to reply.",
])
def test_slow_reply_complaints_detected(text):
    assert places.find_slow_reply_sentence(text) is not None, text


@pytest.mark.parametrize("text", [
    "Great service, they replied within minutes.",
    "Never had to chase, they replied quickly.",
    "Very responsive and professional.",
    "The agent answered all my questions.",
    "Loved the villa, would recommend.",
    "Didn't hesitate to reply and was helpful.",
])
def test_praise_is_not_a_complaint(text):
    assert places.find_slow_reply_sentence(text) is None, text


def test_review_summary_counts_and_trims_quote():
    long = "They never replied to my emails " + "and " * 40 + "ever."
    place = places.normalize_place(raw_place(reviews=[
        raw_review(long),
        raw_review("Lovely team, quick replies."),
    ]))
    summary = places.review_summary(place)
    assert summary["checked"] == 2 and len(summary["mentions"]) == 1
    assert summary["mentions"][0]["quote"].endswith(chr(0x2026))
    assert len(summary["mentions"][0]["quote"].split()) <= 26


def test_review_criteria_detection():
    got = places.review_criteria([
        "because customers complain about slow replies to inquiries",
        "weak online support channels",
        "public reviews mention missed calls"])
    assert len(got) == 2 and "weak online support channels" not in got


def test_validator_capability_changes_with_key(places_off, monkeypatch):
    monkeypatch.setenv("ALLOW_PAID_PROVIDERS", "1")
    assert "public customer reviews" in places.capability_text()
    monkeypatch.setenv("GOOGLE_PLACES_API_KEY", "k")
    text = places.capability_text()
    assert "Google Maps reviews" in text and "optional" in text


def test_validator_prompt_allows_reviews_when_places_configured(places_on, monkeypatch):
    captured = {}
    monkeypatch.setattr(clarify, "_structured_call",
                        lambda p, s: captured.setdefault("p", p) and SimpleNamespace(assessments=[]))
    clarify.validate_criteria(make_request(criteria=["customers complain about slow replies"]))
    assert "up to 5 Google Maps reviews" in captured["p"]
    assert "Only fetched passages" in captured["p"]


# ---- discovery with Places -------------------------------------------------

def test_places_candidates_filtered_and_carry_place(places_on, monkeypatch):
    found = [
        places.normalize_place(raw_place("agency.ae", pid="1")),
        places.normalize_place(raw_place("agency.ae", pid="2")),                 # duplicate
        places.normalize_place(raw_place("closed.ae", status="CLOSED_PERMANENTLY", pid="3")),
        places.normalize_place({**raw_place("x.ae", pid="4"), "websiteUri": ""}),  # no site
        places.normalize_place({**raw_place("x.ae", pid="5"),
                                "websiteUri": "https://facebook.com/page"}),    # blocked
        places.normalize_place(raw_place("other.ae", pid="6")),
    ]
    monkeypatch.setattr(places, "search_text", lambda *a, **k: found)
    out = discovery.search_one_query_places("q", set(), set(), 10, "AE")
    assert [c["domain"] for c in out["candidates"]] == ["agency.ae", "other.ae"]
    assert out["candidates"][0]["place"]["place_id"] == "1"
    assert out["candidates"][0]["title"] == "agency.ae"      # Google's name not stored
    assert out["raw_count"] == 6 and out["blocked_count"] == 3
    assert out["duplicate_count"] == 1


def test_run_one_query_falls_back_to_web_search_on_places_error(places_on, monkeypatch):
    def fail(*a, **k):
        raise places.PlacesError("rate limited (HTTP 429)")
    monkeypatch.setattr(places, "search_text", fail)
    monkeypatch.setattr(discovery, "search_one_query", lambda **k: {"web": True})
    state = {"industry": "x", "location": "y", "plan": {}}
    assert discovery.run_one_query("q", state, set(), set(), 5) == {"web": True}


def test_run_one_query_falls_back_when_places_returns_nothing(places_on, monkeypatch):
    monkeypatch.setattr(places, "search_text", lambda *a, **k: [])
    monkeypatch.setattr(discovery, "search_one_query", lambda **k: {"web": True})
    state = {"industry": "x", "location": "y", "plan": {}}
    assert discovery.run_one_query("q", state, set(), set(), 5) == {"web": True}


def test_run_one_query_uses_web_search_when_not_configured(places_off, monkeypatch):
    monkeypatch.setattr(discovery, "search_one_query", lambda **k: {"web": True})
    state = {"industry": "x", "location": "y", "plan": {}}
    assert discovery.run_one_query("q", state, set(), set(), 5) == {"web": True}


# ---- qualification with review evidence ------------------------------------

def company_with(reviews):
    return {"place": places.normalize_place(raw_place(reviews=reviews))}


PLAN_WITH_REVIEWS = {"review_criteria": ["customers complain about slow replies"]}


def test_review_evidence_is_gap_evidence_and_strong():
    c = company_with([raw_review("They never replied to my emails."),
                      raw_review("Nice office.", 5)])
    summary, items = qualify.review_evidence_for(c, PLAN_WITH_REVIEWS)
    assert summary["checked"] == 2 and len(summary["mentions"]) == 1
    assert items[0]["source"] == "google_reviews"
    merged = qualify.dedupe_evidence(
        [{"indicator": "Serving 4,900+ clients", "source": "website_text"}] + items)
    assert qualify.has_gap_evidence(merged)
    assert qualify.evidence_strength(merged) == "moderate"


def test_no_review_evidence_when_request_does_not_ask_for_reviews():
    c = company_with([raw_review("They never replied.")])
    assert qualify.review_evidence_for(c, {}) == (None, [])


def test_no_complaint_found_gives_summary_but_no_item():
    c = company_with([raw_review("Great agents.", 5)])
    summary, items = qualify.review_evidence_for(c, PLAN_WITH_REVIEWS)
    assert items == [] and summary["checked"] == 1
    assert "None of the 1 Google Maps reviews" in qualify.review_gate_message(c, summary)


def test_gate_message_when_no_place_matched():
    assert "No Google Maps listing" in qualify.review_gate_message({}, None)


def test_review_facts_shown_to_gemini_as_counts_only():
    c = company_with([raw_review("They never replied to my emails.")])
    s = {"footprint": "https: True"}
    qualify.attach_review_facts(c, s, PLAN_WITH_REVIEWS)
    assert "google_maps_reviews_checked: 1" in s["footprint"]
    assert "mentioning_slow_or_missing_replies: 1" in s["footprint"]
    assert "never replied" not in s["footprint"]


def test_google_facts_support_only_review_claims():
    fact = "google_maps_reviews_mentioning_slow_or_missing_replies: 2"
    assert qualify.technical_fact_supports("Public reviews complain about slow replies", fact)
    assert not qualify.technical_fact_supports("Website lacks a live chat widget", fact)


# ---- exports never contain Google content ----------------------------------

def test_exports_keep_links_but_not_review_text(tmp_path):
    lead = {
        "company": "Agency", "website": "https://agency.ae",
        "evidence": [{"indicator": "Public Google Maps reviews mention slow or missing replies",
                      "evidence": "1 of 5 Google Maps reviews checked mention slow or missing replies"}],
        "review_evidence": {
            "checked": 5, "place_id": "abc", "place_maps_uri": "https://maps.google.com/?cid=abc",
            "mentions": [{"quote": "They never replied to my emails.", "author": "A. User",
                          "rating": 1, "maps_uri": "https://maps.google.com/?review=1"}]},
    }
    run_dir = export_run(make_request(), {}, {"qualified_leads": [lead], "rejected_leads": []},
                         {}, base_dir=str(tmp_path))
    csv_text = (run_dir / "leads.csv").read_text(encoding="utf-8-sig")
    json_text = (run_dir / "leads.json").read_text()
    for text in (csv_text, json_text):
        assert "never replied" not in text and "A. User" not in text
    assert "review=1" in csv_text and "abc" in csv_text
    assert json.loads(json_text)[0]["review_evidence"]["mentions"] == [
        {"maps_uri": "https://maps.google.com/?review=1"}]



# ============================================================ speed features

import time as _time


def test_searches_in_a_batch_run_at_the_same_time(monkeypatch):
    monkeypatch.setenv("SEARCH_WORKERS", "3")
    monkeypatch.setattr(discovery, "expand_geo_terms", lambda *a: [])

    def slow(**k):
        _time.sleep(0.3)
        return fake_result(cands=10)

    monkeypatch.setattr(discovery, "search_one_query", slow)
    started = _time.time()
    out = discovery.search_companies(base_state())
    assert _time.time() - started < 0.7            # 3 x 0.3s would be 0.9s
    assert len(out["seen_queries"]) == 3


def test_parallel_queries_dedupe_across_queries_and_respect_the_cap(monkeypatch):
    monkeypatch.setenv("SEARCH_WORKERS", "3")
    monkeypatch.setattr(discovery, "expand_geo_terms", lambda *a: [])
    # every query returns the same 10 domains (d0..d9)
    monkeypatch.setattr(discovery, "search_one_query",
                        lambda **k: fake_result(cands=10))
    state = base_state(number_of_leads=5)           # target = 6 candidates
    out = discovery.search_companies(state)
    domains = [c["domain"] for c in out["companies"]]
    assert len(domains) == len(set(domains)) == 6


def test_search_workers_setting(monkeypatch):
    monkeypatch.delenv("SEARCH_WORKERS", raising=False)
    assert discovery.search_workers() == 3
    monkeypatch.setenv("SEARCH_WORKERS", "1")
    assert discovery.search_workers() == 1
    monkeypatch.setenv("SEARCH_WORKERS", "junk")
    assert discovery.search_workers() == 3


def test_render_workers_setting(monkeypatch):
    monkeypatch.delenv("RENDER_WORKERS", raising=False)
    assert website.render_workers() == 1
    monkeypatch.setenv("RENDER_WORKERS", "3")
    assert website.render_workers() == 3


# ---- measured "no live chat" evidence --------------------------------------

def test_chat_scan_states():
    assert website.chat_widget_scan("<html><body>hi</body></html>") == \
        "none found in page HTML (no tag manager)"
    assert website.chat_widget_scan(
        '<script src="https://embed.tawk.to/abc"></script>').startswith("found tawk.to")
    assert "tag manager present" in website.chat_widget_scan(
        '<script src="https://www.googletagmanager.com/gtm.js?id=X"></script>')


CHAT_PLAN = {"need_indicators": [
    "Website lacks a live chat or chatbot feature",
    "Public mention of multiple office locations in Dubai"]}


def test_clean_scan_becomes_technical_evidence_for_the_planned_indicator():
    s = {"footprint": "chat_widget_scan: none found in page HTML (no tag manager)"}
    items = qualify.chat_scan_evidence(s, CHAT_PLAN)
    assert len(items) == 1 and items[0]["source"] == "technical_fact"
    assert items[0]["indicator"] == CHAT_PLAN["need_indicators"][0]


def test_tag_manager_or_vendor_gives_no_chat_evidence():
    for scan in ("none found in page HTML (tag manager present, widget may load dynamically)",
                 "found tawk.to"):
        assert qualify.chat_scan_evidence(
            {"footprint": f"chat_widget_scan: {scan}"}, CHAT_PLAN) == []


def test_measured_chat_evidence_outranks_screenshot_impression():
    measured = qualify.chat_scan_evidence(
        {"footprint": "chat_widget_scan: none found in page HTML (no tag manager)"}, CHAT_PLAN)
    visual = [{"indicator": "Website lacks a live chat or chatbot feature",
               "evidence": "no bubble in screenshot", "source": "visual_review"}]
    merged = qualify.dedupe_evidence(visual + measured)
    assert len(merged) == 1 and merged[0]["source"] == "technical_fact"


def test_chat_scan_fact_supports_only_matching_claims():
    ok = "chat_widget_scan: none found in page HTML (no tag manager)"
    gtm = "chat_widget_scan: none found in page HTML (tag manager present, widget may load dynamically)"
    assert qualify.technical_fact_supports("Website lacks a live chat or chatbot feature", ok)
    assert not qualify.technical_fact_supports("Website lacks a live chat or chatbot feature", gtm)
    assert not qualify.technical_fact_supports("Presence of a live chat widget", ok)
    assert not qualify.technical_fact_supports("Absence of a contact form", ok)
    assert qualify.technical_fact_supports(
        "Presence of a live chat widget", "chat_widget_scan: found tawk.to")


# ================================================================ result cache

import subprocess
import sys
from pathlib import Path

import result_cache
import warm

PROJECT = Path(__file__).resolve().parent.parent


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(result_cache, "CACHE_DIR", tmp_path / "results")
    monkeypatch.delenv("REFRESH", raising=False)
    monkeypatch.delenv("RESULT_CACHE_HOURS", raising=False)
    return tmp_path / "results"


def final_state(n_qualified=3):
    return {
        "qualified_leads": [
            {"company": f"Firm {i}", "website": f"https://f{i}.ae",
             "tier": "evidence-backed", "reason": "r", "opportunity": "p",
             "visual_summary": "v", "offices": "Dubai", "emails": [], "phones": [],
             "pages_read": ["https://f%d.ae" % i], "evidence_strength": "moderate",
             "evidence": [{"indicator": "i", "evidence": "e", "source": "website_text"}],
             "review_evidence": {"checked": 5, "place_id": "p", "place_maps_uri": "u",
                                 "mentions": [{"quote": "never replied", "author": "A. User",
                                               "maps_uri": "m"}]}}
            for i in range(n_qualified)],
        "rejected_leads": [{"website": "https://x.com", "rejection_type": "disqualified",
                            "rejection_reason": "r", "title": "SECRET TITLE"}],
        "round": 2, "research_stats": {"directory_filtered": 7},
    }


def test_message_key_ignores_case_and_spacing_but_not_fast_mode():
    a = result_cache.message_key("Find 5  dentists in Lahore ", False)
    assert a == result_cache.message_key("find 5 dentists in lahore", False)
    assert a != result_cache.message_key("find 5 dentists in lahore", True)
    assert a != result_cache.message_key("find 6 dentists in lahore", False)


def test_save_then_load_roundtrip_without_google_content(cache_dir):
    assert result_cache.save(["k1", "k2"], 5, final_state(3), False)
    for key in ("k1", "k2"):
        state, age = result_cache.load(key)
        assert len(state["qualified_leads"]) == 3 and age < 5
        assert state["round"] == 2
        assert state["research_stats"] == {"directory_filtered": 7}
    raw = (cache_dir / "k1.json").read_text()
    assert "never replied" not in raw and "A. User" not in raw
    assert "SECRET TITLE" not in raw                      # only website/type/reason kept


def test_empty_runs_are_never_cached(cache_dir):
    assert result_cache.save(["k"], 5, final_state(0), False) is False
    assert result_cache.load("k") is None


def test_requested_count_rules(cache_dir):
    result_cache.save(["k"], 5, final_state(3), False)       # asked 5, found 3
    assert result_cache.load("k", requested=3)[0]["number_of_leads"] == 3
    assert len(result_cache.load("k", requested=2)[0]["qualified_leads"]) == 2
    assert result_cache.load("k", requested=5) is not None   # same ask: shortfall is the answer
    assert result_cache.load("k", requested=8) is None       # bigger ask: research again


def test_ttl_and_refresh(cache_dir, monkeypatch):
    result_cache.save(["k"], 5, final_state(3), False)
    monkeypatch.setenv("REFRESH", "1")
    assert result_cache.load("k") is None
    monkeypatch.delenv("REFRESH")
    assert result_cache.load("k") is not None
    monkeypatch.setenv("RESULT_CACHE_HOURS", "0")
    assert result_cache.load("k") is None
    assert result_cache.save(["z"], 5, final_state(3), False) is False


def test_stale_entries_are_ignored(cache_dir, monkeypatch):
    result_cache.save(["k"], 5, final_state(3), False)
    path = cache_dir / "k.json"
    data = json.loads(path.read_text())
    data["saved_at"] -= 8 * 24 * 3600
    path.write_text(json.dumps(data))
    assert result_cache.load("k") is None


def test_describe_age():
    assert result_cache.describe_age(10) == "less than a minute ago"
    assert result_cache.describe_age(600) == "10 minutes ago"
    assert result_cache.describe_age(7200) == "2 hours ago"
    assert result_cache.describe_age(3 * 86400) == "3 days ago"


def test_main_answers_a_repeated_request_instantly_from_saved_results(tmp_path):
    """Runs the real main.py: no Gemini key, network or browser is used."""
    message = "Find 3 accounting firms in Dubai for AI customer support"
    saved = tmp_path / ".cache" / "results"
    saved.mkdir(parents=True)
    state = final_state(3)
    payload = {
        "saved_at": __import__("time").time(), "fast": False, "requested": 3,
        "qualified_leads": [result_cache.public_lead(lead) for lead in state["qualified_leads"]],
        "rejected_leads": [], "round": 1, "research_stats": {},
    }
    (saved / f"{result_cache.message_key(message, False)}.json").write_text(json.dumps(payload))

    started = __import__("time").time()
    done = subprocess.run(
        [sys.executable, str(PROJECT / "main.py")], input=message + "\n",
        text=True, capture_output=True, cwd=tmp_path, timeout=60,
        env={**__import__("os").environ, "GEMINI_API_KEY": "x",
             "PYTHONPATH": str(PROJECT)})
    out = done.stdout
    assert "Results from an earlier run" in out, out + done.stderr
    assert "Firm 0" in out and "Firm 2" in out, out + done.stderr
    assert "Answered from saved results" in out
    assert "Understanding your request" not in out
    assert done.returncode == 0


def test_warm_reads_requests_and_classifies_outcomes(tmp_path):
    file = tmp_path / "r.txt"
    file.write_text("# comment\n\nFind 5 a in B for C\n  Find 6 d in E for F  \n")
    assert warm.read_requests(str(file)) == ["Find 5 a in B for C", "Find 6 d in E for F"]
    assert warm.classify(0, "x\nSaved: runs\\1\\leads.csv") == "saved"
    assert warm.classify(1, "EOFError").startswith("skipped")
    assert warm.classify(0, "No lead research was started.").startswith("skipped")
    assert warm.classify(3, "boom").startswith("failed")



# ====================================================== round 5 (Wyoming log)

def test_cache_ignores_leads_missing_fields_the_display_needs(cache_dir):
    state = final_state(2)
    del state["qualified_leads"][0]["visual_summary"]
    result_cache.save(["k"], 5, state, False)
    assert result_cache.load("k") is None


def test_same_site_twice_in_one_result_list_counts_once(monkeypatch):
    results = [
        {"href": "https://bustardcares.com/", "title": "Bustard & Jacoby", "body": "a"},
        {"href": "https://www.bustardcares.com/obituaries", "title": "Obituaries", "body": "b"},
        {"href": "https://newcomercasper.com/", "title": "Newcomer", "body": "c"},
    ]

    class FakeDDGS:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def text(self, *a, **k): return results

    monkeypatch.setattr(discovery, "DDGS", FakeDDGS)
    monkeypatch.setattr(discovery, "classify_candidates_parallel", lambda cands, *a: [
        {"candidate": c, "error": None,
         "classification": SimpleNamespace(result_type="single_business",
                                           matches_target_industry=True)}
        for c in cands])
    out = discovery.search_one_query("q", set(), set(), 10, "funeral homes", "Wyoming")
    domains = [c["domain"] for c in out["candidates"]]
    assert domains == ["bustardcares.com", "newcomercasper.com"]
    assert out["duplicate_count"] == 1


# ---- research time limit ---------------------------------------------------

def test_deadline_helpers(monkeypatch):
    assert not discovery.deadline_passed({})
    assert not discovery.deadline_passed({"deadline": 0})
    assert not discovery.deadline_passed({"deadline": _time.time() + 60})
    assert discovery.deadline_passed({"deadline": _time.time() - 1})


def test_max_run_seconds_setting(monkeypatch):
    monkeypatch.delenv("MAX_RUN_SECONDS", raising=False)
    assert discovery.max_run_seconds() == 600
    monkeypatch.setenv("MAX_RUN_SECONDS", "0")
    assert discovery.max_run_seconds() == 0
    monkeypatch.setenv("MAX_RUN_SECONDS", "90")
    assert discovery.max_run_seconds() == 90
    monkeypatch.setenv("MAX_RUN_SECONDS", "junk")
    assert discovery.max_run_seconds() == 600


def test_search_round_stops_when_deadline_passed(monkeypatch):
    called = []
    monkeypatch.setattr(discovery, "run_one_query", lambda *a, **k: called.append(1))
    out = discovery.search_companies(base_state(deadline=_time.time() - 1))
    assert out["companies"] == [] and out["query_pool_exhausted"] is True
    assert not called
    assert called == []


# ---- why a site could not be fetched ---------------------------------------

def test_fetch_page_records_the_reason(monkeypatch):
    def boom(url, **k):
        raise RuntimeError("blocked by firewall")
    monkeypatch.setattr(website.httpx, "get", boom)
    assert website.fetch_page("https://blocked.example/") is None
    assert "RuntimeError: blocked by firewall" in website.FETCH_ERRORS["https://blocked.example/"]


def test_fetch_page_does_not_disable_certificate_verification(monkeypatch):
    calls = []
    def get(url, **kwargs):
        calls.append(kwargs.get("verify", True))
        raise RuntimeError("CERTIFICATE_VERIFY_FAILED")
    monkeypatch.setattr(website.httpx, "get", get)
    assert website.fetch_page("https://badcert.example/") is None
    import ssl
    assert len(calls) == 1
    assert calls[0].verify_mode == ssl.CERT_REQUIRED and calls[0].check_hostname
    assert "CERTIFICATE_VERIFY_FAILED" in website.FETCH_ERRORS["https://badcert.example/"]


def test_planner_cache_key_changes_with_prompt_version(monkeypatch):
    import planner
    before = planner.plan_cache_key(make_request())
    monkeypatch.setattr(planner, "PLAN_PROMPT_VERSION", planner.PLAN_PROMPT_VERSION + 1)
    assert planner.plan_cache_key(make_request()) != before
