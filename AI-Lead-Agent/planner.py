from gemini_llm import structured_call, GEMINI_MODEL
import hashlib
import json
import os
import time
from pathlib import Path

import places
import re

from schemas import SearchRequest, EvidencePlan


# =========================================================
# PLAN CACHE
# Same (industry, service, location, criteria) -> same plan, so repeated
# runs judge the same websites against the same indicators.
# Set NO_PLAN_CACHE=1 to bypass.
# =========================================================

PLAN_CACHE_DIR = Path(".cache") / "plans"
PLAN_CACHE_TTL_SECONDS = 7 * 24 * 3600

# Bump when the planner prompt changes, so old saved plans are not reused.
PLAN_PROMPT_VERSION = 10


def _norm(value):
    return " ".join((value or "").lower().split())


def plan_cache_key(
    request: SearchRequest,
):
    payload = json.dumps(
        {
            "version": PLAN_PROMPT_VERSION,
            "gemini_model": GEMINI_MODEL,
            "location": _norm(request.location),
            "industry": _norm(request.industry),
            "service": _norm(request.service),
            "criteria": sorted(
                _norm(item)
                for item in request.criteria
            ),
        },
        sort_keys=True,
    )

    return hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()[:24]


def make_research_plan(
    request: SearchRequest,
):
    use_cache = not os.getenv(
        "NO_PLAN_CACHE"
    )

    path = PLAN_CACHE_DIR / (
        plan_cache_key(request) + ".json"
    )

    if use_cache and path.exists():
        try:
            age = time.time() - path.stat().st_mtime

            if age < PLAN_CACHE_TTL_SECONDS:
                plan = EvidencePlan.model_validate_json(
                    path.read_text(encoding="utf-8")
                )

                print(
                    "\n(Using the saved research plan for this "
                    "exact request.)"
                )

                return plan
        except Exception:
            pass  # unreadable cache entry: regenerate

    plan = _generate_research_plan(
        request
    )

    if use_cache:
        try:
            PLAN_CACHE_DIR.mkdir(
                parents=True,
                exist_ok=True,
            )

            path.write_text(
                plan.model_dump_json(),
                encoding="utf-8",
            )
        except Exception:
            pass  # caching is best-effort

    return plan


def _generate_research_plan(
    request: SearchRequest,
):
    criteria_text = "\n".join(
        f"- {criterion}"
        for criterion in request.criteria
    ) or "- none"

    observable_sources = (
        "the business's own website, measured website facts, screenshots, and "
        "public third-party pages fetched via DDGS (reviews, forums, news, job pages)"
    )

    prompt = f"""
Create a practical public-web research plan for a B2B lead-generation agent.

LOCATION:
{request.location}

INDUSTRY:
{request.industry}

SERVICE:
{request.service}

NUMBER OF LEADS:
{request.number_of_leads}

USER-APPROVED QUALIFICATION CRITERIA:
{criteria_text}

RULES

1. The user's approved criteria are binding.
   If criteria are present, the plan MUST preserve their meaning and include
   them in the need indicators/search strategy where public evidence can
   support them. Do not silently replace them with a different reason.
   Explicit criteria are mandatory eligibility conditions, not examples.
   For "broken navigation OR unreadable mobile layouts", slow load or a
   missing booking CTA cannot substitute for either requested condition.

2. Do not invent hidden/internal business facts such as churn, revenue loss,
   owner dissatisfaction, ticket volume, staffing problems, or future events.

3. Need indicators must be observable from {observable_sources}.
   Public external sources are supported when readable; internal/private data is not.
   For complaint/review criteria generate 3 complaint_search_queries that find
   review pages naming businesses in the target industry/location and describing
   the requested pain. Do NOT include company placeholders: these run BEFORE
   business discovery. Include one broad query and one Trustpilot review query.
   Use concise pain phrases rather than copying the entire user sentence.
   Generate up to 3 external_search_queries using literal placeholders
   {{company}}, {{domain}}, {{location}} for targeted per-business research.
   Include external searches when reviews, complaints, news, hiring or other
   third-party evidence would help establish service need. Use [] for purely
   website-based requests. Search snippets alone never establish evidence.
   Prefer general searches for public complaints/forums; do not require Yelp
   or Google Maps HTML scraping, which may refuse requests or require JavaScript.
   external_criteria must contain the exact user criteria requiring external
   evidence (reviews, complaints, public reports); never substitute website
   proxies for these. Include each such criterion verbatim in need_indicators.
   Write 5 to 7 service-specific indicators. Each must describe an observable
   problem that SERVICE would solve for INDUSTRY's customers.
   The service is what the user SELLS, not what target businesses sell.
   For web development targeting gyms: missing/broken membership inquiry or
   trial-booking flow, unreadable mobile pages, broken navigation or media,
   missing class/pricing information, or a concrete obsolete layout.
   Never require gyms to offer web development, nor a CTA to buy web development.
   Do not impose chat/after-hours gaps unless relevant to the requested service.
   Company size, a missing carousel, or generic "modern UI" does not prove need.
   Ideal buyer and disqualifiers describe industry/location fit; lack of need
   evidence is insufficient evidence, not automatically a business mismatch.
   Do not add requirements for in-house staffing, ownership model, business
   size or software deployment unless the user requested those restrictions.
   Visual indicators must refer to the target's customer tasks and visible faults.

4. Search queries should find actual business websites, not listicles or
   directories. Generate multiple distinct variations. If LOCATION is broad,
   diversify queries geographically when reasonable rather than repeating the
   same wording. For broad business categories, vary real subtypes/specialties
   and local areas rather than repeating the category verbatim. For medical
   practices use family practice, pediatrician, dentist, urgent care, etc.
   Never invent neighborhood names.

5. Disqualifiers should include:
   - wrong location
   - wrong business type
   - directories/aggregators/listicles
   - clear evidence the requested service problem is already fully solved,
     when that genuinely removes the opportunity
   Never exclude a business solely for being large or part of a network unless
   the user requested that restriction. Do not exclude sole traders unless requested.
   Size does not prove solved call handling.
   Judge a specific local practice when evidence identifies that location.

6. evidence_observable=true only when the need can be judged from public data.

7. visual_review_needed=true only if screenshots materially help judge the
   service/criteria (for example redesign, UI/UX, booking flows, visible CTAs).

8. Do not equate:
   - WhatsApp with AI automation
   - live chat with AI automation
   - a tracker with conversational support
   - absence of a phrase with proof of absence

9. For broad regions/countries, include geographically diverse search queries
   rather than repeating only one city.
"""

    plan = structured_call(prompt, EvidencePlan)
    # External conditions must reference the user's exact criteria. Model
    # paraphrases otherwise create extra gates no returned finding can satisfy.
    import difflib
    requested_external = []
    for criterion in plan.external_criteria:
        matches = difflib.get_close_matches(criterion, request.criteria, n=1, cutoff=0.65)
        if matches and matches[0] not in requested_external:
            requested_external.append(matches[0])
    for criterion in places.review_criteria(request.criteria):
        if criterion not in requested_external:
            requested_external.append(criterion)
    plan.external_criteria = requested_external
    for criterion in requested_external:
        if criterion not in plan.need_indicators:
            plan.need_indicators.append(criterion)
    if plan.external_criteria and not plan.external_search_queries:
        plan.external_search_queries = [
            '"{company}" "{location}" ' + criterion
            for criterion in plan.external_criteria[:3]
        ]
    if requested_external and not plan.complaint_search_queries:
        pain = " ".join(requested_external)
        pain = re.sub(r"^because\s+", "", pain, flags=re.I)
        base = f"{request.industry} {request.location}"
        plan.complaint_search_queries = [
            f'{base} reviews {pain}',
            f'site:trustpilot.com/review/ {base} {pain}',
        ]
    plan.complaint_search_queries = [q for q in plan.complaint_search_queries[:3]
                                   if "{company}" not in q and "{domain}" not in q]
    plan.complaint_search_queries = [
        q if _norm(request.location) in _norm(q) else f"{q} {request.location}"
        for q in plan.complaint_search_queries]
    return plan


def show_research_plan(
    plan: EvidencePlan,
):
    print(
        "\n" + "=" * 70
    )
    print(
        "RESEARCH PLAN"
    )
    print(
        "=" * 70
    )

    print(
        "\nIdeal buyer:"
    )
    print(
        plan.ideal_buyer
    )

    print(
        "\nNeed indicators:"
    )
    for item in plan.need_indicators:
        print(
            f"  + {item}"
        )

    if plan.complaint_search_queries:
        print("\nComplaint-led discovery searches:")
        for query in plan.complaint_search_queries:
            print(f"  - {query}")
    if plan.external_search_queries:
        print("\nExternal searches per business:")
        for query in plan.external_search_queries:
            print(f"  - {query}")

    print(
        "\nDisqualifiers:"
    )
    for item in plan.disqualifiers:
        print(
            f"  - {item}"
        )

    print(
        "\nPages to inspect:"
    )
    for item in plan.pages_to_check:
        print(
            f"  - {item}"
        )

    print(
        "\nEvidence mode: "
        + (
            "EVIDENCE-BACKED"
            if plan.evidence_observable
            else "FIT-ONLY"
        )
    )

    print(
        "Visual review: "
        + (
            "YES"
            if plan.visual_review_needed
            else "NO"
        )
    )

    if plan.visual_review_needed:
        print(
            "\nVisual checks:"
        )
        for item in plan.visual_indicators:
            print(
                f"  + {item}"
            )
