from langchain_google_genai import ChatGoogleGenerativeAI

from schemas import (
    SearchRequest,
    EvidencePlan
)


llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=0
)


plan_llm = llm.with_structured_output(
    EvidencePlan
)


def make_research_plan(
    request: SearchRequest
):

    prompt = f"""
You are designing a B2B lead-research plan.

SERVICE BEING SOLD:
{request.service}

TARGET INDUSTRY:
{request.industry}

TARGET LOCATION:
{request.location}

TARGET NUMBER OF QUALIFIED LEADS:
{request.number_of_leads}


IDEAL BUYER

Describe in one sentence what kind of company would
realistically buy this specific service.


SEARCH QUERIES

Generate 5 different web-search queries.

They should surface actual businesses and preferably
their own websites.

Include the target location.

Avoid intentionally searching for directories,
listicles, review websites or articles.

Vary the wording between queries.


NEED INDICATORS

Generate 3 to 6 specific indicators that would provide
meaningful evidence that this company is a good potential
buyer for the service.

Indicators must be based on information that can reasonably
be observed from public company information, its website,
its rendered interface, or technical facts.

Do not use vague indicators such as:

"company values quality"
"company might benefit"
"business probably receives many calls"

Indicators must be checkable.


DISQUALIFIERS

Generate 2 to 5 concrete reasons the candidate should
be rejected.

Examples can include:

wrong business type
directory rather than a business
wrong location
clearly unsuitable scale
already has the exact solution

Use only reasons relevant to this request.


PAGES TO CHECK

Return 3 to 5 internal-page keywords that would be
particularly useful for researching this service.

Examples can include:

contact
about
services
products
careers
pricing
appointments

Choose dynamically for this specific request.


EVIDENCE OBSERVABLE

Set evidence_observable=true when meaningful evidence
of need can realistically be observed externally.

Set false when the real need is mostly internal and
cannot honestly be established from public information.


VISUAL REVIEW

Set visual_review_needed=true only when actually seeing
the rendered website could materially improve qualification.

Examples where visual review may matter:

website design or redesign
landing page design
conversion optimization
UI/UX work
mobile usability
branding/presentation problems

Visual review may also be useful when visible buttons,
navigation, layout or interfaces are important.

Do NOT enable visual review simply because every company
has a website.

For services where screenshots provide little useful evidence,
set visual_review_needed=false.


VISUAL INDICATORS

If visual_review_needed=true, return 3 to 6 specific things
Gemini should inspect visually.

Examples might include:

poor visual hierarchy
obviously outdated appearance
broken mobile layout
hard-to-find call to action
cluttered navigation
low quality imagery
unprofessional spacing

But choose indicators specifically for THIS service.

If visual review is not needed, return an empty list.
"""

    return plan_llm.invoke(
        prompt
    )


def show_research_plan(
    plan: EvidencePlan
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
        f"\nIdeal buyer:\n{plan.ideal_buyer}"
    )

    print(
        "\nNeed indicators:"
    )

    for indicator in plan.need_indicators:

        print(
            f"  + {indicator}"
        )

    print(
        "\nDisqualifiers:"
    )

    for disqualifier in plan.disqualifiers:

        print(
            f"  - {disqualifier}"
        )

    print(
        "\nPages to inspect:"
    )

    for page in plan.pages_to_check:

        print(
            f"  - {page}"
        )

    print(
        "\nEvidence mode:",
        (
            "EVIDENCE-BACKED"
            if plan.evidence_observable
            else "FIT-ONLY"
        )
    )

    print(
        "Visual review:",
        (
            "YES"
            if plan.visual_review_needed
            else "NO"
        )
    )

    if plan.visual_review_needed:

        print(
            "\nVisual checks:"
        )

        for indicator in plan.visual_indicators:

            print(
                f"  + {indicator}"
            )

            