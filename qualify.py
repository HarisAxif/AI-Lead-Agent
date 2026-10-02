import os
import re
import time

from google import genai

from schemas import (
    AgentState,
    LeadAnalysis
)

from website import inspect_all


GEMINI_MODEL = (
    "gemini-3.1-flash-lite"
)


api_key = (
    os.getenv(
        "GEMINI_API_KEY"
    )
    or os.getenv(
        "GOOGLE_API_KEY"
    )
)


client = genai.Client(
    api_key=api_key
)


def normalize_text(
    value: str
):

    return re.sub(
        r"\s+",
        " ",
        value.lower()
    ).strip()


def verify_findings(
    findings,
    site
):

    page_text = normalize_text(
        site[
            "text"
        ]
    )

    footprint = normalize_text(
        site[
            "footprint"
        ]
    )

    verified = []

    for finding in findings:

        if not finding.present:
            continue

        if (
            finding.evidence_type
            == "text_quote"
            and finding.quote
        ):

            quote = normalize_text(
                finding.quote
            )

            if quote in page_text:

                verified.append(
                    {
                        "indicator":
                            finding.indicator,

                        "evidence":
                            finding.quote,

                        "source":
                            "website_text"
                    }
                )

        elif (
            finding.evidence_type
            == "technical_fact"
            and finding.footprint_fact
        ):

            fact = normalize_text(
                finding.footprint_fact
            )

            if fact in footprint:

                verified.append(
                    {
                        "indicator":
                            finding.indicator,

                        "evidence":
                            finding.footprint_fact,

                        "source":
                            "technical_fact"
                    }
                )

        elif (
            finding.evidence_type
            == "visual_observation"
            and finding.visual_observation
            and (
                site.get(
                    "desktop_screenshot"
                )
                or site.get(
                    "mobile_screenshot"
                )
            )
        ):

            verified.append(
                {
                    "indicator":
                        finding.indicator,

                    "evidence":
                        finding.visual_observation,

                    "source":
                        "visual_review"
                }
            )

    return verified


def judge_company(
    company: dict,
    site: dict,
    state: AgentState
):

    plan = state[
        "plan"
    ]

    indicators = "\n".join(
        f"- {indicator}"
        for indicator
        in plan[
            "need_indicators"
        ]
    )

    visual_indicators = "\n".join(
        f"- {indicator}"
        for indicator
        in plan.get(
            "visual_indicators",
            []
        )
    )

    disqualifiers = "\n".join(
        f"- {disqualifier}"
        for disqualifier
        in plan[
            "disqualifiers"
        ]
    )

    prompt = f"""
You are performing evidence-based B2B lead research.

You have access to:

1. Search discovery information
2. Text extracted directly from the company's own website
3. Technical and rendered DOM facts measured by code
4. Desktop and mobile screenshots when visual review is enabled

SERVICE BEING SOLD:
{state["service"]}

TARGET INDUSTRY:
{state["industry"]}

TARGET LOCATION:
{state["location"]}

IDEAL BUYER:
{plan["ideal_buyer"]}


DISCOVERED COMPANY

Search title:
{company["title"]}

Actual website:
{company["url"]}

Search snippet:
{company["snippet"]}


NEED INDICATORS

{indicators}


VISUAL INDICATORS

{visual_indicators}


DISQUALIFIERS

{disqualifiers}


TECHNICAL AND RENDERED FACTS

{site["footprint"]}


WEBSITE TEXT

{site["text"]}


STRICT RULES

1. Judge the ACTUAL company website.

2. is_single_business_site=true only when this is one
real business's own website.

Directories, listicles, marketplaces, generic blogs,
aggregators and directory profiles are false.

3. office_locations must contain only locations
supported by website evidence.

4. in_target_location=true only when the business
actually appears to operate in the requested location.

5. matches_ideal_buyer must be based on the actual
business shown.

6. hit_disqualifier=true when a planned disqualifier
is genuinely supported.

7. Return one finding for every need indicator.

8. Use evidence_type="text_quote" only when an exact
short quote from WEBSITE TEXT proves the indicator.

9. Use evidence_type="technical_fact" only when an
exact line from TECHNICAL AND RENDERED FACTS supports
the indicator.

10. Use evidence_type="visual_observation" only when
the supplied screenshots visibly support the claim.

11. A fact that merely exists must LOGICALLY support
the indicator.

For example:

html_forms_on_homepage: 0

DOES NOT prove:

"No booking system"
"No CTA"
"Poor conversion"

because booking can happen using buttons, links,
WhatsApp, phone calls or external systems.

12. Likewise:

html_mobile_viewport_tag: True

DOES NOT prove the site is mobile responsive.

Actual visual layout can only be judged from the
mobile screenshot.

13. Never interpret "not detected" as proof that
something does not exist.

14. Check the rendered clickable elements before
claiming that a website lacks a call-to-action.

If "Book Appointment", "Call", "Contact", "WhatsApp",
"Start Chat", or a similar CTA is present, do not claim
that no CTA exists.

15. Screenshots override incorrect assumptions made
from raw HTML structure.

16. Do not invent internal business problems.

17. Do not use unsupported phrases such as:

"they likely receive..."
"they probably need..."
"all businesses need..."

18. For visual claims such as:

outdated design
poor spacing
weak hierarchy
broken mobile layout
unprofessional imagery

use the screenshot itself.

19. A modern, professionally presented website should
NOT be qualified for rebuild merely because it lacks
an HTML form.

20. visual_summary should briefly describe what the
desktop/mobile website actually looks like.

21. The final reason must be supported by the evidence
you were given.

22. The opportunity should map the service to the
verified problem rather than inventing a problem.
"""

    inputs = [
        {
            "type": "text",
            "text": prompt
        }
    ]

    desktop = site.get(
        "desktop_screenshot"
    )

    mobile = site.get(
        "mobile_screenshot"
    )

    if desktop:

        inputs.append(
            {
                "type": "image",
                "data": desktop,
                "mime_type":
                    "image/jpeg"
            }
        )

    if mobile:

        inputs.append(
            {
                "type": "image",
                "data": mobile,
                "mime_type":
                    "image/jpeg"
            }
        )

    interaction = (
        client.interactions.create(
            model=GEMINI_MODEL,

            input=inputs,

            response_format={
                "type": "text",

                "mime_type":
                    "application/json",

                "schema":
                    LeadAnalysis.model_json_schema()
            }
        )
    )

    return LeadAnalysis.model_validate_json(
        interaction.output_text
    )


def research_companies(
    state: AgentState
):

    companies = state[
        "companies"
    ]

    if not companies:

        print(
            "\nNo candidates found in this round."
        )

        return {}

    print(
        "\n" + "=" * 70
    )

    print(
        "INSPECTING ACTUAL WEBSITES"
    )

    print(
        "=" * 70
    )

    site_results = inspect_all(
        companies,
        state[
            "plan"
        ]
    )

    qualified = list(
        state[
            "qualified_leads"
        ]
    )

    rejected = list(
        state[
            "rejected_leads"
        ]
    )

    target = state[
        "number_of_leads"
    ]

    evidence_observable = state[
        "plan"
    ][
        "evidence_observable"
    ]

    for company, site in zip(
        companies,
        site_results
    ):

        if len(
            qualified
        ) >= target:
            break

        print(
            f"\nChecking: "
            f"{company['domain']}"
        )

        if not site[
            "reachable"
        ]:

            print(
                "    SKIPPED - website unreachable"
            )

            rejected.append(
                {
                    "company":
                        company["title"],

                    "website":
                        company["url"],

                    "tier":
                        "unverified",

                    "reason":
                        "Website could not be fetched.",

                    "opportunity":
                        "",

                    "visual_summary":
                        "",

                    "evidence":
                        [],

                    "emails":
                        [],

                    "phones":
                        []
                }
            )

            continue

        try:

            analysis = judge_company(
                company,
                site,
                state
            )

        except Exception as error:

            print(
                f"    Gemini error: {error}"
            )

            print(
                "    Waiting 5 seconds..."
            )

            time.sleep(
                5
            )

            continue

        verified_evidence = (
            verify_findings(
                analysis.findings,
                site
            )
        )

        valid_candidate = (
            analysis.is_single_business_site
            and analysis.in_target_location
            and analysis.matches_ideal_buyer
            and not analysis.hit_disqualifier
        )

        if evidence_observable:

            qualified_result = (
                valid_candidate
                and len(
                    verified_evidence
                ) >= 2
            )

            tier = (
                "evidence-backed"
                if qualified_result
                else "rejected"
            )

        else:

            qualified_result = (
                valid_candidate
            )

            tier = (
                "fit-only"
                if qualified_result
                else "rejected"
            )

        record = {
            "company":
                analysis.company_name
                or company[
                    "title"
                ],

            "website":
                company[
                    "url"
                ],

            "tier":
                tier,

            "reason":
                analysis.reason,

            "opportunity":
                analysis.opportunity,

            "visual_summary":
                analysis.visual_summary,

            "offices":
                analysis.office_locations,

            "evidence":
                verified_evidence,

            "emails":
                site[
                    "emails"
                ],

            "phones":
                site[
                    "phones"
                ],

            "pages_read":
                site[
                    "pages_read"
                ]
        }

        if qualified_result:

            qualified.append(
                record
            )

            print(
                f"    QUALIFIED ({tier})"
            )

        else:

            rejected.append(
                record
            )

            print(
                "    NOT QUALIFIED"
            )

        time.sleep(
            1
        )

    return {
        "qualified_leads":
            qualified,

        "rejected_leads":
            rejected
    }