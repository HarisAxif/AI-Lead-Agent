from dotenv import load_dotenv

load_dotenv()


from langgraph.graph import (
    StateGraph,
    START,
    END,
)

from schemas import AgentState

from clarify import (
    understand_request,
    clarify_request,
)

from planner import (
    make_research_plan,
    show_research_plan,
)

from discovery import (
    search_companies,
    MAX_ROUNDS,
)

from qualify import research_companies


# =========================================================
# SHOW FINAL RESULTS
# =========================================================

def show_results(
    state: AgentState,
):

    qualified = state[
        "qualified_leads"
    ]

    rejected = state[
        "rejected_leads"
    ]

    print(
        "\n\n" + "=" * 70
    )

    print(
        "AI LEAD GENERATION RESULTS"
    )

    print(
        "=" * 70
    )

    print(
        f"\nRequested Leads : "
        f"{state['number_of_leads']}"
    )

    print(
        f"Qualified Leads : "
        f"{len(qualified)}"
    )

    print(
        f"Rejected        : "
        f"{len(rejected)}"
    )

    print(
        f"Search Rounds   : "
        f"{state['round']}"
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "QUALIFIED LEADS"
    )

    print(
        "=" * 70
    )

    if not qualified:

        print(
            "\nNo qualified leads were found."
        )

    for index, lead in enumerate(
        qualified,
        start=1,
    ):

        print(
            "\n" + "─" * 70
        )

        print(
            f"LEAD #{index}"
        )

        print(
            "─" * 70
        )

        print(
            f"Company : "
            f"{lead['company']}"
        )

        print(
            f"Website : "
            f"{lead['website']}"
        )

        print(
            f"Tier    : "
            f"{lead['tier']}"
        )

        # ---------------------------------------------
        # OFFICE LOCATIONS
        # ---------------------------------------------

        if lead.get(
            "offices"
        ):

            print(
                "Offices : "
                + ", ".join(
                    lead[
                        "offices"
                    ]
                )
            )

        # ---------------------------------------------
        # QUALIFICATION REASON
        # ---------------------------------------------

        print(
            f"Reason  : "
            f"{lead['reason']}"
        )

        # ---------------------------------------------
        # NEW: VISUAL REVIEW
        # ---------------------------------------------

        if lead.get(
            "visual_summary"
        ):

            print(
                f"Visual  : "
                f"{lead['visual_summary']}"
            )

        # ---------------------------------------------
        # SALES OPPORTUNITY
        # ---------------------------------------------

        print(
            f"Pitch   : "
            f"{lead['opportunity']}"
        )

        # ---------------------------------------------
        # EMAILS
        # ---------------------------------------------

        if lead.get(
            "emails"
        ):

            print(
                "Emails  : "
                + ", ".join(
                    lead[
                        "emails"
                    ]
                )
            )

        # ---------------------------------------------
        # PHONE NUMBERS
        # ---------------------------------------------

        if lead.get(
            "phones"
        ):

            print(
                "Phones  : "
                + ", ".join(
                    lead[
                        "phones"
                    ]
                )
            )

        # ---------------------------------------------
        # PAGES INSPECTED
        # ---------------------------------------------

        if lead.get(
            "pages_read"
        ):

            print(
                "Pages inspected:"
            )

            for page in lead[
                "pages_read"
            ]:

                print(
                    f"  - {page}"
                )

        # ---------------------------------------------
        # VERIFIED EVIDENCE
        # ---------------------------------------------

        if lead.get(
            "evidence"
        ):

            print(
                "Evidence:"
            )

            for evidence in lead[
                "evidence"
            ]:

                print(
                    f"  - "
                    f"{evidence['indicator']}"
                )

                print(
                    f"    Source: "
                    f"{evidence['source']}"
                )

                print(
                    f"    Evidence: "
                    f"{evidence['evidence']}"
                )

        else:

            print(
                "Evidence: "
                "No individually verified evidence returned."
            )

    # =================================================
    # SEARCH SUMMARY
    # =================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "SEARCH COMPLETE"
    )

    print(
        "=" * 70
    )

    if (
        len(
            qualified
        )
        < state[
            "number_of_leads"
        ]
    ):

        print(
            "\nThe agent did not find the full "
            "requested number of qualified leads."
        )

        print(
            "This is intentional: it is better to "
            "return fewer leads than fill the result "
            "with weak or unsupported candidates."
        )

    else:

        print(
            "\nRequested number of "
            "qualified leads found."
        )

    return {}


# =========================================================
# DECIDE WHETHER TO SEARCH AGAIN
# =========================================================

def route_after_research(
    state: AgentState,
):

    qualified_count = len(
        state[
            "qualified_leads"
        ]
    )

    requested_count = state[
        "number_of_leads"
    ]

    current_round = state[
        "round"
    ]

    # Enough good leads found
    if (
        qualified_count
        >= requested_count
    ):

        return "show_results"

    # Maximum search rounds reached
    if (
        current_round
        >= MAX_ROUNDS
    ):

        return "show_results"

    # Otherwise search again
    return "search_companies"


# =========================================================
# BUILD LANGGRAPH
# =========================================================

graph_builder = StateGraph(
    AgentState
)


# -------------------------
# Nodes
# -------------------------

graph_builder.add_node(
    "search_companies",
    search_companies,
)

graph_builder.add_node(
    "research_companies",
    research_companies,
)

graph_builder.add_node(
    "show_results",
    show_results,
)


# -------------------------
# Start
# -------------------------

graph_builder.add_edge(
    START,
    "search_companies",
)


# -------------------------
# Search → Research
# -------------------------

graph_builder.add_edge(
    "search_companies",
    "research_companies",
)


# -------------------------
# Research → Search again
# OR
# Research → Results
# -------------------------

graph_builder.add_conditional_edges(
    "research_companies",
    route_after_research,
    {
        "search_companies":
            "search_companies",

        "show_results":
            "show_results",
    },
)


# -------------------------
# Finish
# -------------------------

graph_builder.add_edge(
    "show_results",
    END,
)


# Compile graph
graph = graph_builder.compile()


# =========================================================
# PROGRAM START
# =========================================================

print(
    "\n" + "=" * 70
)

print(
    "AI LEAD GENERATION AGENT"
)

print(
    "=" * 70
)


# =========================================================
# USER REQUEST
# =========================================================

user_request = input(
    "\nWhat companies do you want to find?"
    "\n\n> "
)


print(
    "\nUnderstanding your request..."
)


# First extraction
request = understand_request(
    user_request
)


# Clarify anything missing / broad
request = clarify_request(
    request
)


# =========================================================
# FINAL USER SEARCH REQUIREMENTS
# =========================================================

print(
    "\n" + "=" * 70
)

print(
    "SEARCH PLAN"
)

print(
    "=" * 70
)


print(
    f"\nLocation : "
    f"{request.location}"
)

print(
    f"Industry : "
    f"{request.industry}"
)

print(
    f"Leads    : "
    f"{request.number_of_leads}"
)

print(
    f"Service  : "
    f"{request.service}"
)


# =========================================================
# DYNAMIC RESEARCH PLANNER
# =========================================================

print(
    "\nCreating research strategy..."
)


research_plan = make_research_plan(
    request
)


show_research_plan(
    research_plan
)


# =========================================================
# INITIAL LANGGRAPH STATE
# =========================================================

initial_state: AgentState = {

    "location":
        request.location,

    "industry":
        request.industry,

    "number_of_leads":
        request.number_of_leads,

    "service":
        request.service,

    # Dynamic Gemini research plan
    "plan":
        research_plan.model_dump(),

    # Candidates found by DDGS in current round
    "companies":
        [],

    # Final qualified leads
    "qualified_leads":
        [],

    # Rejected candidates
    "rejected_leads":
        [],

    # Domains already checked
    "seen_domains":
        [],

    # Search round counter
    "round":
        0,
}


# =========================================================
# RUN AGENT
# =========================================================

print(
    "\n" + "=" * 70
)

print(
    "STARTING LEAD RESEARCH"
)

print(
    "=" * 70
)


if research_plan.visual_review_needed:

    print(
        "\nVisual website inspection: ENABLED"
    )

    print(
        "Chromium will render candidate websites "
        "and Gemini will receive desktop/mobile views."
    )

else:

    print(
        "\nVisual website inspection: NOT REQUIRED"
    )

    print(
        "The research planner determined that visual "
        "inspection does not materially help this service."
    )


print(
    "\nStarting agent..."
)


try:

    final_state = graph.invoke(
        initial_state
    )

except KeyboardInterrupt:

    print(
        "\n\nAgent stopped by user."
    )

    raise SystemExit


except Exception as error:

    print(
        "\n" + "=" * 70
    )

    print(
        "AGENT ERROR"
    )

    print(
        "=" * 70
    )

    print(
        f"\n{type(error).__name__}: "
        f"{error}"
    )

    print(
        "\nThe program stopped before "
        "the research workflow completed."
    )

    raise


# =========================================================
# FINISHED
# =========================================================

print(
    "\n" + "=" * 70
)

print(
    "AGENT FINISHED"
)

print(
    "=" * 70
)