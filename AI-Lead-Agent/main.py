from gemini_llm import ensure_ready, GeminiModelError, GEMINI_MODEL
import os
import time

from dotenv import load_dotenv

load_dotenv()

from langgraph.graph import (
    StateGraph,
    START,
    END,
)

from schemas import AgentState

from clarify import prepare_request

from planner import (
    make_research_plan,
    show_research_plan,
    plan_cache_key,
)

from discovery import (
    search_companies,
    MAX_ROUNDS,
    deadline_passed,
    max_run_seconds,
)

from qualify import research_companies

from exporter import export_run

import places
import trustpilot_reviews
import result_cache

from timing_utils import (
    start_timer,
    elapsed,
)


# =========================================================
# SHOW FINAL RESULTS
# =========================================================

def show_results(
    state: AgentState,
):
    from gemini_llm import unavailable_reason
    if unavailable_reason():
        state["discovery_stop_reason"] = unavailable_reason()
    elif not state.get("discovery_stop_reason") and len(state["qualified_leads"]) < state["number_of_leads"]:
        if deadline_passed(state):
            state["discovery_stop_reason"] = "time_budget_exhausted"
        elif state["round"] >= MAX_ROUNDS:
            state["discovery_stop_reason"] = "search_round_budget_exhausted"

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

    stats = state.get(
        "research_stats",
        {}
    )

    print(
        "\nOutcome breakdown:"
    )

    print(
        f"  Directory/non-business filtered : "
        f"{stats.get('directory_filtered', 0) + stats.get('prefiltered', 0)}"
    )

    print(
        f"  Unreachable after fallback       : "
        f"{stats.get('unreachable', 0)}"
    )

    print(
        f"  Disqualified by fit/location     : "
        f"{stats.get('disqualified', 0)}"
    )

    print(
        f"  Insufficient verified evidence   : "
        f"{stats.get('insufficient_evidence', 0)}"
    )

    print(
        f"  Gemini/API failures              : "
        f"{stats.get('api_failures', 0)}"
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
            "\n" + "-" * 70
        )

        print(
            f"LEAD #{index}"
        )

        print(
            "-" * 70
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

        if lead.get("evidence_strength"):
            print(
                f"Strength: "
                f"{lead['evidence_strength']}"
                + (
                    " (gap shown by screenshots only)"
                    if lead["evidence_strength"] == "limited"
                    else ""
                )
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
        # VISUAL REVIEW
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

                if evidence.get("source_url"):
                    print(f"    URL: {evidence['source_url']}")
                print(
                    f"    Evidence: "
                    f"{evidence['evidence']}"
                )

        else:

            print(
                "Evidence: "
                "No individually verified evidence returned."
            )

        review_block = lead.get(
            "review_evidence"
        )

        if review_block and review_block.get("mentions"):

            print(
                "\nGoogle Maps reviews that mention slow or missing "
                "replies "
                f"({len(review_block['mentions'])} of "
                f"{review_block['checked']} checked; "
                "Google returns at most 5 per business):"
            )

            for mention in review_block["mentions"]:

                if mention.get("quote"):

                    rating = (
                        f"{mention['rating']}/5, "
                        if mention.get("rating")
                        else ""
                    )

                    print(
                        f"  - \"{mention['quote']}\""
                    )

                    print(
                        f"    {rating}{mention.get('relative_time', '')}, "
                        f"by {mention.get('author') or 'a Google user'}"
                    )

                else:

                    # Results loaded from the saved-results cache keep the
                    # link only: review text is never stored.
                    print(
                        "  - (review text is not kept in saved results)"
                    )

                if mention.get("maps_uri"):
                    print(
                        f"    View on Google Maps: {mention['maps_uri']}"
                    )

            print(
                "    Source: Google Maps"
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

        if deadline_passed(state):

            print(
                f"It stopped at the {max_run_seconds()}s research time "
                "limit. Set MAX_RUN_SECONDS in .env to search longer "
                "(0 = no limit)."
            )

        reasons = []

        if stats.get(
            "unreachable",
            0
        ):
            reasons.append(
                f"{stats['unreachable']} candidate(s) remained unreachable"
            )

        if stats.get(
            "insufficient_evidence",
            0
        ):
            reasons.append(
                f"{stats['insufficient_evidence']} lacked enough verified evidence"
            )

        if stats.get(
            "disqualified",
            0
        ):
            reasons.append(
                f"{stats['disqualified']} failed fit/location checks"
            )

        if stats.get(
            "api_failures",
            0
        ):
            reasons.append(
                f"{stats['api_failures']} could not be judged because of API failures"
            )

        if reasons:

            print(
                "Main reasons: "
                + "; ".join(
                    reasons
                )
                + "."
            )

        print(
            "I kept the criteria strict rather than "
            "filling the result with unsupported leads."
        )

        if state.get("discovery_stop_reason"):
            print(f"Research stopped: {state['discovery_stop_reason']}")
        if stats.get("external_access_incomplete", 0):
            print(f"External-source access was incomplete for {stats['external_access_incomplete']} businesses. "
                  "Review criteria remain unverified; this is not evidence that complaints do not exist.")
        if stats.get("api_failures", 0):
            print("API failures prevented qualification. Check the audit errors and provider quota before changing search criteria.")
        elif stats.get("external_access_incomplete", 0):
            print("Public review sources could not be fully read. Retry later or allow more research time; the review requirements were preserved.")
        elif stats.get("unreachable", 0):
            print("Website access prevented verification. Check the audit and browser setup before changing your criteria.")
        else:
            print("To widen the search, broaden the location/industry or revise one criterion.")

    else:

        print(
            "\nRequested number of "
            "qualified leads found."
        )

    return {"discovery_stop_reason": state.get("discovery_stop_reason", "")}


# =========================================================
# DECIDE WHETHER TO SEARCH AGAIN
# =========================================================

def route_after_research(
    state: AgentState,
):
    from gemini_llm import unavailable_reason
    if unavailable_reason():
        state["discovery_stop_reason"] = unavailable_reason()
        return "show_results"

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

    if state.get("query_pool_exhausted"):
        return "show_results"

    # Maximum search rounds reached
    if (
        current_round
        >= MAX_ROUNDS
    ):

        return "show_results"

    # Research time limit reached (MAX_RUN_SECONDS)
    if deadline_passed(state):

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
# Search -> Research
# -------------------------

graph_builder.add_edge(
    "search_companies",
    "research_companies",
)

# -------------------------
# Research -> Search again
# OR
# Research -> Results
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

def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description="Find and verify business leads using Gemini and public web sources.")
    parser.add_argument("--request", help="Lead request; missing details are still clarified interactively.")
    parser.add_argument("--refresh", action="store_true", help="Ignore saved results and research again.")
    parser.add_argument("--max-seconds", type=int, help="Research time budget; 0 disables the time limit.")
    args = parser.parse_args(argv)
    if args.max_seconds is not None:
        if args.max_seconds < 0:
            parser.error("--max-seconds must be zero or positive")
        os.environ["MAX_RUN_SECONDS"] = str(args.max_seconds)
    if args.refresh:
        os.environ["REFRESH"] = "1"

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

    user_request = args.request or input(
        "\nWhat companies do you want to find?"
        "\n\n> "
    )

    # Do not count the time the user spends typing.
    from gemini_llm import reset_run
    reset_run()
    places.reset_run()
    trustpilot_reviews.reset_run()

    total_start = start_timer()

    fast_mode = os.getenv("FAST_MODE") == "1"


    def show_cached(cached, source):
        state, age = cached

        print(
            "\nResults from an earlier run "
            f"({result_cache.describe_age(age)}). "
            "Set REFRESH=1 in .env to research again."
        )

        show_results(state)

        print(
            f"\nAnswered from saved results ({source}) in "
            f"{elapsed(total_start):.2f}s. No search, browser or Gemini "
            "call was made."
        )

        raise SystemExit


    cached = result_cache.load(
        result_cache.message_key(
            user_request,
            fast_mode,
        )
    )

    if cached:
        show_cached(
            cached,
            "same request text",
        )


    try:
        ensure_ready()
    except GeminiModelError as error:
        print(f"\n{error}\nNo lead research was started.")
        raise SystemExit(1) from None

    print(f"\nGemini model: {GEMINI_MODEL}")

    print(
        "\nUnderstanding your request..."
    )

    request_start = start_timer()

    # Intent routing -> slot extraction -> criteria validation
    # -> slot clarification. Research cannot begin until this passes.
    request = prepare_request(
        user_request
    )

    if request is None:

        print(
            "\nNo lead research was started."
        )

        raise SystemExit

    request_duration = elapsed(
        request_start
    )

    print(
        f"\n[TIMER] Request understanding: "
        f"{request_duration:.2f}s"
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

    if request.criteria:

        print(
            "Criteria :"
        )

        for criterion in request.criteria:

            print(
                f"  - {criterion}"
            )


    # =========================================================
    # DYNAMIC RESEARCH PLANNER
    # =========================================================

    cached = result_cache.load(
        result_cache.request_key(
            plan_cache_key(
                request
            ),
            fast_mode,
        ),
        requested=request.number_of_leads,
    )

    if cached:
        show_cached(
            cached,
            "same request, different wording",
        )

    print(
        "\nCreating research strategy..."
    )

    plan_start = start_timer()

    try:
        research_plan = make_research_plan(request)
    except GeminiModelError as error:
        print(f"\nCould not create a research plan: {error}\nNo lead research was started.")
        raise SystemExit(1) from None

    if fast_mode and research_plan.visual_review_needed:

        research_plan.visual_review_needed = False

        print(
            "\nFAST_MODE: screenshots skipped (website text and measured "
            "facts only)."
        )

    plan_dict = research_plan.model_dump()
    plan_dict["user_criteria"] = list(request.criteria)

    # Country of the search location, used to validate phone numbers.
    plan_dict["country_code"] = request.country_code

    # Criteria that can only be checked from Google Maps reviews. Only present
    # when GOOGLE_PLACES_API_KEY is set (otherwise the validator already
    # removed them before searching).
    plan_dict["review_criteria"] = (
        places.review_criteria(request.criteria)
        if places.is_configured()
        else []
    )

    plan_duration = elapsed(
        plan_start
    )

    show_research_plan(
        research_plan
    )

    print(
        f"\n[TIMER] Research planning: "
        f"{plan_duration:.2f}s"
    )


    # =========================================================
    # INITIAL LANGGRAPH STATE
    # =========================================================

    initial_state: AgentState = {

        # Research stops after the round in which this moment passes
        # (MAX_RUN_SECONDS, default 300; 0 = no limit).
        "deadline":
            (
                time.time() + max_run_seconds()
                if max_run_seconds()
                else 0
            ),


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
            plan_dict,

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

        # Search queries already run. Prevents repeated queries across rounds.
        "seen_queries":
            [],

        # Queries that hit a search-backend error (retried once, then dropped).
        "failed_queries":
            [],

        # Towns/cities inside a broad location. None = not expanded yet.
        "geo_terms":
            None,

        # Pipeline outcome counters used for diagnostics/shortfall reporting.
        "research_stats":
            {
                "directory_filtered": 0,
                "search_failures": 0,
                "unreachable": 0,
                "prefiltered": 0,
                "disqualified": 0,
                "insufficient_evidence": 0,
                "api_failures": 0,
                "qualified": 0,
            },

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

    agent_start = start_timer()
    exit_code = 0

    try:

        final_state = dict(initial_state)
        for update in graph.stream(initial_state, config={"recursion_limit": 2 * MAX_ROUNDS + 5}):
            for changes in update.values():
                final_state.update(changes)

    except KeyboardInterrupt:
        exit_code = 130

        print(
            "\n\nAgent stopped by user."
        )

        final_state["discovery_stop_reason"] = "interrupted_by_user"
        show_results(final_state)

    except Exception as error:
        exit_code = 1

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

        final_state["discovery_stop_reason"] = f"workflow_error: {type(error).__name__}: {error}"
        show_results(final_state)


    agent_duration = elapsed(
        agent_start
    )

    total_duration = elapsed(
        total_start
    )


    # =========================================================
    # PERFORMANCE SUMMARY
    # =========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "PERFORMANCE SUMMARY"
    )

    print(
        "=" * 70
    )

    print(
        f"\nRequest understanding : "
        f"{request_duration:.2f}s"
    )

    print(
        f"Research planning     : "
        f"{plan_duration:.2f}s"
    )

    print(
        f"Lead research graph   : "
        f"{agent_duration:.2f}s"
    )

    print(
        f"TOTAL PROGRAM TIME    : "
        f"{total_duration:.2f}s"
    )

    print(
        "=" * 70
    )


    # =========================================================
    # SAVE RESULTS (CSV + JSON + audit report)
    # =========================================================

    try:

        run_dir = export_run(
            request,
            plan_dict,
            final_state,
            {
                "request_understanding": round(request_duration, 2),
                "research_planning": round(plan_duration, 2),
                "lead_research": round(agent_duration, 2),
                "total": round(total_duration, 2),
            },
        )

        print(
            f"\nSaved: {run_dir / 'leads.csv'}"
        )
        print(
            f"Audit report: {run_dir / 'report.json'}"
        )

    except Exception as error:

        print(
            f"\nCould not save results to disk: {error}"
        )


    # =========================================================
    # SAVE FOR INSTANT REPEATS
    # =========================================================

    try:

        result_cache.save(
            [
                result_cache.message_key(
                    user_request,
                    fast_mode,
                ),
                result_cache.request_key(
                    plan_cache_key(
                        request
                    ),
                    fast_mode,
                ),
            ],
            request.number_of_leads,
            final_state,
            fast_mode,
        )

    except Exception:

        pass


    # =========================================================
    # GOOGLE PLACES USAGE
    # =========================================================

    if places.is_configured():

        print(
            "\nGoogle Places requests this run: "
            f"{places.requests_made()} "
            f"(cap {places.max_requests()}). Each request is billed; see "
            "README for pricing."
        )

        if places.disabled_reason() and plan_dict.get("review_criteria"):

            print(
                "WARNING: Google Places stopped working during this run "
                f"({places.disabled_reason()}). DDGS external research remains "
                "available; only verified accessible evidence can qualify leads."
            )


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
    return exit_code


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (EOFError, KeyboardInterrupt):
        print("\nRequest cancelled.")
