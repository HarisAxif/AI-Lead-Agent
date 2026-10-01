import time
from typing import TypedDict

from dotenv import load_dotenv
from ddgs import DDGS
from pydantic import BaseModel
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()


llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=0
)


class SearchRequest(BaseModel):
    location: str
    industry: str
    number_of_leads: int
    service: str


class LeadAnalysis(BaseModel):
    company: str
    website: str
    qualified: bool
    reason: str
    opportunity: str


request_llm = llm.with_structured_output(
    SearchRequest
)

structured_llm = llm.with_structured_output(
    LeadAnalysis
)


class AgentState(TypedDict):
    location: str
    industry: str
    number_of_leads: int
    service: str
    companies: list
    researched_companies: list


def search_companies(state: AgentState):

    print("\n" + "=" * 70)
    print("SEARCHING FOR COMPANIES")
    print("=" * 70)

    location = state["location"]
    industry = state["industry"]
    number_of_leads = state["number_of_leads"]

    query = f"{industry} in {location}"

    print(f"\nSearch Query : {query}")
    print(f"Target       : {number_of_leads} results")

    companies = []

    try:

        with DDGS() as ddgs:

            results = ddgs.text(
                query,
                max_results=number_of_leads
            )

            for result in results:

                company = {
                    "title": result.get("title", ""),
                    "url": result.get("href", ""),
                    "description": result.get("body", "")
                }

                companies.append(company)

    except Exception as error:

        print(f"\nDDGS Search Error: {error}")

    print(
        f"\nFound {len(companies)} search results."
    )

    for index, company in enumerate(
        companies,
        start=1
    ):

        print(
            f"{index}. {company['title']}"
        )

    return {
        "companies": companies
    }


def research_companies(state: AgentState):

    print("\n" + "=" * 70)
    print("RESEARCHING & QUALIFYING COMPANIES")
    print("=" * 70)

    companies = state["companies"]
    service = state["service"]

    researched_companies = []

    for index, company in enumerate(
        companies,
        start=1
    ):

        print(
            f"\n[{index}/{len(companies)}] "
            f"Checking: {company['title']}"
        )

        prompt = f"""
You are a B2B lead qualification assistant.

We are selling this service:

{service}

COMPANY INFORMATION

Company / Search Result:
{company["title"]}

Website:
{company["url"]}

Search Description:
{company["description"]}

YOUR TASK

Determine whether this company could realistically
be a potential customer for our service.

A qualified company should have a reasonable
business use for the service.

Do not invent information that is not available
in the provided company information.

Return:

company:
The company name.

website:
The website.

qualified:
true if this looks like a potential buyer.
false if it does not.

reason:
Briefly explain why it qualifies or does not qualify.

opportunity:
If qualified, briefly explain how our service could
potentially help this company.

If it is not qualified, explain that there is no
clear opportunity based on the available information.
"""

        max_attempts = 3

        for attempt in range(max_attempts):

            try:

                analysis = structured_llm.invoke(
                    prompt
                )

                researched_companies.append(
                    analysis.model_dump()
                )

                if analysis.qualified:
                    print("    Status: QUALIFIED")
                else:
                    print("    Status: NOT QUALIFIED")

                break

            except Exception as error:

                print(
                    f"    Gemini Error "
                    f"({attempt + 1}/{max_attempts})"
                )

                print(f"    {error}")

                if attempt < max_attempts - 1:

                    print(
                        "    Waiting 5 seconds before retry..."
                    )

                    time.sleep(5)

                else:

                    print(
                        "    Could not analyze this company."
                    )

                    print("    Skipping...")

    return {
        "researched_companies":
            researched_companies
    }


def show_results(state: AgentState):

    results = state["researched_companies"]

    qualified = [
        company
        for company in results
        if company["qualified"]
    ]

    rejected = [
        company
        for company in results
        if not company["qualified"]
    ]

    print("\n\n" + "=" * 70)
    print("AI LEAD GENERATION RESULTS")
    print("=" * 70)

    print(
        f"\nCompanies Researched : {len(results)}"
    )

    print(
        f"Qualified Leads      : {len(qualified)}"
    )

    print(
        f"Not Qualified        : {len(rejected)}"
    )

    print("\n" + "=" * 70)
    print("QUALIFIED LEADS")
    print("=" * 70)

    if not qualified:
        print("\nNo qualified leads were found.")

    for index, lead in enumerate(
        qualified,
        start=1
    ):

        print("\n" + "─" * 70)
        print(f"LEAD #{index}")
        print("─" * 70)

        print(
            f"Company     : {lead['company']}"
        )

        print(
            f"Website     : {lead['website']}"
        )

        print(
            "Status      : QUALIFIED"
        )

        print(
            f"Reason      : {lead['reason']}"
        )

        print(
            f"Opportunity : {lead['opportunity']}"
        )

    if rejected:

        print("\n\n" + "=" * 70)
        print("NOT QUALIFIED")
        print("=" * 70)

        for index, lead in enumerate(
            rejected,
            start=1
        ):

            print("\n" + "─" * 70)

            print(
                f"{index}. {lead['company']}"
            )

            print(
                f"Website : {lead['website']}"
            )

            print(
                f"Reason  : {lead['reason']}"
            )

    print("\n" + "=" * 70)
    print("SEARCH COMPLETE")
    print("=" * 70)

    return {}


graph_builder = StateGraph(
    AgentState
)


graph_builder.add_node(
    "search_companies",
    search_companies
)

graph_builder.add_node(
    "research_companies",
    research_companies
)

graph_builder.add_node(
    "show_results",
    show_results
)


graph_builder.add_edge(
    START,
    "search_companies"
)

graph_builder.add_edge(
    "search_companies",
    "research_companies"
)

graph_builder.add_edge(
    "research_companies",
    "show_results"
)

graph_builder.add_edge(
    "show_results",
    END
)


graph = graph_builder.compile()


print("\n" + "=" * 70)
print("AI LEAD GENERATION AGENT")
print("=" * 70)

user_request = input(
    "\nWhat companies do you want to find?\n\n> "
)


print("\nUnderstanding your request...")


request = request_llm.invoke(
    user_request
)


initial_state = {
    "location": request.location,
    "industry": request.industry,
    "number_of_leads": request.number_of_leads,
    "service": request.service,
    "companies": [],
    "researched_companies": []
}


print("\n" + "=" * 70)
print("SEARCH PLAN")
print("=" * 70)

print(
    f"\nLocation : {initial_state['location']}"
)

print(
    f"Industry : {initial_state['industry']}"
)

print(
    f"Leads    : {initial_state['number_of_leads']}"
)

print(
    f"Service  : {initial_state['service']}"
)

print("\nStarting agent...")


final_state = graph.invoke(
    initial_state
)


print("\n" + "=" * 70)
print("AGENT FINISHED")
print("=" * 70)