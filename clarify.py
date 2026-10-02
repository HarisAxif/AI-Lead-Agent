from langchain_google_genai import ChatGoogleGenerativeAI

from schemas import (
    SearchRequest,
    FieldValidation,
    ServiceSuggestions,
    IndustrySuggestions
)


llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=0
)


request_llm = llm.with_structured_output(
    SearchRequest
)

validation_llm = llm.with_structured_output(
    FieldValidation
)

service_suggestion_llm = llm.with_structured_output(
    ServiceSuggestions
)

industry_suggestion_llm = llm.with_structured_output(
    IndustrySuggestions
)


def understand_request(
    user_message: str,
    current_request: SearchRequest | None = None
):

    if current_request is None:
        current_request = SearchRequest()

    prompt = f"""
Extract lead-generation requirements from the user's latest message.

CURRENT KNOWN REQUIREMENTS

Location:
{current_request.location}

Industry:
{current_request.industry}

Number of Leads:
{current_request.number_of_leads}

Service:
{current_request.service}

LATEST USER MESSAGE

{user_message}

RULES

1. Preserve previously known values unless the user clearly changes them.

2. Extract only information the user actually provides or clearly implies.

3. Never invent missing information.

4. If a field is missing and there is no previous value,
return null for that field.

5. Do not judge whether service, industry, or location is too broad.
Another validator handles that.

6. Preserve broad phrases exactly enough for later validation.

For example:

"AI agents"
→ service = "AI agents"

"web services"
→ service = "web services"

"businesses"
→ industry = "businesses"

"Pakistan"
→ location = "Pakistan"

7. If the user gives a clear positive number,
extract it as an integer.

8. If quantity is vague, such as:
"a few"
"some"
"many"

return null for number_of_leads.
"""

    return request_llm.invoke(
        prompt
    )


def update_field(
    request: SearchRequest,
    field_name: str,
    value
):

    return request.model_copy(
        update={
            field_name: value
        }
    )


def validate_field(
    request: SearchRequest,
    field_name: str
):

    value = getattr(
        request,
        field_name
    )

    prompt = f"""
Validate ONE field of this B2B lead-generation request.

FIELD:
{field_name}

VALUE:
{value}

FULL CONTEXT

Service:
{request.service}

Industry:
{request.industry}

Location:
{request.location}

Number of leads:
{request.number_of_leads}

Return one status:

VALID
TOO_BROAD
UNCLEAR
LIKELY_TYPO

GENERAL RULE

The value should be specific enough to make a useful,
targeted lead search.

Do not demand unnecessary specificity.

SERVICE

A service is valid when the actual thing being sold is clear.

Broad categories like:

AI agents
AI services
web services
software services
marketing services
consulting
automation services
IT services

are usually TOO_BROAD.

Specific examples:

AI voice receptionist
AI appointment booking agent
website redesign
Shopify store development
penetration testing
SEO audit
Google Ads management
bookkeeping

These are examples only.

Apply the same reasoning to ANY field or service.

INDUSTRY

A usable business niche is valid.

Too broad examples:

businesses
companies
organizations

Specific enough examples:

law firms
dental clinics
restaurants
software companies
real estate agencies
logistics companies

LOCATION

Judge location using the complete request.

A country is NOT automatically invalid.

Example:

5 local law firms across an entire large country
may be too broad.

50 software companies across Germany
may be completely reasonable.

5 dentists across Europe
is too broad.

10 restaurants in Dubai
is reasonable.

TYPO

Use LIKELY_TYPO only when the value appears to be
a probable misspelling of a recognizable term.

If possible, provide the likely correction.

Use UNCLEAR when meaning cannot confidently be determined.

Keep the reason short.
"""

    return validation_llm.invoke(
        prompt
    )


def suggest_services(
    request: SearchRequest
):

    prompt = f"""
Suggest exactly 5 specific services the user could sell.

CURRENT SERVICE:
{request.service}

TARGET INDUSTRY:
{request.industry}

TARGET LOCATION:
{request.location}

Suggestions must describe concrete offers,
not broad categories.

If industry is known, tailor suggestions to that niche.

If industry is unknown, suggest specific versions
of the current service category.

Return short service names.
"""

    result = service_suggestion_llm.invoke(
        prompt
    )

    return result.services[:5]


def suggest_industries(
    request: SearchRequest
):

    prompt = f"""
Suggest exactly 5 specific business niches that could
realistically buy this service.

SERVICE:
{request.service}

LOCATION:
{request.location}

CURRENT INDUSTRY:
{request.industry}

Return specific niches, not categories like
businesses or companies.
"""

    result = industry_suggestion_llm.invoke(
        prompt
    )

    return result.industries[:5]


def handle_typo(
    request: SearchRequest,
    field_name: str,
    validation: FieldValidation
):

    correction = validation.suggested_correction

    if not correction:
        return request

    current_value = getattr(
        request,
        field_name
    )

    print(
        f"\nI may have misunderstood '{current_value}'."
    )

    print(
        f"Did you mean '{correction}'? (yes/no)"
    )

    answer = input(
        "\n> "
    ).strip().lower()

    if answer in [
        "yes",
        "y"
    ]:

        return update_field(
            request,
            field_name,
            correction
        )

    print(
        f"\nPlease enter the intended {field_name}."
    )

    answer = input(
        "\n> "
    )

    return understand_request(
        answer,
        request
    )


def resolve_service(
    request: SearchRequest
):

    failed_attempts = 0

    while True:

        if request.service is None:

            print(
                "\nWhat specific service do you want to sell?"
            )

            answer = input(
                "\n> "
            )

            request = understand_request(
                answer,
                request
            )

            continue

        validation = validate_field(
            request,
            "service"
        )

        if validation.status == "VALID":
            return request

        if validation.status == "LIKELY_TYPO":

            request = handle_typo(
                request,
                "service",
                validation
            )

            continue

        failed_attempts += 1

        print(
            "\nThe service needs clarification."
        )

        print(
            f"Reason: {validation.reason}"
        )

        suggestions = suggest_services(
            request
        )

        print(
            "\nHere are some more specific options:"
        )

        for index, service in enumerate(
            suggestions,
            start=1
        ):

            print(
                f"{index}. {service}"
            )

        if failed_attempts >= 3:

            print(
                "\nChoose a number, type your own service, "
                "or type 'skip'."
            )

        else:

            print(
                "\nChoose a number or type your own specific service."
            )

        answer = input(
            "\n> "
        ).strip()

        if (
            failed_attempts >= 3
            and answer.lower() == "skip"
        ):

            print(
                "\nContinuing with the current service."
            )

            return request

        if answer.isdigit():

            choice = int(
                answer
            )

            if 1 <= choice <= len(suggestions):

                request = update_field(
                    request,
                    "service",
                    suggestions[
                        choice - 1
                    ]
                )

                continue

        request = understand_request(
            answer,
            request
        )


def resolve_industry(
    request: SearchRequest
):

    failed_attempts = 0

    while True:

        if request.industry is None:

            print(
                "\nWhich type of businesses do you want to target?"
            )

            answer = input(
                "\n> "
            )

            request = understand_request(
                answer,
                request
            )

            continue

        validation = validate_field(
            request,
            "industry"
        )

        if validation.status == "VALID":
            return request

        if validation.status == "LIKELY_TYPO":

            request = handle_typo(
                request,
                "industry",
                validation
            )

            continue

        failed_attempts += 1

        print(
            "\nThe target industry needs clarification."
        )

        print(
            f"Reason: {validation.reason}"
        )

        suggestions = suggest_industries(
            request
        )

        print(
            "\nPossible niches:"
        )

        for index, industry in enumerate(
            suggestions,
            start=1
        ):

            print(
                f"{index}. {industry}"
            )

        if failed_attempts >= 3:

            print(
                "\nChoose a number, type your own niche, "
                "or type 'skip'."
            )

        else:

            print(
                "\nChoose a number or type your own niche."
            )

        answer = input(
            "\n> "
        ).strip()

        if (
            failed_attempts >= 3
            and answer.lower() == "skip"
        ):

            return request

        if answer.isdigit():

            choice = int(
                answer
            )

            if 1 <= choice <= len(suggestions):

                request = update_field(
                    request,
                    "industry",
                    suggestions[
                        choice - 1
                    ]
                )

                continue

        request = understand_request(
            answer,
            request
        )


def ensure_location(
    request: SearchRequest
):

    while request.location is None:

        print(
            "\nWhich location do you want to target?"
        )

        answer = input(
            "\n> "
        )

        request = understand_request(
            answer,
            request
        )

    return request


def resolve_number_of_leads(
    request: SearchRequest
):

    while True:

        if (
            request.number_of_leads is not None
            and request.number_of_leads > 0
        ):

            return request

        print(
            "\nHow many leads do you want me to find?"
        )

        print(
            "Please enter a positive number, for example 5 or 10."
        )

        answer = input(
            "\n> "
        )

        request = understand_request(
            answer,
            request
        )


def resolve_location(
    request: SearchRequest
):

    failed_attempts = 0

    while True:

        validation = validate_field(
            request,
            "location"
        )

        if validation.status == "VALID":
            return request

        if validation.status == "LIKELY_TYPO":

            request = handle_typo(
                request,
                "location",
                validation
            )

            continue

        failed_attempts += 1

        print(
            "\nThe location may be too broad or unclear for this search."
        )

        print(
            f"Reason: {validation.reason}"
        )

        if failed_attempts >= 3:

            print(
                "\nEnter another location or type 'skip' "
                "to keep the current location."
            )

        else:

            print(
                "\nEnter a more specific location."
            )

        answer = input(
            "\n> "
        ).strip()

        if (
            failed_attempts >= 3
            and answer.lower() == "skip"
        ):

            return request

        request = understand_request(
            answer,
            request
        )


def clarify_request(
    request: SearchRequest
):

    request = resolve_service(
        request
    )

    request = resolve_industry(
        request
    )

    request = ensure_location(
        request
    )

    request = resolve_number_of_leads(
        request
    )

    request = resolve_location(
        request
    )

    return request