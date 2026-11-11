import re

from urllib.parse import urlparse

from ddgs import DDGS

from schemas import AgentState


MAX_ROUNDS = 5

SEARCH_RESULTS_PER_ROUND = 25


BLOCKED_DOMAINS = {
    "linkedin.com",
    "facebook.com",
    "instagram.com",
    "twitter.com",
    "x.com",
    "youtube.com",
    "tiktok.com",
    "pinterest.com",
    "reddit.com",
    "quora.com",
    "wikipedia.org",
    "yelp.com",
    "tripadvisor.com",
    "yellowpages.com",
    "clutch.co",
    "goodfirms.co",
    "g2.com",
    "capterra.com",
    "zoominfo.com",
    "crunchbase.com",
    "f6s.com",
    "builtin.com",
    "glassdoor.com",
    "indeed.com",
    "medium.com",
    "lawzana.com",
    "lawlyhub.com",
    "ekatchery.com",
    "grokipedia.com"
}


JUNK_TITLE_PATTERNS = [
    r"\btop\s+\d+\b",
    r"\btop\s+\d+\s+best\b",
    r"\bbest\s+.*companies\b",
    r"\bbest\s+.*firms\b",
    r"\blist of\b",
    r"\bdirectory\b",
    r"\brankings?\b",
    r"\bcompanies to know\b"
]


def get_domain(
    url: str
):

    try:

        domain = urlparse(
            url
        ).netloc.lower()

        return domain.removeprefix(
            "www."
        )

    except Exception:

        return ""


def get_homepage(
    url: str
):

    parsed = urlparse(
        url
    )

    if not parsed.scheme or not parsed.netloc:
        return ""

    return (
        f"{parsed.scheme}://"
        f"{parsed.netloc}"
    )


def is_blocked_domain(
    url: str
):

    domain = get_domain(
        url
    )

    if not domain:
        return True

    for blocked in BLOCKED_DOMAINS:

        if (
            domain == blocked
            or domain.endswith(
                "." + blocked
            )
        ):

            return True

    return False


def title_looks_like_junk(
    title: str
):

    lowered = title.lower()

    for pattern in JUNK_TITLE_PATTERNS:

        if re.search(
            pattern,
            lowered
        ):

            return True

    return False


def search_companies(
    state: AgentState
):

    round_number = state[
        "round"
    ]

    queries = state[
        "plan"
    ][
        "search_queries"
    ]

    query = queries[
        round_number
        % len(queries)
    ]

    print(
        "\n" + "=" * 70
    )

    print(
        f"SEARCH ROUND {round_number + 1}/{MAX_ROUNDS}"
    )

    print(
        "=" * 70
    )

    print(
        f"\nQuery: {query}"
    )

    seen_domains = set(
        state[
            "seen_domains"
        ]
    )

    candidates = []

    try:

        with DDGS() as ddgs:

            results = ddgs.text(
                query,
                max_results=SEARCH_RESULTS_PER_ROUND
            )

            for result in results:

                url = result.get(
                    "href",
                    ""
                )

                title = result.get(
                    "title",
                    ""
                )

                snippet = result.get(
                    "body",
                    ""
                )

                if not url:
                    continue

                if is_blocked_domain(
                    url
                ):
                    continue

                if title_looks_like_junk(
                    title
                ):
                    continue

                domain = get_domain(
                    url
                )

                if not domain:
                    continue

                if domain in seen_domains:
                    continue

                homepage = get_homepage(
                    url
                )

                if not homepage:
                    continue

                seen_domains.add(
                    domain
                )

                candidates.append(
                    {
                        "title": title,
                        "url": homepage,
                        "domain": domain,
                        "snippet": snippet
                    }
                )

    except Exception as error:

        print(
            f"\nDDGS search error: {error}"
        )

    print(
        f"\nNew candidates after cleaning: "
        f"{len(candidates)}"
    )

    for index, company in enumerate(
        candidates,
        start=1
    ):

        print(
            f"{index}. "
            f"{company['domain']}"
        )

    return {
        "companies": candidates,
        "seen_domains": list(
            seen_domains
        ),
        "round": round_number + 1
    }