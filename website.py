import base64
import re

from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urljoin, urlparse

import httpx

from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright


MAX_INTERNAL_PAGES = 3
MAX_WORKERS = 8


# =========================================================
# BASIC URL HELPERS
# =========================================================

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


# =========================================================
# SIMPLE HTTP FETCH
# Used for text/content/internal pages
# =========================================================

def fetch_page(
    url: str
):

    try:

        response = httpx.get(
            url,
            timeout=12,
            follow_redirects=True,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 "
                    "(Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 "
                    "(KHTML, like Gecko) "
                    "Chrome/120.0 Safari/537.36"
                )
            }
        )

        response.raise_for_status()

        content_type = response.headers.get(
            "content-type",
            ""
        ).lower()

        if "text/html" not in content_type:

            return None

        return response

    except Exception:

        return None


# =========================================================
# CLEAN HTML TEXT
# =========================================================

def clean_page_text(
    html: str
):

    soup = BeautifulSoup(
        html,
        "html.parser"
    )

    for tag in soup(
        [
            "script",
            "style",
            "noscript",
            "svg"
        ]
    ):

        tag.decompose()

    text = soup.get_text(
        " ",
        strip=True
    )

    return re.sub(
        r"\s+",
        " ",
        text
    )


# =========================================================
# FIND INTERNAL PAGES RELEVANT TO RESEARCH PLAN
# =========================================================

def find_relevant_internal_links(
    soup,
    base_url: str,
    page_keywords: list[str]
):

    base_domain = get_domain(
        base_url
    )

    found = []

    seen = set()

    for anchor in soup.find_all(
        "a",
        href=True
    ):

        href = anchor.get(
            "href",
            ""
        )

        label = (
            anchor.get_text(
                " ",
                strip=True
            )
            + " "
            + href
        ).lower()

        if not any(
            keyword.lower() in label
            for keyword in page_keywords
        ):

            continue

        full_url = urljoin(
            base_url,
            href
        )

        if get_domain(
            full_url
        ) != base_domain:

            continue

        if full_url in seen:

            continue

        seen.add(
            full_url
        )

        found.append(
            full_url
        )

        if (
            len(found)
            >= MAX_INTERNAL_PAGES
        ):

            break

    return found


# =========================================================
# SCRIPT HOSTS
# =========================================================

def extract_script_hosts(
    soup
):

    hosts = set()

    for script in soup.find_all(
        "script",
        src=True
    ):

        src = script.get(
            "src",
            ""
        )

        if src.startswith(
            "//"
        ):

            src = "https:" + src

        if not src.startswith(
            "http"
        ):

            continue

        host = get_domain(
            src
        )

        if host:

            hosts.add(
                host
            )

    return sorted(
        hosts
    )[:15]


# =========================================================
# HTML / TEXT INSPECTION
# =========================================================

def inspect_html(
    company: dict,
    plan: dict
):

    homepage_response = fetch_page(
        company[
            "url"
        ]
    )

    if homepage_response is None:

        return {
            "reachable": False,
            "text": "",
            "footprint": "",
            "emails": [],
            "phones": [],
            "pages_read": [],
            "desktop_screenshot": None,
            "mobile_screenshot": None,
            "rendered_facts": ""
        }

    homepage_html = (
        homepage_response.text
    )

    homepage_soup = BeautifulSoup(
        homepage_html,
        "html.parser"
    )

    homepage_text = clean_page_text(
        homepage_html
    )

    pages_text = [
        homepage_text
    ]

    pages_read = [
        str(
            homepage_response.url
        )
    ]

    internal_links = (
        find_relevant_internal_links(
            homepage_soup,
            str(
                homepage_response.url
            ),
            plan[
                "pages_to_check"
            ]
        )
    )

    for link in internal_links:

        page_response = fetch_page(
            link
        )

        if page_response is None:

            continue

        page_text = clean_page_text(
            page_response.text
        )

        if page_text:

            pages_text.append(
                page_text
            )

            pages_read.append(
                str(
                    page_response.url
                )
            )

    combined_text = " ".join(
        pages_text
    )

    emails = sorted(
        set(
            re.findall(
                r"[\w.+-]+@[\w-]+\.[\w.-]+",
                combined_text
            )
        )
    )[:5]

    phones = sorted(
        set(
            re.findall(
                r"\+?\d[\d\s\-()]{8,16}\d",
                combined_text
            )
        )
    )[:5]

    copyright_years = re.findall(
        r"(?:©|copyright)\D{0,10}(\d{4})",
        combined_text,
        re.I
    )[:5]

    script_hosts = extract_script_hosts(
        homepage_soup
    )

    footprint_lines = [
        (
            "https: "
            f"{str(homepage_response.url).startswith('https')}"
        ),
        (
            "http_load_seconds: "
            f"{round(homepage_response.elapsed.total_seconds(), 2)}"
        ),
        (
            "html_mobile_viewport_tag: "
            f"{bool(homepage_soup.find('meta', attrs={'name': 'viewport'}))}"
        ),
        (
            "html_forms_on_homepage: "
            f"{len(homepage_soup.find_all('form'))}"
        ),
        (
            "third_party_script_hosts: "
            f"{script_hosts}"
        ),
        (
            "copyright_years: "
            f"{copyright_years}"
        ),
        (
            "website_text_word_count: "
            f"{len(combined_text.split())}"
        ),
        (
            "pages_read: "
            f"{len(pages_read)}"
        ),
        (
            "emails_found: "
            f"{len(emails)}"
        ),
        (
            "phones_found: "
            f"{len(phones)}"
        )
    ]

    return {
        "reachable": True,

        "text":
            combined_text[:10000],

        "footprint":
            "\n".join(
                footprint_lines
            ),

        "emails":
            emails,

        "phones":
            phones,

        "pages_read":
            pages_read,

        "desktop_screenshot":
            None,

        "mobile_screenshot":
            None,

        "rendered_facts":
            ""
    }


# =========================================================
# RENDERED DOM FACTS
# =========================================================

def get_rendered_dom_facts(
    page
):

    data = page.evaluate(
        """
        () => {

            const visible = (el) => {

                const style =
                    window.getComputedStyle(el);

                const rect =
                    el.getBoundingClientRect();

                return (
                    style.display !== "none" &&
                    style.visibility !== "hidden" &&
                    style.opacity !== "0" &&
                    rect.width > 0 &&
                    rect.height > 0
                );
            };


            const clickable =
                Array.from(
                    document.querySelectorAll(
                        'a, button, [role="button"], input[type="submit"]'
                    )
                )
                .filter(visible)
                .map(el => ({
                    text: (
                        el.innerText ||
                        el.value ||
                        el.textContent ||
                        ""
                    )
                    .trim()
                    .replace(/\\s+/g, " "),

                    href:
                        el.href || "",

                    tag:
                        el.tagName
                }))
                .filter(
                    x => x.text || x.href
                )
                .slice(
                    0,
                    60
                );


            const navigation =
                Array.from(
                    document.querySelectorAll(
                        'nav a, header a'
                    )
                )
                .filter(visible)
                .map(el => (
                    el.innerText ||
                    el.textContent ||
                    ""
                )
                .trim()
                .replace(/\\s+/g, " "))
                .filter(Boolean)
                .slice(
                    0,
                    30
                );


            const telLinks =
                Array.from(
                    document.querySelectorAll(
                        'a[href^="tel:"]'
                    )
                )
                .map(
                    a => a.href
                );


            const mailLinks =
                Array.from(
                    document.querySelectorAll(
                        'a[href^="mailto:"]'
                    )
                )
                .map(
                    a => a.href
                );


            const whatsappLinks =
                Array.from(
                    document.querySelectorAll(
                        'a'
                    )
                )
                .map(
                    a => a.href || ""
                )
                .filter(
                    href =>
                        href.includes("wa.me") ||
                        href.toLowerCase().includes("whatsapp")
                );


            const visibleForms =
                Array.from(
                    document.querySelectorAll(
                        "form"
                    )
                )
                .filter(
                    visible
                );


            const bodyText =
                (
                    document.body.innerText ||
                    ""
                )
                .trim()
                .replace(
                    /\\s+/g,
                    " "
                );


            return {

                title:
                    document.title,

                body_text_length:
                    bodyText.length,

                visible_forms:
                    visibleForms.length,

                clickable_count:
                    clickable.length,

                clickable:
                    clickable,

                navigation:
                    navigation,

                tel_links:
                    telLinks,

                mail_links:
                    mailLinks,

                whatsapp_links:
                    whatsappLinks
            };
        }
        """
    )

    lines = [

        (
            "rendered_page_title: "
            f"{data.get('title', '')}"
        ),

        (
            "rendered_body_text_length: "
            f"{data.get('body_text_length', 0)}"
        ),

        (
            "rendered_clickable_count: "
            f"{data.get('clickable_count', 0)}"
        ),

        (
            "rendered_visible_forms: "
            f"{data.get('visible_forms', 0)}"
        ),

        (
            "rendered_phone_links: "
            f"{data.get('tel_links', [])}"
        ),

        (
            "rendered_email_links: "
            f"{data.get('mail_links', [])}"
        ),

        (
            "rendered_whatsapp_links: "
            f"{data.get('whatsapp_links', [])}"
        ),

        (
            "rendered_navigation: "
            f"{data.get('navigation', [])}"
        ),

        "rendered_clickable_elements:"
    ]

    for element in data.get(
        "clickable",
        []
    ):

        lines.append(
            (
                f"- text="
                f"{element.get('text', '')!r}; "
                f"href="
                f"{element.get('href', '')!r}"
            )
        )

    return "\n".join(
        lines
    )


# =========================================================
# PAGE READINESS / LOADING SCREEN CHECK
# =========================================================

def get_page_health(
    page
):

    try:

        return page.evaluate(
            """
            () => {

                const text =
                    (
                        document.body.innerText ||
                        ""
                    )
                    .trim();


                const links =
                    document.querySelectorAll(
                        'a'
                    ).length;


                const buttons =
                    document.querySelectorAll(
                        'button, [role="button"]'
                    ).length;


                const images =
                    document.querySelectorAll(
                        'img'
                    ).length;


                return {

                    text_length:
                        text.length,

                    links:
                        links,

                    buttons:
                        buttons,

                    images:
                        images
                };
            }
            """
        )

    except Exception:

        return {
            "text_length": 0,
            "links": 0,
            "buttons": 0,
            "images": 0
        }


def page_looks_incomplete(
    health: dict
):

    text_length = health.get(
        "text_length",
        0
    )

    links = health.get(
        "links",
        0
    )

    buttons = health.get(
        "buttons",
        0
    )

    # Very little visible content usually indicates
    # loader, splash page, failed JS render or blocked content.

    if (
        text_length < 250
        and links < 4
        and buttons < 3
    ):

        return True

    return False


# =========================================================
# WAIT FOR REAL PAGE CONTENT
# =========================================================

def wait_for_page_to_settle(
    page
):

    # -----------------------------------------------------
    # 1. Give normal JS execution a chance
    # -----------------------------------------------------

    try:

        page.wait_for_load_state(
            "domcontentloaded",
            timeout=8000
        )

    except Exception:

        pass


    # -----------------------------------------------------
    # 2. Try network idle
    # Some sites never truly reach networkidle due to
    # analytics/chat/widgets, so failure is fine.
    # -----------------------------------------------------

    try:

        page.wait_for_load_state(
            "networkidle",
            timeout=8000
        )

    except Exception:

        pass


    # -----------------------------------------------------
    # 3. Allow animations/loaders/framework hydration
    # -----------------------------------------------------

    page.wait_for_timeout(
        2500
    )


    # -----------------------------------------------------
    # 4. Initial sanity check
    # -----------------------------------------------------

    health = get_page_health(
        page
    )

    if page_looks_incomplete(
        health
    ):

        print(
            "    Page appears incomplete. "
            "Waiting for loader/content..."
        )

        page.wait_for_timeout(
            4000
        )

        health = get_page_health(
            page
        )


    # -----------------------------------------------------
    # 5. Second retry for slow splash-screen websites
    # -----------------------------------------------------

    if page_looks_incomplete(
        health
    ):

        print(
            "    Still little content. "
            "Giving page one final wait..."
        )

        page.wait_for_timeout(
            5000
        )


    # -----------------------------------------------------
    # 6. Scroll down to trigger lazy loading
    # -----------------------------------------------------

    try:

        page.evaluate(
            """
            () => {

                window.scrollTo(
                    0,
                    Math.min(
                        document.body.scrollHeight,
                        2500
                    )
                );

            }
            """
        )

        page.wait_for_timeout(
            1200
        )


        # Back to top for clean screenshot
        page.evaluate(
            """
            () => {

                window.scrollTo(
                    0,
                    0
                );

            }
            """
        )

        page.wait_for_timeout(
            800
        )

    except Exception:

        pass


# =========================================================
# RENDER WEBSITE IN REAL CHROMIUM
# =========================================================

def render_site(
    browser,
    company: dict
):

    context = browser.new_context(
        viewport={
            "width": 1440,
            "height": 900
        }
    )

    page = context.new_page()

    page.set_default_timeout(
        15000
    )

    try:

        response = page.goto(
            company[
                "url"
            ],
            wait_until="domcontentloaded",
            timeout=20000
        )

        if (
            response is not None
            and response.status >= 400
        ):

            context.close()

            return None


        # =================================================
        # NEW:
        # WAIT UNTIL SPLASH / LOADER / JS CONTENT SETTLES
        # =================================================

        wait_for_page_to_settle(
            page
        )


        # =================================================
        # FINAL HEALTH CHECK
        # =================================================

        health = get_page_health(
            page
        )


        print(
            "    Rendered content: "
            f"{health.get('text_length', 0)} chars, "
            f"{health.get('links', 0)} links, "
            f"{health.get('buttons', 0)} buttons"
        )


        # =================================================
        # EXTRACT DOM ONLY AFTER PAGE HAS SETTLED
        # =================================================

        rendered_facts = (
            get_rendered_dom_facts(
                page
            )
        )


        # =================================================
        # DESKTOP SCREENSHOT
        # =================================================

        desktop_bytes = page.screenshot(
            type="jpeg",
            quality=70,
            full_page=False
        )


        # =================================================
        # MOBILE VIEW
        # =================================================

        page.set_viewport_size(
            {
                "width": 390,
                "height": 844
            }
        )


        # Changing viewport can trigger responsive
        # JS/layout changes. Give it time to settle.

        page.wait_for_timeout(
            1800
        )


        mobile_health = get_page_health(
            page
        )


        # If responsive version itself loads slowly,
        # give it another chance.

        if page_looks_incomplete(
            mobile_health
        ):

            page.wait_for_timeout(
                3000
            )


        mobile_bytes = page.screenshot(
            type="jpeg",
            quality=70,
            full_page=False
        )


        context.close()


        return {

            "desktop_screenshot":
                base64.b64encode(
                    desktop_bytes
                ).decode(
                    "utf-8"
                ),

            "mobile_screenshot":
                base64.b64encode(
                    mobile_bytes
                ).decode(
                    "utf-8"
                ),

            "rendered_facts":
                rendered_facts,

            "render_health":
                health
        }


    except Exception as error:

        print(
            f"    Browser render failed: "
            f"{error}"
        )

        context.close()

        return None


# =========================================================
# INSPECT ALL CANDIDATES
# =========================================================

def inspect_all(
    companies: list,
    plan: dict
):

    # =====================================================
    # STEP 1:
    # Fetch normal HTML/text in parallel
    # =====================================================

    with ThreadPoolExecutor(
        max_workers=MAX_WORKERS
    ) as pool:

        html_results = list(
            pool.map(
                lambda company:
                    inspect_html(
                        company,
                        plan
                    ),
                companies
            )
        )


    # =====================================================
    # STEP 2:
    # Skip browser rendering if planner says vision
    # provides no useful evidence.
    # =====================================================

    if not plan.get(
        "visual_review_needed",
        False
    ):

        return html_results


    print(
        "\nRendering websites visually..."
    )


    # =====================================================
    # STEP 3:
    # Open actual Chromium
    # =====================================================

    with sync_playwright() as playwright:

        browser = playwright.chromium.launch(
            headless=False
        )


        for index, (
            company,
            site
        ) in enumerate(
            zip(
                companies,
                html_results
            ),
            start=1
        ):

            if not site[
                "reachable"
            ]:

                continue


            print(
                f"\nRendering "
                f"[{index}/{len(companies)}]: "
                f"{company['domain']}"
            )


            visual = render_site(
                browser,
                company
            )


            if visual is None:

                print(
                    "    Visual render unavailable."
                )

                continue


            # Add screenshots + DOM facts
            site.update(
                visual
            )


            # Add rendered information into footprint
            # Gemini can now compare HTML facts against
            # what Chromium actually found.

            site[
                "footprint"
            ] += (
                "\n"
                + visual[
                    "rendered_facts"
                ]
            )


        browser.close()


    return html_results