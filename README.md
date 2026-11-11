# AI Lead Agent

A command-line agent that finds businesses, checks their fit for your service,
and returns leads supported by readable public evidence. Gemini handles request
understanding, planning and evidence analysis. Public web search and Chromium
supply the source material.

## Setup

Use Python 3.11 or newer. In this folder:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m playwright install chromium
```

Copy `.env.example` to `.env` only if you do not already have `.env`. Add your
Google AI Studio key:

```dotenv
GEMINI_API_KEY=your-google-ai-studio-key
GEMINI_MODEL=gemini-3.5-flash-lite
GEMINI_FALLBACK_MODEL=gemini-3.1-flash-lite
GEMINI_THIRD_MODEL=
ALLOW_PAID_PROVIDERS=0
MAX_RUN_SECONDS=600
```

Keep your key private. Use a Gemini project on the **free tier**. Google lists
free-tier inference for these Flash-Lite models; actual quota depends on the
project. The app cannot inspect or change your project's billing tier. It uses
ordinary Gemini inference, not paid search grounding. It does not switch to
OpenAI, Groq, or another paid model provider. Optional Google Places and
Trustpilot API integrations stay disabled unless `ALLOW_PAID_PROVIDERS=1`, even
if old credentials remain in the environment.

Google references: [pricing](https://ai.google.dev/gemini-api/docs/pricing),
[project rate limits](https://ai.google.dev/gemini-api/docs/rate-limits).

## Run

```powershell
python main.py
```

Describe the number of leads, target industry, target location and the service
you sell. Missing or unclear details are clarified before research begins.
Optional criteria, including specific review complaints, remain mandatory.

Example:

> Find 3 plumbing and heating companies in the United Kingdom for my call-handling and customer update system. Public customer reviews must mention unanswered calls and missed appointments without notification.

You can also pass the request directly:

```powershell
python main.py --request "Find 3 dental clinics in Lahore for my appointment reminder service" --refresh
python main.py --help
```

`--max-seconds 600` overrides the research time budget for one run; `0` removes
the time limit. The action and search-round limits still apply. `--refresh`
bypasses saved results. A missing detail can still trigger an interactive
question when using `--request`.

## How decisions work

1. Understand and validate the request; ask for missing or ambiguous details.
2. Plan searches for the location, industry, service and explicit criteria.
3. Discover candidate businesses. Review requests start with complaint-led searches.
4. Read official pages to establish identity, industry, location and contacts.
   Failed HTTP requests get a normal Chromium text-recovery attempt even when
   visual review is unnecessary. Screenshots are taken only when requested by
   the research plan. TLS verification stays enabled.
5. Fetch relevant public review pages and other sources. Search snippets identify
   sources but never establish a complaint. Refused sources trigger alternatives.
6. Check exact quotations and independently audit the criteria. A second, focused
   check sees raw observations without generated indicator labels and verifies
   each explicit AND clause. Related complaints cannot substitute for a missing
   clause. Evidence can span multiple relevant passages.
7. Investigate missing facts within the budget. Continue unused discovery searches
   when the model returns no new queries or its replanning allowance ends.
8. Return verified leads and explain any shortfall. Never invent contacts,
   complaints, buying intent, or extra leads to meet the requested count.

A suitable prospect is not a confirmed buyer. Internal software use or purchase
intent cannot be inferred solely from missing website features. Blocked or
unreadable pages leave a criterion unverified; they do not prove that no
complaints exist. Free public sources cannot guarantee every requested count.

## Results

Every completed research run writes a timestamped directory under `runs/`:

- `leads.csv`: company, website, contacts, opportunity and evidence for CRM use.
- `leads.json`: qualified leads with structured detail.
- `report.json`: request, plan, search outcomes, rejection reasons, missing facts,
  evidence audits, access errors and timings.

Ctrl+C during research saves state from completed workflow stages. Unexpected
workflow errors also preserve completed stages. Work still in flight may not be
included. Repeating a successful request can reuse dated results; the console
clearly identifies cached output. Incomplete runs caused by access, API, search,
or budget failures do not become cached answers.

## Settings

| Setting | Default | Purpose |
| --- | --- | --- |
| `MAX_RUN_SECONDS` | 600 | Overall research deadline; 0 disables it. |
| `RESEARCH_COMPANY_SECONDS` | 180 | Follow-up investigation budget per company. |
| `RESEARCH_MAX_STEPS_PER_COMPANY` | 3 | Follow-up actions per company. |
| `DISCOVERY_MAX_REPLANS` | 4 | Model replans; unused searches continue afterward. |
| `GEMINI_REQUESTS_PER_MINUTE` | 15 | Local pacing; set to your AI Studio allowance. |
| `GEMINI_MAX_WAIT_SECONDS` | 90 | Maximum local quota wait for a call. |
| `SEARCH_WORKERS` | 3 | Concurrent searches; lower to 1 if search is throttled. |
| `RENDER_WORKERS` | 1 | Concurrent Chromium instances. |
| `DDGS_BACKEND` | duckduckgo | Primary public search engine. |
| `DDGS_FALLBACK_BACKEND` | yahoo | Fallback after failed or empty searches. |
| `FAST_MODE` | 0 | Skip visual screenshots; browser text recovery still runs. |
| `RESULT_CACHE_HOURS` | 168 | Saved-result lifetime; 0 disables caching. |
| `REFRESH` | 0 | Set to 1 to ignore saved results. |
| `NO_PLAN_CACHE` | 0 | Set to 1 to regenerate the research plan. |

Minute quotas are paced and transient errors have bounded retries. A daily-quota
or authentication failure stops new model requests for the run. Partial verified
results are still exported. Wait for free quota to reset instead of enabling a
paid provider. Deadlines prevent new work and bound supported network calls;
third-party search calls may take time to return after a deadline.

## Troubleshooting

- **Chromium unavailable:** run `python -m playwright install chromium` in the same
  environment. The agent records the browser error and continues other work.
- **HTTP 403 or access challenge:** ordinary browser recovery may help an official
  page. The agent does not solve CAPTCHAs or bypass logins. Other candidates and
  review sources remain eligible.
- **Certificate error:** both system and certifi trust roots are used, with host
  verification enabled. Check the machine's certificates/network if it persists.
- **Zero search results:** inspect search errors in the audit. Retry later or use
  `SEARCH_WORKERS=1`; do not interpret an engine failure as an empty market.
- **Quota exhausted:** check your project's allowance in AI Studio. No code
  change can remove a provider's daily free-tier limit.

## Verify changes

```powershell
python -m pytest -q -p no:cacheprovider
```

Tests cover clarification, evidence and identity checks, review conjunctions,
source access, discovery recovery, browser fallback, quotas, caching and export.
They use mocked provider/network calls and do not consume Gemini quota. Live
research is separate and depends on the sources available at run time.
