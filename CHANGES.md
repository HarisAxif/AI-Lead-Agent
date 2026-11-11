# Reliability update ? 2026-10-20

- Official-page browser recovery runs independently of screenshot requirements;
  recovered text carries business identity, final source URL and review links.
- System and certifi certificate roots are trusted; TLS verification remains on.
- Empty/repeated model queries and exhausted replanning allowance preserve unused
  searches. Short adaptive query lists no longer discard the rest of a round.
- Public search retries another backend on empty results as well as exceptions;
  search outcomes and access failures are retained in the audit.
- Explicit criteria receive a second raw-observation audit, clause by clause.
  This fixes a live false positive where a phone complaint was incorrectly
  treated as evidence of an appointment missed without notification. Output
  labels use audited clauses instead of embellished model-generated indicators.
- Optional paid data integrations require explicit opt-in. The configured Gemini
  fallback is another free-tier Flash-Lite model; Vertex AI is disabled.
- Daily Gemini quota/authentication failures stop subsequent model requests.
- The CLI is safe to import, supports --request/--refresh/--max-seconds, allows
  the full search-round budget, and exports completed stages after interruption.
- Default research budget is ten minutes; per-company follow-up is three minutes.
- Transient/incomplete results are not cached as answers for subsequent requests.
- Added regression coverage for browser recovery, source identity, compound
  criteria, provider controls, search fallback, quota stops and interrupted export.

# Changes in this version

## Request handling (clarify.py)
- Replies to a criteria problem are classified: yes / no / cancel / revised
  criteria / insist-on-original / unrelated. "no, search exactly as I said"
  no longer becomes a criterion; the claim is dropped and the user is told.
- "High volume" claims are converted to public proxy signals in code (no LLM
  judgment, so no run-to-run flip). Other clauses of the same sentence are kept.
- Validator knows `self_defeating` and may not suggest alternatives that
  require the prospect to already own the service (the "active live chat" bug).
- `mixed_request` handled: the extra instruction is skipped with a message and
  the valid lead request continues.
- Intent routing and slot extraction run in parallel.
- Retries: 3 attempts with backoff; optional GEMINI_FALLBACK_MODEL; seed=7.
- country_code extracted from the location (used for phone validation).

## Evidence (qualify.py)
- FIX: the lead record used the unguarded reason/pitch; the contradiction
  guard's safe text was computed and discarded. Now used.
- Technical facts only count when their meaning matches the claim
  (form count 0 no longer "proves" a contact form; clickable counts never count).
- New contradiction checks: "no FAQ" vs FAQ on page; "no chat" vs known chat
  vendor scripts.
- Evidence is de-duplicated by category (same fact is no longer counted twice
  toward the two-item rule); hard evidence preferred over screenshots.
- Log wording: "Dropped weak evidence item" (was "Rejected ...", which
  appeared even for qualified leads).
- Domains rejected twice as "not a single-business website" are learned.

## Contacts (website.py)
- tel:/mailto: links first; developer credits ("designed by") skipped.
- Emails must match the site domain or a free-mail provider.
- Phones validated with phonenumbers, must belong to the search country,
  de-duplicated on E.164, shown in international format.

## Discovery (discovery.py)
- Broad locations expanded into up to 30 towns once per run.
- Queries are taken from an unused pool (no skipping when a round ends early);
  a query that hits a search error is retried once, then dropped.
- Larger static blocklist (search engines, review sites, funeral directories).

## New
- planner.py: plan cache + seed (same request -> same criteria).
- exporter.py: CSV / JSON / audit report per run.
- blocklist.py, requirements.txt, tests/ (124 offline tests), README.md.

# Round 2 (after the first live run)

- Subpages were never read: the plan said "Contact Us page" and a link
  label never contains that phrase. Plan phrases are now reduced to keywords,
  Contact and About are ranked first, and same-page/fragment links are skipped.
- WhatsApp links (wa.me, api.whatsapp.com) now count as phone numbers, and
  tel:/mailto:/WhatsApp links found only after JavaScript runs (Chromium) are
  merged into the lead's contacts with the same validation.
- Evidence must map to a PLANNED need indicator with the same meaning.
  "Presence of a booking system" can no longer be stored as evidence for
  "Absence of a booking system"; unplanned paraphrases are dropped; evidence is
  stored under the planned wording.
- Company-size / credibility signals (client counts, testimonials, offices,
  service range) can no longer qualify a lead on their own: at least one
  verified item must show the support gap itself.
- "Directory/non-business filtered" in the final breakdown now includes the
  discovery counts (it always showed 0 before).
- 124 offline tests.

# Round 3 (after the second live run)

- Evidence strength (strong / moderate / limited) on every lead, shown in the
  output and in leads.csv. "limited" = the support gap rests on screenshot
  impressions only.
- Generated reason/pitch no longer state unverified inquiry volume
  ("high-volume inquiries" -> "routine inquiries").
- Role mailboxes (info@, contact@) are preferred; named staff emails are kept
  only when nothing generic exists.
- Pages inspected: redirect duplicates (/contact vs /contact/) are removed;
  blog-style long slugs are not crawled; shorter paths are preferred.
- 124 offline tests.

# Round 3.1

- The criteria validator now knows the agent only reads a business's OWN
  website. Claims that need customer reviews, social media or job postings
  (for example "customers complain about slow replies") are flagged with an
  accurate reason ("reviews are public, but this agent doesn't read them")
  instead of "not publicly accessible". The old prompt listed public reviews
  as checkable, which contradicted what the pipeline can do.
- 124 offline tests.

# Round 4: Google Places (optional)

- places.py: Places API (New) Text Search with places.reviews; request cap,
  run-level disable on key/billing errors, never leaks the key.
- Discovery: when a key is set, queries go to Google Places first (real
  businesses with official websites, no directories); any Places error or an
  empty answer falls back to web search for that query.
- Reviews: complaints about slow or missing replies are detected per sentence
  (praise like "replied quickly" is not a complaint). Counts are shown to
  Gemini; the lead gets a "google_reviews" evidence item (counts as a real gap).
- If your request asks for review-based criteria and Places is on, a lead must
  have at least one such review to qualify (reason given otherwise). Without
  Places the validator explains that reviews are not read (unchanged).
- Console output shows each matching review with author, time and a Google
  Maps link. Saved CSV/JSON/report contain place ids and links only.
- 124 offline tests. Not tested against the live Google API.

# Round 5: speed + fixes from the Wyoming log

Speed
- Saved results: a repeated request (same text, or same request reworded)
  is answered in <1s with no Gemini/search/browser call. warm.py pre-runs a list.
- FAST_MODE=1 skips screenshots; "no live chat" becomes a measured fact
  (scan of the page's scripts, only trusted when no tag manager is present).
- Searches run in parallel batches (SEARCH_WORKERS, default 3).
- RENDER_WORKERS=N renders N sites at once, one browser per worker (default 1;
  not tested against real Chromium).
- MAX_RUN_SECONDS (default 300) stops the run after the current round.

Fixes
- BUG: the same site could enter one round twice (two URLs of one domain in
  the same search results), so it was rendered, analysed and counted twice.
  In the Wyoming run 21 of 56 candidates were such duplicates.
- Planner: need indicators must be checkable from the business's own website,
  5-7 of them, at least 3 about contact/availability gaps. The old plans
  contained review-based and "Presence or absence of..." indicators that could
  never count as evidence, so almost every real funeral home failed the
  two-evidence rule. Plan cache is versioned so old plans are not reused.
- Unreachable sites: the log now says why (HTTP status or error), and a broken
  certificate chain is retried without verification.
- Saved review links no longer crash the result display.
- 124 offline tests.
