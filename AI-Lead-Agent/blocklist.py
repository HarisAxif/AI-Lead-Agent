"""
Learned blocklist of directory / aggregator domains.

A domain is added after the qualifier has rejected it as "not a single
business website" MIN_REJECTIONS times (across runs), so one wrong judgment
can never block a real business permanently.
"""

import json
from pathlib import Path

BLOCKLIST_PATH = Path(".cache") / "learned_blocklist.json"
MIN_REJECTIONS = 2


def _load() -> dict:
    try:
        return json.loads(BLOCKLIST_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_learned_blocklist() -> set:
    return {
        domain
        for domain, count in _load().items()
        if isinstance(count, int) and count >= MIN_REJECTIONS
    }


def learn_rejected_domain(domain: str) -> None:
    domain = (domain or "").lower().strip()

    if not domain:
        return

    try:
        data = _load()
        data[domain] = int(data.get(domain, 0)) + 1
        BLOCKLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
        BLOCKLIST_PATH.write_text(
            json.dumps(data, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    except Exception:
        pass  # best-effort only
