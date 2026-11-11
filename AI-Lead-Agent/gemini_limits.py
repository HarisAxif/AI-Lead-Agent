"""Shared request pacing and server-directed cooldown for Gemini calls."""
import os
import research_budget
import re
import threading
import time
from collections import deque


def retry_delay(error, default=30.0):
    message = str(error)
    delays = [float(m.group(1)) for m in re.finditer(
        r"(?:retry in\s+|retryDelay[^0-9]{0,10})(\d+(?:\.\d+)?)\s*s", message, re.I)]
    return max(delays, default=default) + 1.0


class RequestLimiter:
    def __init__(self, requests_per_minute):
        self.limit = max(0, requests_per_minute)
        self.starts = deque()
        self.cooldown_until = 0.0
        self.lock = threading.Lock()

    def defer(self, seconds):
        with self.lock:
            self.cooldown_until = max(self.cooldown_until, time.monotonic() + seconds)

    def acquire(self):
        started = time.monotonic()
        max_wait = max(1.0, float(os.getenv("GEMINI_MAX_WAIT_SECONDS", "90")))
        available = research_budget.check()
        if available is not None:
            max_wait = min(max_wait, available)
        next_notice = started
        while True:
            with self.lock:
                now = time.monotonic()
                while self.starts and now - self.starts[0] >= 60:
                    self.starts.popleft()
                delay = max(0.0, self.cooldown_until - now)
                if self.limit and len(self.starts) >= self.limit:
                    delay = max(delay, 60 - (now - self.starts[0]) + 0.05)
                if delay <= 0:
                    if self.limit:
                        self.starts.append(now)
                    return
            remaining = max_wait - (time.monotonic() - started)
            if remaining <= 0 or delay > remaining:
                if available is not None and available <= float(os.getenv("GEMINI_MAX_WAIT_SECONDS", "90")):
                    raise research_budget.ResearchBudgetExceeded("Research time budget cannot accommodate the Gemini quota wait")
                raise TimeoutError(
                    f"Gemini quota wait exceeds the {max_wait:g}s limit. "
                    "Check API quota or retry after it resets."
                )
            if time.monotonic() >= next_notice:
                print(f"[Gemini quota] Waiting {delay:.0f}s for request capacity "
                      f"(wait limit {max_wait:g}s).", flush=True)
                next_notice = time.monotonic() + 15
            time.sleep(min(delay, 10.0, remaining))


# Override for your account's quota; zero disables local RPM pacing.
limiter = RequestLimiter(int(os.getenv("GEMINI_REQUESTS_PER_MINUTE", "15")))


def handle_quota_error(error):
    message = str(error).lower()
    if "429" in message or "resource_exhausted" in message:
        limiter.defer(retry_delay(error))
        return True
    return False
