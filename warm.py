"""
Pre-run requests so people get instant answers later.

Usage:
    python warm.py requests.txt

requests.txt holds one complete request per line (blank lines and lines
starting with # are ignored), for example:
    Find 10 accounting firms in Dubai for AI customer support automation.
    Find 10 dental clinics in Lahore for an AI appointment-booking assistant.

Each line is run through main.py with REFRESH=1, which researches it fresh and
saves the result. Asking the same thing later is then answered in under a
second from .cache/results (default freshness: 7 days, see RESULT_CACHE_HOURS).

A request that needs an answer from you (a missing detail, or a criterion that
cannot be verified) cannot run unattended and is reported as skipped. Rewrite
it as a complete request.
"""

import os
import subprocess
import sys
import time

TIMEOUT_SECONDS = 30 * 60


def read_requests(path: str) -> list:
    with open(path, encoding="utf-8") as handle:
        return [
            line.strip()
            for line in handle
            if line.strip() and not line.lstrip().startswith("#")
        ]


def classify(returncode: int, output: str) -> str:
    if "Saved:" in output and returncode == 0:
        return "saved"

    if "EOFError" in output:
        return "skipped (needs an answer from you, make the request complete)"

    if "Could not save" in output:
        return "researched but could not write files"

    if "No lead research was started" in output:
        return "skipped (not a lead request or could not be validated)"

    return f"failed (exit code {returncode})"


def run_one(request: str) -> tuple:
    env = dict(os.environ, REFRESH="1", PYTHONIOENCODING="utf-8")
    started = time.time()

    try:
        completed = subprocess.run(
            [sys.executable, "main.py"],
            input=request + "\n",
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=TIMEOUT_SECONDS,
            env=env,
        )
        status = classify(
            completed.returncode,
            (completed.stdout or "") + (completed.stderr or ""),
        )
    except subprocess.TimeoutExpired:
        status = "failed (timed out)"

    return status, time.time() - started


def main(argv: list) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2

    requests = read_requests(argv[1])

    if not requests:
        print("No requests found in the file.")
        return 2

    failures = 0

    for number, request in enumerate(requests, start=1):
        print(f"\n[{number}/{len(requests)}] {request}")
        status, seconds = run_one(request)
        print(f"    {status} ({seconds:.0f}s)")

        if not status.startswith("saved"):
            failures += 1

    print(f"\nDone: {len(requests) - failures} saved, {failures} not saved.")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
