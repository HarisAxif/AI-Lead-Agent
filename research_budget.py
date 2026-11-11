"""Deadline propagation for network/model actions, including worker entry points."""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import time

_deadline = ContextVar("research_deadline", default=0.0)


class ResearchBudgetExceeded(TimeoutError):
    """No research time remains, including time needed for a retry."""


def remaining():
    deadline = _deadline.get()
    return max(0.0, deadline-time.time()) if deadline else None


def check():
    left = remaining()
    if left is not None and left <= 0:
        raise ResearchBudgetExceeded("Research time budget exhausted")
    return left


@contextmanager
def bounded(deadline):
    previous = _deadline.get()
    effective = min(previous, deadline) if previous and deadline else previous or deadline or 0.0
    token = _deadline.set(effective)
    try:
        yield
    finally:
        _deadline.reset(token)


def state_budget(position):
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            state = kwargs.get("state") or (args[position] if len(args) > position else {})
            with bounded(state.get("deadline", 0)):
                return function(*args, **kwargs)
        return wrapped
    return decorate


def run_with_deadline(deadline, function, *args, **kwargs):
    """Bind an absolute deadline inside a worker (ContextVars aren't inherited)."""
    with bounded(deadline):
        return function(*args, **kwargs)


def request_timeout(default):
    available = check()
    return max(0.001, min(default, available)) if available is not None else default
