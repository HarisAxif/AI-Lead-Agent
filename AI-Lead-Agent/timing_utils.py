import time


def start_timer():
    """Start a high-resolution performance timer."""
    return time.perf_counter()


def elapsed(start):
    """Return elapsed seconds since start."""
    return time.perf_counter() - start
