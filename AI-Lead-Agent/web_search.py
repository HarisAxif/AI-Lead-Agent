"""Public search with bounded backend fallback, including empty responses."""
import os
import research_budget


def search_public(query, factory, max_results=8, details=None):
    backends = list(dict.fromkeys([
        os.getenv("DDGS_BACKEND", "duckduckgo"),
        os.getenv("DDGS_FALLBACK_BACKEND", "yahoo"),
    ]))
    errors = []
    empty = False
    for index, backend in enumerate(backends):
        research_budget.check()
        try:
            with factory() as search:
                results = list(search.text(query, max_results=max_results, backend=backend))
            if details is not None:
                details.update(backend=backend, fallback=index > 0, backend_errors=list(errors))
            if results:
                return results
            empty = True
        except research_budget.ResearchBudgetExceeded:
            raise
        except Exception as error:
            errors.append(f"{backend}: {str(error).splitlines()[0]}")
    if empty:
        return []
    raise RuntimeError("Public search unavailable: " + "; ".join(errors))
