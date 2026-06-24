"""Recursion safety guard — context-manager depth counters.

Used by kernel.ingest() and REJECTED-path auto-extraction to prevent
infinite recursion when a text triggers rule extraction that feeds back
into validation that triggers more extraction.

Two independent counters:
  IngestDepthGuard  — ingest() call-stack depth (book → chapter → section)
  AutoExtractGuard  — REJECTED auto-extract depth (one auto-run is fine, two is a bug)

Usage (in kernel.py ingest() step 0):
    with IngestDepthGuard(self) as g:
        # g.depth is current depth after increment
        ...

Usage (in kernel.py validate() REJECTED auto-extract):
    with AutoExtractGuard(self) as g:
        ...
"""

from __future__ import annotations


class RecursionSafetyError(RuntimeError):
    """Raised when a guarded code path exceeds its maximum depth."""

    def __init__(self, max_depth: int, current_depth: int, context: str = ""):
        self.max_depth = max_depth
        self.current_depth = current_depth
        self.context = context
        super().__init__(
            f"Recursion safety limit reached: depth {current_depth} > "
            f"max {max_depth}. context={context or 'unknown'}"
        )


# ───────────────────────────────────────────────────────────────────
# IngestDepthGuard — guards ingest() call stack (book→chapter→section)
# ───────────────────────────────────────────────────────────────────

class IngestDepthGuard:
    """Context manager that tracks ingest() recursion depth.

    Stores depth on the kernel instance so it persists across the
    ingest → extract → validate → reject → auto-extract → ingest chain.

    Max depth = 3: a book of chapters of sections is the deepest expected chain.
    """

    _ATTR = "_ingest_depth"

    def __init__(self, kernel_instance, max_depth: int = 3):
        self._kernel = kernel_instance
        self.max_depth = max_depth
        self.depth = 0

    def __enter__(self) -> "IngestDepthGuard":
        current = getattr(self._kernel, self._ATTR, 0)
        current += 1
        setattr(self._kernel, self._ATTR, current)
        self.depth = current
        if current > self.max_depth:
            raise RecursionSafetyError(
                max_depth=self.max_depth,
                current_depth=current,
                context=f"ingest() call depth {current} exceeds max {self.max_depth}",
            )
        return self

    def __exit__(self, *exc):
        current = getattr(self._kernel, self._ATTR, 1)
        setattr(self._kernel, self._ATTR, max(0, current - 1))
        return False  # don't suppress exceptions


# ───────────────────────────────────────────────────────────────────
# AutoExtractGuard — guards REJECTED-path auto-extraction
# ───────────────────────────────────────────────────────────────────

class AutoExtractGuard:
    """Context manager that prevents runaway auto-extraction.

    The REJECTED path in validate() may auto-extract rules for a
    candidate domain that has no rules.  One auto-extract is normal
    (the domain was just empty).  A second auto-extract means the
    newly-extracted rules didn't load — that's a bug or a loop.

    Max depth = 1: one auto-extract pass, then block.
    """

    _ATTR = "_auto_extract_depth"

    def __init__(self, kernel_instance, max_depth: int = 1):
        self._kernel = kernel_instance
        self.max_depth = max_depth
        self.depth = 0

    def __enter__(self) -> "AutoExtractGuard":
        current = getattr(self._kernel, self._ATTR, 0)
        current += 1
        setattr(self._kernel, self._ATTR, current)
        self.depth = current
        if current > self.max_depth:
            raise RecursionSafetyError(
                max_depth=self.max_depth,
                current_depth=current,
                context="auto-extract depth exceeded — newly extracted rules may not have loaded",
            )
        return self

    def __exit__(self, *exc):
        current = getattr(self._kernel, self._ATTR, 1)
        setattr(self._kernel, self._ATTR, max(0, current - 1))
        return False
