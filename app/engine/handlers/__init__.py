# wo41-core-split: handler modules extracted from core.py PythonMatcher
# Each module contains exactly one _eval_xxx method, plus its private helpers.
# P3.1: Handler registry — all handlers self-register on import.
# core.py reads from here instead of a hardcoded dispatch dict.
# Version: 2026-06-22

# Import registry primitives first so handler modules can self-register
from app.engine.handlers._registry import HANDLER_REGISTRY, register_handler, get_handler

# Trigger handler self-registration — each module imports _registry.register_handler
# and calls it at module level, populating the shared HANDLER_REGISTRY
from app.engine.handlers import ast_check  # noqa: E402
from app.engine.handlers import co_occurrence  # noqa: E402
from app.engine.handlers import contextual_co_occurrence  # noqa: E402
from app.engine.handlers import definition_contains  # noqa: E402
from app.engine.handlers import forbidden_pattern  # noqa: E402
from app.engine.handlers import logical_chain  # noqa: E402
from app.engine.handlers import mutual_exclusion  # noqa: E402
from app.engine.handlers import numeric_comparison  # noqa: E402
from app.engine.handlers import required_pattern  # noqa: E402
from app.engine.handlers import scope_constraint  # noqa: E402
from app.engine.handlers import sum_numeric_comparison  # noqa: E402
from app.engine.handlers import term_coverage_check  # noqa: E402
from app.engine.handlers import topic_coverage  # noqa: E402
