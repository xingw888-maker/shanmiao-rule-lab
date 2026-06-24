# P3.1: Handler registry — separate module to avoid circular imports.
# Handlers import register_handler from here.
# core.py imports get_handler from here.
# Version: 2026-06-22

from typing import Callable, Optional

HANDLER_REGISTRY: dict[str, Callable] = {}


def register_handler(condition_type: str, handler_fn: Callable) -> None:
    """Register a condition handler. Called by each handler module on import."""
    HANDLER_REGISTRY[condition_type] = handler_fn


def get_handler(condition_type: str) -> Optional[Callable]:
    """Look up a handler by condition type string. Returns None if unregistered."""
    return HANDLER_REGISTRY.get(condition_type)
