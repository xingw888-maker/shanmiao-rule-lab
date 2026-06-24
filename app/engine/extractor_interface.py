"""Extractor interface — Protocol + registry for pluggable extractors.

All extractors (VocabularyExtractor, StructuredRuleExtractor, CandidateExtractor, etc.)
implement this Protocol and register here so the domain builder can discover them.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass
class ExtractionResult:
    """Unified extraction result — all extractors return this."""
    terms: list[dict] = field(default_factory=list)
    bigrams: dict[str, int] = field(default_factory=dict)
    rules: list[dict] = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    extractor_name: str = ""
    domain_hint: str = ""

    def to_dict(self) -> dict:
        return {
            "terms": self.terms,
            "bigrams": dict(self.bigrams),
            "rules": self.rules,
            "stats": self.stats,
            "extractor_name": self.extractor_name,
            "domain_hint": self.domain_hint,
        }


@runtime_checkable
class ExtractorProtocol(Protocol):
    """Protocol that all extractors must implement."""
    name: str  # human-readable name

    def extract(self, text: str, **kwargs) -> ExtractionResult:
        """Extract concepts, terms, bigrams, and/or rules from text."""
        ...


# ── Extractor registry ──
EXTRACTOR_REGISTRY: dict[str, ExtractorProtocol] = {}


def register_extractor(extractor: ExtractorProtocol) -> None:
    """Register an extractor instance."""
    EXTRACTOR_REGISTRY[extractor.name] = extractor


def get_extractor(name: str) -> ExtractorProtocol | None:
    """Get a registered extractor by name."""
    return EXTRACTOR_REGISTRY.get(name)


def list_extractors() -> list[str]:
    """List all registered extractor names."""
    return list(EXTRACTOR_REGISTRY.keys())


def extract_with_all(text: str, domain_hint: str = "") -> list[ExtractionResult]:
    """Run all registered extractors on the same text and return combined results."""
    results = []
    for name, ext in EXTRACTOR_REGISTRY.items():
        try:
            result = ext.extract(text, domain_hint=domain_hint)
            results.append(result)
        except Exception:
            pass
    return results
