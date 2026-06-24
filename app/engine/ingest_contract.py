"""IngestRuleCandidate — system-level ingest contract.

All extractors (StructuredRuleExtractor, LLMRuleExtractor, CandidateExtractor)
MUST normalise their output to IngestRuleCandidate before the result enters
AutoValidator.  This is the single chokepoint every rule candidate passes
through before gate / promote / validate.

Design invariants:
  - condition_type must be in VALID_CONDITION_TYPES (13 values)
  - label must be non-empty
  - source_text must be non-empty (traceability)
  - numeric rules must have expected_value + unit + operator
  - pattern rules must have terms or context_pattern

Normalisation helpers:
  - from_rule_candidate()   ← StructuredRuleExtractor
  - from_pipeline()         ← CandidateExtractor / CandidateProposition
  - from_llm_dict()         ← LLMRuleExtractor (dict with validation)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# ── Whitelist: the 13 condition types the engine knows ──
VALID_CONDITION_TYPES: frozenset[str] = frozenset({
    "numeric_comparison",
    "sum_numeric_comparison",
    "mutual_exclusion",
    "co_occurrence",
    "forbidden_pattern",
    "required_pattern",
    "logical_chain",
    "scope_constraint",
    "contextual_co_occurrence",
    "definition_contains",
    "ast_check",
    "topic_coverage",
    "term_coverage_check",
})

# ── Operator whitelist for numeric rules ──
VALID_OPERATORS: frozenset[str] = frozenset({">=", "<=", ">", "<", "==", "!="})

# ── Extraction method enum ──
EXTRACTION_METHODS: frozenset[str] = frozenset({
    "template", "llm", "llm_extract", "candidate_pipeline", "manual", "legacy",
})


class ContractRejectionError(ValueError):
    """Raised when a candidate fails normalisation — cannot be repaired."""

    def __init__(self, reason: str, raw: dict | None = None):
        self.reason = reason
        self.raw = raw
        super().__init__(reason)


# ═══════════════════════════════════════════════════════════════════════
# IngestRuleCandidate — the contract
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class IngestRuleCandidate:
    """Single rule candidate ready for AutoValidator / promote.

    All extractors MUST produce this exact shape before entering the gate.
    """

    # ── Required (every candidate) ──
    condition_type: str       # Must be in VALID_CONDITION_TYPES
    label: str                # maps to rules.json condition.label
    source_text: str          # original sentence / paragraph (traceability)

    # ── Numeric comparison (only when condition_type == "numeric_comparison") ──
    expected_value: Optional[float] = None
    unit: Optional[str] = None        # 年 月 日 天 %
    operator: str = ">="              # Must be in VALID_OPERATORS

    # ── Pattern / logical rules ──
    terms: list[str] = field(default_factory=list)
    context_pattern: str = ""

    # ── Metadata (every extractor fills these) ──
    extraction_method: str = "template"  # Must be in EXTRACTION_METHODS
    confidence: float = 0.5              # 0..1 — extractor self-reported quality
    source_credibility: float = 0.5      # 0..1 — source authority level
    legal_ref: str = ""                  # e.g. "国务院令第279号第40条"
    legal_hierarchy: str = "unknown"     # law | admin_regulation | dept_rule | GB_std | unknown

    def __post_init__(self):
        """Validate the contract on construction."""
        if self.condition_type not in VALID_CONDITION_TYPES:
            raise ContractRejectionError(
                f"Invalid condition_type '{self.condition_type}'. "
                f"Must be one of {sorted(VALID_CONDITION_TYPES)}",
            )
        if not self.label or not self.label.strip():
            raise ContractRejectionError("label must be non-empty")
        if not self.source_text or not self.source_text.strip():
            raise ContractRejectionError("source_text must be non-empty (traceability)")
        if self.condition_type == "numeric_comparison":
            if self.expected_value is None:
                raise ContractRejectionError(
                    "numeric_comparison requires expected_value",
                )
            if not self.unit:
                raise ContractRejectionError(
                    "numeric_comparison requires unit (年/月/日/%)",
                )
            if self.operator not in VALID_OPERATORS:
                raise ContractRejectionError(
                    f"Invalid operator '{self.operator}'. Must be one of {sorted(VALID_OPERATORS)}",
                )
        # Pattern rules need at least terms or context_pattern
        if self.condition_type in (
            "forbidden_pattern", "required_pattern", "mutual_exclusion",
            "co_occurrence", "contextual_co_occurrence",
        ):
            if not self.terms and not self.context_pattern:
                raise ContractRejectionError(
                    f"{self.condition_type} requires at least 'terms' or 'context_pattern'",
                )
        if self.extraction_method not in EXTRACTION_METHODS:
            raise ContractRejectionError(
                f"Invalid extraction_method '{self.extraction_method}'. "
                f"Must be one of {sorted(EXTRACTION_METHODS)}",
            )

    # ── Normalisation factories ──

    @classmethod
    def from_rule_candidate(cls, rc, **overrides) -> "IngestRuleCandidate":
        """Normalise from StructuredRuleExtractor.RuleCandidate."""
        # Map RuleCandidate.operator to numeric operator if applicable
        numeric_ops = {">=": ">=", "<=": "<=", ">": ">", "<": "<"}
        op = numeric_ops.get(getattr(rc, "operator", ">="), ">=")

        return cls(
            condition_type=getattr(rc, "condition_type", "required_pattern"),
            label=getattr(rc, "subject", "") or "",
            source_text=getattr(rc, "source_text", "") or "",
            expected_value=getattr(rc, "expected_value", None),
            unit=getattr(rc, "unit", None),
            operator=op,
            terms=getattr(rc, "required_terms", None) or [],
            context_pattern=getattr(rc, "context_pattern", "") or "",
            extraction_method=overrides.pop("extraction_method", "template"),
            confidence=float(getattr(rc, "confidence", 0.5) or 0.5),
            source_credibility=overrides.pop("source_credibility", 0.5),
            legal_ref=overrides.pop("legal_ref", ""),
            legal_hierarchy=overrides.pop("legal_hierarchy", "unknown"),
            **overrides,
        )

    @classmethod
    def from_pipeline(cls, cp, **overrides) -> "IngestRuleCandidate":
        """Normalise from candidate_pipeline.CandidateProposition."""
        # Map predicate → condition_type
        pred_map = {
            "REQUIRES": "required_pattern",
            "FORBIDS": "forbidden_pattern",
            "IMPLIES": "contextual_co_occurrence",
            "MUTUALLY_EXCLUSIVE_WITH": "mutual_exclusion",
        }
        pred = getattr(cp, "predicate", "REQUIRES")
        cond_type = pred_map.get(pred, "required_pattern")

        return cls(
            condition_type=overrides.pop("condition_type", cond_type),
            label=getattr(cp, "rule_name", "") or getattr(cp, "subject", ""),
            source_text=getattr(cp, "source_text", "") or "",
            terms=[getattr(cp, "subject", ""), getattr(cp, "object", "")],
            extraction_method=getattr(cp, "extraction_method", "candidate_pipeline"),
            confidence=float(getattr(cp, "confidence", 0.5) or 0.5),
            source_credibility=overrides.pop("source_credibility", 0.5),
            legal_ref=overrides.pop("legal_ref", ""),
            legal_hierarchy=overrides.pop("legal_hierarchy", "unknown"),
            **overrides,
        )

    @classmethod
    def from_llm_dict(cls, d: dict, **overrides) -> "IngestRuleCandidate":
        """Normalise from LLMRuleExtractor output dict — with strict validation.

        This is the most important normaliser because LLM output is
        untyped and can drift.  We validate condition_type against the
        whitelist before constructing.
        """
        cond_type = d.get("condition_type") or d.get("condition", {}).get("type") or ""
        if not cond_type:
            raise ContractRejectionError(
                "LLM dict missing 'condition_type' or 'condition.type'",
                raw=d,
            )
        cond_type = cond_type.strip()
        if cond_type not in VALID_CONDITION_TYPES:
            raise ContractRejectionError(
                f"LLM output has unknown condition_type '{cond_type}'. "
                f"Must be one of {sorted(VALID_CONDITION_TYPES)}",
                raw=d,
            )

        # Extract numeric fields
        condition = d.get("condition", {})
        label = (
            d.get("label") or d.get("subject") or d.get("name")
            or condition.get("label") or ""
        )
        expected = d.get("expected_value") or condition.get("expected") or condition.get("expected_value")
        unit_val = d.get("unit") or condition.get("unit") or ""
        op = d.get("operator") or condition.get("operator") or ">="
        terms_val = (
            d.get("terms") or condition.get("terms")
            or (d.get("required_terms")) or []
        )
        ctx_pat = d.get("context_pattern") or condition.get("context_pattern") or ""
        src_text = d.get("source_text") or d.get("source") or ""

        return cls(
            condition_type=cond_type,
            label=overrides.pop("label", label) or "",
            source_text=overrides.pop("source_text", src_text) or "",
            expected_value=float(expected) if expected is not None else None,
            unit=str(unit_val) if unit_val else None,
            operator=str(op) if op in VALID_OPERATORS else ">=",
            terms=list(terms_val) if terms_val else [],
            context_pattern=str(ctx_pat) if ctx_pat else "",
            extraction_method=overrides.pop("extraction_method", "llm"),
            confidence=float(d.get("confidence", 0.5) or 0.5),
            source_credibility=float(d.get("source_credibility", 0.5) or 0.5),
            legal_ref=overrides.pop("legal_ref", d.get("legal_ref", "") or ""),
            legal_hierarchy=overrides.pop("legal_hierarchy", d.get("legal_hierarchy", "unknown") or "unknown"),
            **overrides,
        )

    def to_rule_dict(self, rule_id: str = "") -> dict:
        """Convert to rules.json-compatible dict for AutoValidator / promote."""
        condition: dict = {"type": self.condition_type}

        if self.condition_type == "numeric_comparison":
            condition.update({
                "label": self.label,
                "context_pattern": self.context_pattern or self.label,
                "unit": self.unit or "年",
                "operator": self.operator,
                "expected": self.expected_value or 0,
                "legal_ref": self.legal_ref,
            })
        elif self.condition_type in ("forbidden_pattern", "required_pattern", "mutual_exclusion"):
            condition["terms"] = self.terms
            if self.context_pattern:
                condition["context_pattern"] = self.context_pattern
        elif self.condition_type in ("co_occurrence", "contextual_co_occurrence"):
            condition["antecedent"] = self.terms[0] if len(self.terms) > 0 else ""
            condition["consequent"] = self.terms[1] if len(self.terms) > 1 else ""
        else:
            condition["label"] = self.label

        return {
            "id": rule_id or self.label,
            "name": self.label,
            "condition": condition,
            "severity": "warning",
            "message": self.label,
            "category": f"auto.{self.condition_type}",
            "source": f"Auto-extracted: {self.source_text[:120]}",
            "source_credibility": self.source_credibility,
            "extraction_method": self.extraction_method,
        }
