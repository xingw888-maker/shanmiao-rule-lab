"""Traceability Tracer — source-to-verdict mapping.

Every proposition, constraint, and violation carries a source trail back to
the original document. When Z3 reports UNSAT, the tracer pinpoints exactly
which lines of text caused the conflict.

This is the bridge from "black-box logic engine" to "academic research tool."

Architecture:
  Source Document (line N, snippet)
    → Extracted Proposition (source_id, line_ref, context)
      → Z3 Constraint (carries source ref)
        → Violation Report (pinpoints conflicting source lines)
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# Source Reference
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class SourceRef:
    """A pointer to a specific location in a source document."""
    document_id: str              # e.g. "regulation_279", "contract_yongjin"
    document_title: str = ""      # e.g. "建设工程质量管理条例"
    line_number: int = 0          # approximate line in source
    char_offset: int = 0          # character offset
    snippet: str = ""             # the actual text (50-150 chars)
    section: str = ""             # chapter/article reference
    confidence: float = 1.0       # how sure we are about this reference

    def to_dict(self) -> dict:
        return {
            "document_id": self.document_id,
            "document_title": self.document_title,
            "line_number": self.line_number,
            "char_offset": self.char_offset,
            "snippet": self.snippet[:200],
            "section": self.section,
            "confidence": self.confidence,
        }


# ═══════════════════════════════════════════════════════════════════════
# Traceable Proposition
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class TraceableProposition:
    """A proposition with full source traceability."""
    field: str
    value: float
    unit: str
    source: SourceRef
    extraction_method: str = ""   # "llm", "regex", "manual", "auto"
    extraction_confidence: float = 0.0
    rule_id: str = ""             # which engine rule extracted this
    metadata: dict = field(default_factory=dict)


@dataclass
class TraceableConstraint:
    """A legal constraint with source traceability."""
    field: str
    operator: str
    threshold: float
    unit: str
    severity: str
    source: SourceRef              # where in the law/regulation this comes from
    weight: float = 1.0


# ═══════════════════════════════════════════════════════════════════════
# Traceable Violation
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class TraceableViolation:
    """A violation with both sides traceable to source documents."""
    field: str
    contract_value: float
    expected_threshold: float
    operator: str
    unit: str
    gap: float
    severity: str

    # Source traceability
    contract_source: Optional[SourceRef] = None    # where in the contract
    regulation_source: Optional[SourceRef] = None  # where in the law

    proof_chain: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "field": self.field,
            "contract_value": self.contract_value,
            "expected_threshold": self.expected_threshold,
            "operator": self.operator,
            "unit": self.unit,
            "gap": self.gap,
            "severity": self.severity,
            "contract_source": self.contract_source.to_dict() if self.contract_source else None,
            "regulation_source": self.regulation_source.to_dict() if self.regulation_source else None,
            "proof_chain": self.proof_chain,
        }


# ═══════════════════════════════════════════════════════════════════════
# Evidence Registry — central registry of all traceable items
# ═══════════════════════════════════════════════════════════════════════

class EvidenceRegistry:
    """Central registry mapping logical items back to source documents.

    This is the "one-stop shop" for traceability. Every proposition,
    constraint, and violation is registered here with its source ref.
    When a user asks "why did this fail?", the registry provides
    the exact source snippet.
    """

    def __init__(self):
        self._propositions: dict[str, TraceableProposition] = {}
        self._constraints: dict[str, TraceableConstraint] = {}
        self._violations: list[TraceableViolation] = []
        self._documents: dict[str, dict] = {}  # doc_id → metadata

    # ── Registration ──

    def register_document(self, doc_id: str, title: str, content: str = "") -> None:
        """Register a source document."""
        self._documents[doc_id] = {
            "id": doc_id,
            "title": title,
            "content_hash": hashlib.md5(content.encode()).hexdigest() if content else "",
            "line_count": content.count('\n') + 1 if content else 0,
        }

    def register_proposition(self, prop: TraceableProposition) -> None:
        """Register a traceable proposition."""
        self._propositions[prop.field] = prop

    def register_constraint(self, constraint: TraceableConstraint) -> None:
        """Register a traceable constraint."""
        self._constraints[constraint.field] = constraint

    def register_violation(self, violation: TraceableViolation) -> None:
        """Register a violation with full traceability."""
        self._violations.append(violation)

    # ── Lookup ──

    def get_contract_source(self, field: str) -> Optional[SourceRef]:
        """Get the contract source for a field."""
        prop = self._propositions.get(field)
        return prop.source if prop else None

    def get_regulation_source(self, field: str) -> Optional[SourceRef]:
        """Get the regulation source for a constraint field."""
        constraint = self._constraints.get(field)
        return constraint.source if constraint else None

    def get_document(self, doc_id: str) -> Optional[dict]:
        """Get document metadata."""
        return self._documents.get(doc_id)

    # ── Report generation ──

    def generate_trace_report(self) -> str:
        """Generate a full traceability report for all violations.

        Format: each violation shows the exact source text from both
        the contract and the regulation, with line numbers.
        """
        if not self._violations:
            return "No violations to trace."

        lines = ["═" * 60, "TRACEABILITY REPORT", "═" * 60, ""]

        for i, v in enumerate(self._violations, 1):
            lines.append(f"Violation #{i}: {v.field}")
            lines.append(f"  Severity: {v.severity.upper()}")
            lines.append(f"  Contract: {v.contract_value}{v.unit} vs "
                         f"Regulation: {v.operator} {v.expected_threshold}{v.unit}")
            lines.append(f"  Gap: {v.gap}{v.unit}")
            lines.append("")

            # Contract side
            if v.contract_source:
                cs = v.contract_source
                lines.append(f"  📄 Contract Source:")
                lines.append(f"     Document: {cs.document_title} ({cs.document_id})")
                if cs.section:
                    lines.append(f"     Section: {cs.section}")
                lines.append(f"     Line: ~{cs.line_number}, Offset: {cs.char_offset}")
                lines.append(f"     Text: \"{cs.snippet}\"")
                lines.append("")

            # Regulation side
            if v.regulation_source:
                rs = v.regulation_source
                lines.append(f"  ⚖️  Regulation Source:")
                lines.append(f"     Document: {rs.document_title} ({rs.document_id})")
                if rs.section:
                    lines.append(f"     Section: {rs.section}")
                lines.append(f"     Text: \"{rs.snippet}\"")
                lines.append("")

            # Proof chain
            if v.proof_chain:
                lines.append(f"  🔗 Proof Chain:")
                for step in v.proof_chain:
                    lines.append(f"     {step}")
                lines.append("")

            lines.append("-" * 60)

        lines.append("")
        lines.append(f"Total violations: {len(self._violations)}")
        lines.append(f"Documents referenced: {len(self._documents)}")
        return "\n".join(lines)

    def generate_evidence_index(self) -> dict:
        """Generate a JSON-serializable evidence index.

        Maps every field to its source references for programmatic access.
        """
        return {
            "documents": self._documents,
            "propositions": {
                field: {
                    "value": prop.value,
                    "unit": prop.unit,
                    "source": prop.source.to_dict(),
                    "extraction_method": prop.extraction_method,
                    "extraction_confidence": prop.extraction_confidence,
                }
                for field, prop in self._propositions.items()
            },
            "constraints": {
                field: {
                    "operator": str(c.operator),
                    "threshold": c.threshold,
                    "unit": c.unit,
                    "severity": c.severity,
                    "source": c.source.to_dict(),
                }
                for field, c in self._constraints.items()
            },
            "violations": [v.to_dict() for v in self._violations],
        }

    # ── Clear ──

    def clear_violations(self) -> None:
        """Clear violation list between runs."""
        self._violations = []


# ═══════════════════════════════════════════════════════════════════════
# Source Annotator — extracts line-numbered snippets from text
# ═══════════════════════════════════════════════════════════════════════

class SourceAnnotator:
    """Extracts source references from raw text.

    Given a position or term in a document, returns a SourceRef with
    line number, surrounding text snippet, and section context.
    """

    @staticmethod
    def annotate_position(
        text: str,
        char_offset: int,
        document_id: str,
        document_title: str = "",
        context_chars: int = 100,
    ) -> SourceRef:
        """Create a SourceRef for a character position in text.

        Args:
            text: Full document text.
            char_offset: Character offset in text.
            document_id: Document identifier.
            document_title: Human-readable title.
            context_chars: Characters of context to include.

        Returns:
            SourceRef with line number, snippet, and section info.
        """
        # Compute line number
        line_number = text[:char_offset].count('\n') + 1

        # Extract snippet
        start = max(0, char_offset - context_chars // 2)
        end = min(len(text), char_offset + context_chars // 2)
        snippet = text[start:end].strip()
        if start > 0:
            snippet = "…" + snippet
        if end < len(text):
            snippet = snippet + "…"

        # Try to find enclosing section (chapter/article header)
        section = ""
        before = text[:char_offset]
        # Look for the nearest "第X章" or "第X条" heading before this position
        import re
        headings = list(re.finditer(
            r'(第[一二三四五六七八九十百千]+[章节条])\s*(.*?)(?:\n|$)',
            before
        ))
        if headings:
            last = headings[-1]
            section = f"{last.group(1)} {last.group(2).strip()}"

        return SourceRef(
            document_id=document_id,
            document_title=document_title,
            line_number=line_number,
            char_offset=char_offset,
            snippet=snippet,
            section=section,
        )

    @staticmethod
    def annotate_term(
        text: str,
        term: str,
        document_id: str,
        document_title: str = "",
    ) -> Optional[SourceRef]:
        """Create a SourceRef for the first occurrence of a term in text."""
        idx = text.find(term)
        if idx == -1:
            return None
        return SourceAnnotator.annotate_position(
            text, idx, document_id, document_title,
        )

    @staticmethod
    def annotate_all_occurrences(
        text: str,
        term: str,
        document_id: str,
        document_title: str = "",
    ) -> list[SourceRef]:
        """Create SourceRefs for all occurrences of a term in text."""
        refs = []
        pos = 0
        while True:
            idx = text.find(term, pos)
            if idx == -1:
                break
            refs.append(SourceAnnotator.annotate_position(
                text, idx, document_id, document_title,
            ))
            pos = idx + 1
        return refs


# ═══════════════════════════════════════════════════════════════════════
# Traced Solver — Z3 with full source traceability
# ═══════════════════════════════════════════════════════════════════════

class TracedSolver:
    """Wraps CittaZ3Solver with evidence registry for full traceability.

    Every check() run produces a trace report showing exactly which
    source lines caused each violation.
    """

    def __init__(self, registry: EvidenceRegistry | None = None):
        self.registry = registry or EvidenceRegistry()
        from app.engine.solver import CittaZ3Solver
        self._solver = CittaZ3Solver()

    def load_propositions(
        self,
        traceable_props: list[TraceableProposition],
    ) -> None:
        """Load traceable propositions into both solver and registry."""
        from app.engine.solver import NumericProposition

        for tp in traceable_props:
            self.registry.register_proposition(tp)
            self._solver.load_propositions([
                NumericProposition(
                    field=tp.field,
                    value=tp.value,
                    unit=tp.unit,
                    source_rule_id=tp.rule_id,
                )
            ])

    def load_constraints(
        self,
        traceable_constraints: list[TraceableConstraint],
    ) -> None:
        """Load traceable constraints into both solver and registry."""
        from app.engine.solver import LegalConstraint, ViolationSeverity

        for tc in traceable_constraints:
            self.registry.register_constraint(tc)
            sev = ViolationSeverity.FATAL if tc.severity == "fatal" else \
                  ViolationSeverity.MAJOR if tc.severity == "major" else \
                  ViolationSeverity.MINOR if tc.severity == "minor" else \
                  ViolationSeverity.ADVISORY

            self._solver.load_constraints([
                LegalConstraint(
                    field=tc.field,
                    operator=tc.operator,
                    threshold=tc.threshold,
                    unit=tc.unit,
                    severity=sev,
                    legal_ref=tc.source.snippet if tc.source else "",
                    weight=tc.weight,
                )
            ])

    def check(self) -> dict:
        """Run Z3 check and produce traceable violations."""
        from app.engine.solver import SolverVerdict

        result = self._solver.check()
        self.registry.clear_violations()

        for v in result.violations:
            contract_src = self.registry.get_contract_source(v.field)
            regulation_src = self.registry.get_regulation_source(v.field)

            proof_chain = []
            if v.proof:
                proof_chain = v.proof.split('\n')

            # Parse expected string like ">= 5.0 年" into components
            parts = v.expected.split() if v.expected else ["?", "0", ""]
            op = parts[0] if len(parts) > 0 else "?"
            threshold_str = parts[1] if len(parts) > 1 else "0"
            unit_str = parts[2] if len(parts) > 2 else ""
            try:
                threshold_val = float(threshold_str)
            except (ValueError, TypeError):
                threshold_val = 0.0

            tv = TraceableViolation(
                field=v.field,
                contract_value=v.actual_value,
                expected_threshold=threshold_val,
                operator=op,
                unit=unit_str,
                gap=v.gap,
                severity=v.severity.value,
                contract_source=contract_src,
                regulation_source=regulation_src,
                proof_chain=proof_chain,
            )
            self.registry.register_violation(tv)

        trace_report = self.registry.generate_trace_report()
        evidence_index = self.registry.generate_evidence_index()

        return {
            "verdict": result.verdict.value,
            "violations": [v.to_dict() for v in self.registry._violations],
            "trace_report": trace_report,
            "evidence_index": evidence_index,
            "summary": result.proof_summary,
        }
