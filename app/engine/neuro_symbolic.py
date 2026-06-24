"""Neuro-symbolic pipeline — LLM extracts propositions, Z3 verifies.

Position in architecture:
  Raw text → LLM extractor → Structured propositions → Z3 Solver → Verdict

This is the bridge between the two weakest points of Citta:
- Translation layer: regex has ~36% noise (doesn't know semantic equivalence)
- Verification layer: Z3 is deterministic but needs clean structured input

LLM fills the semantic gap. Z3 guarantees the deterministic audit trail.

Capability summary:
1. LLM proposition extraction — given a contract paragraph, extract all
   numeric fields with values, units, and context.
2. Z3 constraint verification — same as solver.py, now fed by LLM output.
3. Cross-validation — compare LLM results against regex engine results,
   flag disagreements for human review.
4. Evidence fusion — merge LLM and regex testimation chains into a single
   weighted evidence chain.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from app.engine.solver import (
    CittaZ3Solver,
    LegalConstraint,
    NumericProposition,
    SolverResult,
    SolverVerdict,
    ViolationSeverity,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# Proposition Schema
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class ExtractedProposition:
    """A single proposition extracted by any method (LLM, regex, manual)."""
    field: str               # e.g. "屋面防水保修期限"
    value: float             # e.g. 1.0
    unit: str                # e.g. "年"
    context: str             # surrounding text snippet (for human verification)
    confidence: float        # 0-1, extraction confidence
    extraction_method: str   # "llm", "regex", "keyword", "manual"
    source_text: str = ""    # original text segment
    legal_ref: str = ""      # relevant statute


@dataclass
class ExtractionResult:
    """Complete extraction result from any extractor."""
    propositions: list[ExtractedProposition]
    method: str                    # "llm" | "regex" | "hybrid"
    extraction_time_ms: int = 0
    raw_output: str = ""           # LLM raw response for debugging


# ═══════════════════════════════════════════════════════════════════════
# LLM Proposition Extractor
# ═══════════════════════════════════════════════════════════════════════

class LLMPropositionExtractor:
    """Uses an LLM API to extract structured propositions from contract text.

    The LLM receives a prompt template asking it to identify all numeric
    clauses (durations, percentages, amounts) and output them as a JSON
    array. This is the NEURO half of the neuro-symbolic pipeline.
    """

    PROMPT_TEMPLATE = """你是一名建设工程合同合规审查专家。请从以下合同段落中提取所有包含数值的条款。

对每个数值条款，提取：
- field: 字段名（如"屋面防水保修期限"、"质量保证金比例"、"工期"）
- value: 数值（纯数字）
- unit: 单位（年、月、日、元、%）
- context: 包含该数值的原文段落（20-50字）
- legal_ref: 如果合同引用了法规，列出法规名称

请以 JSON 数组格式输出，每个条目包含 field、value、unit、context、legal_ref 字段。
只输出 JSON，不要加其他文字。

合同段落：
{text}

JSON 输出："""

    def __init__(self, api_url: str = "", api_key: str = "", model: str = ""):
        self.api_url = api_url
        self.api_key = api_key
        self.model = model or "gpt-4"

    async def extract(self, text: str, max_chars: int = 8000) -> ExtractionResult:
        """Extract propositions from contract text via LLM.

        Args:
            text: Contract text (will be truncated to max_chars if needed).
            max_chars: Maximum characters to send to LLM.

        Returns:
            ExtractionResult with structured propositions.
        """
        import time
        start = time.time()

        if not self.api_url or not self.api_key:
            # No LLM configured — return empty with note
            return ExtractionResult(
                propositions=[],
                method="llm",
                raw_output="LLM not configured (no API credentials)",
            )

        text_chunk = text[:max_chars]
        prompt = self.PROMPT_TEMPLATE.format(text=text_chunk)

        try:
            # Use OpenAI-compatible API
            import aiohttp
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,  # deterministic
                "max_tokens": 2000,
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.api_url}/v1/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=60,
                ) as resp:
                    data = await resp.json()
                    content = data["choices"][0]["message"]["content"]
        except Exception as e:
            logger.warning("LLM extraction failed: %s", e)
            return ExtractionResult(
                propositions=[],
                method="llm",
                raw_output=f"Error: {e}",
            )

        # Parse JSON output
        propositions = self._parse_llm_output(content)

        elapsed_ms = int((time.time() - start) * 1000)
        return ExtractionResult(
            propositions=propositions,
            method="llm",
            extraction_time_ms=elapsed_ms,
            raw_output=content,
        )

    def extract_sync(self, text: str, max_chars: int = 8000) -> ExtractionResult:
        """Synchronous wrapper for extract()."""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Already in async context — run in new loop
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run, self.extract(text, max_chars)
                    )
                    return future.result(timeout=120)
            else:
                return loop.run_until_complete(self.extract(text, max_chars))
        except RuntimeError:
            return asyncio.run(self.extract(text, max_chars))
        except Exception as e:
            logger.warning("Sync extraction failed: %s", e)
            return ExtractionResult(propositions=[], method="llm", raw_output=str(e))

    def _parse_llm_output(self, raw: str) -> list[ExtractedProposition]:
        """Parse LLM JSON output into ExtractedProposition objects."""
        propositions = []

        # Strip markdown code fences if present
        clean = raw.strip()
        if clean.startswith("```"):
            clean = re.sub(r'^```\w*\s*', '', clean)
            clean = re.sub(r'\s*```$', '', clean)

        try:
            items = json.loads(clean)
            if isinstance(items, dict):
                items = [items]
        except json.JSONDecodeError:
            # Try to extract JSON array from the response
            match = re.search(r'\[.*\]', clean, re.DOTALL)
            if match:
                try:
                    items = json.loads(match.group(0))
                except json.JSONDecodeError:
                    logger.warning("Could not parse LLM output as JSON: %s", raw[:200])
                    return propositions
            else:
                logger.warning("No JSON found in LLM output: %s", raw[:200])
                return propositions

        for item in items:
            try:
                value = float(item.get("value", 0))
            except (ValueError, TypeError):
                continue

            propositions.append(ExtractedProposition(
                field=item.get("field", ""),
                value=value,
                unit=item.get("unit", ""),
                context=item.get("context", ""),
                confidence=0.85,  # LLM extraction default confidence
                extraction_method="llm",
                source_text=item.get("context", ""),
                legal_ref=item.get("legal_ref", ""),
            ))

        return propositions


# ═══════════════════════════════════════════════════════════════════════
# Evidence Fusion — cross-validates LLM vs regex
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class FusedEvidence:
    """Evidence merged from multiple extraction methods."""
    field: str
    llm_value: Optional[float] = None
    regex_value: Optional[float] = None
    llm_confidence: float = 0.0
    regex_confidence: float = 0.0
    agreement: bool = False        # do both methods agree?
    unit: str = ""
    context: str = ""


class EvidenceFusionEngine:
    """Cross-validates LLM and regex extraction results.

    When both methods agree: confidence is boosted (multiplied by 1.2).
    When they disagree: flag for human review, report both values.
    When only one method found the field: keep at base confidence.
    """

    def fuse(
        self,
        llm_extractions: list[ExtractedProposition],
        regex_extractions: list[dict],  # from engine evidence chain
    ) -> list[FusedEvidence]:
        """Merge LLM and regex extraction results into fused evidence."""
        # Index by field name
        llm_map: dict[str, ExtractedProposition] = {}
        for p in llm_extractions:
            llm_map[p.field] = p

        regex_map: dict[str, dict] = {}
        for r in regex_extractions:
            # Try to match field names
            name = r.get("rule_name", r.get("rule_id", ""))
            regex_map[name] = r

        all_fields = set(list(llm_map.keys()) + list(regex_map.keys()))

        fused_list = []
        for field in all_fields:
            llm_prop = llm_map.get(field)
            regex_ev = regex_map.get(field)

            llm_val = llm_prop.value if llm_prop else None
            regex_val = None
            if regex_ev:
                matched = regex_ev.get("matched_terms", [])
                if matched:
                    try:
                        regex_val = float(re.search(r'[\d.]+', matched[0]).group(0))
                    except (ValueError, AttributeError):
                        pass

            llm_conf = llm_prop.confidence if llm_prop else 0.0
            regex_conf = 0.95 if regex_ev else 0.0  # regex is deterministic within its limits

            # Check agreement
            agreement = False
            if llm_val is not None and regex_val is not None:
                agreement = abs(llm_val - regex_val) < 0.01

            # Boost confidence when both agree
            if agreement:
                llm_conf = min(1.0, llm_conf * 1.2)
                regex_conf = min(1.0, regex_conf * 1.2)

            unit = llm_prop.unit if llm_prop else ""
            context = llm_prop.context if llm_prop else ""

            fused_list.append(FusedEvidence(
                field=field,
                llm_value=llm_val,
                regex_value=regex_val,
                llm_confidence=llm_conf,
                regex_confidence=regex_conf,
                agreement=agreement,
                unit=unit,
                context=context,
            ))

        return fused_list


# ═══════════════════════════════════════════════════════════════════════
# Neuro-Symbolic Pipeline (main orchestrator)
# ═══════════════════════════════════════════════════════════════════════

class NeuroSymbolicPipeline:
    """Full neuro-symbolic pipeline: LLM extract → fuse → Z3 verify.

    Usage:
        pipe = NeuroSymbolicPipeline(llm_url="...", llm_key="...")
        result = await pipe.run(contract_text)
        print(result["proof_summary"])
    """

    def __init__(self, llm_url: str = "", llm_key: str = "", llm_model: str = ""):
        self.extractor = LLMPropositionExtractor(
            api_url=llm_url, api_key=llm_key, model=llm_model,
        )
        self.fuser = EvidenceFusionEngine()
        self.solver = CittaZ3Solver()
        self._has_llm = bool(llm_url and llm_key)

    async def run(
        self,
        text: str,
        constraints: list[LegalConstraint] | None = None,
        engine_evidence: list[dict] | None = None,
    ) -> dict:
        """Run the full neuro-symbolic pipeline.

        Args:
            text: Contract text.
            constraints: Legal constraints (uses built-in construction set if None).
            engine_evidence: Evidence from the regex engine (for fusion).

        Returns:
            Dict with extraction, fusion, and verification results.
        """
        if constraints is None:
            try:
                from domains.construction.constraints import build_constraints as _bcc
                constraints = _bcc()
            except ImportError:
                logger.warning("No constraints available for neuro-symbolic pipeline")
                constraints = []

        # Phase 1: LLM extraction
        if self._has_llm:
            llm_result = await self.extractor.extract(text)
        else:
            # No-LLM fallback: extract from engine evidence chain
            llm_result = ExtractionResult(propositions=[], method="regex")

        # Phase 2: Convert to Z3 propositions
        propositions = []
        for p in llm_result.propositions:
            propositions.append(NumericProposition(
                field=p.field,
                value=p.value,
                unit=p.unit,
                legal_ref=p.legal_ref,
                source_rule_id=f"llm_{p.field}",
            ))

        # Also load from engine evidence if available
        if engine_evidence:
            self.solver.load_from_evidence_chain(engine_evidence)

        # Phase 3: Z3 verification
        if propositions:
            self.solver.load_propositions(propositions)
        self.solver.load_constraints(constraints)
        solver_result = self.solver.check()

        # Phase 4: Evidence fusion (if engine_evidence provided)
        fused = []
        if engine_evidence:
            fused = self.fuser.fuse(llm_result.propositions, engine_evidence)

        # Build response
        disagreements = [f for f in fused if not f.agreement and f.llm_value is not None and f.regex_value is not None]
        agreements = [f for f in fused if f.agreement]

        return {
            "pipe_verdict": solver_result.verdict.value,
            "proof_summary": solver_result.proof_summary,
            "violations": [v.to_dict() for v in solver_result.violations],
            "extraction": {
                "method": llm_result.method,
                "propositions_found": len(llm_result.propositions),
                "extraction_time_ms": llm_result.extraction_time_ms,
                "raw_llm_output": llm_result.raw_output[:500] if llm_result.raw_output else "",
            },
            "fusion": {
                "total_fused": len(fused),
                "agreements": len(agreements),
                "disagreements": len(disagreements),
                "disputed_fields": [
                    {
                        "field": d.field,
                        "llm_value": d.llm_value,
                        "regex_value": d.regex_value,
                        "llm_confidence": d.llm_confidence,
                        "regex_confidence": d.regex_confidence,
                    }
                    for d in disagreements
                ],
            },
            "solver": solver_result.to_dict(),
        }
