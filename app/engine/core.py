"""Pure Python validation engine — full implementation.
This engine is the authoritative implementation of the Citta validation logic.
It serves as the fallback when the Rust `citta_core` native module is unavailable,
but implements the exact same semantics.
The engine handles all 6 rule condition types:
- mutual_exclusion
- co_occurrence
- forbidden_pattern
- required_pattern
- logical_chain
- scope_constraint
- numeric_comparison (v2: extract numeric values from text, compare against legal thresholds)
"""
import json
import logging
import os
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
logger = logging.getLogger(__name__)
# ======================================================================
# Enums
# ======================================================================
class Verdict(str, Enum):
    PASSED = "PASSED"
    FAILED = "FAILED"
    NOT_APPLICABLE = "NOT_APPLICABLE"
class ValidationStatus(str, Enum):
    VALIDATED = "VALIDATED"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    UNDEFINED = "UNDEFINED"
    CONFLICTED = "CONFLICTED"
class Severity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
# ======================================================================
# Data structures
# ======================================================================
@dataclass
class EvidenceItem:
    """A single evidence entry in the chain."""
    trace_id: str
    rule_id: str
    rule_name: str
    rule_version: str
    package_id: str
    package_version: str
    severity: str
    status: str  # PASSED or FAILED
    input_fragment: str
    segment_id: Optional[str]
    matched_terms: list[str]
    rationale: str
    suggestion: str
    category: str = ""
    # ── Knowledge layer fields (added for knowledge layering) ──
    source_type: str = ""            # "官方机构" | "行业标准" | "LLM提取" | "个人编写" | ""
    source_credibility: float = 0.5  # 0-1
    extraction_method: str = ""      # "manual" | "keyword_scan" | "llm_extract" | "conjecture_mine"
    layer: str = ""                  # "L0_VALIDATED" | "L1_CONJECTURE" | "L2_SOURCE_UNCERTAIN" | "L3_OUTER_POSSIBILITY"
    legal_hierarchy: str = ""        # constitution|law|admin_regulation|local_regulation|dept_rule|gb_standard|industry_standard|other
@dataclass
class Conflict:
    """A conflict between two rule verdicts."""
    conflict_id: str
    conflict_type: str  # intra_package | cross_package | scope_overlap
    rule_a: dict
    rule_b: dict
    description: str
@dataclass
class ValidationResult:
    """Complete validation result."""
    status: ValidationStatus
    evidence_chain: list[EvidenceItem]
    conflicts: list[Conflict]
    summary: dict
    package_versions: dict[str, str]
    processing_ms: int
    def to_dict(self) -> dict:
        return {
            "status": self.status.value,
            "evidence_chain": [
                {
                    "trace_id": e.trace_id,
                    "rule_id": e.rule_id,
                    "rule_name": e.rule_name,
                    "rule_version": e.rule_version,
                    "package_id": e.package_id,
                    "package_version": e.package_version,
                    "severity": e.severity,
                    "status": e.status,
                    "input_fragment": e.input_fragment,
                    "segment_id": e.segment_id,
                    "matched_terms": e.matched_terms,
                    "rationale": e.rationale,
                    "suggestion": e.suggestion,
                    "category": e.category,
                    "source_type": e.source_type,
                    "source_credibility": e.source_credibility,
                    "extraction_method": e.extraction_method,
                    "layer": e.layer,
                    "legal_hierarchy": e.legal_hierarchy,
                }
                for e in self.evidence_chain
            ],
            "conflicts": [
                {
                    "conflict_id": c.conflict_id,
                    "conflict_type": c.conflict_type,
                    "rule_a": c.rule_a,
                    "rule_b": c.rule_b,
                    "description": c.description,
                }
                for c in self.conflicts
            ],
            "summary": self.summary,
            "package_versions": self.package_versions,
        }
@dataclass
class CompiledRule:
    """A compiled rule ready for matching."""
    id: str
    name: str
    condition_type: str
    condition_params: dict
    severity: str
    message: str
    category: str
    version: str
    package_id: str
    package_version: str
    # Pre-compiled regex patterns
    compiled_pattern: Optional[re.Pattern] = None
    # ── Source tracking fields ──
    source: str = ""                 # 来源描述（机构/URL/提取方式）
    source_credibility: float = 0.5  # 来源可信度 0-1
    extraction_method: str = ""      # "manual" | "keyword_scan" | "llm_extract" | "conjecture_mine"
    # ── Clause type for context scoping ──
    clause_type: str = ""            # "保修", "付款", "验收", "违约", "安全", "其他", etc.
    # ── Contract broad-type exclusion ──
    exclude_contract_types: list = field(default_factory=list)
    # ── Minimum contract value filtering ──
    min_contract_value: float = 0.0
    # ── Legal hierarchy level ──
    legal_hierarchy: str = ""  # constitution|law|admin_regulation|local_regulation|dept_rule|gb_standard|industry_standard|other
    # ── Pre-computed direction for numeric_comparison (P1.1 → P1.2) ──
    precomputed_direction: str = ""  # ">=", "<=", "==" or "" if not pre-computed
@dataclass
class CompiledPackage:
    """A compiled rule package."""
    id: str
    name: str
    version: str
    domain: str
    rules: list[CompiledRule] = field(default_factory=list)
# ======================================================================
# Tokeniser
# ======================================================================
class Tokeniser:
    """Tokenises input text for rule matching.
    Splits on whitespace and punctuation; for CJK text, emits single characters
    plus sliding-window substrings (2-6 chars) so multi-character Chinese terms
    are directly matchable in the ngram set.
    """
    _CJK = re.compile(r'[一-鿿㐀-䶿豈-﫿]+')
    _TOKEN = re.compile(r"[一-鿿㐀-䶿豈-﫿]+|[a-z0-9_]+")

    @staticmethod
    def tokenise(text: str) -> list[str]:
        """Tokenise text into lowercase tokens (unigrams + CJK substrings)."""
        text_lower = text.lower()
        tokens: list[str] = []
        for m in Tokeniser._TOKEN.finditer(text_lower):
            segment = m.group()
            if Tokeniser._CJK.match(segment):
                # Emit each single char
                tokens.extend(segment)
                # Emit sliding-window substrings for phrase matching
                n = len(segment)
                for length in range(2, min(7, n + 1)):
                    for i in range(n - length + 1):
                        tokens.append(segment[i:i + length])
            else:
                tokens.append(segment)
        return tokens

    @staticmethod
    def ngrams(tokens: list[str], n: int = 3) -> set[str]:
        """Generate n-gram sequences up to n for phrase matching."""
        ngrams_set: set[str] = set()
        ngrams_set.update(tokens)
        for size in range(2, n + 1):
            for i in range(len(tokens) - size + 1):
                ngrams_set.add(" ".join(tokens[i : i + size]))
        return ngrams_set
# ======================================================================
# Rule Compiler
# ======================================================================
class PythonRuleCompiler:
    """Compiles RulePackage JSON into internal CompiledPackage objects.

    Entity alias expansion: when aliases are provided (from domain ontology.json),
    every term in every rule is expanded to all known synonyms before compilation.
    E.g. "保修期" → "保修期|保修期限|保修期间|保修时间" in regex patterns,
    and to all individual variants in term lists.  This means rules written once
    in standard terminology match contracts written in any known variant.
    """

    @staticmethod
    def load_aliases(domain_path: str) -> dict[str, list[str]]:
        """Load entity alias groups from a domain's ontology.json.

        Returns a flat dict mapping each canonical term to ALL its variants
        (including itself), e.g.:
          {"保修期": ["保修期","保修期限","保修期间","保修时间"],
           "发包人": ["发包人","建设单位","甲方","业主","招标人"], ...}

        If ontology.json does not exist or has no entity_groups, returns {}.
        """
        onto_path = os.path.join(domain_path, "ontology.json")
        if not os.path.isfile(onto_path):
            return {}
        try:
            with open(onto_path, "r", encoding="utf-8") as f:
                onto = json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}
        groups = onto.get("entity_groups", {})
        if not isinstance(groups, dict):
            return {}
        # Build flat lookup: every variant → list of all variants in its group
        aliases: dict[str, list[str]] = {}
        for group_key, variants in groups.items():
            if not isinstance(variants, list) or len(variants) < 2:
                continue
            all_variants = list(variants)
            for v in all_variants:
                if isinstance(v, str) and v.strip():
                    aliases[v.strip()] = all_variants
        return aliases

    @staticmethod
    def _expand_terms(terms: list[str], aliases: dict[str, list[str]]) -> list[str]:
        """Expand a list of terms through entity aliases.

        Each input term is replaced by all its known synonym variants.
        Terms not found in aliases are kept as-is.
        """
        if not aliases or not terms:
            return list(terms)
        seen: set[str] = set()
        result: list[str] = []
        for t in terms:
            if not t:
                continue
            variants = aliases.get(t, [t])
            for v in variants:
                v_lower = v.lower()
                if v_lower not in seen:
                    seen.add(v_lower)
                    result.append(v)
        return result

    @staticmethod
    def _expand_pattern(pattern_str: str, aliases: dict[str, list[str]]) -> str:
        """Expand a regex pattern by replacing known terms with their alias groups.

        E.g. pattern "(?i)(保修)" with aliases {"保修": ["保修","保修期限","保修期"]}
        → "(?i)(保修|保修期限|保修期)"

        Only replaces terms that are STRICTLY wrapped in regex group parentheses,
        i.e. they were originally generated from terms by RulePackageBuilder.
        Patterns that don't look like term-based expressions are returned unchanged.
        """
        if not aliases or not pattern_str:
            return pattern_str
        # Only expand patterns that look like term-based OR groups: (?i)(a|b|c)
        # or simple term patterns: (?i)(a)
        m = re.match(r'^\(\?i\)\((.+)\)$', pattern_str.strip())
        if not m:
            return pattern_str
        inner = m.group(1)
        parts = inner.split('|')
        expanded_parts: list[str] = []
        for part in parts:
            part_stripped = part.strip()
            if part_stripped in aliases:
                # Expand this term to all its aliases
                for v in aliases[part_stripped]:
                    expanded_parts.append(re.escape(v))
            else:
                expanded_parts.append(part)
        if expanded_parts == list(parts):
            return pattern_str  # nothing changed
        return f"(?i)({'|'.join(expanded_parts)})"

    @staticmethod
    def compile(package_data: dict, aliases: dict[str, list[str]] | None = None,
                llm_url: str = "", llm_key: str = "",
                llm_model: str = "") -> CompiledPackage:
        """Compile a single rule package with optional entity alias expansion.

        When llm_url and llm_key are provided, numeric_comparison rules get
        their semantic direction pre-computed at compile time.
        """
        pkg_id = package_data.get("id", package_data.get("name", "unknown"))
        pkg_name = package_data.get("name", pkg_id)
        version = package_data.get("version", "0.0.0")
        domain = package_data.get("domain", "")
        rules_data = package_data.get("rules", [])
        compiled_rules: list[CompiledRule] = []
        for rule_data in rules_data:
            compiled = PythonRuleCompiler._compile_rule(
                rule_data, pkg_id, version, aliases,
                llm_url, llm_key, llm_model,
            )
            compiled_rules.append(compiled)
        return CompiledPackage(
            id=pkg_id,
            name=pkg_name,
            version=version,
            domain=domain,
            rules=compiled_rules,
        )
    @staticmethod
    def _compile_rule(rule_data: dict, pkg_id: str, pkg_version: str,
                      aliases: dict[str, list[str]] | None = None,
                      llm_url: str = "", llm_key: str = "",
                      llm_model: str = "") -> CompiledRule:
        """Compile a single rule, expanding terms through entity aliases first.

        When llm_url and llm_key are provided, numeric_comparison rules with
        direction-ambiguous context_patterns get their direction pre-computed
        at compile time, avoiding LLM calls during evaluation.
        """
        rule_id = rule_data.get("id", str(uuid.uuid4()))
        condition = rule_data.get("condition", {})
        cond_type = condition.get("type", "")

        if aliases:
            condition = dict(condition)  # shallow copy so we don't mutate the original
            # Expand terms in term-list based types
            if cond_type == "mutual_exclusion" and "terms" in condition:
                condition["terms"] = PythonRuleCompiler._expand_terms(
                    condition["terms"], aliases)
            elif cond_type == "co_occurrence":
                if "antecedent" in condition:
                    expanded = PythonRuleCompiler._expand_terms(
                        [condition["antecedent"]], aliases)
                    condition["antecedent"] = expanded[0] if expanded else condition["antecedent"]
                if "consequent" in condition:
                    expanded = PythonRuleCompiler._expand_terms(
                        [condition["consequent"]], aliases)
                    condition["consequent"] = expanded[0] if expanded else condition["consequent"]
            elif cond_type == "logical_chain":
                if "premises" in condition:
                    condition["premises"] = PythonRuleCompiler._expand_terms(
                        condition["premises"], aliases)
                if "conclusion" in condition:
                    expanded = PythonRuleCompiler._expand_terms(
                        [condition["conclusion"]], aliases)
                    condition["conclusion"] = expanded[0] if expanded else condition["conclusion"]
            # Expand context_pattern for numeric_comparison (helps region-finding)
            if "context_pattern" in condition and isinstance(condition["context_pattern"], str):
                condition["context_pattern"] = PythonRuleCompiler._expand_pattern(
                    condition["context_pattern"], aliases)

        compiled_pattern = None
        if cond_type in ("forbidden_pattern", "required_pattern"):
            pattern_str = condition.get("pattern", "")
            if aliases and pattern_str:
                pattern_str = PythonRuleCompiler._expand_pattern(pattern_str, aliases)
            try:
                compiled_pattern = re.compile(pattern_str, re.IGNORECASE)
            except re.error as e:
                from app.exceptions import CompilationError
                raise CompilationError(
                    f"Rule {rule_id}: invalid regex pattern '{pattern_str}': {e}"
                )
        # Extract exclude_contract_types from scope
        scope = rule_data.get("scope", {})
        exclude_contract_types = scope.get("exclude_contract_types", []) if isinstance(scope, dict) else []
        # Extract min_contract_value from scope
        min_contract_value = 0.0
        if isinstance(scope, dict):
            min_contract_value = float(scope.get("min_contract_value", 0))

        # ── Direction pre-computation for numeric_comparison rules ──
        # Pre-compute semantic direction at compile time so the evaluation
        # path does not need to call LLM (constitutional constraint P1.1).
        # LLM unavailability is silently degraded: no direction field written.
        if cond_type == "numeric_comparison" and llm_url and llm_key:
            context_pattern = condition.get("context_pattern", "")
            if isinstance(context_pattern, str) and _needs_direction_resolution(context_pattern):
                # Extract the numeric value from context_pattern (e.g. "不低于 1000 万元")
                import re as _re
                nums = _re.findall(r'[\d.]+', context_pattern)
                for num_str in nums:
                    direction = _call_llm_direction(
                        context_pattern, num_str,
                        llm_url, llm_key, llm_model or "deepseek-chat",
                    )
                    if direction:
                        condition["direction"] = direction
                        break  # first successful parse is enough

        return CompiledRule(
            id=rule_id,
            name=rule_data.get("name", rule_id),
            condition_type=cond_type,
            condition_params=condition,
            severity=rule_data.get("severity", "error"),
            message=rule_data.get("message", ""),
            category=rule_data.get("category", ""),
            version=rule_data.get("version", pkg_version),
            package_id=pkg_id,
            package_version=pkg_version,
            compiled_pattern=compiled_pattern,
            source=rule_data.get("source", ""),
            source_credibility=rule_data.get("source_credibility", 0.5),
            extraction_method=rule_data.get("extraction_method", ""),
            clause_type=rule_data.get("clause_type", ""),
            exclude_contract_types=exclude_contract_types,
            min_contract_value=min_contract_value,
            legal_hierarchy=rule_data.get("legal_hierarchy", ""),
            precomputed_direction=condition.get("direction", ""),
        )
# ======================================================================
# Module-level helpers
# ======================================================================

_DIRECTION_KEYWORDS: set[str] = {
    # Chinese (simplified)
    "不低于", "不低於", "不足", "不少于", "不多于", "不高于", "不低於",
    "不超过", "超出", "达到", "不到", "至少", "最少", "最多", "至多",
    "高于", "低于", "超過", "低於", "以上", "以下", "仅为",
    # Chinese (traditional)
    "不低於", "超過", "低於",
}


def _call_llm_direction(snippet: str, number: str,
                        llm_url: str, llm_key: str, llm_model: str) -> str | None:
    """Call LLM to determine the semantic direction of a numeric clause.

    Module-level function so both compile-time and run-time paths can use it
    without duplicating the HTTP call logic.

    Returns '>=', '<=', '==' or None on failure.
    """
    if not llm_url or not llm_key:
        return None

    prompt = (
        '你是一个数值语义分析器。给定一句话和一个数值，判断这句话中该数值是：\n'
        '- 承诺的最小值/下限（>=）：如不少于、至少、不低于、达到、满足\n'
        '- 承诺的最大值/上限（<=）：如不超过、不高于、不足、少于、低于、仅、限于\n'
        '- 精确值（==）：如为、约定为、确定为、等于\n\n'
        '只回复 >= 、 <= 或 == 三个符号之一，不要加任何文字。\n\n'
        '句子：{snippet}\n'
        '数值：{number}\n'
        '方向：'
    ).format(snippet=snippet, number=number)

    try:
        import aiohttp
        import asyncio

        async def _call():
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{llm_url.rstrip('/')}/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {llm_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": llm_model,
                        "max_tokens": 4,
                        "temperature": 0.0,
                        "messages": [{"role": "user", "content": prompt}],
                    },
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status != 200:
                        return None
                    data = await resp.json()
                    return data["choices"][0]["message"]["content"].strip()

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, _call())
                    answer = future.result(timeout=15)
            else:
                answer = loop.run_until_complete(_call())
        except RuntimeError:
            answer = asyncio.run(_call())
    except Exception:
        return None

    if answer and answer.strip() in (">=", "<=", "=="):
        return answer.strip()
    return None


def _needs_direction_resolution(context_pattern: str) -> bool:
    """Check whether a context_pattern contains direction-ambiguous keywords.

    Returns True if any known direction keyword appears in the pattern,
    meaning LLM-based resolution is needed.
    """
    pattern_lower = context_pattern.lower()
    return any(kw.lower() in pattern_lower for kw in _DIRECTION_KEYWORDS)


# ======================================================================
# Matcher
# ======================================================================
class PythonMatcher:
    """Evaluates compiled rules against tokenised input."""
    def __init__(self, llm_url: str = "", llm_key: str = "", llm_model: str = "",
                 entity_groups: Optional[dict] = None):
        self._tokeniser = Tokeniser()
        self._llm_url = llm_url
        self._llm_key = llm_key
        self._llm_model = llm_model or "deepseek-chat"
        self._direction_cache: dict[str, str] = {}  # text hash → direction op
        # Entity groups from domain ontology for term expansion at match time
        # Format: {"group_name": ["term1", "term2", ...]}
        self._entity_groups: dict[str, list[str]] = entity_groups or {}
        # Road 2: sidecar structured_inputs, keyed by rule_id (survives rule copies)
        self._structured_inputs: dict[str, dict] = {}

    def _resolve_direction(self, snippet: str, raw_str: str, label: str) -> str | None:
        """Ask LLM to determine the semantic direction of a numeric clause.

        Returns '>=', '<=', '==' or None on failure.  Results are cached by
        text hash so identical snippets only call LLM once per session.

        Delegates to the module-level _call_llm_direction to avoid code
        duplication with compile-time pre-computation.
        """
        import hashlib
        cache_key = hashlib.sha256((snippet + raw_str).encode()).hexdigest()[:16]
        if cache_key in self._direction_cache:
            return self._direction_cache[cache_key]

        answer = _call_llm_direction(
            snippet, raw_str,
            self._llm_url, self._llm_key, self._llm_model,
        )

        if answer and answer.strip() in (">=", "<=", "=="):
            self._direction_cache[cache_key] = answer.strip()
            return answer.strip()
        return None
    def evaluate(
        self,
        rule: CompiledRule,
        text: str,
        tokens: list[str],
        ngrams: set[str],
        segment_text: Optional[str] = None,
        clause_blocks: Optional[list[dict]] = None,
        contract_broad_type: str = "",
        estimated_value: float = 0.0,
    ) -> EvidenceItem:
        """Evaluate a single compiled rule against input.
        Args:
            rule: The compiled rule to evaluate.
            text: The full input text (for fragment extraction).
            tokens: Tokenised text (unigrams).
            ngrams: N-gram token set for phrase matching.
            segment_text: If evaluating a specific segment, its text.
            clause_blocks: Optional list of clause block dicts ({clause_type, content, ...})
                           for clause-type scoped matching. Only used by numeric_comparison
                           and sum_numeric_comparison handlers.
            contract_broad_type: Broad contract type (e.g. "建设工程", "购销合同").
                                 If this value is in rule.exclude_contract_types, the rule
                                 returns NOT_APPLICABLE immediately.
            estimated_value: Estimated contract value in yuan. If below rule.min_contract_value,
                             the rule returns NOT_APPLICABLE.
        Returns:
            An EvidenceItem with the verdict.
        """
        # ── Contract broad-type exclusion ──
        # If the rule has exclude_contract_types and the current contract's broad_type
        # is in that list, skip the rule entirely.  This check happens BEFORE clause_type
        # filtering so the exclusion takes precedence.
        if contract_broad_type and rule.exclude_contract_types:
            if contract_broad_type in rule.exclude_contract_types:
                return self._make_evidence(
                    rule, Verdict.NOT_APPLICABLE, text[:200], [],
                    f"Rule '{rule.name}' excludes contract type '{contract_broad_type}'. "
                    f"Excluded types: {rule.exclude_contract_types}.",
                    "", None,
                )

        # ── Minimum contract value filtering ──
        # If the rule has min_contract_value and the contract's estimated value
        # is below that threshold, skip the rule.  This guards against expensive
        # or inappropriate rules (e.g. cn-003主体结构保修 on a 15万维修合同).
        if rule.min_contract_value > 0 and estimated_value > 0 and estimated_value < rule.min_contract_value:
            return self._make_evidence(
                rule, Verdict.NOT_APPLICABLE, text[:200], [],
                f"Rule '{rule.name}' requires minimum contract value of {rule.min_contract_value:.0f} yuan. "
                f"Estimated contract value: {estimated_value:.0f} yuan.",
                "", None,
            )

        # ── Material purchase contract detection ──
        # Even if broad_type is not "购销合同", detect clear material purchase/supply
        # contracts by their specific terminology (买受方/出卖方).  This prevents
        # cn-016 (安全生产) from firing on material purchase contracts that happen
        # to contain the word "工程质量" (which triggers "建设工程" broad_type).
        if "买受方" in text and "出卖方" in text and rule.exclude_contract_types:
            if "购销合同" in rule.exclude_contract_types:
                return self._make_evidence(
                    rule, Verdict.NOT_APPLICABLE, text[:200], [],
                    f"Rule '{rule.name}' excluded: text uses purchase terminology (买受方/出卖方).",
                    "", None,
                )
        target_text = segment_text or text
        target_tokens = self._tokeniser.tokenise(target_text) if segment_text else tokens
        target_ngrams = self._tokeniser.ngrams(target_tokens) if segment_text else ngrams
        from app.engine.handlers import get_handler
        handler = get_handler(rule.condition_type)
        if handler is None:
            return self._make_evidence(
                rule, Verdict.NOT_APPLICABLE, target_text, [],
                f"Unknown condition type: {rule.condition_type}", "", None
            )

        # ── Clause-type text scoping ──
        # Store clause_blocks on self so handlers (numeric_comparison, etc.)
        # can access it for regex fallback scoping even when no matching blocks exist.
        self._clause_blocks = clause_blocks

        # If the rule has a clause_type AND clause_blocks were provided,
        # restrict the search text to only matching clause blocks.
        # This prevents cross-clause numeric cross-talk (e.g. a "保修" rule
        # picking up a "50天" from a "工期" clause).
        #
        # Fallback: if no matching clause blocks, evaluate on full text
        # instead of returning NOT_APPLICABLE.  This handles contracts
        # where the clause splitter misclassifies blocks or the contract
        # format isn't recognised by the splitter.
        if rule.clause_type and clause_blocks:
            clause_text_parts = []
            for cb in clause_blocks:
                if cb.get("clause_type") == rule.clause_type:
                    clause_text_parts.append(cb.get("content", ""))
            if clause_text_parts:
                clause_text = "\n".join(clause_text_parts)
                # Re-tokenise for the scoped text
                clause_tokens = self._tokeniser.tokenise(clause_text)
                clause_ngrams = self._tokeniser.ngrams(clause_tokens)
                # handler is a function (not a bound method) — pass self explicitly
                return handler(self, rule, clause_text, clause_tokens, clause_ngrams, text)
            # else: no matching clause blocks found — fall through to
            # full-text evaluation instead of returning NOT_APPLICABLE.
            # This ensures rules still fire even when the clause splitter
            # fails to identify the correct clause type.

        return handler(self, rule, target_text, target_tokens, target_ngrams, text)
    def _make_evidence(
        self,
        rule: CompiledRule,
        verdict: Verdict,
        fragment: str,
        matched_terms: list[str],
        rationale: str,
        suggestion: str,
        segment_id: Optional[str],
    ) -> EvidenceItem:
        """Create an EvidenceItem from rule evaluation results."""
        status_str = verdict.value
        return EvidenceItem(
            trace_id=f"evt_{uuid.uuid4().hex[:8]}",
            rule_id=rule.id,
            rule_name=rule.name,
            rule_version=rule.version,
            package_id=rule.package_id,
            package_version=rule.package_version,
            severity=rule.severity,
            status=status_str,
            input_fragment=fragment[:500],  # Cap fragment length
            segment_id=segment_id,
            matched_terms=matched_terms,
            rationale=f"{status_str}: {rationale}",
            suggestion=suggestion,
            category=rule.category,
            source_type=rule.source or "",
            source_credibility=rule.source_credibility,
            extraction_method=rule.extraction_method,
            layer="",  # populated later by knowledge layering engine
            legal_hierarchy=rule.legal_hierarchy,
        )

    # ── Negation / compliance-conditional markers ──
    _NEGATION_WORDS = re.compile(
        r'不得|禁止|不应|不能|不可|严禁|不许|不准|切勿|不得将|不允许'
    )
    _PERMISSIVE_PREFIX = re.compile(
        r'可低于|可不|可以低于|可以不|有权|须垫|必须垫|应当垫|需垫|应予垫|由.*垫'
    )
    _OBLIGATION_MARKERS = re.compile(
        r'应当|必须|应(?=对|当|按|由|在|向|于|经)'
    )
    # Narrow negation words for required_pattern — only compound negators
    # that indicate explicit refusal to provide/grant/warrant something.
    # Single-character negators (不, 无, 非, 未) are excluded because they
    # are too ambiguous in Chinese: "协商不成的" ("if negotiation fails")
    # negates 成 (result), not 协商 (negotiation).  "未约定付款期限"
    # ("payment deadline not specified") is a statement of absence, not
    # a declaration of refusal.
    _BROAD_NEGATION_WORDS = re.compile(
        r'不另行|不单独|不包含|不出具|不再出具|不另出具|不予出具|'
        r'不提供|不另提供|不单独提供|不签订|不另签订|不另行签订|'
        r'不约定|不另约定|'
        r'不得|禁止|不应|不能|不可|严禁|不许|不准|不允许'
    )

    @classmethod
    def _check_negation_context(cls, text: str, term: str, window: int = 50) -> bool:
        """Return True if term appears in a negation/compliance-conditional context."""
        text_lower = text.lower()
        term_lower = term.lower()
        if term_lower not in text_lower:
            return False
        idx = 0
        compliant_count = 0
        total_count = 0
        while True:
            pos = text_lower.find(term_lower, idx)
            if pos == -1:
                break
            total_count += 1
            before_start = max(0, pos - window)
            before = text_lower[before_start:pos]
            after_end = min(len(text_lower), pos + len(term_lower) + window)
            after = text_lower[pos + len(term_lower):after_end]
            ctx_start = max(0, pos - window)
            ctx_end = min(len(text_lower), pos + len(term_lower) + window)
            surrounding = text_lower[ctx_start:ctx_end]
            has_before = bool(cls._NEGATION_WORDS.search(before))
            has_after = bool(cls._NEGATION_WORDS.search(after))
            has_obligation = bool(cls._OBLIGATION_MARKERS.search(surrounding))
            has_permissive = bool(cls._PERMISSIVE_PREFIX.search(surrounding))
            if has_permissive:
                pass
            elif has_before or has_after or has_obligation:
                compliant_count += 1
            idx = pos + 1
        if total_count == 0:
            return False
        return compliant_count > 0 and compliant_count >= total_count / 2

    _SENTENCE_BOUNDARY = re.compile(r'[。；;！!\n##]')

    # Punctuation that breaks a clause (used to prevent false negation detection
    # across clause boundaries, e.g. "协商不成的" should NOT make "协商" negated).
    _CLAUSE_BREAK = re.compile(r'[，,。；;！!\n：:、]')

    @classmethod
    def _all_occurrences_negated(cls, text: str, term: str, window: int = 50) -> bool:
        """Return True if EVERY occurrence of term in text is in a negation context.

        Unlike _check_negation_context (which returns True if at least half of
        occurrences are negated), this requires 100% of occurrences to be negated.
        Used by required_pattern to detect terms that are mentioned but explicitly
        negated (e.g. "不另行签订质量保修书").

        CRITICAL: negator must be in the same sentence as the term AND within 5
        characters before the term with no punctuation between them.  A "协商不成的"
        pattern ("if negotiation fails") must NOT make "协商" appear negated —
        the negation word "不" modifies "成" (fails), not "协商" (negotiation).
        We enforce this by requiring:
          1. Negator is strictly BEFORE the term (not after)
          2. No comma/semicolon/period between negator and term
          3. Distance from negator end to term start <= 5 characters
        """
        text_lower = text.lower()
        term_lower = term.lower()
        if term_lower not in text_lower:
            return False
        idx = 0
        negated_count = 0
        total_count = 0
        while True:
            pos = text_lower.find(term_lower, idx)
            if pos == -1:
                break
            total_count += 1
            before_start = max(0, pos - window)
            before = text_lower[before_start:pos]
            ctx_start = max(0, pos - window)
            ctx_end = min(len(text_lower), pos + len(term_lower) + window)
            surrounding = text_lower[ctx_start:ctx_end]
            has_permissive = bool(cls._PERMISSIVE_PREFIX.search(surrounding))

            if not has_permissive:
                # Find the LAST negation match in the `before` segment
                neg_matches = list(cls._BROAD_NEGATION_WORDS.finditer(before))
                if neg_matches:
                    neg_match = neg_matches[-1]  # closest negator to the term
                    neg_end = neg_match.end()
                    neg_in_text = before_start + neg_end
                    gap = text_lower[neg_in_text:pos]
                    # Condition 1: no clause-breaking punctuation between negator and term
                    # Condition 2: distance <= 5 chars (negator end to term start)
                    if not cls._CLAUSE_BREAK.search(gap) and len(gap) <= 5:
                        negated_count += 1
            idx = pos + 1
        if total_count == 0:
            return False
        return negated_count == total_count

    def _eval_mutual_exclusion(
        self, rule: CompiledRule, text: str, tokens: list[str], ngrams: set[str], full_text: str
    ) -> EvidenceItem:
        """mutual_exclusion: N terms cannot co-occur above a threshold."""
        terms = rule.condition_params.get("terms", [])
        threshold = rule.condition_params.get("threshold", 2)
        matched_terms = [t for t in terms if t.lower() in ngrams]
        if len(matched_terms) >= threshold:
            fragment = self._extract_fragment(text, matched_terms)
            msg = rule.message.format(matched=", ".join(matched_terms), threshold=threshold)
            return self._make_evidence(
                rule, Verdict.FAILED, fragment, matched_terms,
                f"Two mutually-exclusive terms co-occur. Threshold is {threshold - 1}, found {len(matched_terms)}.",
                f"Remove or qualify at least one of: {', '.join(matched_terms)}.",
                None,
            )
        return self._make_evidence(
            rule, Verdict.PASSED, text[:200], matched_terms,
            f"No mutual exclusion violation. Found {len(matched_terms)} of threshold {threshold}.",
                "", None,
        )
    def _eval_co_occurrence(
        self, rule: CompiledRule, text: str, tokens: list[str], ngrams: set[str], full_text: str
    ) -> EvidenceItem:
        """co_occurrence: If antecedent appears, consequent must also appear."""
        antecedent = rule.condition_params.get("antecedent", "").lower()
        consequent = rule.condition_params.get("consequent", "").lower()
        antecedent_present = antecedent in ngrams
        consequent_present = consequent in ngrams
        if antecedent_present and not consequent_present:
            fragment = self._extract_fragment(text, [antecedent])
            return self._make_evidence(
                rule, Verdict.FAILED, fragment, [antecedent],
                f"Antecedent '{antecedent}' present but consequent '{consequent}' absent.",
                f"Add '{consequent}' to satisfy co-occurrence requirement.",
                None,
            )
        return self._make_evidence(
            rule, Verdict.PASSED, text[:200], [antecedent, consequent],
            f"Co-occurrence condition met: antecedent '{antecedent}' and consequent '{consequent}'.",
            "", None,
        )
    def _eval_forbidden_pattern(
        self, rule: CompiledRule, text: str, tokens: list[str], ngrams: set[str], full_text: str
    ) -> EvidenceItem:
        """forbidden_pattern: any forbidden term found → FAILED.

        Uses ngrams (CJK sliding-window tokens) for matching rather than raw regex
        on the original text.  This means the Tokeniser's Unicode segmentation and
        entity-alias expansion both reach this handler.
        """
        terms = rule.condition_params.get("terms", [])
        # If no explicit term list, extract from compiled pattern for backward compat
        if not terms and rule.compiled_pattern is not None:
            raw = rule.compiled_pattern.pattern
            # Strip (?i) prefix and split OR groups
            inner = re.sub(r'^\(\?[imsx]*-?[imsx]*\)', '', raw)
            inner = inner.strip('()')
            terms = [t.strip() for t in inner.split('|') if t.strip()]

        matched = [t for t in terms if t.lower() in ngrams]
        # Fallback: check raw text for multi-char Chinese terms not in ngrams
        if not matched:
            for t in terms:
                if t in full_text or t in text:
                    matched.append(t)
        if matched:
            # Negation context filter
            filtered = [t for t in matched
                        if not self._check_negation_context(full_text or text, t)]
            if not filtered:
                return self._make_evidence(
                    rule, Verdict.PASSED, text[:200], matched,
                    'All forbidden terms found in negation context — contract is compliant.',
                    '', None,
                )
            matched = filtered
            fragment = self._extract_fragment(text, matched)
            return self._make_evidence(
                rule, Verdict.FAILED, fragment, matched,
                f"Forbidden term(s) found: {', '.join(matched)}.",
                f"Remove or rephrase: {', '.join(matched)}.",
                None,
            )
        if rule.compiled_pattern is not None and not terms:
            # Legacy fallback: no term extraction possible, use regex
            match = rule.compiled_pattern.search(text)
            if match:
                fragment = text[max(0, match.start()-20):min(len(text), match.end()+20)]
                return self._make_evidence(
                    rule, Verdict.FAILED, fragment, [match.group()],
                    f"Forbidden pattern matched: '{match.group()}'.",
                    "Remove or rephrase the matched content.",
                    None,
                )
        return self._make_evidence(
            rule, Verdict.PASSED, text[:200], [],
            f"No forbidden terms found.",
            "", None,
        )
    def _build_entity_term_lookup(self) -> dict[str, set[str]]:
        """Build a flat lookup: every entity term -> group's full term set.

        From entity_groups like {"质量保修": ["质量保修书", "保修书", ...]},
        produces: {"质量保修书": {"质量保修书", "保修书", ...}, "保修书": {...}, ...}
        Returns empty dict if no entity_groups are loaded.
        """
        lookup: dict[str, set[str]] = {}
        for group_key, group_terms in self._entity_groups.items():
            if not isinstance(group_terms, list) or len(group_terms) < 1:
                continue
            term_set = set(group_terms)
            for t in group_terms:
                if isinstance(t, str) and t.strip():
                    lookup[t.strip()] = term_set
        return lookup

    def _expand_required_terms(self, terms: list[str],
                                entity_lookup: dict[str, set[str]]) -> list[str]:
        """Expand a list of terms using entity group lookup.

        Each term that appears as a key in entity_lookup is replaced by ALL
        terms in its group.  Terms not found in any group are kept as-is.
        Duplicates are removed while preserving rough order.
        """
        if not entity_lookup or not terms:
            return list(terms)
        seen: set[str] = set()
        result: list[str] = []
        for t in terms:
            if not t:
                continue
            variants = entity_lookup.get(t, {t})
            for v in variants:
                v_lower = v.lower()
                if v_lower not in seen:
                    seen.add(v_lower)
                    result.append(v)
        return result

    def _eval_required_pattern(
        self, rule: CompiledRule, text: str, tokens: list[str], ngrams: set[str], full_text: str
    ) -> EvidenceItem:
        """required_pattern: required term NOT found → FAILED.

        Uses ngrams (CJK sliding-window tokens) for matching, so entity-alias
        expansion and Tokeniser segmentation both apply.

        Entity group expansion: if entity_groups were loaded from domain ontology,
        each term in the rule's terms list is expanded to all synonyms in its
        entity group before matching.  This catches real-world variants that
        rules.json could never exhaustively enumerate.
        """
        terms = rule.condition_params.get("terms", [])
        # If no explicit term list, extract from compiled pattern for backward compat
        if not terms and rule.compiled_pattern is not None:
            raw = rule.compiled_pattern.pattern
            inner = re.sub(r'^\(\?[imsx]*-?[imsx]*\)', '', raw)
            inner = inner.strip('()')
            terms = [t.strip() for t in inner.split('|') if t.strip()]

        # Save original terms BEFORE entity expansion — the negation check
        # (below) operates on these so that entity-expanded variants do not
        # override a negation of the actual required term.
        original_terms = list(terms)

        # ── Ontology-based term expansion ──
        # Build flat lookup once per matcher session (entity_groups don't change)
        if self._entity_groups:
            entity_lookup = self._build_entity_term_lookup()
            expanded = self._expand_required_terms(terms, entity_lookup)
            if expanded != terms:
                import logging as _log
                _log.getLogger(__name__).debug("Entity expansion[%s]: %s -> %s", rule.id, terms, expanded)
                terms = expanded
            else:
                # Debug: even when no change, log what groups are loaded
                import logging as _log
                _log.getLogger(__name__).debug("Entity expansion[%s]: no change (groups=%d, terms=%s)",
                                               rule.id, len(self._entity_groups), terms)

        matched = [t for t in terms if t.lower() in ngrams]
        if matched:
            # \u2500\u2500 Negation-context check for required_pattern \u2500\u2500
            # Check ORIGINAL terms (from rules.json, before entity expansion).
            # If all original terms appear only in negation contexts (e.g.
            # "\u4e0d\u53e6\u884c\u7b7e\u8ba2\u8d28\u91cf\u4fdd\u4fee\u4e66"), the contract explicitly refuses to
            # provide what's required => FAILED.  Entity-expanded terms alone
            # (e.g. "\u4fdd\u4fee" from "\u4fdd\u4fee\u4e8b\u5b9c") do NOT override a negation of
            # the original term.
            src = full_text or text
            original_matched = [t for t in original_terms if t.lower() in ngrams]
            if original_matched:
                all_negated = all(self._all_occurrences_negated(src, t)
                                  for t in original_matched)
                if all_negated:
                    return self._make_evidence(
                        rule, Verdict.FAILED, text[:200], matched,
                        f"Required term(s) negated in contract: {', '.join(original_matched)}. "
                        f"The contract explicitly says it will NOT provide these.",
                        f"Ensure '{', '.join(original_matched)}' is positively stated in the contract.",
                        None,
                    )
            return self._make_evidence(
                rule, Verdict.PASSED, text[:200], matched,
                f"Required term(s) found: {', '.join(matched)}.",
                "", None,
            )
        if rule.compiled_pattern is not None and not terms:
            # Legacy fallback
            match = rule.compiled_pattern.search(text)
            if match:
                return self._make_evidence(
                    rule, Verdict.PASSED, text[:200], [match.group()],
                    f"Required pattern found: '{match.group()}'.",
                    "", None,
                )
        missing_desc = ', '.join(terms) if terms else rule.condition_params.get('pattern', 'unknown')
        return self._make_evidence(
            rule, Verdict.FAILED, text[:200], [],
            f"Required term(s) NOT found: {missing_desc}.",
            f"Add '{missing_desc}' to the text.",
            None,
        )
    def _eval_logical_chain(
        self, rule: CompiledRule, text: str, tokens: list[str], ngrams: set[str], full_text: str
    ) -> EvidenceItem:
        """logical_chain: All premises present but conclusion absent -> FAILED."""
        premises = rule.condition_params.get("premises", [])
        conclusion = rule.condition_params.get("conclusion", "").lower()
        premise_terms = [p.lower() for p in premises]
        premises_present = [p for p in premise_terms if p in ngrams]
        conclusion_present = conclusion in ngrams
        if len(premises_present) == len(premise_terms) and not conclusion_present:
            fragment = self._extract_fragment(text, premise_terms)
            return self._make_evidence(
                rule, Verdict.FAILED, fragment, premise_terms,
                f"All premises present ({', '.join(premises)}) but conclusion '{conclusion}' absent.",
                f"Add '{conclusion}' to complete the logical chain.",
                None,
            )
        return self._make_evidence(
            rule, Verdict.PASSED, text[:200], premise_terms + ([conclusion] if conclusion_present else []),
            "Logical chain condition satisfied.",
            "", None,
        )
    def _eval_scope_constraint(
        self, rule: CompiledRule, text: str, tokens: list[str], ngrams: set[str], full_text: str
    ) -> EvidenceItem:
        """scope_constraint: Evaluate delegate rule only within scoped segments."""
        scope = rule.condition_params.get("scope", "").lower()
        delegate = rule.condition_params.get("delegate")
        if delegate is None:
            return self._make_evidence(rule, Verdict.NOT_APPLICABLE, "", [], "No delegate rule defined.", "", None)
        # Tokenise scope for matching
        scope_tokens = self._tokeniser.tokenise(scope)
        # Find text portions matching the scope
        # Simple approach: look at the full text for scope indicators
        scope_matches = [t for t in scope_tokens if t in ngrams]
        if not scope_matches:
            return self._make_evidence(
                rule, Verdict.NOT_APPLICABLE, "", [],
                f"Scope '{scope}' not applicable to this input.", "", None,
            )
        # Compile delegate rule on the fly
        try:
            if isinstance(delegate, dict):
                delegate_compiled = PythonRuleCompiler._compile_rule(
                    delegate, rule.package_id, rule.package_version
                )
            else:
                return self._make_evidence(rule, Verdict.NOT_APPLICABLE, "", [], "Invalid delegate rule format.", "", None)
        except Exception as e:
            return self._make_evidence(rule, Verdict.NOT_APPLICABLE, "", [], f"Delegate compilation failed: {e}", "", None)
        # Evaluate delegate against text containing scope
        return self.evaluate(delegate_compiled, text, tokens, ngrams,
                             contract_broad_type="", estimated_value=0.0)
    # ── numeric_comparison ──
    # Chinese number parser (一=1, 二=2, ..., 十=10, 百=100, 十五=15, 三百=300, etc.)
    _CN_DIGIT = {
        "零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
        "六": 6, "七": 7, "八": 8, "九": 9, "十": 10, "百": 100,
        "千": 1000, "万": 10000,
    }
    _CN_UNIT_RE = re.compile(r'[万亿千百十]')
    _NUM_VALUE_RE = re.compile(r'(\d+(?:\.\d+)?)\s*(年|月|日|天|元|万|%|％)')
    _CN_NUM_RE = re.compile(r'[零一二两三四五六七八九十百千万亿]+')
    # Chinese percentage/permille patterns: 百分之三 → 3%, 万分之五 → 0.05%
    _CN_PERCENT_RE = re.compile(r'百分之([零一二两三四五六七八九十百千万]+)')
    _CN_PERMYRIAD_RE = re.compile(r'万分之([零一二两三四五六七八九十百千万]+)')
    @classmethod
    def _parse_cn_number(cls, s: str) -> int | None:
        """Parse Chinese numeral string to int. E.g. '十五' → 15, '三百' → 300."""
        s = s.strip()
        if not s:
            return None
        # Try pure Arabic first
        try:
            return int(s)
        except ValueError:
            pass
        result = 0
        current = 0
        for i, ch in enumerate(s):
            if ch in cls._CN_DIGIT:
                val = cls._CN_DIGIT[ch]
                if val >= 10:
                    if current == 0:
                        current = 1
                    current *= val
                    result += current
                    current = 0
                else:
                    current = val
            else:
                return None
        result += current
        return result if result > 0 else None
    @classmethod
    def _extract_numbers(cls, text: str) -> list[dict]:
        """Extract numeric values with their units from text.
        Returns list of {value: float, unit: str, raw: str, position: int}.
        """
        found = []
        # Arabic numbers with units
        for m in cls._NUM_VALUE_RE.finditer(text):
            found.append({
                "value": float(m.group(1)),
                "unit": m.group(2),
                "raw": m.group(0),
                "position": m.start(),
            })
        # Chinese percentage/permyriad patterns: "百分之三" → 3%, "万分之五" → 0.05%
        # These MUST be processed before general Chinese number matching to avoid
        # "百分之三" being incorrectly split to 百=100 + 三=3.
        for m in cls._CN_PERCENT_RE.finditer(text):
            cn_num = m.group(1)
            val = cls._parse_cn_number(cn_num)
            if val is not None:
                found.append({
                    "value": float(val),
                    "unit": "%",
                    "raw": m.group(0),
                    "position": m.start(),
                })
        for m in cls._CN_PERMYRIAD_RE.finditer(text):
            cn_num = m.group(1)
            val = cls._parse_cn_number(cn_num)
            if val is not None:
                found.append({
                    "value": float(val) * 0.01,  # 万分之N = N/10000 = N*0.01%
                    "unit": "%",
                    "raw": m.group(0),
                    "position": m.start(),
                })
        # Chinese number phrases: e.g. "保修期限为五年", "保修一年"
        for m in cls._CN_NUM_RE.finditer(text):
            cn_str = m.group()
            val = cls._parse_cn_number(cn_str)
            if val is None:
                continue
            # Look for unit immediately after
            end = m.end()
            unit = ""
            if end < len(text) and text[end] in "年月日天元个百分点":
                unit = text[end]
            found.append({
                "value": float(val),
                "unit": unit,
                "raw": m.group(0) + unit,
                "position": m.start(),
            })
        return found
    def _eval_numeric_comparison(
        self, rule: CompiledRule, text: str, tokens: list[str], ngrams: set[str], full_text: str
    ) -> EvidenceItem:
        """numeric_comparison: Extract a numeric value from text and compare against a threshold.
        Condition params:
          - label: human name of the field being compared (e.g. "屋面防水保修期限")
          - context_pattern: regex to locate the relevant region in text (e.g. "保修|防水|屋面")
          - unit: expected unit (e.g. "年", "月", "元", "%")
          - operator: ">=", "<=", ">", "<", "==", "!="
          - expected: numeric threshold (from legal text)
          - legal_ref: citation string (e.g. "国务院令第279号第40条(二)")
        """
        label = rule.condition_params.get("label", rule.name)
        context_pat = rule.condition_params.get("context_pattern", "")
        unit = rule.condition_params.get("unit", "年")
        operator = rule.condition_params.get("operator", ">=")
        expected = rule.condition_params.get("expected", 0)
        legal_ref = rule.condition_params.get("legal_ref", "")
        try:
            expected = float(expected)
        except (ValueError, TypeError):
            expected = 0
        # Narrow text to context region if context_pattern provided.
        # Strategy: find ALL regex matches, score each surrounding region by how
        # many numbers with the target unit it contains, and pick the best region.
        # This fixes the bug where context_pattern (e.g. "保修|防水|屋面")
        # hits an irrelevant section (e.g. "防水材料") before the operative
        # clause (e.g. "保修期为一年"), causing a false NOT_APPLICABLE.
        search_text = text
        best_start, best_end, best_ctx_pos = 0, 0, 0
        best_count = -1
        all_matches = []
        if context_pat:
            try:
                all_matches = list(re.finditer(context_pat, text))
                if not all_matches:
                    return self._make_evidence(
                        rule, Verdict.NOT_APPLICABLE, text[:200], [],
                        f"Rule '{rule.name}' not applicable: context_pattern '{context_pat}' did not match.",
                        "", None,
                    )
                for ctx_match in all_matches:
                    start = max(0, ctx_match.start() - 200)
                    end = min(len(text), ctx_match.end() + 200)
                    region = text[start:end]
                    nums = self._extract_numbers(region)
                    count = sum(1 for n in nums if n["unit"] == unit)
                    # Also count alternative accepted units for region scoring
                    if count == 0 and unit == "月":
                        accept_units = rule.condition_params.get("accept_units", None)
                        if accept_units and "年" in accept_units:
                            count = sum(1 for n in nums if n["unit"] == "年")
                    if count > best_count:
                        best_count = count
                        best_start, best_end = start, end
                        best_ctx_pos = ctx_match.start()
                    elif count == best_count and count > 0:
                        # Same-count tiebreaker: prefer the region where the
                        # context match appears CLOSER to a number, by checking
                        # the minimum distance from context match to nearest number.
                        region_nums = [n for n in nums if n["unit"] == unit]
                        if region_nums:
                            ctx_rel = ctx_match.start() - start
                            min_dist = min(abs(n["position"] - ctx_rel) for n in region_nums)
                            curr_region = text[best_start:best_end]
                            curr_nums_ = self._extract_numbers(curr_region)
                            curr_rel = [n for n in curr_nums_ if n["unit"] == unit]
                            curr_min_dist = min(abs(n["position"] - (best_ctx_pos - best_start)) for n in curr_rel) if curr_rel else float('inf')
                            if min_dist < curr_min_dist:
                                best_count = count
                                best_start, best_end = start, end
                                best_ctx_pos = ctx_match.start()
                if best_count > 0:
                    search_text = text[best_start:best_end]
            except re.error:
                pass
        # Extract numbers
        numbers = self._extract_numbers(search_text)
        if not numbers:
            # Fall back to full text
            numbers = self._extract_numbers(text)
        # Filter by unit
        relevant = [n for n in numbers if n["unit"] == unit]
        if not relevant:
            # ── accept_units fallback ──
            # If the rule specifies alternative accepted units (e.g. "年" when
            # the primary unit is "月"), accept numbers with those units and
            # convert values to the primary unit.
            # Supported conversions: 年 -> 月 (multiply by 12).
            accept_units = rule.condition_params.get("accept_units", None)
            if accept_units and unit == "月" and "年" in accept_units:
                # Look for year values and convert to months
                year_nums = [n for n in numbers if n["unit"] == "年"]
                if year_nums:
                    for yn in year_nums:
                        yn["value"] = yn["value"] * 12
                        yn["unit"] = "月"
                        yn["raw"] = str(int(yn["value"])) + "月"
                    relevant = year_nums
            if not relevant:
                return self._make_evidence(
                    rule, Verdict.NOT_APPLICABLE, search_text[:200], [],
                    f"Cannot find any numeric value with unit '{unit}' for field '{label}'.",
                    f"Ensure the contract specifies {label} as a number with unit '{unit}'.",
                    None,
                )
        # Compare each found value — pick the closest AFTER the best context match.
        # "屋面5年 / 主体50年" with context "主体" should pick 50年, not 5年.
        # Prefer: min(pos - ctx_pos) for pos >= ctx_pos. Fall back to absolute distance.
        contract_value = relevant[0]
        if context_pat and len(relevant) > 1:
            best_dist = float('inf')
            best_after = float('inf')
            best_after_val = None
            # Collect all context match positions for proximity scoring.
            # When accept_units (e.g. 年->月) fallback is active, the numbers
            # come from the full text rather than a narrowed region, so we
            # must search all context-match positions to find the nearest number.
            try:
                ctx_positions = [m.start() for m in re.finditer(context_pat, text)]
                for n in relevant:
                    full_pos = n.get("position", 0)
                    # First priority: prefer numbers that appear AFTER a context match
                    for ctx_pos in ctx_positions:
                        if full_pos >= ctx_pos:
                            dist = full_pos - ctx_pos
                            if dist < best_after:
                                best_after = dist
                                best_after_val = n
                    if best_after_val is not None:
                        continue  # skip absolute distance if we found after-match
                    # Fallback: absolute distance (for numbers before context)
                    for ctx_pos in ctx_positions:
                        dist = abs(full_pos - ctx_pos)
                        if dist < best_dist:
                            best_dist = dist
                            contract_value = n
                if best_after_val is not None:
                    contract_value = best_after_val
            except re.error:
                pass
        val = contract_value["value"]
        raw_str = contract_value["raw"]
        raw_pos = contract_value.get("position", 0)

        # ── Direction word check (pre-computed at compile time) ──
        # P1.1 moved LLM direction resolution from the execution path to
        # compile time.  The precomputed_direction field is populated by
        # PythonRuleCompiler._compile_rule() when the rule's context_pattern
        # contains direction-ambiguous phrases ("不低于", "不足", etc.).
        # If no pre-computed direction exists (e.g. LLM was unavailable at
        # compile time), direction_op stays None and the rule's declared
        # operator is used as-is.
        direction_op: str | None = None
        direction_note: str = ""
        pre_dir = rule.precomputed_direction.strip() if rule.precomputed_direction else ""
        if pre_dir and pre_dir in (">=", "<=", "=="):
            direction_op = pre_dir
            direction_note = f" [precomputed direction: {direction_op}]"

        # If LLM returned a specific direction, use it as the effective operator
        # for the comparison.  "不足三年" → LLM says "<=" → contract says ≤ 3,
        # rule says ≥ 5 → FAILED (correctly catches under-compliance).
        effective_op = direction_op if direction_op else operator
        op_map = {
            ">=": lambda a, b: a >= b, "<=": lambda a, b: a <= b,
            ">": lambda a, b: a > b, "<": lambda a, b: a < b,
            "==": lambda a, b: a == b, "!=": lambda a, b: a != b,
        }
        compare = op_map.get(effective_op, op_map.get(operator, lambda a, b: a >= b))
        ok = compare(val, expected)
        if ok:
            return self._make_evidence(
                rule, Verdict.PASSED, contract_value["raw"], [contract_value["raw"]],
                f"Field '{label}': contract value {val}{unit} {'meets' if effective_op in ('>=', '>') else 'is within'} legal threshold ({effective_op} {expected}{unit}){'. Ref: ' + legal_ref if legal_ref else ''}.{direction_note}",
                "", None,
            )
        else:
            gap = abs(expected - val)
            if direction_op:
                suggest = (f"'{label}' is {val}{unit} ({direction_op}), "
                          f"does not meet legal threshold {operator} {expected}{unit}.")
            else:
                suggest = f"Increase '{label}' from {val}{unit} to at least {expected}{unit}."
            return self._make_evidence(
                rule, Verdict.FAILED, contract_value["raw"], [contract_value["raw"]],
                f"Field '{label}': contract value {val}{unit} FAILS legal threshold ({effective_op} {expected}{unit}){'. Ref: ' + legal_ref if legal_ref else ''}. Gap: {gap}{unit}.{direction_note}",
                suggest,
                None,
            )
    # ── sum_numeric_comparison ──
    def _eval_sum_numeric_comparison(
        self, rule: CompiledRule, text: str, tokens: list[str], ngrams: set[str], full_text: str
    ) -> EvidenceItem:
        """Sum all numbers matching a context, compare total against threshold."""
        label = rule.condition_params.get("label", rule.name)
        context_pat = rule.condition_params.get("context_pattern", "%%")
        unit = rule.condition_params.get("unit", "%%")
        operator = rule.condition_params.get("operator", "<=")
        expected = rule.condition_params.get("expected", 100)
        legal_ref = rule.condition_params.get("legal_ref", "")
        try: expected = float(expected)
        except (ValueError, TypeError): expected = 100
        search_text = text
        if context_pat:
            try:
                all_ctx = list(re.finditer(context_pat, text))
                if all_ctx:
                    s = max(0, all_ctx[0].start() - 300)
                    e = min(len(text), all_ctx[-1].end() + 300)
                    search_text = text[s:e]
            except re.error: pass
        numbers = self._extract_numbers(search_text)
        relevant = [n for n in numbers if n["unit"] == unit and n.get("kind") != "date"]
        if not relevant:
            numbers = self._extract_numbers(text)
            relevant = [n for n in numbers if n["unit"] == unit and n.get("kind") != "date"]
        if not relevant:
            return self._make_evidence(rule, Verdict.NOT_APPLICABLE, search_text[:200], [],
                "Cannot find any value with unit '%s' for sum." % unit, "", None)
        total = sum(n["value"] for n in relevant)
        raw_terms = [n["raw"] for n in relevant]
        op = {">=": lambda a,b: a>=b, "<=": lambda a,b: a<=b}
        ok = op.get(operator, lambda a,b: a<=b)(total, expected)
        if ok:
            return self._make_evidence(rule, Verdict.PASSED, ", ".join(raw_terms[:5]), raw_terms[:5],
                "Sum of %d values = %s%s, meets (%s %s%s)" % (len(relevant), total, unit, operator, expected, unit)
                + (". Ref: %s" % legal_ref if legal_ref else ""), "", None)
        else:
            return self._make_evidence(rule, Verdict.FAILED, ", ".join(raw_terms[:5]), raw_terms[:5],
                "Sum of %d values = %s%s, FAILS (%s %s%s)" % (len(relevant), total, unit, operator, expected, unit)
                + (". Ref: %s" % legal_ref if legal_ref else ""),
                "Total should be %s %s%s, is %s%s." % (operator, expected, unit, total, unit), None)

    # ── contextual_co_occurrence ──
    def _eval_contextual_co_occurrence(
        self, rule: CompiledRule, text: str, tokens: list[str], ngrams: set[str], full_text: str
    ) -> EvidenceItem:
        """contextual_co_occurrence: term A and term B must appear within N characters
        of each other in the text.  Unlike regular co_occurrence (which only checks
        global presence), this checks LOCAL adjacency — essential for conceptual
        domains where two terms are only meaningfully related if they appear in
        the same paragraph/context.

        Condition params:
          - term_a: str — first concept
          - term_b: str — second concept (must appear near term_a)
          - window_chars: int — max distance between the two terms (default 500)
          - min_occurrences: int — minimum number of co-occurrence windows (default 1)
        """
        term_a = rule.condition_params.get("term_a", "").lower()
        term_b = rule.condition_params.get("term_b", "").lower()
        window_chars = int(rule.condition_params.get("window_chars", 500))
        min_occurrences = int(rule.condition_params.get("min_occurrences", 1))

        if not term_a or not term_b:
            return self._make_evidence(
                rule, Verdict.NOT_APPLICABLE, "", [],
                "Missing term_a or term_b for contextual_co_occurrence.", "", None,
            )

        text_lower = text.lower()

        # Find all positions of term_a
        positions_a = []
        idx = 0
        while True:
            pos = text_lower.find(term_a, idx)
            if pos == -1:
                break
            positions_a.append(pos)
            idx = pos + 1

        if not positions_a:
            return self._make_evidence(
                rule, Verdict.NOT_APPLICABLE, text[:200], [],
                f"Term A '{term_a}' not found in text — contextual check not applicable.",
                "", None,
            )

        # For each occurrence of term_a, check if term_b appears within window_chars
        co_occurrences = 0
        for pos_a in positions_a:
            window_start = max(0, pos_a - window_chars)
            window_end = min(len(text_lower), pos_a + len(term_a) + window_chars)
            window = text_lower[window_start:window_end]
            if term_b in window:
                co_occurrences += 1

        if co_occurrences >= min_occurrences:
            return self._make_evidence(
                rule, Verdict.PASSED, text[:200], [term_a, term_b],
                f"'{term_a}' and '{term_b}' co-occur within {window_chars} chars "
                f"in {co_occurrences} instance(s) (min required: {min_occurrences}).",
                "", None,
            )
        else:
            return self._make_evidence(
                rule, Verdict.FAILED, text[:200], [term_a],
                f"'{term_a}' found but '{term_b}' does not appear within {window_chars} chars "
                f"in any of {len(positions_a)} occurrence(s) (found {co_occurrences}, need {min_occurrences}). "
                f"This may indicate the text mentions '{term_a}' without connecting it to '{term_b}'.",
                f"Ensure '{term_b}' is discussed near each occurrence of '{term_a}'.",
                None,
            )

    # ── definition_contains ──
    def _eval_definition_contains(
        self, rule: CompiledRule, text: str, tokens: list[str], ngrams: set[str], full_text: str
    ) -> EvidenceItem:
        """definition_contains: when a concept is mentioned, its definition context
        must contain certain required sub-terms.  This checks STRUCTURED co-occurrence
        rather than pattern matching — it first locates the concept, then checks the
        surrounding text for required sub-terms.

        Condition params:
          - concept: str — the main concept to locate (e.g. "认知失调")
          - required_terms: list[str] — terms that must appear near the concept (e.g. ["态度", "不一致"])
          - window_chars: int — context window around concept (default 400)
          - min_ratio: float — fraction of required_terms that must appear (default 0.5)
        """
        concept = rule.condition_params.get("concept", "").lower()
        required_terms = rule.condition_params.get("required_terms", [])
        window_chars = int(rule.condition_params.get("window_chars", 400))
        min_ratio = float(rule.condition_params.get("min_ratio", 0.5))

        if not concept or not required_terms:
            return self._make_evidence(
                rule, Verdict.NOT_APPLICABLE, "", [],
                "Missing concept or required_terms for definition_contains.", "", None,
            )

        text_lower = text.lower()

        # Find all occurrences of the concept
        positions = []
        idx = 0
        while True:
            pos = text_lower.find(concept, idx)
            if pos == -1:
                break
            positions.append(pos)
            idx = pos + 1

        if not positions:
            return self._make_evidence(
                rule, Verdict.NOT_APPLICABLE, text[:200], [],
                f"Concept '{concept}' not found in text.", "", None,
            )

        # For each occurrence, check which required_terms appear nearby
        best_matched = []
        best_missing = []
        best_ratio = 0.0

        for pos in positions:
            window_start = max(0, pos - window_chars)
            window_end = min(len(text_lower), pos + len(concept) + window_chars)
            window = text_lower[window_start:window_end]

            matched = [t for t in required_terms if t.lower() in window]
            missing = [t for t in required_terms if t.lower() not in window]
            ratio = len(matched) / len(required_terms) if required_terms else 0.0

            if ratio > best_ratio:
                best_matched = matched
                best_missing = missing
                best_ratio = ratio

        if best_ratio >= min_ratio:
            return self._make_evidence(
                rule, Verdict.PASSED, text[:200], [concept] + best_matched,
                f"'{concept}' definition context contains {len(best_matched)}/{len(required_terms)} "
                f"required terms ({best_ratio:.0%}, threshold {min_ratio:.0%}). "
                f"Matched: {best_matched}. Missing: {best_missing}.",
                "", None,
            )
        else:
            return self._make_evidence(
                rule, Verdict.FAILED, text[:200], [concept] + best_matched,
                f"'{concept}' found but only {len(best_matched)}/{len(required_terms)} "
                f"required terms appear in its context ({best_ratio:.0%}, need {min_ratio:.0%}). "
                f"Matched: {best_matched}. Missing: {best_missing}. "
                f"The text may discuss '{concept}' without properly defining it.",
                f"Include the missing terms {best_missing} when discussing '{concept}'.",
                None,
            )
    def _eval_topic_coverage(
        self, rule: CompiledRule, text: str, tokens: list[str], ngrams: set[str], full_text: str
    ) -> EvidenceItem:
        """topic_coverage: check whether the response text covers the user's key terms.
        Condition params:
          - source_keywords: list of keywords from user input
          - min_coverage_ratio: float 0-1, minimum fraction of keywords that must appear (default 0.5)
        Returns PASSED if coverage >= min_coverage_ratio, FAILED otherwise.
        Does NOT perform semantic analysis - only surface token matching.
        """
        source_keywords = rule.condition_params.get("source_keywords", [])
        min_coverage_ratio = float(rule.condition_params.get("min_coverage_ratio", 0.5))
        if not source_keywords:
            return self._make_evidence(
                rule, Verdict.NOT_APPLICABLE, text[:200], [],
                "No source_keywords provided for topic_coverage check.",
                "", None,
            )
        text_lower = text.lower()
        matched = []
        missing = []
        for kw in source_keywords:
            if kw.lower() in text_lower:
                matched.append(kw)
            else:
                missing.append(kw)
        coverage = len(matched) / len(source_keywords) if source_keywords else 0.0
        if coverage >= min_coverage_ratio:
            return self._make_evidence(
                rule, Verdict.PASSED, text[:200], matched,
                f"Topic coverage: {len(matched)}/{len(source_keywords)} keywords matched ({coverage:.0%}), threshold {min_coverage_ratio:.0%}. "
                f"Matched: {matched}. Missing: {missing}.",
                "", None,
            )
        else:
            return self._make_evidence(
                rule, Verdict.FAILED, text[:200], matched,
                f"Topic coverage: {len(matched)}/{len(source_keywords)} keywords matched ({coverage:.0%}), below threshold {min_coverage_ratio:.0%}. "
                f"Missing: {missing}. The response may have drifted away from the user's core question.",
                f"Consider addressing the missing keywords: {missing}.",
                None,
            )
    def _extract_fragment(self, text: str, terms: list[str], context_chars: int = 100) -> str:
        """Extract a relevant fragment of text surrounding matched terms."""
        if not terms or not text:
            return text[:200]
        text_lower = text.lower()
        positions = []
        for term in terms:
            idx = text_lower.find(term.lower())
            if idx >= 0:
                positions.append(idx)
        if not positions:
            return text[:200]
        start = max(0, min(positions) - context_chars)
        end = min(len(text), max(positions) + context_chars)
        fragment = text[start:end].strip()
        if start > 0:
            fragment = "..." + fragment
        if end < len(text):
            fragment = fragment + "..."
        return fragment
    def _eval_ast_check(
        self, rule, text, tokens, ngrams, full_text
    ):
        """ast_check: Tree-sitter AST structure patterns for code review."""
        lang = rule.condition_params.get("language", "javascript")
        check = {
            "search_type": rule.condition_params.get("search_type", "forbidden_node"),
            "node_type": rule.condition_params.get("node_type", ""),
            "node_pattern": rule.condition_params.get("node_pattern", ""),
            "min_count": int(rule.condition_params.get("min_count", 1)),
        }
        try:
            from app.engine.ast_checker import check_ast
            hits = check_ast(text, lang, check)
        except Exception as e:
            return self._make_evidence(
                rule, Verdict.NOT_APPLICABLE, text[:200], [],
                f"AST check error: {e}", "", None,
            )
        if not hits:
            return self._make_evidence(
                rule, Verdict.PASSED, text[:200], [],
                f"AST check '{check['search_type']}' passed: no violations.", "", None,
            )
        first = hits[0]
        return self._make_evidence(
            rule, Verdict.FAILED, first.node_text, [],
            f"AST check '{check['search_type']}' found {len(hits)} violation(s). "
            f"First: line {first.line} [{first.node_type}] {first.node_text[:120]}",
            f"Review line {first.line}: {first.node_text[:100]}",
            None,
        )

# ======================================================================
# Conflict Detector
# ======================================================================
class PythonConflictDetector:
    """Finds contradictory verdicts across rules and packages."""
    @staticmethod
    def detect_conflicts(evidence_chain: list[EvidenceItem]) -> list[Conflict]:
        conflicts: list[Conflict] = []
        fragment_groups: dict[str, list[EvidenceItem]] = {}
        for ev in evidence_chain:
            fragment_groups.setdefault(ev.input_fragment, []).append(ev)
        for fragment, items in fragment_groups.items():
            if len(items) < 2:
                continue
            for i in range(len(items)):
                for j in range(i + 1, len(items)):
                    a, b = items[i], items[j]
                    if a.status != b.status:
                        if a.package_id == b.package_id:
                            conflict_type = "intra_package"
                        else:
                            conflict_type = "cross_package"
                        conflict = Conflict(
                            conflict_id=f"cfl_{uuid.uuid4().hex[:8]}",
                            conflict_type=conflict_type,
                            rule_a={
                                "rule_id": a.rule_id,
                                "package_id": a.package_id,
                                "verdict": a.status,
                                "input_fragment": a.input_fragment,
                            },
                            rule_b={
                                "rule_id": b.rule_id,
                                "package_id": b.package_id,
                                "verdict": b.status,
                                "input_fragment": b.input_fragment,
                            },
                            description=(
                                f"Rule {a.rule_id} in {a.package_id} returns {a.status} "
                                f"while rule {b.rule_id} in {b.package_id} returns {b.status} "
                                f"on the same input fragment. No unified resolution."
                            ),
                        )
                        conflicts.append(conflict)
        return conflicts
# ======================================================================
# Validation Engine
# ======================================================================
class PythonValidationEngine:
    def __init__(self, llm_url: str = "", llm_key: str = "", llm_model: str = ""):
        self._llm_url = llm_url
        self._llm_key = llm_key
        self._llm_model = llm_model or "deepseek-chat"
        self._packages: dict = {}
        self._compiler = PythonRuleCompiler()
        self._entity_groups: dict[str, list[str]] = {}
        self._matcher = PythonMatcher(llm_url=llm_url, llm_key=llm_key, llm_model=llm_model)
        self._conflict_detector = PythonConflictDetector()
        self._lock = threading.RLock()

    def _load_entity_groups(self, domain_id: str) -> dict[str, list[str]]:
        """Load raw entity_groups from domain ontology.json.

        Returns the entity_groups dict (group_key -> list of terms),
        or empty dict if ontology doesn't exist or has no entity_groups.
        """
        domain_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__)))), "domains", domain_id)
        onto_path = os.path.join(domain_path, "ontology.json")
        if not os.path.isfile(onto_path):
            return {}
        try:
            with open(onto_path, "r", encoding="utf-8") as f:
                onto = json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}
        groups = onto.get("entity_groups", {})
        if not isinstance(groups, dict):
            return {}
        return groups

    def load_package(self, package_data: dict, domain_id: str = "") -> None:
        aliases: dict[str, list[str]] = {}
        domain_id = domain_id or package_data.get("domain", "")
        if domain_id:
            domain_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(
                    os.path.abspath(__file__)))), "domains", domain_id)
            aliases = PythonRuleCompiler.load_aliases(domain_path)
            if aliases:
                logger.info("Loaded %d entity alias groups for domain '%s'",
                            len(aliases), domain_id)
            # Load entity_groups for required_pattern runtime expansion
            eg = self._load_entity_groups(domain_id)
            if eg:
                self._entity_groups = eg
                self._matcher._entity_groups = eg

        compiled = self._compiler.compile(package_data, aliases=aliases or None,
                                          llm_url=self._llm_url, llm_key=self._llm_key,
                                          llm_model=self._llm_model)
        with self._lock:
            if compiled.id in self._packages:
                from app.exceptions import PackageAlreadyLoadedError
                raise PackageAlreadyLoadedError(compiled.id)
            self._packages[compiled.id] = compiled

    
    def reload_package(self, package_id: str, new_package_data: dict) -> None:
        new_compiled = self._compiler.compile(new_package_data,
                                             llm_url=self._llm_url, llm_key=self._llm_key,
                                             llm_model=self._llm_model)
        with self._lock:
            if package_id not in self._packages:
                from app.exceptions import PackageNotFoundError
                raise PackageNotFoundError(package_id)
            self._packages[package_id] = new_compiled

    def unload_package(self, package_id: str) -> None:
        with self._lock:
            self._packages.pop(package_id, None)

    def list_packages(self) -> list:
        with self._lock:
            return [{
                "id": pkg.id, "version": pkg.version,
                "domain": pkg.domain, "rule_count": len(pkg.rules),
            } for pkg in self._packages.values()]

    def validate(self, input_data: dict, packages: list, options: dict | None = None) -> dict:
        opts = options or {}
        max_evidence = opts.get("max_evidence", 100)
        include_warnings = opts.get("include_warnings", True)
        text = input_data.get("text", "")
        clause_blocks = input_data.get("clause_blocks", None)
        contract_broad_type = input_data.get("contract_broad_type", "")
        estimated_value = float(input_data.get("estimated_value", 0))
        tokeniser = Tokeniser()
        tokens = tokeniser.tokenise(text)
        ngrams = tokeniser.ngrams(tokens)
        compiled_packages = [self._packages[p] for p in packages if p in self._packages]
        evidence_list = []
        for compiled_pkg in compiled_packages:
            for rule in compiled_pkg.rules:
                evidence_list.append(self._matcher.evaluate(
                    rule, text, tokens, ngrams,
                    clause_blocks=clause_blocks,
                    contract_broad_type=contract_broad_type,
                    estimated_value=estimated_value,
                ))

        # ── Deduplicate evidence: same rule_id + same status → merge ──
        dedup_map: dict[tuple[str, str], EvidenceItem] = {}
        for ev in evidence_list:
            key = (ev.rule_id, ev.status)
            if key in dedup_map:
                existing = dedup_map[key]
                # Merge: concatenate fragments and matched_terms, keep first rationale
                if ev.input_fragment not in existing.input_fragment:
                    existing.input_fragment = existing.input_fragment[:300] + chr(10) + "---" + chr(10) + ev.input_fragment[:200]
                for mt in ev.matched_terms:
                    if mt not in existing.matched_terms:
                        existing.matched_terms.append(mt)
            else:
                dedup_map[key] = ev


        # Preserve original order
        seen_keys: set[tuple[str, str]] = set()
        deduped_list: list[EvidenceItem] = []
        for ev in evidence_list:
            key = (ev.rule_id, ev.status)
            if key not in seen_keys:
                seen_keys.add(key)
                deduped_list.append(dedup_map[key])
        evidence_list = deduped_list

        if max_evidence and len(evidence_list) > max_evidence:
            evidence_list = evidence_list[:max_evidence]
        if not include_warnings:
            evidence_list = [ev for ev in evidence_list if ev.severity != warning]
        conflicts = self._conflict_detector.detect_conflicts(evidence_list)
        result = ValidationResult(
            status=ValidationStatus.VALIDATED,
            evidence_chain=evidence_list, conflicts=conflicts, summary={},
            package_versions={pkg.id: pkg.version for pkg in compiled_packages},
            processing_ms=0,
        )
        failed = any(ev.status == Verdict.FAILED.value for ev in evidence_list)
        result.status = ValidationStatus.CONFLICTED if failed else result.status
        return result.to_dict()

    
    def _preserve_structured_inputs(self, extractions: list[dict], rule_label_map: dict) -> None:
        """R2.10: Preserve model structured_inputs into matcher before validation.
        
        Called by kernel.py validate() before self.validate() so that
        handler._eval_numeric_comparison() can read null-value signals
        from the matcher._structured_inputs sidecar.
        """
        from app.engine.structured_input import inject_structured_fields
        si = inject_structured_fields(extractions, rule_label_map)
        if hasattr(self, '_matcher'):
            self._matcher._structured_inputs.update(si)
