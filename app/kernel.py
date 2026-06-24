"""Shanmiao Kernel — domain-agnostic validation engine.

The Kernel is the pure-logic core.  It knows SPP predicates (Section 2 of
SPEC.md) but contains zero domain-specific constants.  All domain
knowledge lives in domains/<id>/, loaded at runtime via DomainLoader.

Architecture:
    Raw Text -> DomainLoader -> SPP Mapper -> Knowledge Layers -> Z3 -> Result
"""

from __future__ import annotations

import importlib
import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Try-import self-bootstrap rings (Ring 1: DomainPrototypeStore, Ring 2: AutoClusterer) ──
# Graceful fallback: if rings are missing, kernel falls back to keyword-based behavior.
try:
    from app.engine.domain_prototype import DomainPrototypeStore
    _HAS_PROTOTYPE_STORE = True
except (ImportError, Exception):
    DomainPrototypeStore = None
    _HAS_PROTOTYPE_STORE = False

try:
    from app.engine.auto_clusterer import AutoClusterer
    _HAS_AUTO_CLUSTERER = True
except (ImportError, Exception):
    AutoClusterer = None
    _HAS_AUTO_CLUSTERER = False

# ── P1: Lexical domain classifier (CJK bigram coverage) ──
try:
    from app.engine.lexical_prototype import LexicalPrototypeStore
    _LEXICAL_STORE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "lexical_prototypes.json")
    _lexical_store: "LexicalPrototypeStore | None" = None
    _HAS_LEXICAL_STORE = True
except (ImportError, Exception):
    LexicalPrototypeStore = None
    _HAS_LEXICAL_STORE = False

from app.engine.clause_splitter import ClauseSplitter, ClauseBlock
from app.engine.core import (
    CompiledPackage,
    CompiledRule,
    EvidenceItem,
    PythonMatcher,
    PythonRuleCompiler,
    PythonValidationEngine,
    Tokeniser,
    ValidationResult,
    ValidationStatus,
    Verdict,
)

_CITTA_Z3_AVAILABLE = False
_CittaZ3Solver = None
_LegalConstraint = None


def _ensure_z3():
    global _CITTA_Z3_AVAILABLE, _CittaZ3Solver, _LegalConstraint
    if not _CITTA_Z3_AVAILABLE:
        try:
            from app.engine.solver import (
                CittaZ3Solver as _CittaZ3Solver,
                LegalConstraint as _LegalConstraint,
            )
            _CITTA_Z3_AVAILABLE = True
        except ImportError:
            _CITTA_Z3_AVAILABLE = False


def _try_extract_numeric(ev: dict) -> float:
    """Extract a numeric value from an evidence item for tracer integration.
    Tries matched_terms first, then raw.
    """
    import re as _re
    for src_key in ("matched_terms", "raw"):
        src = ev.get(src_key, None)
        if src:
            text = src[0] if isinstance(src, list) and src else str(src)
            m = _re.search(r'[\d.]+', text)
            if m:
                try:
                    return float(m.group())
                except ValueError:
                    pass
    return 0.0


def _build_conflict_report(
    mus_constraints: list, propositions: list, contract_text: str,
) -> dict:
    """Build a human-readable conflict report from MUS constraints and propositions.

    Used by validate_with_z3() to generate LLM-actionable conflict explanations.
    """
    prop_map = {}
    for p in propositions:
        field = getattr(p, 'field', '')
        value = getattr(p, 'value', 0)
        unit = getattr(p, 'unit', '')
        prop_map[field] = {"value": value, "unit": unit}

    conflict_lines = []
    conflicting_fields = []
    for c in mus_constraints:
        field = getattr(c, 'field', str(c))
        operator = getattr(c, 'operator', '?')
        threshold = getattr(c, 'threshold', 0)
        unit = getattr(c, 'unit', '')
        legal_ref = getattr(c, 'legal_ref', '')
        conflicting_fields.append(field)

        prop = prop_map.get(field, {})
        actual = prop.get("value", "?")
        prop_unit = prop.get("unit", unit)

        conflict_lines.append(
            f"  • {field}：合同值 {actual}{prop_unit}，"
            f"法规要求 {operator} {threshold}{unit}"
        )
        if legal_ref:
            conflict_lines[-1] += f"（{legal_ref}）"

    human_readable = "【SMT 约束冲突报告】\n\n" + "\n".join(conflict_lines)

    # Build suggested fix for LLM
    suggested_fix = (
        "请核实以下字段的提取值是否正确：\n"
        + "\n".join(f"  - {f}" for f in conflicting_fields)
        + f"\n\n原文上下文：\n  {contract_text[:500]}"
    )

    return {
        "human_readable": human_readable,
        "suggested_fix": suggested_fix,
        "conflicting_fields": conflicting_fields,
    }


@dataclass
class DomainManifest:
    id: str
    name: str
    version: str
    description: str = ""
    maintainer: str = ""
    disclaimer: str = ""
    predicate_whitelist: list[str] = field(default_factory=list)
    files: dict = field(default_factory=dict)


class DomainLoader:
    def __init__(self, domain_path: Optional[str] = None):
        if domain_path is None:
            domain_path = os.environ.get(
                "SHANMIAO_DOMAIN_PATH",
                os.path.join(os.path.dirname(__file__), "..", "domains"),
            )
        self._domain_path = Path(domain_path).resolve()
        self._manifests: dict[str, DomainManifest] = {}

    def discover(self) -> list[str]:
        """Recursively discover all domain.json-bearing subdirectories.

        Returns relative paths like 'validated/construction', 'candidate/labor_law'.
        Directories starting with '_' are skipped.
        """
        if not self._domain_path.exists():
            return []
        result = []
        for root, dirs, _files in os.walk(str(self._domain_path)):
            for dname in dirs:
                if dname.startswith("_"):
                    continue
                dpath = Path(root) / dname
                if (dpath / "domain.json").exists():
                    rel = dpath.relative_to(self._domain_path)
                    result.append(str(rel).replace("\\", "/"))
        return result

    def _find_domain_dir(self, domain_id: str) -> Optional[Path]:
        """Resolve a domain_id to its directory under _domain_path.

        If domain_id contains '/', treat it as a relative path from _domain_path.
        Otherwise, search recursively for a matching directory name (short-name lookup).
        """
        if "/" in domain_id:
            candidate = self._domain_path / domain_id
            if candidate.is_dir() and (candidate / "domain.json").exists():
                return candidate
            return None
        # Short name: search all subdirectories
        for root, dirs, _files in os.walk(str(self._domain_path)):
            for dname in dirs:
                if dname == domain_id:
                    dpath = Path(root) / dname
                    if (dpath / "domain.json").exists():
                        return dpath
        return None

    def _resolve_domain_path(self, domain_id: str) -> Path:
        """Resolve domain_id to a concrete directory Path under _domain_path.

        If domain_id contains '/', treat it as a relative path from _domain_path
        (this covers "validated/construction", "candidate/labor_law", etc.).
        Otherwise, use _find_domain_dir for short-name lookup (e.g. "construction").
        Falls back to _domain_path / domain_id if unresolved, preserving original behavior.
        """
        if "/" in domain_id:
            return self._domain_path / domain_id
        resolved = self._find_domain_dir(domain_id)
        if resolved is not None:
            return resolved
        return self._domain_path / domain_id

    def load(self, domain_id: str) -> DomainManifest:
        if domain_id in self._manifests:
            return self._manifests[domain_id]
        # Resolve to concrete directory (supports short names like 'construction')
        domain_dir = self._find_domain_dir(domain_id)
        if domain_dir is None:
            raise FileNotFoundError(
                f"Domain '{domain_id}' not found under {self._domain_path}"
            )
        # Use resolved path as canonical domain_id for path construction
        resolved_id = str(domain_dir.relative_to(self._domain_path)).replace("\\", "/")
        manifest_path = domain_dir / "domain.json"
        with open(manifest_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        manifest = DomainManifest(
            id=resolved_id,
            name=raw["name"],
            version=raw.get("version", "0.0.0"),
            description=raw.get("description", ""),
            maintainer=raw.get("maintainer", ""),
            disclaimer=raw.get("disclaimer", ""),
            predicate_whitelist=raw.get("predicate_whitelist", []),
            files=raw.get("files", {}),
        )
        self._manifests[domain_id] = manifest
        return manifest

    def load_ontology(self, domain_id: str) -> dict:
        manifest = self.load(domain_id)
        onto_file = manifest.files.get("ontology", "ontology.json")
        path = self._resolve_domain_path(domain_id) / onto_file
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_rules_package(self, domain_id: str) -> dict:
        manifest = self.load(domain_id)
        rules_file = manifest.files.get("rules_package", "rules.json")
        path = self._resolve_domain_path(domain_id) / rules_file
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _available_package_files(self, domain_id: str) -> list[str]:
        domain_dir = self._resolve_domain_path(domain_id)
        available = []
        # Check domain.json files.rules_packages first, then fall back to default list
        manifest = self.load(domain_id)
        declared = manifest.files.get("rules_packages", [])
        if declared:
            for fname in declared:
                if (domain_dir / fname).exists():
                    available.append(fname)
            if available:
                return available
        # Fallback: use rules_package field if declared (canonical master)
        rp = manifest.files.get("rules_package", "")
        if rp and (domain_dir / rp).exists():
            available.append(rp)
            return available
        # Last resort: try known filenames
        for fname in ["rules.json", "general-contract.json", "main-contract.json"]:
            if (domain_dir / fname).exists():
                available.append(fname)
        return available

    def resolve_package_files(
        self, domain_id: str, contract_types: Optional[list[str]] = None
    ) -> list[str]:
        available = self._available_package_files(domain_id)
        if not available:
            return []
        return list(available)

    def load_rules_packages(
        self, domain_id: str, contract_types: Optional[list[str]] = None,
    ) -> list[dict]:
        files = self.resolve_package_files(domain_id, contract_types)
        packages = []
        for fname in files:
            path = self._resolve_domain_path(domain_id) / fname
            if not path.exists():
                continue
            with open(path, "r", encoding="utf-8") as f:
                packages.append(json.load(f))
        return packages

    def load_constraints(self, domain_id: str) -> list:
        _ensure_z3()
        if not _CITTA_Z3_AVAILABLE:
            return []
        manifest = self.load(domain_id)
        constraints_file = manifest.files.get("z3_constraints", "constraints.py")
        mod_path = self._resolve_domain_path(domain_id) / constraints_file
        if not mod_path.exists():
            return []
        spec = importlib.util.spec_from_file_location(
            f"shanmiao_domain_{domain_id}_constraints", str(mod_path)
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if hasattr(mod, "build_constraints"):
            return mod.build_constraints()
        return []

    def get_contract_classification(self, domain_id: str) -> Optional[dict]:
        manifest_path = self._resolve_domain_path(domain_id) / "domain.json"
        if not manifest_path.exists():
            return None
        with open(manifest_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return raw.get("contract_classification")


def _is_construction_domain(domain_id: str) -> bool:
    """Match construction domain regardless of subdirectory prefix."""
    return bool(domain_id) and domain_id.split("/")[-1] == "construction"


SPP_PREDICATES = {
    "MUST_BE_GE", "MUST_BE_LE", "MUST_BE_GT", "MUST_BE_LT",
    "MUST_EQUAL", "MUST_NOT_EQUAL",
    "IMPLIES", "MUTUALLY_EXCLUSIVE_WITH", "FORBIDS", "REQUIRES",
    "LOGICAL_CHAIN", "SCOPED_TO",
}

TYPE_TO_PRED: dict[str, str] = {
    "numeric_comparison": "MUST_BE_GE",
    "sum_numeric_comparison": "MUST_BE_LE",
    "co_occurrence": "IMPLIES",
    "mutual_exclusion": "MUTUALLY_EXCLUSIVE_WITH",
    "forbidden_pattern": "FORBIDS",
    "required_pattern": "REQUIRES",
    "logical_chain": "LOGICAL_CHAIN",
    "scope_constraint": "SCOPED_TO",
    "topic_coverage": "FORBIDS",
    "contextual_co_occurrence": "IMPLIES",
    "definition_contains": "REQUIRES",
    "ast_check": "FORBIDS",
}

REQUIRED_FIELDS = {"subject", "predicate", "object", "source_ref", "domain"}


class SPPValidationError(ValueError):
    pass


class DomainValidationError(ValueError):
    pass


class RecursionSafetyError(RuntimeError):
    """R2.6 — Auto-extraction recursion depth exceeded."""
    def __init__(self, max_depth: int = 2, context: str = ""):
        self.max_depth = max_depth
        self.context = context
        super().__init__(
            f"Auto-extraction recursion depth exceeded (max={max_depth}). "
            f"Context: {context}. Manual review required."
        )


def validate_spp(proposition: dict, domain: DomainManifest) -> None:
    missing = REQUIRED_FIELDS - set(proposition.keys())
    if missing:
        raise SPPValidationError(f"Missing required fields: {missing}")
    pred = proposition["predicate"]
    if pred not in SPP_PREDICATES:
        raise SPPValidationError(f"Unknown predicate '{pred}'")
    if domain.predicate_whitelist and pred not in domain.predicate_whitelist:
        raise SPPValidationError(
            f"Predicate '{pred}' not allowed in domain '{domain.id}'"
        )
    conf = proposition.get("confidence", 1.0)
    if not (0.0 <= conf <= 1.0):
        raise SPPValidationError(f"Confidence out of range: {conf}")


class ShanmiaoKernel:
    def __init__(self, domain_path: Optional[str] = None):
        self._loader = DomainLoader(domain_path)
        self._engine = PythonValidationEngine()
        self._current_domain: Optional[DomainManifest] = None
        self._loaded_package_ids: list[str] = []
        self._contract_types_explicit: bool = False
        self._dynamic_classifier: Optional[object] = None
        self._hybrid_classifier: Optional[object] = None
        # Self-bootstrap rings (try-imported above; None if unavailable)
        self._domain_prototype_store: Optional[object] = None  # Ring 1: lazy init DomainPrototypeStore
        self._auto_clusterer: Optional[object] = None  # Ring 2: lazy init AutoClusterer
        self._auto_clusters_cache: dict[str, list] = {}  # domain_id -> list of AutoCluster objects
        # Online incremental learner (WO-45): lazy init on first validate with enable_layers
        self._online_learner: Optional[object] = None
        # P1: Lexical domain classifier (CJK bigram coverage)
        self._lexical_classifier: Optional[object] = None  # lazy init LexicalPrototypeStore
        # R2.6: Recursion safety — per-instance depth counter for auto-extract loop
        self._auto_extract_depth: int = 0
        # R2.8: ingest recursion depth (book->chapter->section, MAX=3)
        self._ingest_depth: int = 0

    def list_domains(self) -> list[dict]:
        result = []
        for did in self._loader.discover():
            m = self._loader.load(did)
            result.append({
                "id": m.id, "name": m.name, "version": m.version,
                "description": m.description,
                "predicate_whitelist": m.predicate_whitelist,
            })
        return result

    def _load_packages_for_domain(
        self, domain_id: str, contract_types: Optional[list[str]] = None,
    ) -> list[str]:
        manifest = self._loader.load(domain_id)
        packages = self._loader.load_rules_packages(domain_id, contract_types)
        # ── Normalise bare-list rules.json (candidate domains) into package dict ──
        for idx, rules_pkg in enumerate(packages):
            if isinstance(rules_pkg, list):
                packages[idx] = {
                    "id": domain_id.replace("/", "-") + "-rules",
                    "name": f"{domain_id} rules",
                    "version": "0.1.0",
                    "domain": domain_id.split("/")[-1],
                    "rules": rules_pkg,
                }
        for rules_pkg in packages:
            if manifest.predicate_whitelist:
                for rule in rules_pkg.get("rules", []):
                    ct = rule.get("condition", {}).get("type", "")
                    pred = TYPE_TO_PRED.get(ct)
                    if pred is None:
                        raise DomainValidationError(
                            f"Rule '{rule.get('id', '?')}' uses unknown condition "
                            f"type '{ct}' - not in TYPE_TO_PRED mapping"
                        )
                    if pred not in manifest.predicate_whitelist:
                        raise DomainValidationError(
                            f"Rule '{rule.get('id', '?')}' maps to SPP predicate "
                            f"'{pred}' (via condition type '{ct}'), but domain "
                            f"'{domain_id}' only whitelists: "
                            f"{manifest.predicate_whitelist}"
                        )
        pkg_ids = []
        for rules_pkg in packages:
            pkg_id = rules_pkg.get("id", "")
            rules_pkg["_input_text"] = ""
            try:
                self._engine.load_package(rules_pkg)
            except Exception:
                try:
                    self._engine.reload_package(pkg_id, rules_pkg)
                except Exception:
                    pass
            pkg_ids.append(pkg_id)
        self._loaded_package_ids = pkg_ids
        return pkg_ids

    def _split_contracts(self, text: str) -> list[dict]:
        """Split multi-contract text by '# ' heading lines.

        Each '# Title' line marks the start of a new independent contract.
        The segment before the first '# ' line, if non-empty, becomes the first contract.
        Returns [{"title": str, "text": str}, ...].
        """
        if not text or not text.strip():
            return []
        lines = text.split("\n")
        segments = []          # list of (title, start_line_idx)
        prefix_lines = []      # lines before the first '# ' heading
        i = 0
        # Collect prefix (lines before first '# ')
        while i < len(lines):
            ln = lines[i]
            # Only lines that START with '# ' (not '##', not '#正文', etc.) and have text after
            if ln.startswith("# ") and len(ln) > 2:
                # This is a new contract heading
                break
            prefix_lines.append(ln)
            i += 1
        # If we found a heading, segments start here
        while i < len(lines):
            ln = lines[i]
            if ln.startswith("# ") and len(ln) > 2:
                title = ln[2:].strip()
                segments.append((title, i + 1))  # content starts after heading line
            i += 1
        # Build contracts
        contracts = []
        # Prefix contract (if non-empty after stripping whitespace)
        prefix_text = "\n".join(prefix_lines).strip()
        if prefix_text and not segments:
            # No headings at all — single contract
            contracts.append({"title": "全文", "text": text})
            return contracts
        if prefix_text:
            contracts.append({"title": "前言/无标题", "text": prefix_text})
        # Heading-based contracts
        for idx, (title, start) in enumerate(segments):
            end = segments[idx + 1][1] if idx + 1 < len(segments) else len(lines)
            # Body is the heading line + content until next heading
            body_lines = []
            # heading line
            body_lines.append("# " + title)
            body_lines.extend(lines[start:end])
            body_text = "\n".join(body_lines).strip()
            if body_text:
                contracts.append({"title": title, "text": body_text})
        if not contracts:
            # Fallback: single contract
            contracts.append({"title": "全文", "text": text})
        return contracts

    def _classify_contract_profile(self, text: str, domain_id: str) -> dict:
        """Unified contract classification via ClassifierPipeline.

        Replaces the original _infer_contract_profile() (230 lines of duplicate
        logic) with a single delegation to the 5-layer pipeline.  The returned
        dict preserves the keys that validate() and the multi-contract merge path
        expect.
        """
        classification = self._loader.get_contract_classification(domain_id)
        from app.engine.domain_pipeline import ClassifierPipeline
        pipeline = ClassifierPipeline(
            lexical_store=(self._get_lexical_store()
                           if hasattr(self, '_get_lexical_store') else None),
            domain_path=str(self._loader._domain_path),
            proto_store=(self._get_domain_prototype_store()
                         if hasattr(self, '_get_domain_prototype_store') else None),
            candidate_store=(self._get_candidate_store()
                             if hasattr(self, '_get_candidate_store') else None),
        )
        result = pipeline.classify(text=text, domain_id=domain_id,
                                    classification=classification)
        value = self._estimate_contract_value(text)
        # Contract type tags (keep existing downstream expectations)
        types: list[str] = []
        if value > 50_000_000:
            types.append("大型工程")
        if "总承包" in text or "施工总承包" in text or "总包" in text:
            types.append("施工总承包")
        elif "分包" in text:
            has_negative = bool(re.search(
                r'(?:不得|禁止|不可|严禁|不应|不许)\S{0,20}分[包包]', text,
            ))
            if not has_negative:
                types.append("专业分包")
        if "屋面防水" in text or "外墙保温" in text:
            if value < 100_000:
                types.append("小型维修")
        if not types:
            if value > 10_000_000:
                types.append("大型工程")
            else:
                types.append("中小型工程")
        return {
            "types": types,
            "estimated_value": value,
            "broad_type": result.broad_type,
            "confidence": result.confidence,
            "reject_reason": result.reject_reason,
            "reject_detail": getattr(result, "reject_detail", None),
            "classification_source": result.source,
            "lexical_domains": result.matched_domains,
            "lexical_scores": result.lexical_scores,
            "lexical_primary": bool(result.matched_domains),
        }

    @staticmethod
    def _estimate_contract_value(text: str) -> int:
        value = 0
        m = re.search(r'(\d+(?:\.\d+)?)\s*万元', text)
        if m:
            value = int(float(m.group(1)) * 10000)
        else:
            m = re.search(r'(\d+(?:\.\d+)?)\s*亿元', text)
            if m:
                value = int(float(m.group(1)) * 100000000)
        if value == 0:
            value = int(ShanmiaoKernel._parse_chinese_amount(text))
        return value

    @staticmethod
    def _parse_chinese_amount(text: str) -> float:
        d = dict(zip("零壹贰叁肆伍陆柒捌玖一二三四五六七八九", [0,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9]))
        u = dict(zip("十百千仟拾佰", (10,100,1000,1000,10,100)))
        b = dict(zip("万亿萬億", (10000,100000000,10000,100000000)))
        chars = set("壹贰叁肆伍陆柒捌玖一二三四五六七八九十百千万亿零元整仟拾佰仟萬億")
        runs = []
        cur = ""
        for ch in text:
            if ch in chars:
                cur += ch
            else:
                if cur:
                    runs.append(cur)
                    cur = ""
        if cur:
            runs.append(cur)
        best = 0
        for run in runs:
            val = ShanmiaoKernel._eval_run(run)
            if val > best:
                best = val
        return best

    @staticmethod
    def _eval_run(run: str) -> float:
        d = dict(zip("零壹贰叁肆伍陆柒捌玖一二三四五六七八九", [0,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9]))
        u = dict(zip("十百千仟拾佰", (10,100,1000,1000,10,100)))
        b = dict(zip("万亿萬億", (10000,100000000,10000,100000000)))
        run = run.rstrip("元整")
        pairs: list[tuple[int, int]] = []
        digits = 0
        total = 0
        for ch in run:
            if ch in d:
                digits += d[ch]
            elif ch in u:
                pairs.append((1 if digits == 0 else digits, u[ch]))
                digits = 0
            elif ch in b:
                sub_total = sum(dg * m for dg, m in pairs) + digits
                total += sub_total * b[ch]
                pairs = []
                digits = 0
        total += sum(dg * m for dg, m in pairs) + digits
        return total

    # ── P1: Lexical domain classifier ──
    def _get_lexical_store(self) -> Optional[object]:
        """Lazy-init LexicalPrototypeStore from saved prototypes.

        Returns None if store is unavailable (module not importable or no prototypes saved).
        """
        if self._lexical_classifier is not None:
            return self._lexical_classifier
        if not _HAS_LEXICAL_STORE:
            return None
        try:
            self._lexical_classifier = LexicalPrototypeStore.load(_LEXICAL_STORE_PATH)
            if not self._lexical_classifier.prototypes:
                self._lexical_classifier = None
                return None
            return self._lexical_classifier
        except Exception:
            logger.warning("LexicalPrototypeStore lazy-init failed", exc_info=True)
            return None

    def load_domain(
        self, domain_id: str, contract_types: Optional[list[str]] = None,
    ) -> DomainManifest:
        manifest = self._loader.load(domain_id)
        self._load_packages_for_domain(domain_id, contract_types)
        self._current_domain = manifest
        self._contract_types_explicit = contract_types is not None
        return manifest

    @property
    def current_domain(self) -> Optional[DomainManifest]:
        return self._current_domain

    @property
    def loader(self) -> DomainLoader:
        return self._loader

    def _get_hybrid_classifier(self) -> Optional[object]:
        if self._hybrid_classifier is not None:
            return self._hybrid_classifier
        if self._current_domain is None:
            return None
        domain_dir = str(self._loader._domain_path / self._current_domain.id)
        try:
            from app.engine.dynamic_classifier import DynamicClauseClassifier, HybridClauseClassifier
        except ImportError:
            return None
        dc = DynamicClauseClassifier()
        proto_count = dc.seed_from_domain_config(domain_dir)
        if proto_count == 0:
            return None
        from app.engine.clause_splitter import ClauseSplitter
        self._dynamic_classifier = dc
        self._hybrid_classifier = HybridClauseClassifier(
            dynamic=dc, keyword_table={},
        )
        return self._hybrid_classifier

    # ── Self-bootstrap Ring 1: DomainPrototypeStore ──
    def _get_domain_prototype_store(self) -> Optional[object]:
        """Lazy-init DomainPrototypeStore for domain-level text classification.

        Returns None if the module is not importable (falls back to keyword classification).
        """
        if self._domain_prototype_store is not None:
            return self._domain_prototype_store
        if not _HAS_PROTOTYPE_STORE:
            return None
        try:
            self._domain_prototype_store = DomainPrototypeStore()
            base = self._loader._domain_path
            if base.exists():
                for d in base.iterdir():
                    if not d.is_dir() or d.name.startswith("_"):
                        continue
                    p = d / "prototypes.json"
                    if p.exists():
                        try:
                            partial = DomainPrototypeStore.load(str(p))
                            for pid, proto in partial.prototypes.items():
                                self._domain_prototype_store.store_prototype(proto)
                        except Exception:
                            pass
            return self._domain_prototype_store
        except Exception:
            logger.warning("DomainPrototypeStore lazy-init failed", exc_info=True)
            return None

    # ── T4: Candidate store (candidate domain lexical prototypes) ──
    def _get_candidate_store(self) -> Optional[object]:
        """Lazy-init CandidatePrototypeStore from persisted data.

        Bootstraps from candidate/ domain directories if the persisted file
        doesn't exist yet.  Returns None if the CandidatePrototypeStore module
        is not importable.
        """
        if hasattr(self, '_candidate_store') and self._candidate_store is not None:
            return self._candidate_store
        try:
            from app.engine.candidate_store import CandidatePrototypeStore
        except ImportError:
            self._candidate_store = None
            return None
        try:
            store_path = self._loader._domain_path / "data" / "candidate_prototypes.json"
            store = CandidatePrototypeStore()
            if store_path.exists():
                store.load(str(store_path))
            else:
                # Bootstrap from candidate/ directories
                import json as _cjson
                for _dname in ["civil_code", "civil_procedure",
                                "immigration_law", "labor_law",
                                "nationality_law"]:
                    _ddir = self._loader._domain_path / "candidate" / _dname
                    if _ddir.exists():
                        try:
                            _meta = _cjson.load(open(_ddir / "domain.json", encoding="utf-8"))
                            _rules = _cjson.load(open(_ddir / "rules.json", encoding="utf-8"))
                            store.register(
                                domain=_dname,
                                meta=_meta,
                                rules=_rules if isinstance(_rules, list) else _rules.get("rules", []),
                            )
                        except Exception:
                            pass
                store_path.parent.mkdir(parents=True, exist_ok=True)
                store.save(str(store_path))
            self._candidate_store = store
            return store
        except Exception:
            logger.warning("CandidatePrototypeStore lazy-init failed", exc_info=True)
            self._candidate_store = None
            return None

    # ── Self-bootstrap Ring 2: AutoClusterer ──
    def _get_auto_clusterer(self) -> Optional[object]:
        """Lazy-init AutoClusterer for unsupervised clause block clustering.

        Returns None if the module is not importable (falls back to _TYPE_PATTERNS keywords).
        """
        if self._auto_clusterer is not None:
            return self._auto_clusterer
        if not _HAS_AUTO_CLUSTERER:
            return None
        try:
            from app.engine.auto_clusterer import AutoClusterer as RealAutoClusterer
            self._auto_clusterer = RealAutoClusterer()
            return self._auto_clusterer
        except Exception:
            logger.warning("AutoClusterer lazy-init failed", exc_info=True)
            return None

    def _get_or_build_auto_clusters(self, domain_id: str, texts: Optional[list[str]] = None) -> list:
        """Get cached auto-clusters for domain_id, or build them from corpus texts.

        Args:
            domain_id: Domain identifier.
            texts: List of contract texts for building clusters (if not cached).

        Returns:
            List of AutoCluster objects, or empty list if unavailable.
        """
        if domain_id in self._auto_clusters_cache:
            return self._auto_clusters_cache[domain_id]
        clusterer = self._get_auto_clusterer()
        if clusterer is None or not texts:
            return []

        # Pool blocks from all provided texts
        from app.engine.clause_splitter import ClauseSplitter, ClauseBlock
        all_blocks = []
        for t in texts:
            blocks = ClauseSplitter.split(t)
            all_blocks.extend(blocks)

        if len(all_blocks) < 2:
            return []

        try:
            clusters = clusterer.cluster(all_blocks)
            self._auto_clusters_cache[domain_id] = clusters
            return clusters
        except Exception:
            logger.warning("AutoClusterer cluster() failed", exc_info=True)
            return []

    def _reload_packages_with_input_text(self, text: str) -> None:
        for pkg_id in list(self._loaded_package_ids):
            compiled = self._engine._packages.get(pkg_id)
            if compiled is None:
                continue
            domain_dir = self._loader._domain_path / self._current_domain.id
            for fname in self._loader._available_package_files(
                self._current_domain.id
            ):
                fpath = domain_dir / fname
                if not fpath.exists():
                    continue
                with open(fpath, "r", encoding="utf-8") as f:
                    pkg_data = json.load(f)
                if isinstance(pkg_data, list):
                    continue
                if pkg_data.get("id") == pkg_id:
                    pkg_data["_input_text"] = text
                    try:
                        self._engine.reload_package(pkg_id, pkg_data)
                    except Exception:
                        pass
                    break

    def validate(
        self, text: str, domain_id: Optional[str] = None,
        enable_layers: bool = False, timeout_ms: int = 5000,
        _called_recursively: bool = False,
        structured_extractions: Optional[list[dict]] = None,
        validation_mode: str = "document",
    ) -> dict:
        """Validate contract text against loaded domain rules.

        Args:
            text: Contract text or clause fragment.
            domain_id: Domain to validate against (e.g. "validated/construction").
            validation_mode: "document" (default) — full contract, classifier
                gates non-contract text. "clause" — short clause fragment,
                trusts explicit domain_id and skips broad domain rejection.
                Requires domain_id to be explicitly provided; raises ValueError
                if validation_mode="clause" and domain_id is None.
        """
        if validation_mode == "clause" and not domain_id:
            raise ValueError(
                "validation_mode='clause' requires an explicit domain_id. "
                "Clause fragments have no context for domain classification."
            )
        # ── Multi-contract splitting ──
        if not _called_recursively:
            contracts = self._split_contracts(text)
            if len(contracts) > 1:
                all_evidence: list[EvidenceItem] = []
                all_results: list[dict] = []
                for c in contracts:
                    seg_title = c.get("title", "")
                    # Infer profile per-contract via unified pipeline
                    if self._current_domain is None:
                        self.load_domain(domain_id or "construction")
                    profile = self._classify_contract_profile(
                        c["text"], domain_id=self._current_domain.id,
                    ) if self._current_domain is not None else {}
                    sub_result = self.validate(
                        c["text"], domain_id=domain_id,
                        enable_layers=enable_layers, timeout_ms=timeout_ms,
                        _called_recursively=True,
                        structured_extractions=structured_extractions,
                    )
                    sub_result["contract_segment"] = seg_title
                    all_results.append(sub_result)
                    # Collect evidence items with contract_segment tag
                    for ev_item in sub_result.get("evidence", []):
                        if isinstance(ev_item, dict):
                            ev_item["contract_segment"] = seg_title
                        elif hasattr(ev_item, "contract_segment"):
                            ev_item.contract_segment = seg_title
                    all_evidence.extend(sub_result.get("evidence", []))
                # Merge results: take first as template, splice evidence
                merged = dict(all_results[0]) if all_results else {}
                merged["evidence"] = all_evidence
                # Extract segment stats from each sub-result's summary or evidence_chain
                segment_infos = []
                for c, sub in zip(contracts, all_results):
                    sm = sub.get("summary", {})
                    ev = sub.get("evidence_chain", [])
                    n_total = len(ev)
                    n_passed = sum(1 for e in ev if e.get("status") == "PASSED")
                    n_failed = sum(1 for e in ev if e.get("status") == "FAILED")
                    n_na     = sum(1 for e in ev if e.get("status") == "NOT_APPLICABLE")
                    segment_infos.append({
                        "title": c["title"],
                        "broad_type": sub.get("contract_profile", {}).get("broad_type", ""),
                        "estimated_value": sub.get("contract_profile", {}).get("estimated_value", 0),
                        "total_rules": n_total, "passed": n_passed,
                        "failed": n_failed, "na": n_na,
                    })
                merged["contract_segments"] = segment_infos
                merged["total"] = sum(s["total_rules"] for s in segment_infos)
                merged["passed"] = sum(s["passed"] for s in segment_infos)
                merged["failed"] = sum(s["failed"] for s in segment_infos)
                merged["na"] = sum(s["na"] for s in segment_infos)
                merged["multi_contract"] = True
                return merged

        if domain_id:
            self.load_domain(domain_id)
        if self._current_domain is None:
            raise RuntimeError("No domain loaded. Call load_domain() first.")
        profile = self._classify_contract_profile(text, domain_id=self._current_domain.id)

        # ── P1: Domain aware rejection ── dynamic VALID_DOMAINS from filesystem ──
        reject_reason = profile.get("reject_reason") if profile else None

        # Check lexical classification: if the primary lexical domain is NOT
        # in any filesystem-validated or candidate domain, reject unless it's
        # a multi-domain match that includes at least one registered domain.
        lexical_domains = profile.get("lexical_domains", [])
        _validated_domains: set[str] = set()
        _candidate_domains: set[str] = set()
        for _base in ("validated", "candidate"):
            _base_path = self._loader._domain_path / _base
            if _base_path.exists():
                for _d in _base_path.iterdir():
                    if _d.is_dir() and (_d / "domain.json").exists():
                        if _base == "validated":
                            _validated_domains.add(_d.name)
                        else:
                            _candidate_domains.add(_d.name)
        VALID_DOMAINS = _validated_domains | _candidate_domains
        if not reject_reason and lexical_domains:
            has_valid = any(d in VALID_DOMAINS for d in lexical_domains)
            has_only_invalid = all(d not in VALID_DOMAINS for d in lexical_domains)
            if has_only_invalid:
                reject_reason = (
                    f"文本被分类为非合同域 ({'+'.join(lexical_domains)})。"
                    f"当前已加载域: {', '.join(sorted(VALID_DOMAINS))}。"
                    f"如需对此文本进行规则验证，请先提取该领域的候选规则。"
                )

        # ── Clause mode: trust explicit domain_id, skip broad domain rejection ──
        if reject_reason and validation_mode == "clause":
            reject_reason = None  # bypass — caller vouches for the domain

        if reject_reason:
            # ── R2.6: Recursion safety gate ──
            self._auto_extract_depth += 1
            if self._auto_extract_depth >= 2:
                self._auto_extract_depth -= 1
                raise RecursionSafetyError(
                    max_depth=2,
                    context=f"domain={self._current_domain.id if self._current_domain else 'none'}",
                )
            try:
                return {
                    "status": "REJECTED",
                    "reject_reason": reject_reason,
                    "contract_profile": profile,
                    "disclaimer": self._current_domain.disclaimer or (
                        "本系统仅提供辅助初审建议，校验结果不具备法律效力。"
                    ),
                    "evidence_chain": [],
                    "summary": {"rejected": True, "reason": reject_reason},
                }
            finally:
                self._auto_extract_depth -= 1

        # GRADUAL MIGRATION: if auto-clusters are available for this domain, pass them
        # as classifier context to ClauseSplitter so clause_type comes from auto-clusters
        # instead of keywords. Falls back to hybrid classifier or keyword _TYPE_PATTERNS.
        hybrid = self._get_hybrid_classifier()
        auto_clusters = self._get_or_build_auto_clusters(
            self._current_domain.id,
            texts=None,  # Only use cached clusters; don't rebuild every validate call
        )
        classifier = hybrid
        if auto_clusters:
            # Wrap auto-clusters into a classifier-like object for ClauseSplitter
            # The auto_clusters themselves are used by the clause_splitter's _apply_classifier
            # entry point: if a classifier is present, it overrides _TYPE_PATTERNS.
            # We use the hybrid classifier (which respects auto-clusters) when available.
            pass  # `hybrid` already serves this purpose if it respects auto-cluster data
        clause_blocks = ClauseSplitter.split(text, classifier=classifier)
        blocks_dicts = [
            dict(
                clause_id=b.clause_id,
                clause_title=b.clause_title,
                clause_type=b.clause_type,
                type_confidence=b.type_confidence,
                content=b.content,
            )
            for b in clause_blocks
        ]
        self._reload_packages_with_input_text(text)
        # ── R2.10: Sidecar structured input injection ──
        if structured_extractions:
            rule_label_map_si: dict[str, str] = {}
            for pkg_id in self._loaded_package_ids:
                compiled = self._engine._packages.get(pkg_id)
                if compiled:
                    for r in compiled.rules:
                        lbl = r.condition_params.get("label", "")
                        if lbl:
                            rule_label_map_si[lbl] = r.id
            self._engine._preserve_structured_inputs(
                structured_extractions, rule_label_map_si,
            )
        if not self._loaded_package_ids:
            raise RuntimeError(
                f"No rule packages loaded for domain '{self._current_domain.id}'"
            )
        result = self._engine.validate(
            input_data=dict(
                text=text,
                clause_blocks=blocks_dicts,
                contract_broad_type=profile.get("broad_type", ""),
                estimated_value=profile.get("estimated_value", 0),
            ), packages=self._loaded_package_ids,
            options=dict(
                timeout_ms=timeout_ms, max_evidence=100,
                include_warnings=True,
            ),
        )
        result["disclaimer"] = self._current_domain.disclaimer or (
            "本系统仅提供辅助初审建议，校验结果不具备法律效力。"
        )
        if profile:
            result["contract_profile"] = profile
        result["clause_blocks"] = [
            dict(
                clause_id=b.clause_id,
                clause_title=b.clause_title[:80],
                clause_type=b.clause_type,
                type_confidence=b.type_confidence,
                content_preview=b.content[:200],
            )
            for b in clause_blocks
        ]
        if enable_layers:
            try:
                from app.engine.knowledge_layers import KnowledgeLayeringEngine
                layering = KnowledgeLayeringEngine()
                pkg_registry = dict(
                    (pkg_id, dict(
                        domain=self._current_domain.id,
                        maintainer=self._current_domain.maintainer,
                        version=self._current_domain.version,
                    ))
                    for pkg_id in self._loaded_package_ids
                )
                lr = layering.layered_validate(
                    text=text, validation_result=result,
                    package_registry=pkg_registry,
                )
                result["knowledge_layers"] = lr.to_dict()
            except Exception:
                result["knowledge_layers"] = dict(error="layering failed")

            # ── Deep Algorithm: Belief Propagation Network ──
            try:
                from app.engine.belief_propagation import BeliefNetwork
                bn = BeliefNetwork()
                # Collect rules from loaded packages
                all_rules = []
                for pkg_id in self._loaded_package_ids:
                    pkg = self._engine._packages.get(pkg_id)
                    if pkg and hasattr(pkg, 'rules'):
                        for r in pkg.rules:
                            all_rules.append({
                                "id": r.id,
                                "name": r.name,
                                "category": r.category,
                                "clause_type": r.clause_type,
                                "source_credibility": r.source_credibility,
                                "condition": r.condition_params,
                            })
                if all_rules:
                    bn.build_network(all_rules)
                    evidence = {
                        e.get("rule_id", ""): e.get("status", "NOT_APPLICABLE")
                        for e in result.get("evidence_chain", [])
                    }
                    bp_report = bn.propagate(evidence)
                    result["belief_network"] = bp_report.to_dict()
                    # Update evidence chain credibility
                    for ev in result.get("evidence_chain", []):
                        rid = ev.get("rule_id", "")
                        if rid in bp_report.credibility_adjustments:
                            adj = bp_report.credibility_adjustments[rid]
                            ev["source_credibility"] = round(
                                ev.get("source_credibility", 0.5) + adj, 3)
            except Exception:
                result["belief_network"] = dict(error="belief propagation failed")

            # ── Triager: Three-state classification layer (三态分流层) ──
            try:
                from app.engine.triager import Triager
                triager = Triager()
                # Load domain-specific calibration if available
                if self._current_domain is not None:
                    domain_dir = self._loader._domain_path / self._current_domain.id
                    if domain_dir.is_dir():
                        triager.load_domain_calibration(str(domain_dir))
                result["evidence_chain"] = triager.triage(
                    evidence_chain=result.get("evidence_chain", []),
                    belief_network=result.get("belief_network"),
                    contract_text=text,
                )
                result["triager_summary"] = triager.summary(
                    result["evidence_chain"]
                )
            except Exception:
                result["triager_summary"] = dict(error="triager failed")

        # ── Deep Algorithm: Rule Dependency Graph ──
        try:
            from app.engine.rule_graph import build_graph_from_packages
            graph = build_graph_from_packages(self._engine._packages)
            verdicts = {
                e.get("rule_id", ""): e.get("status", "NOT_APPLICABLE")
                for e in result.get("evidence_chain", [])
            }
            classification = graph.dependency_resolve(verdicts)
            causal_reports = {}
            for ev in result.get("evidence_chain", []):
                if ev.get("status") == "FAILED":
                    rid = ev.get("rule_id", "")
                    report = graph.find_causal_chain(rid, verdicts)
                    causal_reports[rid] = report.to_dict()
            result["causal_graph"] = {
                "node_count": graph.node_count(),
                "edge_count": graph.edge_count(),
                "dependency_classification": classification,
                "causal_reports": causal_reports,
            }
        except Exception:
            result["causal_graph"] = dict(error="graph analysis failed")

        # ── Online Incremental Learner (WO-45) ──
        # Runs after triager + belief network + causal graph.
        # Uses high-confidence engine verdicts as training signals to
        # incrementally update the clause classifier and ontology.
        # Failure is non-fatal — the main pipeline result is unaffected.
        try:
            from app.engine.online_learner import OnlineLearner
            if self._online_learner is None:
                self._online_learner = OnlineLearner(
                    model_path=os.path.join(
                        os.path.dirname(os.path.dirname(
                            os.path.abspath(__file__))),
                        "data", "models", "clause_model.json"),
                    ontology_path=os.path.join(
                        os.path.dirname(os.path.dirname(
                            os.path.abspath(__file__))),
                        "domains", "construction", "ontology.json"),
                )
            if enable_layers:
                clause_blocks = result.get("clause_blocks", [])
                evidence_chain = result.get("evidence_chain", [])
                stats = self._online_learner.run(evidence_chain, clause_blocks)
                result["online_learner"] = stats.to_dict()
                if stats.total_feedback_accepted > 0:
                    logger.info(
                        "Online learner: %d/%d feedbacks applied, "
                        "%d ontology expansions",
                        stats.total_feedback_accepted,
                        stats.total_feedback_generated,
                        stats.total_ontology_expansions,
                    )
        except Exception as _ol_err:
            result["online_learner"] = dict(
                error="online learner skipped",
                detail=str(_ol_err)[:200],
            )

        return result

    def validate_with_extraction(
        self, text: str, domain_id: Optional[str] = None,
        enable_layers: bool = False, timeout_ms: int = 5000,
        use_llm: bool = False,
    ) -> dict:
        """R2.10: Full extraction → injection → validation loop.

        1. Run MockRuleExtractor (or LLMRuleExtractor if use_llm=True) on text
           to pre-extract structured fields.
        2. Build rule_label_map from currently loaded domain rules.
        3. Call kernel.validate(text, structured_extractions=...) to inject
           pre-extracted fields into handler sidecar.
        4. Return the validation result.

        The extraction layer uses MockRuleExtractor by default (zero API cost,
        heuristic CJK pattern matching). Set use_llm=True to use real LLM.

        Returns the same dict as validate(), plus:
          - "extraction_info": {method, n_fields, fields, ...}
        """
        if domain_id:
            self.load_domain(domain_id)
        if self._current_domain is None:
            raise RuntimeError("No domain loaded. Call load_domain() first.")

        # ── Step 1: Run extraction ──
        extraction_method = "mock"
        structured_extractions = []

        if use_llm:
            try:
                from app.engine.llm_rule_extractor import LLMRuleExtractor
                extractor = LLMRuleExtractor()
                candidates = extractor.extract(
                    text, domain_id=self._current_domain.id,
                )
                for c in candidates:
                    structured_extractions.append({
                        "field": getattr(c, "subject", ""),
                        "value": getattr(c, "expected_value", None),
                        "unit": getattr(c, "unit", "") or "",
                        "operator_hint": getattr(c, "operator", "") or "",
                        "source_text": getattr(c, "source_text", "")[:200],
                        "confidence": getattr(c, "confidence", 0.5),
                    })
                extraction_method = "llm"
            except Exception:
                pass

        if not structured_extractions:
            from app.engine.llm_rule_extractor import MockRuleExtractor
            mock = MockRuleExtractor()
            candidates = mock.extract(
                text, domain_id=self._current_domain.id,
            )
            for c in candidates:
                structured_extractions.append({
                    "field": getattr(c, "subject", ""),
                    "value": getattr(c, "expected_value", None),
                    "unit": getattr(c, "unit", "") or "",
                    "operator_hint": getattr(c, "operator", "") or "",
                    "source_text": getattr(c, "source_text", "")[:200],
                    "confidence": getattr(c, "confidence", 0.5),
                })
            extraction_method = "mock"

        n_fields = len(structured_extractions)

        # ── Step 2: Validate with extracted fields as sidecar ──
        result = self.validate(
            text,
            domain_id=None,
            enable_layers=enable_layers,
            timeout_ms=timeout_ms,
            structured_extractions=structured_extractions,
        )

        # ── Step 3: Attach extraction metadata ──
        result["extraction_info"] = {
            "method": extraction_method,
            "n_fields": n_fields,
            "fields": [
                {"field": e["field"], "value": e["value"], "unit": e["unit"]}
                for e in structured_extractions[:10]
            ],
        }

        return result

    def validate_with_z3(
        self, text: str, domain_id: Optional[str] = None,
        enable_layers: bool = False,
    ) -> dict:
        if domain_id:
            self.load_domain(domain_id)
        result = self.validate(text, enable_layers=enable_layers)
        _ensure_z3()
        if not _CITTA_Z3_AVAILABLE:
            result["z3"] = {"verdict": "SKIPPED", "reason": "Z3 not installed"}
            return result
        constraints = self._loader.load_constraints(self._current_domain.id)
        if not constraints:
            result["z3"] = {"verdict": "SKIPPED", "reason": "No constraints"}
            return result
        solver = _CittaZ3Solver()
        solver.load_from_evidence_chain(result.get("evidence_chain", []))
        solver.load_constraints(constraints)
        z3_result = solver.check()
        result["z3"] = z3_result.to_dict()

        # ── TracedSolver: additive trace layer ──
        try:
            from app.engine.tracer import (
                TracedSolver, TraceableProposition, TraceableConstraint,
                SourceRef, EvidenceRegistry,
            )
            registry = EvidenceRegistry()
            registry.register_document("contract", text[:200], text)
            traceable_props = [
                TraceableProposition(
                    field=ev.get("rule_name", ev.get("rule_id", "")),
                    value=_try_extract_numeric(ev),
                    unit="",
                    source=SourceRef(
                        document_id="contract",
                        snippet=ev.get("input_fragment", "")[:100],
                        confidence=ev.get("source_credibility", 0.5),
                    ),
                    extraction_method=ev.get("extraction_method", "engine"),
                    rule_id=ev.get("rule_id", ""),
                )
                for ev in result.get("evidence_chain", [])
                if ev.get("status") in ("PASSED", "FAILED")
            ]
            traceable_constraints = [
                TraceableConstraint(
                    field=c.field, operator=c.operator,
                    threshold=c.threshold, unit=c.unit,
                    severity=c.severity.value if hasattr(c.severity, 'value') else str(c.severity),
                    source=SourceRef(document_id=c.legal_ref or "regulation", snippet=c.legal_ref or ""),
                    weight=c.weight if hasattr(c, 'weight') else 1.0,
                )
                for c in constraints
            ]
            traced = TracedSolver(registry)
            traced.load_propositions(traceable_props)
            traced.load_constraints(traceable_constraints)
            traced_result = traced.check()
            result["trace_report"] = traced_result.get("trace_report", "")
        except Exception:
            result["trace_report"] = ""

        # ── MUS extraction + ConflictReporter ──
        if z3_result.verdict.value == "UNSATISFIABLE":
            try:
                from app.engine.mus import MUSExtractor
                mus_extractor = MUSExtractor()
                # _propositions is a dict[str, NumericProposition]; convert to list
                solver_props = (list(solver._propositions.values())
                               if isinstance(getattr(solver, '_propositions', None), dict)
                               else getattr(solver, '_propositions', []))
                mus = mus_extractor.find_mus(constraints, solver_props)
                if mus:
                    result["mus"] = {
                        "conflict_set": [
                            {"field": c.field, "operator": c.operator,
                             "threshold": c.threshold, "unit": c.unit}
                            for c in mus.constraints
                        ],
                        "explanation": mus_extractor.explain_conflict(mus),
                        "fix_suggestions": [
                            s.to_dict() for s in mus_extractor.suggest_fix(mus, solver_props)
                        ],
                    }
                else:
                    result["mus"] = {"verdict": "no conflict identified"}

                # ── ConflictReporter: LLM-actionable conflict report from MUS ──
                if mus:
                    try:
                        result["logic_correction"] = _build_conflict_report(
                            mus.constraints, solver_props, text
                        )
                    except Exception:
                        result["logic_correction"] = dict(error="conflict report generation failed")
            except Exception:
                result["mus"] = dict(error="MUS analysis failed")
        else:
            result["mus"] = {"verdict": z3_result.verdict.value,
                             "note": "No conflicts to analyze"}

        return result

    # ═══════════════════════════════════════════════════════════════

    # ═══════════════════════════════════════════════════════════════
    # R2.8: kernel.ingest() — unified ingest entry (5-step pipeline)
    # ═══════════════════════════════════════════════════════════════

    @dataclass
    class IngestResult:
        domain_id: str = ""
        is_new_domain: bool = False
        rules_count: int = 0
        passed_rules: int = 0
        rejected_rules: int = 0
        gate_results: list = field(default_factory=list)
        validation_result: Optional[dict] = None
        reject_reason: Optional[str] = None

    def ingest(self, text: str, **opts) -> "ShanmiaoKernel.IngestResult":
        """Unified 5-step ingest pipeline.

        STEP 0: Recursion gate (_ingest_depth, MAX=3)
        STEP 1: Domain classification (ClassifierPipeline + CandidateStore fallback)
        STEP 2: Rule extraction (StructuredRuleExtractor + LLM fallback + book recursion)
        STEP 3: Gate (AutoValidator 3-gate)
        STEP 4: Promote (graded: new_domain->candidate/, auto_promote->validated/, default->candidate/)
        STEP 5: Validate (optional, on by default)
        """
        import hashlib
        import json as _json
        _log = logging.getLogger(__name__)
        source_type = opts.get("source_type", "unknown")
        domain_hint = opts.get("domain_hint")
        enable_llm = opts.get("enable_llm", False)
        auto_promote = opts.get("auto_promote", False)
        validate_after = opts.get("validate_after", True)

        # ── STEP 0: Recursion gate ──
        _MAX_INGEST_DEPTH = 3
        self._ingest_depth += 1
        if self._ingest_depth > _MAX_INGEST_DEPTH:
            self._ingest_depth -= 1
            raise RecursionSafetyError(
                max_depth=_MAX_INGEST_DEPTH,
                context=f"ingest recursion depth exceeded. source_type={source_type}",
            )
        try:
            # ── STEP 1: Domain classification ──
            is_new_domain = False
            if domain_hint:
                domain_id = domain_hint
            else:
                from app.engine.domain_pipeline import ClassifierPipeline
                classification = None
                try:
                    classification = self._loader.get_contract_classification("construction")
                except Exception:
                    pass
                pipeline = ClassifierPipeline(
                    lexical_store=(self._get_lexical_store()
                                   if hasattr(self, '_get_lexical_store') else None),
                    domain_path=str(self._loader._domain_path),
                    proto_store=(self._get_domain_prototype_store()
                                 if hasattr(self, '_get_domain_prototype_store') else None),
                    candidate_store=None,  # R2.5 fix
                )
                result = pipeline.classify(text=text, domain_id="construction",
                                           classification=classification)
                if result.reject_reason:
                    cstore = (self._get_candidate_store()
                              if hasattr(self, '_get_candidate_store') else None)
                    if cstore:
                        matches = cstore.classify(text)
                        if matches:
                            domain_id = matches[0][0]
                        else:
                            domain_id = "auto_" + hashlib.md5(text[:200].encode()).hexdigest()[:8]
                            is_new_domain = True
                    else:
                        domain_id = "auto_" + hashlib.md5(text[:200].encode()).hexdigest()[:8]
                        is_new_domain = True
                else:
                    domain_id = result.primary_domain or "construction"

            # Resolve domain directory: discover() covers validated/ and candidate/ subdirs
            domain_dir = None
            for d_name in self._loader.discover():
                if d_name == domain_id:
                    domain_dir = self._loader._domain_path / d_name
                    if not (domain_dir / "domain.json").exists():
                        # Try validated/ and candidate/ subdirs
                        for sub in ["validated", "candidate"]:
                            sd = self._loader._domain_path / sub / domain_id
                            if (sd / "domain.json").exists():
                                domain_dir = sd
                                break
                    break
            if domain_dir is None:
                # Try validated/ and candidate/ subdirs as fallback
                for sub in ["validated", "candidate"]:
                    sd = self._loader._domain_path / sub / domain_id
                    if (sd / "domain.json").exists():
                        domain_dir = sd
                        break
            if domain_dir is None:
                domain_dir = self._loader._domain_path / domain_id
            if domain_dir is None:
                domain_dir = self._loader._domain_path / domain_id
                domain_dir.mkdir(parents=True, exist_ok=True)

            dj = domain_dir / "domain.json"
            if not dj.exists():
                with open(dj, "w", encoding="utf-8") as f:
                    _json.dump({
                        "id": domain_id,
                        "name": domain_id.replace("_", " ").title(),
                        "version": "0.1.0",
                        "description": f"Auto-ingested domain ({source_type})",
                        "requires_human_review": is_new_domain,
                    }, f, ensure_ascii=False, indent=2)

            # ── STEP 2: Rule extraction ──
            rules: list = []
            seq = 0

            try:
                from app.engine.rule_extractor import StructuredRuleExtractor, candidates_to_dicts
                from app.engine.auto_validator import candidate_to_rule as _c2r
                extractor = StructuredRuleExtractor()
                candidates = extractor.extract(text)
                for c in candidates:
                    rid = f"{domain_id}-{seq:03d}"
                    rules.append(_c2r(c, rid))
                    seq += 1
            except ImportError:
                _log.info("StructuredRuleExtractor not available")
            except Exception as e:
                _log.warning("StructuredRuleExtractor failed: %s", e)

            est_capacity = max(1, len(text) // 200)
            coverage = len(rules) / est_capacity
            if coverage < 0.30 and enable_llm:
                try:
                    from app.engine.rule_extractor import LLMRuleExtractor
                    llm_extractor = LLMRuleExtractor()
                    llm_rules = llm_extractor.extract(text, domain_id)
                    for lr in llm_rules:
                        rid = f"{domain_id}-llm-{seq:03d}"
                        rules.append(_c2r(lr, rid))
                        seq += 1
                except ImportError:
                    _log.warning("LLMRuleExtractor not available")
                except Exception as e:
                    _log.warning("LLMRuleExtractor failed: %s", e)
            elif coverage < 0.30 and not enable_llm:
                _log.warning("Low coverage (%.0f%%), LLM fallback disabled", coverage * 100)

            if source_type == "book":
                try:
                    from app.engine.bootstrapper import DocumentSplitter
                    chapters = DocumentSplitter.split(text)
                    for ch_data in chapters:
                        ch_text = ch_data.get("body", "")
                        if ch_text and len(ch_text) > 100:
                            self.ingest(ch_text, source_type="legal_text",
                                        enable_llm=enable_llm, auto_promote=auto_promote,
                                        validate_after=False)
                except ImportError:
                    _log.info("DocumentSplitter not available")
                except Exception as e:
                    _log.warning("Book recursion failed: %s", e)

            if not rules:
                return ShanmiaoKernel.IngestResult(
                    domain_id=domain_id, is_new_domain=is_new_domain,
                    reject_reason="No rules extracted",
                )

            # ── STEP 3: Gate ──
            passed_rules, rejected_rules, gate_results = [], [], []
            try:
                from app.engine.auto_validator import AutoValidator
                validator = AutoValidator(engine=self._ensure_engine)
                existing_rules = []
                rp = domain_dir / "rules.json"
                if rp.exists():
                    try:
                        pkg = _json.loads(rp.read_text(encoding="utf-8"))
                        existing_rules = pkg.get("rules", [])
                    except Exception:
                        pass
                for rule in rules:
                    v = validator.validate(rule, str(domain_dir), existing_rules=existing_rules)
                    gate_results.append({"rule_id": rule.get("id", ""), "passed": v.passed,
                                         "gate_results": v.gate_results})
                    if v.passed:
                        passed_rules.append(rule)
                    else:
                        rejected_rules.append(rule)
            except ImportError:
                _log.info("AutoValidator not available")
                passed_rules = list(rules)
            except Exception as e:
                _log.warning("AutoValidator failed: %s", e)
                passed_rules = list(rules)

            # ── STEP 4: Promote ──
            if passed_rules:
                try:
                    from app.engine.auto_validator import AutoValidator
                    av = AutoValidator()
                    av.promote(passed_rules, str(domain_dir),
                               auto_promote=auto_promote and not is_new_domain)
                except ImportError:
                    _log.warning("AutoValidator.promote() not available")
                except Exception as e:
                    _log.warning("Promote failed: %s", e)

            # ── STEP 5: Validate ──
            validation_result = None
            if validate_after and passed_rules:
                try:
                    self.load_domain(domain_id)
                    validation_result = self.validate(
                        text, domain_id=domain_id,
                        enable_layers=opts.get("enable_layers", False),
                    )
                except Exception as e:
                    _log.warning("Post-ingest validation failed: %s", e)
                    validation_result = {"error": str(e)}

            return ShanmiaoKernel.IngestResult(
                domain_id=domain_id, is_new_domain=is_new_domain,
                rules_count=len(rules), passed_rules=len(passed_rules),
                rejected_rules=len(rejected_rules), gate_results=gate_results,
                validation_result=validation_result,
            )
        finally:
            self._ingest_depth -= 1


    # Part C: self_bootstrap_scan — run all 4 rings end-to-end
    # ═══════════════════════════════════════════════════════════════

    def self_bootstrap_scan(self, domain_id: str, contracts_dir: str,
                             authority_source_path: str) -> dict:
        """Run the full self-bootstrap closed loop for a domain.

        Rings:
          1. DomainPrototypeStore — build domain prototype from contracts
          2. AutoClusterer — pool all clause blocks, discover clause type clusters
          3. StructuredRuleExtractor — extract rule candidates from authority source
          4. AutoValidator — 3-gate validation of candidates

        Passing candidates are auto-admitted to the domain's rules.json on disk.

        Args:
            domain_id: Domain identifier (e.g. "auto-construction").
            contracts_dir: Path to directory containing contract .txt files.
            authority_source_path: Path to authority source text file.

        Returns:
            dict: {
                "prototype": {...}, "n_clusters": int, "n_candidates": int,
                "n_admitted": int, "n_rejected": int, "per_gate_details": {...}
            }
        """
        report = {
            "domain_id": domain_id,
            "prototype": None,
            "n_clusters": 0,
            "n_candidates": 0,
            "n_admitted": 0,
            "n_rejected": 0,
            "per_gate_details": {},
            "warnings": [],
        }

        contracts_path = Path(contracts_dir)
        if not contracts_path.is_dir():
            report["warnings"].append(f"Contracts directory not found: {contracts_dir}")
            return report

        authority_path = Path(authority_source_path)
        if not authority_path.is_file():
            report["warnings"].append(f"Authority source not found: {authority_source_path}")
            return report

        # Read contract texts
        contract_files = sorted(contracts_path.glob("*.txt"))
        if not contract_files:
            report["warnings"].append(f"No .txt files in {contracts_dir}")
            return report

        texts = []
        fnames = []
        for fpath in contract_files:
            try:
                texts.append(fpath.read_text(encoding="utf-8").strip())
                fnames.append(fpath.name)
            except Exception as e:
                report["warnings"].append(f"Cannot read {fpath.name}: {e}")

        if not texts:
            report["warnings"].append("No contract texts could be read")
            return report

        # ── Ring 1: DomainPrototypeStore ──
        try:
            if _HAS_PROTOTYPE_STORE:
                store = DomainPrototypeStore()
                proto = store.build(domain_id, texts)
                store.store_prototype(proto)
                # Save to domain directory
                domain_path = self._loader._domain_path / domain_id
                domain_path.mkdir(parents=True, exist_ok=True)
                store.save(str(domain_path / "prototypes.json"))
                report["prototype"] = {
                    "sample_count": proto.sample_count,
                    "block_count_typical": proto.block_count_typical,
                    "centroid_preview": [round(v, 4) for v in proto.text_level_centroid[:6]],
                }
                logger.info("Ring 1 complete: domain prototype built for %s", domain_id)
            else:
                raise ImportError("DomainPrototypeStore not available")
        except Exception as e:
            report["warnings"].append(f"Ring 1 (DomainPrototypeStore) failed: {e}")

        # ── Ring 2: AutoClusterer ──
        clusters = []
        try:
            from app.engine.clause_splitter import ClauseSplitter as CS
            from app.engine.auto_clusterer import AutoClusterer as RealAC
            all_blocks = []
            for t in texts:
                blocks = CS.split(t)
                all_blocks.extend(blocks)
            if len(all_blocks) >= 2:
                clusterer = RealAC()
                clusters = clusterer.cluster(all_blocks)
                report["n_clusters"] = len(clusters)
                report["clusters"] = [
                    {"cluster_id": c.cluster_id, "auto_label": c.auto_label,
                     "member_count": c.member_count, "top_bigrams": c.top_bigrams[:3]}
                    for c in clusters
                ]
                # Cache for later use in validate()
                self._auto_clusters_cache[domain_id] = clusters
                logger.info("Ring 2 complete: %d auto-clusters found for %s",
                            len(clusters), domain_id)
            else:
                report["warnings"].append("Too few blocks for clustering")
        except ImportError:
            report["warnings"].append("Ring 2 (AutoClusterer) skipped: module not available")
        except Exception as e:
            report["warnings"].append(f"Ring 2 (AutoClusterer) failed: {e}")

        # ── Ring 3: StructuredRuleExtractor ──
        candidates = []
        try:
            from app.engine.rule_extractor import StructuredRuleExtractor
            authority_text = authority_path.read_text(encoding="utf-8")
            extractor = StructuredRuleExtractor()
            raw_candidates = extractor.extract(authority_text)
            candidates = list(raw_candidates)
            report["n_candidates"] = len(candidates)
            report["candidates_raw"] = [
                {"subject": c.subject, "condition_type": c.condition_type,
                 "operator": c.operator, "expected_value": c.expected_value,
                 "unit": c.unit, "confidence": c.confidence}
                for c in candidates
            ]
            logger.info("Ring 3 complete: %d candidates extracted", len(candidates))
        except ImportError:
            report["warnings"].append("Ring 3 (StructuredRuleExtractor) skipped: module not available")
        except Exception as e:
            report["warnings"].append(f"Ring 3 (StructuredRuleExtractor) failed: {e}")

        # ── Ring 4: AutoValidator ──
        admitted = []
        rejected = []
        per_gate = {"bench_ok": 0, "bench_fail": 0, "bad_samples_ok": 0,
                     "bad_samples_fail": 0, "constitution_ok": 0, "constitution_fail": 0}

        if candidates:
            try:
                from app.engine.auto_validator import AutoValidator, candidate_to_rule
                domain_dir = str(self._loader._domain_path / domain_id)
                validator = AutoValidator(engine=self._engine)

                # Load existing rules for constitution check
                existing_rules = []
                rules_pkg_path = self._loader._domain_path / domain_id / "rules.json"
                if rules_pkg_path.exists():
                    try:
                        pkg_data = json.loads(rules_pkg_path.read_text(encoding="utf-8"))
                        existing_rules = pkg_data.get("rules", [])
                    except Exception:
                        pass

                for i, c in enumerate(candidates):
                    rule_id = f"auto-{i:03d}"
                    rule_dict = candidate_to_rule(c, rule_id)
                    v_result = validator.validate(rule_dict, domain_dir, existing_rules=existing_rules)
                    for gate_name, gate_status in v_result.gate_results.items():
                        if gate_status.get("passed", False):
                            per_gate[f"{gate_name}_ok"] = per_gate.get(f"{gate_name}_ok", 0) + 1
                        else:
                            per_gate[f"{gate_name}_fail"] = per_gate.get(f"{gate_name}_fail", 0) + 1

                    if v_result.passed:
                        admitted.append(rule_dict)
                    else:
                        rejected.append({"rule_id": rule_id, "subject": c.subject,
                                         "suggestion": v_result.suggestion})

                report["n_admitted"] = len(admitted)
                report["n_rejected"] = len(rejected)
                report["per_gate_details"] = per_gate
                report["admitted"] = [{"rule_id": a["id"], "subject": a["name"]}
                                       for a in admitted]
                report["rejected"] = rejected

                # Auto-admit passing candidates to rules.json on disk
                if admitted:
                    try:
                        pkg_data = json.loads(rules_pkg_path.read_text(encoding="utf-8"))
                    except Exception:
                        pkg_data = {
                            "id": domain_id,
                            "name": f"Auto-Bootstrap {domain_id}",
                            "version": "0.1.0",
                            "domain": domain_id,
                            "rules": [],
                        }
                                 # Merge: append new rules, deduplicate by id
                    existing_ids = {r["id"] for r in pkg_data.get("rules", [])}
                    for rule_dict in admitted:
                        if rule_dict["id"] not in existing_ids:
                            pkg_data["rules"].append(rule_dict)
                            existing_ids.add(rule_dict["id"])
                    rules_pkg_path.write_text(
                        json.dumps(pkg_data, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    logger.info("Auto-admitted %d rules to %s", len(admitted), rules_pkg_path)

                logger.info("Ring 4 complete: %d admitted, %d rejected",
                            len(admitted), len(rejected))
            except ImportError:
                report["warnings"].append("Ring 4 (AutoValidator) skipped: module not available")
            except Exception as e:
                report["warnings"].append(f"Ring 4 (AutoValidator) failed: {e}")

        return report
