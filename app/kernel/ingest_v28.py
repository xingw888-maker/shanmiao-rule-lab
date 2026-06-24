"""
R2.8 — kernel.ingest() 统一摄取入口（五步法）

Independent module for syntax verification before insertion into kernel.py.
"""
from __future__ import annotations
import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

_log = logging.getLogger(__name__)

MAX_INGEST_DEPTH = 3


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


def ingest(self, text: str, **opts) -> IngestResult:
    """Unified 5-step ingest pipeline.

    STEP 0: Recursion gate (_ingest_depth, MAX=3)
    STEP 1: Domain classification (ClassifierPipeline + CandidateStore fallback)
    STEP 2: Rule extraction (StructuredRuleExtractor + LLM fallback + book recursion)
    STEP 3: Gate (AutoValidator 3-gate)
    STEP 4: Promote (graded: new_domain→candidate/, auto_promote→validated/, default→candidate/)
    STEP 5: Validate (optional, on by default)
    """
    source_type = opts.get("source_type", "unknown")
    domain_hint = opts.get("domain_hint")
    enable_llm = opts.get("enable_llm", False)
    auto_promote = opts.get("auto_promote", False)
    validate_after = opts.get("validate_after", True)

    # ── STEP 0: Recursion gate ──
    self._ingest_depth += 1
    if self._ingest_depth > MAX_INGEST_DEPTH:
        self._ingest_depth -= 1
        raise RecursionSafetyError(
            max_depth=MAX_INGEST_DEPTH,
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
                candidate_store=None,
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

        domain_dir = self._loader._resolve_domain_dir(domain_id)
        if domain_dir is None:
            domain_dir = self._loader._domain_path / domain_id
            domain_dir.mkdir(parents=True, exist_ok=True)

        dj = domain_dir / "domain.json"
        if not dj.exists():
            with open(dj, "w", encoding="utf-8") as f:
                json.dump({
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
            return IngestResult(
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
                    pkg = json.loads(rp.read_text(encoding="utf-8"))
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

        return IngestResult(
            domain_id=domain_id, is_new_domain=is_new_domain,
            rules_count=len(rules), passed_rules=len(passed_rules),
            rejected_rules=len(rejected_rules), gate_results=gate_results,
            validation_result=validation_result,
        )
    finally:
        self._ingest_depth -= 1
