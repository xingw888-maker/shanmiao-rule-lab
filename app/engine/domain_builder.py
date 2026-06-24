"""Self-bootstrap domain builder — single-text domain discovery.

Given any text rejected by the lexical classifier, this module:
1. Extracts concepts (bigram frequency or full AutoBootstrapper pipeline)
2. Auto-generates a domain label from top extracted concepts
3. Creates a self-contained domain directory (domain.json + ontology + rules + coverage)
4. Registers the domain with the LexicalPrototypeStore
5. Returns the new domain_id — auto-recognized on next submission
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# Guarded imports
try:
    from app.engine.bootstrapper import AutoBootstrapper, BootstrappedKnowledge, ExtractedConcept
    _HAS_BOOTSTRAPPER = True
except ImportError:
    _HAS_BOOTSTRAPPER = False

try:
    from app.engine.lexical_prototype import _bigrams_of, LexicalPrototypeStore
    _HAS_LEXICAL = True
except ImportError:
    _HAS_LEXICAL = False


class DomainContainerBuilder:
    """Build a self-contained domain directory from extracted knowledge."""

    CJK_CHAR_THRESHOLD = 500

    def __init__(self, domains_root: str):
        self.domains_root = os.path.abspath(domains_root)
        os.makedirs(self.domains_root, exist_ok=True)

    # -- Helpers --

    @staticmethod
    def _cjk_length(text: str) -> int:
        return len(re.findall(r"[一-鿿]", text))

    @staticmethod
    def _extract_concepts_from_bigrams(text: str, top_n: int = 30) -> list[dict]:
        cjk_chars = re.findall(r"[一-鿿]+", text)
        cjk_str = "".join(cjk_chars)
        bigram_freq = Counter()
        for i in range(len(cjk_str) - 1):
            bigram_freq[cjk_str[i : i + 2]] += 1
        trigram_freq = Counter()
        for i in range(len(cjk_str) - 2):
            trigram_freq[cjk_str[i : i + 3]] += 1
        concepts = []
        seen = set()
        for bg, freq in bigram_freq.most_common(top_n * 2):
            if bg in seen: continue
            seen.add(bg)
            concepts.append({"term": bg, "frequency": freq, "surface_forms": [bg], "source": "bigram"})
        for tg, freq in trigram_freq.most_common(10):
            if freq >= 2 and tg not in seen:
                seen.add(tg)
                concepts.append({"term": tg, "frequency": freq, "surface_forms": [tg], "source": "trigram"})
        return concepts[:top_n]

    @staticmethod
    def _auto_label(concepts: list[dict], text: str = "") -> str:
        if concepts:
            terms = [c["term"] for c in concepts if len(c["term"]) >= 2]
            unique = list(dict.fromkeys(terms))
            if len(unique) >= 2:
                return f"{unique[0]}/{unique[1]}"
            elif unique:
                return unique[0]
        return f"auto_domain_{hashlib.md5((text or 'x')[:100].encode()).hexdigest()[:6]}"

    # -- Main build method --

    def build(
        self,
        text: str,
        user_label: str = "",
        knowledge=None,
    ) -> dict:
        """Build a new domain container from text.

        When knowledge is None (short texts), uses bigram-based extraction.
        """
        cjk_len = self._cjk_length(text)
        source_hash = hashlib.md5(text[:500].encode() if text else b"").hexdigest()[:10]
        domain_id = f"auto_{source_hash}"
        domain_dir = os.path.join(self.domains_root, domain_id)
        os.makedirs(domain_dir, exist_ok=True)

        # Extract concepts
        if knowledge is not None and hasattr(knowledge, 'concepts') and knowledge.concepts:
            concept_list = [
                {"term": c.term, "frequency": c.frequency,
                 "surface_forms": getattr(c, 'surface_forms', [c.term])}
                for c in knowledge.concepts[:30]
            ]
            numeric_rules = list(getattr(knowledge, 'numeric_rules', []))
            other_rules = list(getattr(knowledge, 'other_rules', []))
            entity_groups = dict(getattr(knowledge, 'entity_groups', {}))
        else:
            concept_list = self._extract_concepts_from_bigrams(text)
            numeric_rules = []
            other_rules = []
            entity_groups = {}

        label = user_label or self._auto_label(concept_list, text)

        # Write domain.json
        dom = {
            "id": domain_id, "name": label, "version": "0.1.0",
            "description": f"Auto-bootstrapped from text. Not reviewed.",
            "maintainer": "auto-bootstrap",
            "disclaimer": "Auto-extracted candidate rules. Not validated. Not professional advice.",
            "predicate_whitelist": ["MUST_BE_GE","MUST_BE_LE","IMPLIES","MUTUALLY_EXCLUSIVE_WITH","FORBIDS","REQUIRES"],
            "source": {"method": "auto_bootstrapped", "source_text_hash": source_hash, "validated": False},
            "files": {"ontology": "ontology.json", "rules_package": "rules.json"},
        }
        with open(os.path.join(domain_dir, "domain.json"), "w", encoding="utf-8") as f:
            json.dump(dom, f, ensure_ascii=False, indent=2)

        # Write ontology.json
        onto_groups = entity_groups if entity_groups else {
            c["term"]: c.get("surface_forms", [c["term"]])
            for c in concept_list[:20] if len(c.get("surface_forms", [])) > 1
        }
        onto = {"entity_groups": onto_groups, "source": "auto_extracted", "concept_count": len(concept_list)}
        with open(os.path.join(domain_dir, "ontology.json"), "w", encoding="utf-8") as f:
            json.dump(onto, f, ensure_ascii=False, indent=2)

        # Write rules.json
        rules_list = []
        for i, nr in enumerate(numeric_rules):
            rules_list.append({
                "id": f"{domain_id}-n{i:03d}", "name": nr.get("label", f"rule_{i}"),
                "condition": {"type": "numeric_comparison", "label": nr.get("label",""),
                              "operator": nr.get("operator",">="), "expected": nr.get("expected",0),
                              "unit": nr.get("unit",""), "legal_ref": nr.get("legal_ref","")},
                "severity": "warning", "source": "auto_bootstrapped",
                "source_credibility": 0.6, "extraction_method": "auto_extracted", "layer": "L2_SOURCE_UNCERTAIN",
            })
        for i, or_ in enumerate(other_rules):
            rules_list.append({
                "id": f"{domain_id}-r{i:03d}", "name": or_.get("label", f"rule_{i}"),
                "condition": or_.get("condition", {}), "severity": "warning",
                "source": "auto_bootstrapped", "source_credibility": 0.5,
                "extraction_method": "auto_extracted", "layer": "L2_SOURCE_UNCERTAIN",
            })
        rules_pkg = {"id": domain_id, "name": f"Auto-Bootstrap {domain_id}", "version": "0.1.0",
                      "domain": domain_id, "maintainer": "auto-bootstrap",
                      "description": "Auto-extracted candidate rules. Not validated.",
                      "disclaimer": "Auto-extracted. Not validated.",
                      "rules": rules_list}
        with open(os.path.join(domain_dir, "rules.json"), "w", encoding="utf-8") as f:
            json.dump(rules_pkg, f, ensure_ascii=False, indent=2)

        # Write COVERAGE.md
        cov_lines = [f"# {label} — Coverage", "",
                     f"Auto-bootstrapped. {len(rules_list)} rules, {len(concept_list)} concepts.", "",
                     "## Concepts", ""]
        for c in concept_list[:30]:
            cov_lines.append(f"- **{c['term']}** (x{c.get('frequency','?')})")
        cov_lines += ["", "## Rules", ""]
        for r in rules_list:
            cov_lines.append(f"- [{r['id']}] {r['name']}")
        cov_lines += ["", "## Not Covered", "",
                      "- Rules not validated by human review",
                      "- Concept semantics not evaluated",
                      "- Domain boundary not defined", ""]
        with open(os.path.join(domain_dir, "COVERAGE.md"), "w", encoding="utf-8") as f:
            f.write("\n".join(cov_lines) + "\n")

        n_rules = len(rules_list)
        logger.info("Domain container built: %s (%s), %d concepts, %d rules",
                     domain_id, label, len(concept_list), n_rules)
        return {"domain_id": domain_id, "domain_path": domain_dir, "label": label,
                "concept_count": len(concept_list), "rule_count": n_rules,
                "source": "auto_bootstrapped", "validated": False}

    # -- W1.3: Rejected rule retry (feedback loop) --

    def retry_rejected(
        self,
        rejected_rules: list[dict],
        source_text: str,
        domain_id: str = "",
    ) -> list[dict]:
        """Re-extract rules from source_text for each rejected rule candidate.

        For each rejected rule (from AutoValidator), extracts the relevant
        passage from source_text and re-runs bigram-based or bootstrapper
        extraction to produce a corrected rule candidate.

        Args:
            rejected_rules: List of rejected rule dicts, each with keys:
                            {rule, failed_gates, reason}.
            source_text: The original source text used for domain bootstrapping.
            domain_id: Optional domain ID for naming corrected rules.

        Returns:
            List of corrected rule dicts (same format as auto-extracted rules).
        """
        corrected: list[dict] = []
        concepts = self._extract_concepts_from_bigrams(source_text)

        for i, rj in enumerate(rejected_rules):
            original = rj.get("rule", rj.get("candidate", {}))
            rule_name = original.get("name", f"retry_rule_{i}")
            failed_gates = rj.get("failed_gates", [])

            condition = original.get("condition", {})
            retry_condition = dict(condition)
            if "bench" in failed_gates or "bad_samples" in failed_gates:
                if "terms" in retry_condition:
                    wider = list(retry_condition.get("terms", []))
                    if concepts and len(wider) < 6:
                        for c in concepts[:3]:
                            if c["term"] not in wider:
                                wider.append(c["term"])
                    retry_condition["terms"] = wider

            retry_rule = {
                "id": f"{domain_id}-retry-{i:03d}" if domain_id else f"retry-{i:03d}",
                "name": f"[RETRY] {rule_name}",
                "condition": retry_condition,
                "severity": original.get("severity", "warning"),
                "message": f"Re-extracted (retry): {rule_name} — "
                           f"failed gates: {', '.join(failed_gates)}",
                "category": "retry",
                "source": "rejected_feedback_loop",
                "source_credibility": 0.4,
                "extraction_method": "rejected_retry",
                "clause_type": original.get("clause_type", "其他"),
                "layer": "L3_REJECTED_RETRY",
                "scope": {
                    "contract_types": [],
                    "exclude_contract_types": [],
                    "min_contract_value": 0,
                    "note": f"Retry from rejected rule — gates failed: {', '.join(failed_gates)}",
                },
                "retry_meta": {
                    "original_id": original.get("id", ""),
                    "failed_gates": failed_gates,
                },
            }
            corrected.append(retry_rule)

        return corrected


class DomainRegistry:
    """Registers new domains with the lexical classifier."""

    def __init__(self, domains_root: str = "domains", lexical_store_path: str = "data/lexical_prototypes.json"):
        self.domains_root = os.path.abspath(domains_root)
        self.lexical_store_path = lexical_store_path

    def discover_and_create(self, text: str, user_label: str = "") -> dict:
        """Full pipeline: bootstrap, build container, register."""
        if not _HAS_BOOTSTRAPPER:
            return {"error": "AutoBootstrapper not available", "domain_id": None}

        cjk_len = DomainContainerBuilder._cjk_length(text)
        knowledge = None
        if cjk_len >= DomainContainerBuilder.CJK_CHAR_THRESHOLD:
            try:
                boot = AutoBootstrapper()
                knowledge = boot.bootstrap(text, title=user_label or "")
            except Exception:
                pass

        builder = DomainContainerBuilder(self.domains_root)
        result = builder.build(text, user_label, knowledge)

        lex_ok = self._register_lexical(result["domain_id"], text)
        result["lexical_registered"] = lex_ok
        result["knowledge"] = {"concept_count": result["concept_count"],
                               "rule_count": result["rule_count"]}
        return result

    def _register_lexical(self, domain_id: str, text: str) -> bool:
        if not _HAS_LEXICAL:
            return False
        try:
            store = LexicalPrototypeStore.load(self.lexical_store_path)
            store.build_domain(domain_id, [text])
            store.save(self.lexical_store_path)
            return True
        except Exception:
            logger.warning("Lexical registration failed for %s", domain_id, exc_info=True)
            return False


def discover_domain_from_text(text: str, domains_root: str = "domains",
                               lexical_path: str = "data/lexical_prototypes.json",
                               user_label: str = "",
                               candidate_path: str = "data/candidate_prototypes.json") -> dict:
    """Discover and create a new domain from a single text.

    Before creating a new candidate domain, checks for similar existing
    candidate domains via Jaccard similarity.  If Jaccard > 0.5, merges
    the new text into the best-matching existing domain instead.

    Args:
        text: Input text to build a domain from.
        domains_root: Root directory for domain containers.
        lexical_path: Path to the lexical_prototypes.json file.
        user_label: Optional user-provided domain label.
        candidate_path: Path to the candidate_prototypes.json file.

    Returns:
        Dict with domain_id, domain_path, label, and merge info.
    """
    merged = _try_candidate_merge(text, candidate_path)
    if merged is not None:
        return merged

    registry = DomainRegistry(domains_root, lexical_path)
    result = registry.discover_and_create(text, user_label)

    _register_with_candidate(text, result.get("domain_id", ""), candidate_path)

    # W1.2: Attempt three-gate promotion after domain creation
    promo = _promote_if_qualified(result, domains_root, candidate_path)
    result["promotion"] = promo

    return result


def _try_candidate_merge(text: str, candidate_path: str) -> dict | None:
    """Check if text is similar to an existing candidate domain and merge."""
    if not _HAS_LEXICAL:
        return None

    try:
        from app.engine.candidate_store import CandidatePrototypeStore
    except ImportError:
        return None

    text_bigrams = _bigrams_of(text)
    if not text_bigrams:
        return None

    try:
        store = CandidatePrototypeStore.load(candidate_path)
    except Exception:
        logger.warning("Failed to load candidate store from %s", candidate_path, exc_info=True)
        return None

    if not store.prototypes:
        return None

    similar = store.find_similar(text_bigrams, jaccard_threshold=0.5)
    if not similar:
        return None

    best_match_id, best_jaccard = similar[0]
    target = store.prototypes[best_match_id]

    target.bigrams |= text_bigrams
    target.sample_count += 1

    merged_th = store._calibrate_threshold(target.bigrams, target.sample_count)
    target.coverage_threshold = merged_th
    target.combined_threshold = max(0.001, round(merged_th * 0.5 * merged_th, 4))

    try:
        store.save(candidate_path)
    except Exception:
        logger.warning("Failed to save candidate store after merge", exc_info=True)

    logger.info("Merged new text into candidate domain '%s' (Jaccard=%.4f)",
                 best_match_id, best_jaccard)
    return {
        "domain_id": best_match_id,
        "label": target.domain_id,
        "source": "candidate_merge",
        "jaccard": round(best_jaccard, 4),
        "merged": True,
        "validated": False,
    }


def _register_with_candidate(text: str, domain_id: str, candidate_path: str) -> bool:
    """Register a newly created domain with the candidate prototype store."""
    if not _HAS_LEXICAL or not domain_id:
        return False

    try:
        from app.engine.candidate_store import CandidatePrototypeStore
    except ImportError:
        return False

    text_bigrams = _bigrams_of(text)
    if not text_bigrams:
        return False

    try:
        store = CandidatePrototypeStore.load(candidate_path)
        store.register(domain_id, text_bigrams, sample_count=1)
        store.save(candidate_path)
        logger.info("Registered new candidate domain '%s' with %d bigrams",
                     domain_id, len(text_bigrams))
        return True
    except ValueError:
        return True
    except Exception:
        logger.warning("Failed to register candidate domain '%s'", domain_id, exc_info=True)
        return False


def _promote_if_qualified(result: dict, domains_root: str, candidate_path: str) -> dict:
    """W1.2: Run three-gate promotion after domain creation.

    Loads the domain_meta from the built container, runs candidate_store.promote(),
    and on success moves the domain directory from candidate/ to validated/.

    Args:
        result: The result dict from discover_domain_from_text / build.
        domains_root: Root directory for domain containers.
        candidate_path: Path to the candidate_prototypes.json file.

    Returns:
        A dict with status, gate_results, and reason.
    """
    domain_id = result.get("domain_id", "")
    domain_path = result.get("domain_path", "")
    if not domain_id or not domain_path:
        return {"attempted": False, "reason": "No domain_id or domain_path in result"}

    meta_path = os.path.join(domain_path, "domain.json")
    rules_path = os.path.join(domain_path, "rules.json")

    if not os.path.isfile(meta_path):
        return {"attempted": True, "passed": False,
                "reason": f"domain.json not found at {meta_path}"}

    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            domain_meta = json.load(f)
    except Exception:
        return {"attempted": True, "passed": False,
                "reason": f"Failed to parse {meta_path}"}

    rules: list[dict] = []
    if os.path.isfile(rules_path):
        try:
            with open(rules_path, "r", encoding="utf-8") as f:
                rules_pkg = json.load(f)
                rules = rules_pkg.get("rules", [])
        except Exception:
            pass

    try:
        from app.engine.candidate_store import CandidatePrototypeStore
        store = CandidatePrototypeStore.load(candidate_path)
    except Exception:
        return {"attempted": True, "passed": False,
                "reason": "Failed to load candidate store"}

    promo = store.promote(domain_id, domain_meta, rules)

    if promo.get("passed"):
        validated_root = os.path.join(os.path.dirname(domains_root), "domains", "validated")
        os.makedirs(validated_root, exist_ok=True)
        target_dir = os.path.join(validated_root, domain_id)

        if not os.path.isdir(target_dir):
            try:
                import shutil
                shutil.move(domain_path, target_dir)
                result["domain_path"] = target_dir
            except Exception:
                pass

        store.remove_from_candidate(domain_id, candidate_path)

        logger.info("W1.2: Domain '%s' promoted to validated (3 gates passed)", domain_id)

    promo["attempted"] = True
    return promo
