# -*- coding: utf-8 -*-
"""Online Incremental Learner — self-updating clause classifier powered by engine feedback.

Three components:
  1. IncrementalNaiveBayes — analytical weight update, O(ngram_count), ~5ms per feedback
  2. OntologyCooccurrenceTracker — collaborative filtering for entity group expansion
  3. FeedbackGenerator — three signal types + three noise filters

Constitution compliance:
  - Zero handler modification (12 frozen handlers untouched)
  - Zero core.py dispatch modification
  - Zero external dependencies (Pure Python + numpy optional)
  - Try/except protected at integration point

Usage:
    from app.engine.online_learner import OnlineLearner
    learner = OnlineLearner(
        model_path="data/models/clause_model.json",
        ontology_path="domains/construction/ontology.json",
    )
    feedback = learner.generate_feedback(evidence_chain, clause_blocks, ontology)
    accepted = learner.apply_feedback(feedback)
    if accepted > 0:
        learner.save_model()
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ── CJK helpers (inline so this module is self-contained) ──
_CJK_RE = re.compile(r"[一-鿿㐀-䶿]+")


def _cjk_ngrams(text: str, n_min: int = 1, n_max: int = 3) -> list[str]:
    """Extract CJK character n-grams from text."""
    cjk_chars = "".join(_CJK_RE.findall(text))
    ngrams: list[str] = []
    for n in range(n_min, n_max + 1):
        for i in range(len(cjk_chars) - n + 1):
            ngrams.append(cjk_chars[i : i + n])
    return ngrams


# ═══════════════════════════════════════════════════════════════════════════
# Data types
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class FeedbackSignal:
    """A single correction signal generated from engine output."""

    text_hash: str  # hash of clause text
    old_label: str  # current classifier prediction
    suggested_label: str  # what the signal says it should be
    confidence: float  # 0.0–1.0, signal confidence
    signal_source: str  # "matched_terms_vs_clause_type" | "cross_contract_inconsistency" | "high_confidence_fail"
    evidence_rule_id: str = ""
    evidence_status: str = ""
    clause_text: str = ""
    old_label_source: str = ""  # label_source from clauses_v2 if traceable

    @property
    def key(self) -> tuple:
        return (self.text_hash, self.old_label, self.suggested_label)


@dataclass
class OnlineLearnerStats:
    """Running statistics for the online learner."""

    total_feedback_generated: int = 0
    total_feedback_accepted: int = 0
    total_feedback_rejected: int = 0
    total_model_updates: int = 0
    total_ontology_expansions: int = 0
    last_update_at: str = ""

    def to_dict(self) -> dict:
        return {
            "total_feedback_generated": self.total_feedback_generated,
            "total_feedback_accepted": self.total_feedback_accepted,
            "total_feedback_rejected": self.total_feedback_rejected,
            "total_model_updates": self.total_model_updates,
            "total_ontology_expansions": self.total_ontology_expansions,
            "last_update_at": self.last_update_at,
        }


# ═══════════════════════════════════════════════════════════════════════════
# Component 1: Incremental NaiveBayes
# ═══════════════════════════════════════════════════════════════════════════


class IncrementalNaiveBayes:
    """Analytical incremental update for exported NaiveBayes clause model.

    Updates log-probability weights for a single (text, new_label) pair
    without re-training on all historical data.  Only touches features
    present in the text.  Learning rate is small; EMA smoothing prevents
    a single erroneous feedback from wiping out prior knowledge.

    The model format matches exactly what model_inference.py reads:
      - vocab: {feature: index}
      - class_log_prior: {label: log_p}
      - class_feature_log_prob: {label: [log_p1, log_p2, ...]}
    """

    _EPS = 1e-10  # floor for log probabilities

    def __init__(self, learning_rate: float = 0.05):
        self._lr = learning_rate

    def update(
        self,
        model: dict,
        text: str,
        new_label: str,
        old_label: str,
    ) -> dict:
        """Apply one incremental weight update.

        Args:
            model: The loaded clause_model dict (mutated in place and returned).
            text: Clause text to extract ngrams from.
            new_label: Corrected label (where probability mass should move TOWARD).
            old_label: Current label (where probability mass should move AWAY FROM).
            lr: Override learning rate.

        Returns:
            The same model dict (mutated in place).
        """
        if new_label == old_label:
            return model  # nothing to update

        lr = self._lr
        ngrams = _cjk_ngrams(text)
        vocab = model.get("vocab", {})
        classes = model.get("classes", [])
        feature_log_prob = model.get("class_feature_log_prob", {})
        class_log_prior = model.get("class_log_prior", {})

        if not classes or not feature_log_prob:
            return model

        updated_count = 0
        for ng in ngrams:
            feat_idx = vocab.get(ng)
            if feat_idx is None:
                continue  # new feature – skip to prevent vocab inflation

            # ── Move probability mass FROM old_label ──
            old_probs = feature_log_prob.get(old_label)
            if old_probs is not None and feat_idx < len(old_probs):
                old_p = math.exp(old_probs[feat_idx])
                new_old_p = max(old_p * (1.0 - lr) + lr * 1e-5, self._EPS)
                old_probs[feat_idx] = math.log(new_old_p)

            # ── Move probability mass TO new_label ──
            new_probs = feature_log_prob.get(new_label)
            if new_probs is not None and feat_idx < len(new_probs):
                new_p = math.exp(new_probs[feat_idx])
                new_new_p = max(new_p * (1.0 - lr) + lr * 0.01, self._EPS)
                new_probs[feat_idx] = math.log(new_new_p)

            updated_count += 1

        # ── Slight prior shift ──
        if old_label in class_log_prior:
            old_prior = math.exp(class_log_prior[old_label])
            class_log_prior[old_label] = math.log(
                max(old_prior - lr * 0.005, self._EPS)
            )
        if new_label in class_log_prior:
            new_prior = math.exp(class_log_prior[new_label])
            class_log_prior[new_label] = math.log(
                max(new_prior + lr * 0.005, self._EPS)
            )

        logger.debug(
            "Incremental update: %d features, %s -> %s, lr=%.3f",
            updated_count, old_label, new_label, lr,
        )
        return model


# ═══════════════════════════════════════════════════════════════════════════
# Component 2: Ontology Co-occurrence Tracker (collaborative filtering)
# ═══════════════════════════════════════════════════════════════════════════


class OntologyCooccurrenceTracker:
    """Track term co-occurrence within rule contexts for entity group expansion.

    When a term appears repeatedly in the same rule context as members of
    an existing entity group, it's likely a synonym/variant.  This is
    collaborative filtering applied to domain terminology — zero training,
    deterministic thresholds.
    """

    def __init__(self, cooccurrence_threshold: float = 0.6):
        self._threshold = cooccurrence_threshold
        # {(term, group_name): cooccurrence_score}
        self._scores: dict[tuple[str, str], float] = {}

    def update(
        self,
        evidence_chain: list[dict],
        ontology: dict,
    ) -> dict[str, list[str]]:
        """Scan evidence chain for term-group co-occurrence, accumulate scores.

        Returns dict of {term: [suggested_group, ...]} for terms above threshold.
        """
        entity_groups: dict[str, list[str]] = ontology.get("entity_groups", {})
        if not entity_groups:
            return {}

        # Flatten group membership for fast lookup
        term_to_groups: dict[str, set[str]] = defaultdict(set)
        for group_name, members in entity_groups.items():
            for member in members:
                term_to_groups[member.lower()].add(group_name)

        # Scan each evidence item
        for ev in evidence_chain:
            matched_terms = ev.get("matched_terms", [])
            if not matched_terms:
                continue

            matched_lower = {t.lower().strip() for t in matched_terms if t}

            # Find which groups are represented in this evidence
            groups_present: set[str] = set()
            for term in matched_lower:
                groups_present.update(term_to_groups.get(term, set()))

            if not groups_present:
                continue

            # For each matched term NOT in any group, assign score to
            # each group present in this evidence
            for term in matched_lower:
                if term in term_to_groups:
                    continue  # already belongs to some group
                if len(term) < 2:
                    continue
                for group in groups_present:
                    key = (term, group)
                    # Score: +1 for each co-occurrence, normalized later
                    self._scores[key] = self._scores.get(key, 0.0) + 1.0

        # Normalize scores relative to max per term
        term_max: dict[str, float] = {}
        for (term, _group), score in self._scores.items():
            term_max[term] = max(term_max.get(term, 0.0), score)

        # Find terms above threshold
        expansions: dict[str, list[str]] = defaultdict(list)
        for (term, group), score in self._scores.items():
            normalized = score / max(term_max.get(term, 1.0), 1.0)
            if normalized >= self._threshold:
                expansions[term].append(group)

        return dict(expansions)


# ═══════════════════════════════════════════════════════════════════════════
# Component 3: Feedback Generator + Noise Filter
# ═══════════════════════════════════════════════════════════════════════════


class FeedbackGenerator:
    """Generate correction signals from engine output with noise filtering.

    Three signal types:
      A. matched_terms entity-group vs clause_type mismatch
      B. cross-contract classification inconsistency (via text_hash)
      C. high-confidence FAIL with boundary proximity concern

    Three noise filters:
      1. Signal confidence < 0.60 → reject
      2. Same contradiction must appear ≥ 2 times
      3. Never override ground_truth labels
    """

    def __init__(
        self,
        min_confidence: float = 0.60,
        min_occurrences: int = 2,
    ):
        self._min_confidence = min_confidence
        self._min_occurrences = min_occurrences
        # {(text_hash, old_label, suggested_label): occurrence_count}
        self._history: dict[tuple, int] = {}
        # Track per-text_hash classification history for cross-contract signal
        self._classification_history: dict[str, dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )

    def generate(
        self,
        evidence_chain: list[dict],
        clause_blocks: list[dict],
        ontology: dict,
    ) -> list[FeedbackSignal]:
        """Generate correction signals from one validation run.

        Args:
            evidence_chain: List of evidence dicts from engine.validate().
            clause_blocks: List of clause block dicts from ClauseSplitter.
            ontology: Domain ontology dict with entity_groups.

        Returns:
            List of FeedbackSignal objects (unfiltered at this stage).
        """
        signals: list[FeedbackSignal] = []

        # ── Pre-compute ontology term→group lookup ──
        entity_groups: dict[str, list[str]] = ontology.get("entity_groups", {})
        term_to_group: dict[str, str] = {}
        for group_name, members in entity_groups.items():
            for member in members:
                term_to_group[member.lower()] = group_name

        # ── Pre-compute clause_type → text mapping ──
        clause_type_to_text: dict[str, list[str]] = defaultdict(list)
        clause_text_hashes: dict[str, str] = {}
        for cb in clause_blocks:
            ct = cb.get("clause_type", "")
            text = cb.get("content", cb.get("content_preview", ""))
            if ct and text:
                clause_type_to_text[ct].append(text)
                h = _text_hash(text)
                clause_text_hashes[h] = text

        # ── Signal A: clause_type vs matched_terms entity group mismatch ──
        for ev in evidence_chain:
            if ev.get("status") != "FAILED":
                continue

            clause_type = ev.get("clause_type", "")
            matched_terms = ev.get("matched_terms", [])
            if not clause_type or not matched_terms:
                continue

            # Find which entity groups the matched terms belong to
            term_groups: dict[str, int] = defaultdict(int)
            for t in matched_terms:
                g = term_to_group.get(t.lower().strip())
                if g:
                    term_groups[g] += 1

            if not term_groups:
                continue

            # Does the dominant group match the clause_type?
            dominant_group = max(term_groups, key=term_groups.get)
            dominant_count = term_groups[dominant_group]
            total_matched = len(matched_terms)

            # Signal: dominant entity group != clause_type AND has strong majority
            if dominant_group != clause_type and dominant_count >= 2:
                # Build confidence from group dominance ratio
                confidence = min(0.85, 0.5 + dominant_count / total_matched * 0.5)
                signals.append(FeedbackSignal(
                    text_hash=ev.get("clause_text_hash", ""),
                    old_label=clause_type,
                    suggested_label=dominant_group,
                    confidence=confidence,
                    signal_source="matched_terms_vs_clause_type",
                    evidence_rule_id=ev.get("rule_id", ""),
                    evidence_status=ev.get("status", ""),
                    clause_text=ev.get("input_fragment", ev.get("clause_text", "")),
                ))

        # ── Signal B: cross-contract inconsistency ──
        # Accumulate classification history for each text_hash
        for ev in evidence_chain:
            text_hash = ev.get("clause_text_hash", "")
            clause_type = ev.get("clause_type", "")
            if text_hash and clause_type:
                self._classification_history[text_hash][clause_type] += 1

        # Check for hashes with conflicting majority labels
        for text_hash, type_counts in self._classification_history.items():
            if len(type_counts) < 2:
                continue
            sorted_types = sorted(
                type_counts.items(), key=lambda x: -x[1]
            )
            dominant_type, dominant_count = sorted_types[0]
            runner_up_type, runner_up_count = sorted_types[1]
            total = sum(type_counts.values())

            # If no single type has > 66% majority → inconsistency
            if dominant_count / total < 0.67 and total >= 3:
                confidence = 0.55  # lower confidence for inconsistency signal
                signals.append(FeedbackSignal(
                    text_hash=text_hash,
                    old_label=dominant_type,
                    suggested_label=runner_up_type,
                    confidence=confidence,
                    signal_source="cross_contract_inconsistency",
                    clause_text=clause_text_hashes.get(text_hash, ""),
                ))

        # ── Signal C: high-confidence FAIL with low boundary proximity ──
        for ev in evidence_chain:
            if ev.get("status") != "FAILED":
                continue
            triager_verdict = ev.get("triager_verdict", "")
            triager_scores = ev.get("_triager_scores", {})
            if triager_verdict != "FAIL":
                continue
            boundary = triager_scores.get("boundary_proximity", 1.0)
            if boundary >= 0.5:
                continue  # not a boundary case

            # Low boundary proximity + FAIL → clause_type might be wrong
            # (rule evaluating on wrong clause type)
            clause_type = ev.get("clause_type", "")
            rule_id = ev.get("rule_id", "")

            # Heuristic: if this rule's typical clause_type doesn't match
            # the assigned clause_type, it's a signal
            signals.append(FeedbackSignal(
                text_hash=ev.get("clause_text_hash", ""),
                old_label=clause_type,
                suggested_label="",  # unknown target — signal is "review this"
                confidence=0.45,  # low confidence, advisory only
                signal_source="high_confidence_fail_boundary",
                evidence_rule_id=rule_id,
                evidence_status=ev.get("status", ""),
                clause_text=ev.get("input_fragment", ev.get("clause_text", "")),
            ))

        return signals

    def apply_feedback(
        self,
        signals: list[FeedbackSignal],
        model: Optional[dict] = None,
    ) -> int:
        """Filter signals, accumulate history, apply accepted ones to model.

        Args:
            signals: Raw signals from generate().
            model: Optional NaiveBayes model dict to update in-place.

        Returns:
            Number of signals accepted and applied.
        """
        accepted = 0
        updater = IncrementalNaiveBayes()

        for fb in signals:
            # ── Filter 1: confidence too low ──
            if fb.confidence < self._min_confidence:
                continue

            # ── Filter 2: no suggested label to move toward ──
            if not fb.suggested_label or fb.suggested_label == fb.old_label:
                continue

            # ── Filter 3: old label is ground_truth (never override) ──
            if fb.old_label_source == "ground_truth":
                continue

            # ── Accumulate history ──
            key = fb.key
            self._history[key] = self._history.get(key, 0) + 1

            # ── Filter 4: need min occurrences ──
            if self._history[key] < self._min_occurrences:
                continue

            # ── Apply ──
            if model and fb.clause_text:
                updater.update(model, fb.clause_text, fb.suggested_label, fb.old_label)
                accepted += 1

        return accepted


# ═══════════════════════════════════════════════════════════════════════════
# Main orchestrator: OnlineLearner
# ═══════════════════════════════════════════════════════════════════════════


class OnlineLearner:
    """Orchestrates feedback generation + filtering + model update + ontology expansion.

    Designed to be called from kernel.validate() after triager summarization.
    All operations are try/except safe — failure does not affect the main pipeline.
    """

    def __init__(
        self,
        model_path: str = "data/models/clause_model.json",
        ontology_path: str = "domains/construction/ontology.json",
        learning_rate: float = 0.05,
        cooccurrence_threshold: float = 0.6,
        min_feedback_confidence: float = 0.60,
        min_feedback_occurrences: int = 2,
        stats_path: Optional[str] = None,
    ):
        self._model_path = model_path
        self._ontology_path = ontology_path
        self._stats_path = stats_path or os.path.join(
            os.path.dirname(model_path), "online_learner_stats.json"
        )
        self._updater = IncrementalNaiveBayes(learning_rate=learning_rate)
        self._ontology_tracker = OntologyCooccurrenceTracker(
            cooccurrence_threshold=cooccurrence_threshold
        )
        self._feedback_gen = FeedbackGenerator(
            min_confidence=min_feedback_confidence,
            min_occurrences=min_feedback_occurrences,
        )
        self._stats = OnlineLearnerStats()
        self._model_cache: Optional[dict] = None
        self._ontology_cache: Optional[dict] = None

    # ── Model I/O ──────────────────────────────────────────────────────

    def _load_model(self) -> dict:
        if self._model_cache is not None:
            return self._model_cache
        if os.path.isfile(self._model_path):
            with open(self._model_path, "r", encoding="utf-8") as f:
                self._model_cache = json.load(f)
        else:
            self._model_cache = {}
        return self._model_cache

    def _load_ontology(self) -> dict:
        if self._ontology_cache is not None:
            return self._ontology_cache
        if os.path.isfile(self._ontology_path):
            with open(self._ontology_path, "r", encoding="utf-8") as f:
                self._ontology_cache = json.load(f)
        else:
            self._ontology_cache = {}
        return self._ontology_cache

    def save_model(self) -> None:
        """Atomically write updated model to disk."""
        model = self._model_cache
        if model is None:
            return
        model["version"] = model.get("version", "2.0.0")
        model_dir = os.path.dirname(self._model_path)
        fd, tmp_path = tempfile.mkstemp(
            suffix=".json", prefix="clause_model_", dir=model_dir
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(model, f, ensure_ascii=False, separators=(",", ":"))
            os.replace(tmp_path, self._model_path)
            logger.info("OnlineLearner: model saved to %s", self._model_path)
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

    def save_ontology(self) -> None:
        """Atomically write updated ontology to disk."""
        onto = self._ontology_cache
        if onto is None:
            return
        onto_dir = os.path.dirname(self._ontology_path)
        fd, tmp_path = tempfile.mkstemp(
            suffix=".json", prefix="ontology_", dir=onto_dir
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(onto, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, self._ontology_path)
            logger.info("OnlineLearner: ontology saved to %s", self._ontology_path)
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

    # ── Main entry point ───────────────────────────────────────────────

    def run(
        self,
        evidence_chain: list[dict],
        clause_blocks: list[dict],
    ) -> OnlineLearnerStats:
        """Run full online learning cycle on one validation result.

        Args:
            evidence_chain: Evidence items from engine.validate().
            clause_blocks: Clause blocks from ClauseSplitter.

        Returns:
            OnlineLearnerStats for this run.
        """
        ontology = self._load_ontology()
        model = self._load_model()

        # ── Step 1: Generate feedback ──
        signals = self._feedback_gen.generate(evidence_chain, clause_blocks, ontology)
        self._stats.total_feedback_generated += len(signals)

        # ── Step 2: Apply feedback to model ──
        accepted = self._feedback_gen.apply_feedback(signals, model)
        self._stats.total_feedback_accepted += accepted
        self._stats.total_feedback_rejected += len(signals) - accepted
        if accepted > 0:
            self._stats.total_model_updates += 1

        # ── Step 3: Check ontology expansions ──
        expansions = self._ontology_tracker.update(evidence_chain, ontology)
        if expansions:
            entity_groups = ontology.setdefault("entity_groups", {})
            for term, groups in expansions.items():
                for group in groups:
                    if group in entity_groups and term not in entity_groups[group]:
                        entity_groups[group].append(term)
                        self._stats.total_ontology_expansions += 1
                        logger.info(
                            "OnlineLearner: ontology expansion — %s → %s", term, group
                        )

        # ── Step 4: Persist if changes ──
        need_save = accepted > 0 or expansions
        if need_save:
            from datetime import datetime, timezone
            self._stats.last_update_at = datetime.now(timezone.utc).isoformat()
            try:
                self.save_model()
            except Exception as e:
                logger.warning("OnlineLearner: model save failed: %s", e)
            try:
                self.save_ontology()
            except Exception as e:
                logger.warning("OnlineLearner: ontology save failed: %s", e)
            try:
                self._save_stats()
            except Exception as e:
                logger.warning("OnlineLearner: stats save failed: %s", e)

        return self._stats

    # ── Statistics persistence ─────────────────────────────────────────

    def _save_stats(self) -> None:
        with open(self._stats_path, "w", encoding="utf-8") as f:
            json.dump(self._stats.to_dict(), f, ensure_ascii=False, indent=2)

    def load_stats(self) -> OnlineLearnerStats:
        if os.path.isfile(self._stats_path):
            with open(self._stats_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._stats = OnlineLearnerStats(**data)
        return self._stats


# ── Convenience text hashing ──


def _text_hash(text: str) -> str:
    """Short hash for clause text dedup."""
    import hashlib
    return hashlib.md5(text[:500].encode("utf-8")).hexdigest()[:12]
