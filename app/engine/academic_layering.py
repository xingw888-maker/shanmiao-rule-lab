"""Academic source credibility layering — reuses KnowledgeLayeringEngine structure.

Adapts the 4-layer knowledge hierarchy (L0-L3) to academic paper sources.
Scoring dimensions: venue tier, citation count, study type, extraction method.

Constitution compliance:
  - §2 Domain separation: no academic constants in engine core
  - §4 LLM role: layering is deterministic, no LLM judgments
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ── Venue scoring ──
VENUE_SCORES = {
    "Nature": 1.0,
    "Science": 1.0,
    "Cell": 1.0,
    "The Lancet": 1.0,
    "NEJM": 1.0,
    "New England Journal of Medicine": 1.0,
    "Q1": 0.85,
    "1区": 0.85,
    "top conference": 0.80,
    "ICML": 0.80,
    "NeurIPS": 0.80,
    "ICLR": 0.80,
    "CVPR": 0.80,
    "ACL": 0.80,
    "Q2": 0.65,
    "2区": 0.65,
    "conference": 0.55,
    "CCF-A": 0.75,
    "CCF-B": 0.60,
    "CCF-C": 0.45,
    "preprint": 0.30,
    "arXiv": 0.30,
    "bioRxiv": 0.30,
    "SSRN": 0.25,
    "unpublished": 0.10,
    "working paper": 0.10,
    "blog": 0.05,
    "unknown": 0.30,
}

# ── Study type scoring ──
STUDY_TYPE_SCORES = {
    "meta_analysis": 1.0,
    "systematic_review": 0.95,
    "rct": 0.90,
    "randomized_controlled_trial": 0.90,
    "cohort": 0.70,
    "longitudinal": 0.70,
    "case_control": 0.55,
    "observational": 0.40,
    "cross_sectional": 0.35,
    "case_report": 0.15,
    "unknown": 0.30,
}

# ── Citation count brackets ──
def citation_score(count: int) -> float:
    if count >= 1000:
        return 1.0
    elif count >= 100:
        return 0.80
    elif count >= 10:
        return 0.50
    elif count > 0:
        return 0.30
    return 0.15

# ── Extraction method scoring ──
EXTRACTION_SCORES = {
    "manual": 1.0,
    "structured_parse": 0.85,
    "llm_extract": 0.50,
    "keyword_scan": 0.35,
}


@dataclass
class AcademicSourceProfile:
    """Credibility profile for a single academic proposition."""
    prop_id: str = ""
    paper_id: str = ""

    # Raw dimensions
    venue: str = "unknown"
    citation_count: int = 0
    study_type: str = "unknown"
    extraction_method: str = "llm_extract"

    # Scored dimensions (0.0-1.0)
    venue_score: float = 0.30
    citation_score: float = 0.15
    study_type_score: float = 0.30
    extraction_score: float = 0.50

    # Composite (weighted average)
    composite_credibility: float = 0.30

    # Layer assignment
    layer: str = "L2_SOURCE_UNCERTAIN"


class AcademicLayeringEngine:
    """Assign credibility scores and knowledge layers to academic propositions.

    Reuses the L0-L3 layer structure from KnowledgeLayeringEngine:
      L0_VALIDATED — manual extraction, high-venue, high-cite, validated
      L1_CONJECTURE — moderate credibility, needs cross-check
      L2_SOURCE_UNCERTAIN — low credibility source (preprint, low cites)
      L3_OUTER_POSSIBILITY — very low credibility, speculative
    """

    def __init__(self, ontology_path: str = ""):
        self._ontology_path = ontology_path
        # Load custom venue/study_type mappings from ontology if available
        if ontology_path:
            self._load_ontology()

    def _load_ontology(self):
        """Load custom scoring overrides from domain ontology."""
        onto_file = os.path.join(self._ontology_path, "ontology.json")
        if not os.path.isfile(onto_file):
            return
        try:
            with open(onto_file, "r") as f:
                onto = json.load(f)
            # Custom venue tiers override defaults
            venue_tiers = onto.get("venue_tiers", {})
            if venue_tiers:
                for tier, venues in venue_tiers.items():
                    score_map = {
                        "tier_1": 1.0, "tier_2": 0.80, "tier_3": 0.60,
                        "tier_4": 0.35, "unpublished": 0.10,
                    }
                    sc = score_map.get(tier, 0.30)
                    for v in venues:
                        VENUE_SCORES[v.lower()] = sc
        except (json.JSONDecodeError, OSError):
            pass

    def score_proposition(self, prop) -> AcademicSourceProfile:
        """Score a single academic proposition.

        Args:
            prop: An AcademicProposition or any object with venue,
                  citation_count, study_type, extraction_method attributes.
                  Or a dict with the same keys.

        Returns:
            AcademicSourceProfile with scores and layer assignment.
        """
        # Handle both object and dict input
        if isinstance(prop, dict):
            venue = str(prop.get("venue", "unknown"))
            citation_count = int(prop.get("citation_count", 0))
            study_type = str(prop.get("study_type", "unknown"))
            extraction_method = str(prop.get("extraction_method", "llm_extract"))
            prop_id = str(prop.get("prop_id", "unknown"))
            paper_id = str(prop.get("paper_id", "unknown"))
        else:
            venue = str(getattr(prop, "venue", "unknown"))
            citation_count = int(getattr(prop, "citation_count", 0))
            study_type = str(getattr(prop, "study_type", "unknown"))
            extraction_method = str(getattr(prop, "extraction_method", "llm_extract"))
            prop_id = str(getattr(prop, "prop_id", "unknown"))
            paper_id = str(getattr(prop, "paper_id", "unknown"))

        # Score each dimension
        v_score = _fuzzy_match_score(venue.lower(), VENUE_SCORES, default=0.30)
        c_score = citation_score(citation_count)
        s_score = _fuzzy_match_score(study_type.lower(), STUDY_TYPE_SCORES, default=0.30)
        e_score = EXTRACTION_SCORES.get(extraction_method, 0.50)

        # Composite: venue 40%, citations 25%, study_type 20%, extraction 15%
        composite = (
            v_score * 0.40 +
            c_score * 0.25 +
            s_score * 0.20 +
            e_score * 0.15
        )

        # Layer assignment
        if composite >= 0.80:
            layer = "L0_VALIDATED"
        elif composite >= 0.55:
            layer = "L1_CONJECTURE"
        elif composite >= 0.25:
            layer = "L2_SOURCE_UNCERTAIN"
        else:
            layer = "L3_OUTER_POSSIBILITY"

        return AcademicSourceProfile(
            prop_id=prop_id,
            paper_id=paper_id,
            venue=venue,
            citation_count=citation_count,
            study_type=study_type,
            extraction_method=extraction_method,
            venue_score=v_score,
            citation_score=c_score,
            study_type_score=s_score,
            extraction_score=e_score,
            composite_credibility=composite,
            layer=layer,
        )

    def score_all(self, propositions: list) -> list[AcademicSourceProfile]:
        """Score a list of propositions and return sorted profiles."""
        profiles = [self.score_proposition(p) for p in propositions]
        profiles.sort(key=lambda p: p.composite_credibility, reverse=True)
        return profiles

    def layer_summary(self, profiles: list[AcademicSourceProfile]) -> dict:
        """Produce a summary of layer distribution."""
        from collections import Counter
        layer_counts = Counter(p.layer for p in profiles)
        avg_cred = sum(p.composite_credibility for p in profiles) / len(profiles) if profiles else 0.0
        return {
            "total": len(profiles),
            "layers": dict(layer_counts),
            "average_credibility": round(avg_cred, 3),
            "L0_count": layer_counts.get("L0_VALIDATED", 0),
            "L1_count": layer_counts.get("L1_CONJECTURE", 0),
            "L2_count": layer_counts.get("L2_SOURCE_UNCERTAIN", 0),
            "L3_count": layer_counts.get("L3_OUTER_POSSIBILITY", 0),
        }


def _fuzzy_match_score(value: str, score_map: dict[str, float],
                       default: float = 0.30) -> float:
    """Match a value against a score map, supporting partial matches.

    E.g. "Q1期刊" should match "Q1" in the map.
    """
    # Direct match
    if value in score_map:
        return score_map[value]

    # Substring match — check if any key is contained in the value
    for key, score in sorted(score_map.items(), key=lambda x: -len(x[0])):
        if key in value or value in key:
            return score

    return default
