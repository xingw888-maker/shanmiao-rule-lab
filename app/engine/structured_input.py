"""Model-to-engine structured input injector -- Road 2 sidecar pattern.

Stores structured_inputs on matcher._structured_inputs keyed by rule.id,
NOT on rule.condition_params.  This survives rule object copies/recompiles.

R2.10: null-value injection -- when model returns value=null with confidence >= 0.3,
       inject a null-value marker so handler can return NOT_APPLICABLE instead of
       falling back to regex (which would produce false positives).

R2.11: ontology normalization -- field names from model extraction are normalized
       against global_ontology.json before lookup, so "保修期限" -> "warranty_document"
       canonical -> standard field name for handler consumption.

Interface contract: ARCHITECTURE.md §模型-引擎接口契约
"""

import json
import os
from pathlib import Path

NULL_CONFIDENCE_THRESHOLD = 0.3

# -- Ontology cache (lazy-loaded singleton) --
_ontology_cache = None
_ontology_surface_map = None


def _load_ontology():
    """Load global_ontology.json and build surface->canonical lookup.

    Returns (raw_ontology, surface_map).
    """
    global _ontology_cache, _ontology_surface_map
    if _ontology_cache is not None and _ontology_surface_map is not None:
        return _ontology_cache, _ontology_surface_map

    # Resolve path relative to this file
    onto_path = Path(__file__).resolve().parent.parent.parent / "domains" / "global_ontology.json"
    if not onto_path.exists():
        _ontology_cache = {}
        _ontology_surface_map = {}
        return _ontology_cache, _ontology_surface_map

    with open(onto_path, "r", encoding="utf-8") as f:
        _ontology_cache = json.load(f)

    _ontology_surface_map = {}
    for concept_id, data in _ontology_cache.items():
        if concept_id.startswith("_"):
            continue
        canonical = data.get("canonical", concept_id)
        for surface in data.get("surfaces", []):
            _ontology_surface_map[surface] = canonical
        # Map canonical to itself
        _ontology_surface_map[canonical] = canonical

    return _ontology_cache, _ontology_surface_map


def _normalize_field_name(raw_field, surface_map):
    """Normalize a field name through ontology lookup.

    If raw_field matches a surface form in the ontology, return its canonical name.
    Otherwise return raw_field unchanged (backward compatible).
    """
    # Direct surface match
    if raw_field in surface_map:
        return surface_map[raw_field]

    # Try partial/fuzzy: if any surface term appears inside raw_field
    # Match longest surface first to avoid greedy short matches
    sorted_surfaces = sorted(surface_map.keys(), key=len, reverse=True)
    for surface in sorted_surfaces:
        if surface in raw_field:
            return surface_map[surface]

    return raw_field


def inject_structured_fields(extractions, rule_lookup):
    """Convert model extractions to {rule_id: structured_input}.

    Normalizes field names through global_ontology.json before lookup (R2.11).

    Args:
        extractions: [{"field": ..., "value": ..., ...}, ...]
        rule_lookup: {field_name: rule_id} mapping built from rules.json

    Returns:
        {rule_id: {value, unit, operator_hint, source_text, confidence}}
        value may be None (R2.10: null-value signal from model).
    """
    _, surface_map = _load_ontology()

    result = {}
    for ext in extractions:
        raw_field = ext.get("field", "")
        if not raw_field:
            continue

        # R2.11: ontology normalization -- standardize field name
        normalized_field = _normalize_field_name(raw_field, surface_map)

        # Lookup: try normalized first, then raw as fallback
        rid = rule_lookup.get(normalized_field, "") or rule_lookup.get(raw_field, "")
        if not rid:
            continue

        raw_value = ext.get("value")
        confidence = float(ext.get("confidence", 0.5))

        # R2.10: null-value -- model explicitly says "no such value"
        if raw_value is None:
            if confidence >= NULL_CONFIDENCE_THRESHOLD:
                result[rid] = {
                    "value": None,
                    "unit": ext.get("unit", ""),
                    "operator_hint": ext.get("operator_hint", ""),
                    "source_text": ext.get("source_text", ""),
                    "confidence": confidence,
                }
            # Low confidence null -> skip entirely (do not override regex)
            continue

        result[rid] = {
            "value": float(raw_value),
            "unit": ext.get("unit", ""),
            "operator_hint": ext.get("operator_hint", ""),
            "source_text": ext.get("source_text", ""),
            "confidence": confidence,
        }
    return result
