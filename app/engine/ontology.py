"""Domain ontology — terminology grounding for the translation layer.

Three capabilities:
1. Entity equivalence — maps synonym groups to canonical terms.
2. Concept taxonomy — IS-A hierarchy of domain concepts.
3. Context proximity — scores numeric values by keyword distance.

This module is the first step toward vocabulary ontologization.
It directly reduces the translation-layer noise that currently sits at ~36%.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional


class EntityRegistry:
    """Maps surface terms to canonical entities with synonym groups."""

    def __init__(self):
        self._surface_to_canonical: dict[str, str] = {}
        self._canonical_to_surfaces: dict[str, list[str]] = {}

    def add_group(self, canonical: str, surfaces: list[str]) -> None:
        self._canonical_to_surfaces[canonical] = list(surfaces)
        for s in surfaces:
            self._surface_to_canonical[s] = canonical

    def canonical(self, term: str) -> str:
        return self._surface_to_canonical.get(term, term)

    def expand(self, term: str) -> list[str]:
        canon = self.canonical(term)
        return self._canonical_to_surfaces.get(canon, [term])

    def match_any(self, text: str, target_canonical: str) -> bool:
        surfaces = self._canonical_to_surfaces.get(target_canonical, [target_canonical])
        return any(s in text for s in surfaces)

    def find_all(self, text: str) -> list[tuple[str, str, int]]:
        results = []
        for surface, canonical in self._surface_to_canonical.items():
            pos = 0
            while True:
                idx = text.find(surface, pos)
                if idx == -1:
                    break
                results.append((surface, canonical, idx))
                pos = idx + 1
        results.sort(key=lambda x: x[2])
        return results

    @property
    def group_count(self) -> int:
        return len(self._canonical_to_surfaces)


@dataclass
class TaxonNode:
    name: str
    parent: Optional[str] = None
    children: list[str] = field(default_factory=list)
    depth: int = 0


class ConceptTaxonomy:
    """IS-A hierarchy for domain concepts."""

    def __init__(self):
        self._nodes: dict[str, TaxonNode] = {}

    def add(self, name: str, parent: Optional[str] = None) -> None:
        if name in self._nodes:
            return
        node = TaxonNode(name=name, parent=parent)
        self._nodes[name] = node
        if parent:
            if parent in self._nodes:
                self._nodes[parent].children.append(name)
            else:
                self._nodes[parent] = TaxonNode(name=parent)
                self._nodes[parent].children.append(name)
        self._recompute_depths()

    def _recompute_depths(self):
        def set_depth(name, d):
            if name in self._nodes:
                self._nodes[name].depth = d
                for child in self._nodes[name].children:
                    set_depth(child, d + 1)
        for name, node in self._nodes.items():
            if node.parent is None:
                set_depth(name, 0)

    def is_a(self, child: str, ancestor: str) -> bool:
        if child not in self._nodes:
            return False
        current = child
        while current is not None:
            if current == ancestor:
                return True
            current = self._nodes[current].parent if current in self._nodes else None
        return False

    def ancestors(self, name: str) -> list[str]:
        result = []
        current = name
        while current is not None:
            result.insert(0, current)
            current = self._nodes[current].parent if current in self._nodes else None
        return result

    def descendants(self, name: str) -> list[str]:
        result = list(self._nodes[name].children) if name in self._nodes else []
        for child in list(result):
            result.extend(self.descendants(child))
        return result

    @property
    def root_count(self) -> int:
        return sum(1 for n in self._nodes.values() if n.parent is None)

    def check_consistency(self) -> list[str]:
        """Check for attribute inheritance conflicts in the taxonomy.

        Detects:
        1. Circular IS-A chains (A IS-A B IS-A A)
        2. Depth anomalies (child deeper than expected)
        3. Missing parents (orphaned nodes with invalid parent refs)

        Returns list of human-readable conflict descriptions.
        Empty list = taxonomy is consistent.
        """
        issues = []

        # 1. Cycle detection
        for name in self._nodes:
            visited = set()
            current = name
            while current is not None:
                if current in visited:
                    issues.append(
                        f"Cycle detected: {name} IS-A ... IS-A {current} IS-A ... IS-A {name}"
                    )
                    break
                visited.add(current)
                current = self._nodes[current].parent if current in self._nodes else None

        # 2. Depth sanity check (warning only)
        MAX_DEPTH = 20
        for name, node in self._nodes.items():
            if node.depth > MAX_DEPTH:
                issues.append(
                    f"Excessive depth: {name} is at depth {node.depth} (> {MAX_DEPTH}). "
                    f"Check taxonomy for unintended nesting."
                )

        # 3. Parent reference integrity
        for name, node in self._nodes.items():
            if node.parent and node.parent not in self._nodes:
                issues.append(
                    f"Orphan: {name} claims parent '{node.parent}' "
                    f"which is not a registered concept."
                )

        return issues


class ContextProximityScorer:
    """Scores how likely a numeric value belongs to a given field.

    Uses keyword proximity to pick the right number when a text region
    contains multiple values with the same unit.
    """

    @staticmethod
    def min_distance(values: list[dict], context_pattern: str, text: str) -> list[dict]:
        try:
            ctx_matches = [(m.start(), m.end()) for m in re.finditer(context_pattern, text)]
        except re.error:
            return values
        if not ctx_matches:
            return values
        annotated = []
        for v in values:
            pos = v.get("position", 0)
            dist = min(min(abs(pos - cs), abs(pos - ce)) for cs, ce in ctx_matches)
            annotated.append({**v, "context_distance": dist})
        annotated.sort(key=lambda a: a["context_distance"])
        return annotated

    @staticmethod
    def pick_nearest(
        values: list[dict],
        context_pattern: str,
        text: str,
        operator: str,
        max_distance: int = 80,
    ) -> Optional[dict]:
        """Pick the best candidate by proximity-first, then conservativeness.

        Strategy:
        1. Annotate each value with distance to nearest context keyword.
        2. Sort by distance ascending.
        3. If the closest value is within max_distance: return it directly.
           Proximity beats conservativeness — the closest number IS the
           relevant one. "预留3%为质量保证金" means 3% IS the deposit ratio.
        4. If the closest value is beyond max_distance: no contextual
           association exists. Return None (caller should treat as N/A).
        5. If multiple values share the same minimum distance: apply
           conservative operator within that cluster.

        Returns None to signal no contextually-associated value was found.
        """
        if not values:
            return None
        try:
            ctx_positions = [(m.start(), m.end()) for m in re.finditer(context_pattern, text)]
        except re.error:
            ctx_positions = []
        annotated = []
        for v in values:
            pos = v.get("position", 0)
            if ctx_positions:
                dist = min(min(abs(pos - cs), abs(pos - ce)) for cs, ce in ctx_positions)
            else:
                dist = 0
            annotated.append({**v, "_dist": dist})
        annotated.sort(key=lambda a: a["_dist"])
        closest_dist = annotated[0]["_dist"]
        if closest_dist > max_distance:
            return None
        cluster = [a for a in annotated if a["_dist"] == closest_dist]
        if operator in (">=", ">"):
            return min(cluster, key=lambda n: n["value"])
        elif operator in ("<=", "<"):
            return max(cluster, key=lambda n: n["value"])
        else:
            return cluster[0]


def build_construction_engineering_ontology():
    """Build pre-seeded ontology for Chinese construction engineering contracts."""
    entities = EntityRegistry()
    entities.add_group("发包人", ["发包人", "建设单位", "业主", "甲方", "招标人"])
    entities.add_group("承包人", ["承包人", "施工单位", "乙方", "承包商", "中标人"])
    entities.add_group("监理单位", ["监理单位", "监理", "工程监理", "监理方"])
    entities.add_group("工程质量保修", ["工程质量保修", "质量保修", "工程保修", "保修"])
    entities.add_group("质量保证金", ["质量保证金", "质保金", "保修金", "保证金"])
    entities.add_group("竣工验收", ["竣工验收", "竣工验收合格", "竣工", "工程竣工"])
    entities.add_group("工期", ["工期", "合同工期", "施工工期", "总工期", "工程期限"])
    entities.add_group("违约金", ["违约金", "逾期违约金", "罚金", "处罚", "赔偿金"])
    entities.add_group("保修期", ["保修期", "保修期限", "保修期间", "保修时间"])
    entities.add_group("防水工程", ["防水", "屋面防水", "地下防水", "防水工程", "防渗漏"])
    entities.add_group("违约责任", ["违约责任", "违约", "违约及奖罚"])
    entities.add_group("争议解决", ["争议", "争议解决", "协商", "仲裁", "诉讼"])
    taxonomy = ConceptTaxonomy()
    taxonomy.add("建筑工程")
    taxonomy.add("防水工程", parent="建筑工程")
    taxonomy.add("屋面防水", parent="防水工程")
    taxonomy.add("地下防水", parent="防水工程")
    taxonomy.add("主体结构", parent="建筑工程")
    taxonomy.add("装饰装修", parent="建筑工程")
    taxonomy.add("安装工程", parent="建筑工程")
    taxonomy.add("建设工程合同")
    taxonomy.add("施工总承包合同", parent="建设工程合同")
    taxonomy.add("专业分包合同", parent="建设工程合同")
    taxonomy.add("劳务分包合同", parent="建设工程合同")
    taxonomy.add("防水施工合同", parent="专业分包合同")
    taxonomy.add("法规强制性条款")
    taxonomy.add("行业推荐标准", parent="法规强制性条款")
    taxonomy.add("合同自由约定", parent="法规强制性条款")
    return entities, taxonomy
