"""Auto-bootstrapper — self-assembling knowledge from raw text.

Given any document (law, regulation, textbook, contract), the bootstrapper:
1. Splits into chapters/sections
2. Extracts domain concepts (terms, entities, relationships)
3. Builds a concept taxonomy (IS-A hierarchy) from extracted terms
4. Extracts rules (numeric constraints, mutual exclusions, etc.)
5. Compiles into an installable rule package + ontology

This is the "扔一本PDF进去，自动归位" engine.
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# Data types
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class ExtractedConcept:
    """A domain concept auto-extracted from text."""
    term: str                     # canonical term
    surface_forms: list[str]      # all surface variations found
    frequency: int                # occurrence count
    context_snippets: list[str]   # surrounding text samples
    parent_term: str = ""         # inferred parent in taxonomy
    domain: str = ""              # e.g. "建设工程", "经济学", "医疗"


@dataclass
class ExtractedRelationship:
    """A relationship between two concepts."""
    subject: str
    predicate: str               # "IS_A", "HAS", "REQUIRES", "FORBIDS", "EQUALS"
    object: str
    confidence: float
    source_text: str = ""


@dataclass
class BootstrappedKnowledge:
    """Complete bootstrapped knowledge from a document."""
    source_id: str                # document identifier
    source_title: str = ""
    concepts: list[ExtractedConcept] = field(default_factory=list)
    relationships: list[ExtractedRelationship] = field(default_factory=list)
    concepts_taxonomy: dict[str, list[str]] = field(default_factory=dict)  # parent -> children
    entity_groups: dict[str, list[str]] = field(default_factory=dict)      # canonical -> surfaces
    numeric_rules: list[dict] = field(default_factory=list)
    other_rules: list[dict] = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    # W1.3: Feedback-loop rejected rules from AutoValidator
    rejected_rules: list[dict] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════
# Chapter / Section Splitter
# ═══════════════════════════════════════════════════════════════════════

class DocumentSplitter:
    """Splits a document into structural units (chapters, articles, sections)."""

    # Match Chinese legal/regulation structure
    _CHAPTER_RE = re.compile(r'(第[一二三四五六七八九十百千]+章)\s*(.*)')
    _ARTICLE_RE = re.compile(r'(第[一二三四五六七八九十百千]+条)\s*')
    _SECTION_RE = re.compile(r'(第[一二三四五六七八九十百千]+节)\s*(.*)')
    _NUM_HEADING_RE = re.compile(r'^(\d+[.、]\s*.+)$', re.MULTILINE)

    @classmethod
    def split(cls, text: str) -> list[dict]:
        """Split text into structural units with hierarchy.

        Returns list of {title, level, body, children}.
        """
        units = []
        # Try chapter/article structure first
        chapters = cls._CHAPTER_RE.split(text)
        if len(chapters) > 1:
            # chapters[0] = preamble, then [tag, title, body, tag, title, ...]
            for i in range(1, len(chapters), 3):
                if i + 2 <= len(chapters):
                    title = (chapters[i] + chapters[i+1]).strip()
                    body = chapters[i+2] if i+2 < len(chapters) else ""
                elif i + 1 <= len(chapters):
                    title = chapters[i].strip()
                    body = ""
                else:
                    break
                # Split body into articles
                articles = []
                article_parts = cls._ARTICLE_RE.split(body)
                if len(article_parts) > 1:
                    for j in range(1, len(article_parts), 2):
                        art_title = article_parts[j].strip()
                        art_body = article_parts[j+1] if j+1 < len(article_parts) else ""
                        articles.append({"title": art_title, "level": 2, "body": art_body.strip()})

                units.append({
                    "title": title,
                    "level": 1,
                    "body": body.strip(),
                    "children": articles,
                })
        else:
            # No chapter structure -- try numbered sections
            parts = cls._NUM_HEADING_RE.split(text)
            if len(parts) > 1:
                current_title = parts[0].strip()  # preamble
                for i in range(1, len(parts), 2):
                    title = parts[i].strip()
                    body = parts[i+1] if i+1 < len(parts) else ""
                    units.append({"title": title, "level": 1, "body": body.strip()})
            else:
                # Flat text
                units.append({"title": "全文", "level": 0, "body": text.strip()})

        return units


# ═══════════════════════════════════════════════════════════════════════
# Concept Extractor
# ═══════════════════════════════════════════════════════════════════════

class ConceptExtractor:
    """Extracts domain concepts from text using statistical + heuristic methods.

    1. Term frequency analysis (TF with domain-specific stopword filtering)
    2. Noun phrase detection (Chinese: 2-6 character CJK sequences with high TF-IDF)
    3. Parent term inference (e.g. "屋面防水" IS-A "防水工程")
    4. Co-occurrence based relationship mining
    """

    # Structural markers that suggest definitions
    _DEFINITION_MARKERS = re.compile(
        r'(?:是指|指的是|系指|即|包括|包含|分为|应当|不得|必须|严禁|禁止)'
    )
    # Numeric + unit patterns
    _NUMERIC_CLAUSE_RE = re.compile(
        r'(?:不少于|不超过|不小于|不大于|大于|小于|等于|至少|至多|不得[少低]于|不得[多高]于|应当[少低]于|应当[多高]于'
        r'|最低|最高|不得[多少高低于])'
        r'\s*([\d.]+)\s*(年|月|日|天|元|万|%|％)'
    )
    # Match bare numeric + unit near key regulatory terms
    _NUMERIC_NEAR_KEYWORD_RE = re.compile(
        r'(保修|质量|保证|期限|工期|处罚|罚款|赔偿|违约金).{0,20}?'
        r'([\d.]+)\s*(年|月|日|天|元|万|%|％)'
    )
    _NUMERIC_CLOSE_RE = re.compile(
        r'([\d.]+)\s*(年|月|日|天|元|万|%|％)\s*(?:以[上下内])?'
    )
    _CN_NUM_CLAUSE_RE = re.compile(
        r'(?:不少于|不超过|不小于|不大于|不得[少低]于|不得[多高]于)'
        r'\s*([零一二两三四五六七八九十百千万]+)\s*(年|月|日|天|元|%|％)'
    )

    def __init__(self, min_term_freq: int = 2, max_concepts: int = 100):
        self.min_term_freq = min_term_freq
        self.max_concepts = max_concepts

    def extract(self, text: str, domain_hint: str = "") -> list[ExtractedConcept]:
        """Extract domain concepts from text."""
        concepts = []
        seen_terms = set()

        # 1. Extract CJK multi-character terms (2-6 chars) with frequency
        cjk_terms = re.findall(r'[一-鿿]{2,8}', text)
        cjk_freq = Counter(t for t in cjk_terms if len(t) >= 2)

        # 2. Extract English/proper noun terms
        en_terms = re.findall(r'[A-Z][a-z]+(?:[\s-][A-Z][a-z]+)*', text)
        en_freq = Counter(t.lower() for t in en_terms)

        # 3. Filter low-frequency and noise
        domain_stopwords = {
            '可以', '进行', '或者', '没有', '一个', '这种', '那种',
            '这个', '那个', '他们', '我们', '什么', '怎么', '因为',
            '所以', '但是', '如果', '虽然', '而且', '然后', '之后',
            '之前', '已经', '正在', '将要', '必须', '应该', '可能',
            '需要', '能够', '不能', '不会', '不是', '所有', '其他',
            '其中', '之间', '之后', '以前', '以上', '以下', '以内',
            '以外', '以及', '及其', '关于', '对于', '根据', '按照',
        }

        # 4. Score and select top concepts
        scored = []
        for term, freq in cjk_freq.most_common(200):
            if freq < self.min_term_freq:
                continue
            if term in domain_stopwords:
                continue
            if term in seen_terms:
                continue

            # Extract context snippets
            contexts = []
            for m in re.finditer(re.escape(term), text):
                s = max(0, m.start() - 20)
                e = min(len(text), m.end() + 20)
                contexts.append(text[s:e].strip())
                if len(contexts) >= 3:
                    break

            # Check if term appears in definition context
            def_score = sum(1 for c in contexts if self._DEFINITION_MARKERS.search(c))

            scored.append({
                "term": term,
                "freq": freq,
                "contexts": contexts,
                "def_score": def_score,
                "score": freq * (1 + def_score * 0.5),
            })

        scored.sort(key=lambda x: -x["score"])

        # 5. Infer parent-child relationships
        for item in scored[:self.max_concepts]:
            term = item["term"]
            if term in seen_terms:
                continue
            seen_terms.add(term)

            # Find surface forms
            surface_forms = self._find_surface_forms(term, text)

            # Infer parent
            parent = self._infer_parent(term, [s["term"] for s in scored[:50]], text)

            concepts.append(ExtractedConcept(
                term=term,
                surface_forms=surface_forms,
                frequency=item["freq"],
                context_snippets=item["contexts"],
                parent_term=parent,
                domain=domain_hint,
            ))

        return concepts

    def _find_surface_forms(self, term: str, text: str) -> list[str]:
        """Find common variations of a term in text."""
        forms = {term}
        for prefix in ["", "工程", "项目", "合同"]:
            for suffix in ["", "工程", "项目", "管理", "施工"]:
                variant = prefix + term + suffix
                if variant != term and variant in text:
                    forms.add(variant)
        return list(forms)

    def _infer_parent(self, term: str, all_terms: list[str], text: str) -> str:
        """Infer the parent of a term in the concept taxonomy."""
        # Heuristic 1: suffix matching
        for i in range(len(term) - 2, 0, -1):
            suffix = term[-i:]
            if suffix in all_terms and suffix != term:
                return suffix

        # Heuristic 2: IS-A pattern detection
        patterns = [
            re.compile(rf'{re.escape(term)}.*(?:是|属于|为).*?([一-鿿]{{2,6}})'),
        ]
        for pat in patterns:
            m = pat.search(text)
            if m:
                candidate = m.group(1)
                if candidate in all_terms and candidate != term:
                    return candidate

        return ""

    def extract_numeric_rules(self, text: str, concepts: list[ExtractedConcept]) -> list[dict]:
        """Extract numeric constraints that can become rules."""
        rules = []

        # Pattern 1: keyword + numeric + unit nearby
        for m in self._NUMERIC_NEAR_KEYWORD_RE.finditer(text):
            keyword = m.group(1)
            val = float(m.group(2))
            unit = m.group(3)

            start = max(0, m.start() - 80)
            prefix = text[start:m.start()]
            fields = re.findall(r'[一-鿿]{2,10}', prefix)
            keyword_idx = prefix.rfind(keyword)
            field = keyword
            if keyword_idx >= 0 and keyword_idx + len(keyword) < len(prefix):
                after_keyword = prefix[keyword_idx + len(keyword):]
                next_field = re.match(r'[一-鿿]{1,6}', after_keyword)
                if next_field:
                    field = keyword + next_field.group(0)
            if keyword_idx > 0:
                before = prefix[:keyword_idx]
                prev_fields = re.findall(r'[一-鿿]{2,6}', before)
                if prev_fields:
                    field = prev_fields[-1] + keyword

            if re.search(r'不少于|不小于|至少|不得[少低]于|最低', prefix):
                operator = ">="
            elif re.search(r'不超过|不大于|至多|不得[多高]于|最高', prefix):
                operator = "<="
            else:
                operator = ">="

            ref_match = re.search(
                r'(?:《[^》]+》|第[一二三四五六七八九十百千]+条[之]?[一二三四五六七八九十]?)',
                prefix
            )
            legal_ref = ref_match.group(0) if ref_match else ""

            rules.append({
                "type": "numeric_comparison",
                "label": field,
                "context_pattern": keyword,
                "unit": unit,
                "operator": operator,
                "expected": val,
                "legal_ref": legal_ref,
                "confidence": 0.7,
            })

        # Pattern 2: bare numeric + unit near structural keywords
        for m in self._NUMERIC_CLAUSE_RE.finditer(text):
            val = float(m.group(1))
            unit = m.group(2)
            start = max(0, m.start() - 50)
            prefix = text[start:m.start()]
            fields = re.findall(r'[一-鿿]{2,8}', prefix)
            field = fields[-1] if fields else f"field_{hashlib.md5(prefix.encode()).hexdigest()[:6]}"

            if re.search(r'不少于|不小于|至少|不得[少低]于|应当[多高]于', prefix):
                operator = ">="
            elif re.search(r'不超过|不大于|至多|不得[多高]于|应当[少低]于', prefix):
                operator = "<="
            else:
                operator = ">="

            ref_match = re.search(
                r'(?:《[^》]+》|第[一二三四五六七八九十百千]+条[之]?[一二三四五六七八九十]?)',
                prefix
            )
            legal_ref = ref_match.group(0) if ref_match else ""

            rules.append({
                "type": "numeric_comparison",
                "label": field,
                "context_pattern": field,
                "unit": unit,
                "operator": operator,
                "expected": val,
                "legal_ref": legal_ref,
                "confidence": 0.7,
            })

        # Deduplicate
        seen = set()
        unique = []
        for r in rules:
            key = (r["label"], r["expected"], r["unit"])
            if key not in seen:
                seen.add(key)
                unique.append(r)

        return unique


# ═══════════════════════════════════════════════════════════════════════
# AutoBootstrapper
# ═══════════════════════════════════════════════════════════════════════

class AutoBootstrapper:
    """Self-assembling knowledge from raw text.

    Given any document:
    1. Split into structural units
    2. Extract concepts and build taxonomy
    3. Extract numeric rules
    4. Compile into installable package + ontology
    """

    def __init__(self):
        self.splitter = DocumentSplitter()
        self.extractor = ConceptExtractor()

    def bootstrap(self, text: str, title: str = "", domain_hint: str = "",
                  rejected_rules: list[dict] | None = None) -> BootstrappedKnowledge:
        """Run the full bootstrap pipeline.

        Args:
            text: Raw document text.
            title: Document title (used as package name).
            domain_hint: Domain classification hint.
            rejected_rules: W1.3 -- previously rejected rules to re-enter
                           the extraction pipeline for feedback-loop retry.

        Returns:
            BootstrappedKnowledge with concepts, taxonomy, rules.
        """
        source_id = hashlib.md5(text[:200].encode()).hexdigest()[:12]
        knowledge = BootstrappedKnowledge(
            source_id=source_id,
            source_title=title or f"document_{source_id}",
        )

        # Phase 1: Split
        units = self.splitter.split(text)
        knowledge.stats["structural_units"] = len(units)
        knowledge.stats["chapters"] = sum(1 for u in units if u.get("children"))

        # Phase 2: Extract concepts
        all_text = text.lower()
        concepts = self.extractor.extract(all_text, domain_hint)
        knowledge.concepts = concepts
        knowledge.stats["concepts_extracted"] = len(concepts)

        # Build taxonomy
        taxonomy: dict[str, list[str]] = defaultdict(list)
        for c in concepts:
            if c.parent_term:
                taxonomy[c.parent_term].append(c.term)
        knowledge.concepts_taxonomy = dict(taxonomy)

        # Build entity groups (from surface forms)
        entity_groups: dict[str, list[str]] = {}
        for c in concepts:
            if len(c.surface_forms) > 1:
                entity_groups[c.term] = c.surface_forms
        knowledge.entity_groups = entity_groups

        # Phase 3: Extract relationships
        relationships = self._extract_relationships(all_text, concepts)
        knowledge.relationships = relationships
        knowledge.stats["relationships"] = len(relationships)

        # Phase 4: Extract numeric rules
        numeric_rules = self.extractor.extract_numeric_rules(all_text, concepts)
        knowledge.numeric_rules = numeric_rules
        knowledge.stats["numeric_rules"] = len(numeric_rules)

        # Phase 5: Extract other rules (mutual exclusion, co-occurrence)
        other_rules = self._extract_prohibition_rules(all_text, concepts)
        knowledge.other_rules = other_rules
        knowledge.stats["other_rules"] = len(other_rules)

        knowledge.stats["total_rules"] = len(numeric_rules) + len(other_rules)

        # W1.3: Store rejected rules received from AutoValidator for feedback-loop
        if rejected_rules:
            knowledge.rejected_rules = list(rejected_rules)
            knowledge.stats["rejected_rules"] = len(rejected_rules)

        return knowledge

    def _extract_relationships(
        self, text: str, concepts: list[ExtractedConcept]
    ) -> list[ExtractedRelationship]:
        """Extract relationships between concepts."""
        relationships = []
        concept_set = {c.term for c in concepts}

        # IS-A relationships from taxonomy inference
        for c in concepts:
            if c.parent_term:
                relationships.append(ExtractedRelationship(
                    subject=c.term,
                    predicate="IS_A",
                    object=c.parent_term,
                    confidence=0.6,
                ))

        # REQUIRES: "X 应当 Y" / "X 必须 Y"
        for c1 in concepts:
            for c2 in concepts:
                if c1.term == c2.term:
                    continue
                for ctx in c1.context_snippets:
                    if c2.term in ctx:
                        if re.search(r'(?:应当|必须|应|需)', ctx):
                            relationships.append(ExtractedRelationship(
                                subject=c1.term,
                                predicate="REQUIRES",
                                object=c2.term,
                                confidence=0.4,
                                source_text=ctx,
                            ))
                        break

        # Deduplicate
        seen = set()
        unique = []
        for r in relationships:
            key = (r.subject, r.predicate, r.object)
            if key not in seen:
                seen.add(key)
                unique.append(r)

        return unique[:200]

    def _extract_prohibition_rules(
        self, text: str, concepts: list[ExtractedConcept]
    ) -> list[dict]:
        """Extract prohibition / mandatory rules.

        Patterns:
        - "不得 X" -> forbidden_pattern rule
        - "X 必须 Y" -> required_pattern or co_occurrence rule
        - "X 和 Y 不能同时" -> mutual_exclusion rule
        """
        rules = []

        # "不得" patterns -> forbidden
        for m in re.finditer(r'(.{0,30})不得(.{0,30})', text):
            prefix = m.group(1)
            suffix = m.group(2)
            context = f"{prefix}不得{suffix}"
            terms = re.findall(r'[一-鿿]{2,8}', context)
            if terms:
                rules.append({
                    "type": "forbidden_pattern",
                    "name": f"禁止: {''.join(terms[:2])}",
                    "pattern": re.escape(terms[0]),
                    "severity": "error",
                    "message": f"文档禁止出现 '{terms[0]}' 相关行为。",
                    "confidence": 0.5,
                })

        # "必须" / "应当" -> co_occurrence patterns
        for m in re.finditer(r'(.{0,20})(?:应当|必须|需)(.{0,20})', text):
            prefix_terms = re.findall(r'[一-鿿]{2,8}', m.group(1))
            suffix_terms = re.findall(r'[一-鿿]{2,8}', m.group(2))
            if prefix_terms and suffix_terms:
                rules.append({
                    "type": "co_occurrence",
                    "name": f"共存: {prefix_terms[-1]} => {suffix_terms[0]}",
                    "antecedent": prefix_terms[-1],
                    "consequent": suffix_terms[0],
                    "severity": "warning",
                    "message": f"当出现 '{prefix_terms[-1]}' 时，必须同时出现 '{suffix_terms[0]}'。",
                    "confidence": 0.4,
                })

        return rules[:50]

    def to_package(self, knowledge: BootstrappedKnowledge) -> dict:
        """Convert bootstrapped knowledge into an installable rule package."""
        package = {
            "id": f"bootstrapped-{knowledge.source_id}",
            "name": knowledge.source_title,
            "version": "0.1.0-auto",
            "domain": knowledge.concepts[0].domain if knowledge.concepts else "auto-extracted",
            "description": (
                f"Auto-bootstrapped from '{knowledge.source_title}'. "
                f"{knowledge.stats.get('concepts_extracted', 0)} concepts, "
                f"{knowledge.stats.get('total_rules', 0)} rules. "
                f"PLEASE REVIEW before production use."
            ),
            "maintainer": "auto-bootstrapper",
            "rules": [],
            "ontology": {
                "concepts": [
                    {
                        "term": c.term,
                        "surface_forms": c.surface_forms,
                        "frequency": c.frequency,
                        "parent": c.parent_term,
                    }
                    for c in knowledge.concepts
                ],
                "entity_groups": knowledge.entity_groups,
                "taxonomy": knowledge.concepts_taxonomy,
            },
        }
        return package
