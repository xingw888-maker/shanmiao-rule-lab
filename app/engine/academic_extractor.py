"""Academic proposition extractor — reuses LLMRuleExtractor API logic, new system prompt.

Extracts structured propositions from academic papers:
  claim | method | finding | citation | contradiction

Each proposition carries source metadata:
  author, year, venue, citation_count, study_type

Constitution compliance:
  - §2 Domain separation: no academic constants in extractor (prompt is domain-aware)
  - §4 LLM role: extraction only, no final judgments
"""

import asyncio
import json
import logging
import os
import re
import uuid
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class AcademicProposition:
    """A single extracted proposition from an academic paper."""
    prop_id: str = ""
    proposition_type: str = ""     # claim | method | finding | citation | contradiction
    subject: str = ""              # entity / phenomenon
    predicate: str = ""            # assertion
    object: str = ""               # conclusion / value
    source_ref: str = ""           # Author (Year)
    citation_count: int = 0
    venue: str = ""                # Nature | Q1 | Q2 | conference | preprint | unpublished
    study_type: str = ""           # meta_analysis | rct | cohort | observational | case_report
    confidence: float = 0.5        # 0.0-1.0 from LLM extraction
    source_text: str = ""          # original sentence(s)
    paper_id: str = ""             # which paper this came from
    conflicts_with: list[str] = field(default_factory=list)  # other prop_ids this conflicts with


class AcademicExtractor:
    """Extract academic propositions from papers using LLM.

    Reuses the LLM API calling pattern from LLMRuleExtractor but with a
    different system prompt and output schema optimized for academic papers.

    LLM only extracts — does not judge credibility or detect contradictions.
    Those are the responsibility of AcademicLayeringEngine.
    """

    SYSTEM_PROMPT = """You are an academic proposition extraction engine.
Given the text of an academic paper, extract structured propositions.

Output ONLY a JSON array. No explanation before or after.

Each proposition:
{
  "proposition_type": "claim" | "method" | "finding" | "citation" | "contradiction",
  "subject": "the entity or phenomenon being discussed",
  "predicate": "the assertion made (a short verb phrase: 'supports', 'refutes', 'finds evidence for', 'uses', 'cites')",
  "object": "the conclusion, value, or target of the assertion",
  "source_ref": "Author (Year) format if available, otherwise empty string",
  "citation_count": 0,
  "venue": "Nature|Science|Cell|Lancet|NEJM|Q1|Q2|conference|preprint|unpublished|unknown",
  "study_type": "meta_analysis|rct|cohort|case_control|observational|case_report|unknown",
  "confidence": 0.0-1.0,
  "source_text": "the original sentence(s) from which this proposition was extracted",
  "conflicts_with": []
}

Proposition types:
- claim: a theoretical claim or hypothesis the paper makes
- method: a description of the research method used
- finding: an empirical finding or result
- citation: a reference to another work with context (what role the cited work plays)
- contradiction: an explicit statement that this paper's findings disagree with another work

Rules:
- Extract ONLY what is explicitly stated in the text. Do not invent.
- Each source_text must be a verbatim quote or close paraphrase from the paper.
- If a paper claims "X causes Y" but another paper you see claims "X does NOT cause Y", mark BOTH with proposition_type "contradiction" and fill conflicts_with with the other's prop_id (use paper_id:prop_N format).
- For venue: guess from context if not explicitly stated. "Nature" / "Science" / "Cell" / "The Lancet" / "NEJM" are obvious. "Q1" for top field journals. "Q2" for good journals. "conference" for conference proceedings. "preprint" for arXiv/bioRxiv/SSRN. "unpublished" for working papers or blogs. "unknown" if really cannot determine.
- For study_type: "meta_analysis" for systematic reviews and meta-analyses. "rct" for randomized controlled trials. "cohort" for cohort/longitudinal studies. "case_control" for case-control studies. "observational" for cross-sectional and other observational designs. "case_report" for single case reports. "unknown" if unclear.
- citation_count: use the actual number if mentioned, otherwise 0.
- confidence: how confident you are that this proposition accurately reflects the paper's content (0.8+ for direct quotes, 0.5-0.7 for paraphrases).
- Return at most 30 propositions. Return empty [] if no clear propositions.
- Assign each proposition an ID like "prop_1", "prop_2", etc. in order of appearance.
"""

    def __init__(self, api_url: str = "", api_key: str = "", model: str = ""):
        self.api_url = api_url
        self.api_key = api_key
        self.model = model

    @property
    def enabled(self) -> bool:
        return bool(self.api_url and self.api_key)

    async def extract(self, text: str, paper_id: str = "",
                      max_chars: int = 12000) -> list[AcademicProposition]:
        """Extract propositions from paper text via LLM.

        Args:
            text: The full paper text (or abstract if full text unavailable).
            paper_id: Unique identifier for this paper (e.g. filename or DOI).
            max_chars: Maximum characters to send to LLM (truncated if longer).

        Returns:
            List of AcademicProposition objects.
        """
        if not self.enabled:
            return []

        truncated = text[:max_chars]
        if len(text) > max_chars:
            truncated += f"\n\n[... full text is {len(text)} chars, showing first {max_chars} chars]"

        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.api_url.rstrip('/')}/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.model or "deepseek-chat",
                        "max_tokens": 4096,
                        "temperature": 0.1,
                        "messages": [
                            {"role": "system", "content": self.SYSTEM_PROMPT},
                            {"role": "user",
                             "content": f"Extract academic propositions from this paper (paper_id: {paper_id}):\n\n{truncated}"},
                        ],
                    },
                    timeout=aiohttp.ClientTimeout(total=60),
                ) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        logger.warning("Academic LLM HTTP %s: %s", resp.status, body[:200])
                        return []
                    data = await resp.json()
                    content = data["choices"][0]["message"]["content"]
                    return self._parse_response(content, paper_id)
        except Exception as e:
            logger.warning("Academic extractor error: %s", e)
            return []

    def _parse_response(self, content: str, paper_id: str) -> list[AcademicProposition]:
        """Parse LLM JSON response into AcademicProposition objects."""
        json_match = re.search(r'\[[\s\S]*\]', content)
        if not json_match:
            return []
        try:
            items = json.loads(json_match.group(0))
        except json.JSONDecodeError:
            return []

        props = []
        for item in items:
            if not isinstance(item, dict):
                continue
            props.append(AcademicProposition(
                prop_id=item.get("prop_id", f"prop_{len(props)+1}"),
                proposition_type=item.get("proposition_type", "claim"),
                subject=item.get("subject", ""),
                predicate=item.get("predicate", ""),
                object=item.get("object", ""),
                source_ref=item.get("source_ref", ""),
                citation_count=int(item.get("citation_count", 0)),
                venue=item.get("venue", "unknown"),
                study_type=item.get("study_type", "unknown"),
                confidence=float(item.get("confidence", 0.5)),
                source_text=item.get("source_text", ""),
                paper_id=paper_id,
                conflicts_with=item.get("conflicts_with", []),
            ))
        return props

    def extract_sync(self, text: str, paper_id: str = "",
                     max_chars: int = 12000) -> list[AcademicProposition]:
        """Synchronous wrapper for extract()."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(self.extract(text, paper_id, max_chars))


# ── Multi-paper cross-comparison ──

@dataclass
class CrossPaperResult:
    """Result of cross-comparing propositions from multiple papers."""
    paper_id: str
    propositions: list[AcademicProposition]
    # Contradictions found: list of (prop_a, prop_b, reason)
    contradictions: list[tuple[AcademicProposition, AcademicProposition, str]] = field(default_factory=list)
    # Summary statistics
    total_props: int = 0
    claims: int = 0
    methods: int = 0
    findings: int = 0
    contradictions_count: int = 0


def cross_compare_papers(papers: dict[str, list[AcademicProposition]],
                         ontology_path: str = "") -> dict[str, CrossPaperResult]:
    """Cross-compare propositions from multiple papers for contradictions.

    Uses string matching on subject/predicate/object fields to detect
    potential contradictions.  A contradiction is flagged when:
    - Two propositions share the same or synonymous subject
    - Their predicates express opposite directions (supports vs refutes,
      increases vs decreases, is vs is_not)

    Args:
        papers: Dict of paper_id -> list of AcademicProposition.
        ontology_path: Path to academic domain ontology.json for term expansion.

    Returns:
        Dict of paper_id -> CrossPaperResult with contradictions populated.
    """
    # Load alias map for term expansion
    aliases: dict[str, list[str]] = {}
    if ontology_path and os.path.isfile(os.path.join(ontology_path, "ontology.json")):
        try:
            with open(os.path.join(ontology_path, "ontology.json"), "r") as f:
                onto = json.load(f)
            groups = onto.get("entity_groups", {})
            for _group_key, variants in groups.items():
                if isinstance(variants, list) and len(variants) >= 2:
                    for v in variants:
                        if isinstance(v, str) and v.strip():
                            aliases[v.strip().lower()] = [x.lower() for x in variants]
        except (json.JSONDecodeError, OSError):
            pass

    # ── Contradiction predicates ──
    OPPOSING_PAIRS = [
        ({"supports", "confirms", "proves", "demonstrates", "shows", "finds"},
         {"refutes", "disproves", "contradicts", "fails to support", "no evidence", "does not support"}),
        ({"increases", "raises", "improves", "enhances", "boosts", "positively affects"},
         {"decreases", "lowers", "reduces", "diminishes", "harms", "negatively affects"}),
        ({"is", "equals", "represents"},
         {"is not", "differs from", "is distinct from"}),
        ({"effective", "works", "successful", "beneficial"},
         {"ineffective", "does not work", "unsuccessful", "harmful", "no effect"}),
        ({"significant", "statistically significant", "meaningful"},
         {"not significant", "non-significant", "no statistically significant"}),
    ]

    def _predicate_direction(pred: str) -> int:
        """Return 0=neutral, 1=positive/affirmative, -1=negative/refuting."""
        pred_lower = pred.lower().strip()
        for pos_set, neg_set in OPPOSING_PAIRS:
            if any(w in pred_lower for w in pos_set):
                return 1
            if any(w in pred_lower for w in neg_set):
                return -1
        return 0

    def _expand_term(term: str) -> set[str]:
        """Expand a term through ontology aliases."""
        term_lower = term.lower().strip()
        if term_lower in aliases:
            return set(aliases[term_lower])
        return {term_lower}

    # Build results
    results: dict[str, CrossPaperResult] = {}
    for paper_id, props in papers.items():
        results[paper_id] = CrossPaperResult(
            paper_id=paper_id,
            propositions=props,
            total_props=len(props),
            claims=sum(1 for p in props if p.proposition_type == "claim"),
            methods=sum(1 for p in props if p.proposition_type == "method"),
            findings=sum(1 for p in props if p.proposition_type == "finding"),
        )

    # Cross-compare all pairs of papers
    paper_ids = list(papers.keys())
    for i in range(len(paper_ids)):
        for j in range(i + 1, len(paper_ids)):
            pid_a = paper_ids[i]
            pid_b = paper_ids[j]
            props_a = papers[pid_a]
            props_b = papers[pid_b]

            for pa in props_a:
                subj_a = _expand_term(pa.subject)
                dir_a = _predicate_direction(pa.predicate)

                for pb in props_b:
                    subj_b = _expand_term(pb.subject)
                    dir_b = _predicate_direction(pb.predicate)

                    # Check if subjects overlap (same topic area)
                    if not (subj_a & subj_b):
                        continue

                    # Check if predicates are opposing
                    if dir_a != 0 and dir_b != 0 and dir_a != dir_b:
                        reason = (f"'{pa.subject}' — '{pa.predicate}' (direction={dir_a}) "
                                  f"vs '{pb.subject}' — '{pb.predicate}' (direction={dir_b})")
                        contradiction = (pa, pb, reason)
                        results[pid_a].contradictions.append(contradiction)
                        results[pid_b].contradictions.append(contradiction)
                        results[pid_a].contradictions_count += 1
                        results[pid_b].contradictions_count += 1

    return results
