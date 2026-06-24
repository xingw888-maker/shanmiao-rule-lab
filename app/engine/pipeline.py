"""Book-to-rules pipeline — split, classify, extract in one call.

Wire the three pieces together:
  ChapterSplitter → LLMChapterClassifier → group_by_domain
  → LLMRuleExtractor (per domain) → merged rule package

New API endpoint: POST /v1/packages/extract/book
"""

import json
import logging
import os
from app.engine.book import ChapterSplitter
from app.engine.classifier import DomainGroup, LLMChapterClassifier, group_by_domain
from app.engine.extractor import (
    LLMRuleExtractor,
    RulePackageBuilder,
    normalize_extracted_rules,
)

logger = logging.getLogger(__name__)


async def extract_rules_from_book(
    text: str,
    llm_url: str = "",
    llm_key: str = "",
    llm_model: str = "",
    source_filename: str = "",
) -> dict:
    """Split a book/document into chapters, classify each, extract rules
    per domain, and merge into a single installable rule package.

    Returns the combined rule package dict.
    """
    # ── Step 1: Split ──
    splitter = ChapterSplitter(min_chapter_chars=30, max_chapters=50)
    chapters = splitter.split(text)
    logger.info("Split into %d chapters", len(chapters))

    if not chapters:
        return _empty_package(source_filename, "No chapters detected in document")

    # ── Step 2: Classify ──
    has_llm = bool(llm_url and llm_key)
    classifier = LLMChapterClassifier(
        api_url=llm_url, api_key=llm_key, model=llm_model,
    )
    if has_llm:
        classified = await classifier.classify(chapters)
    else:
        classified = classifier.classify_heuristic(chapters)
        logger.info("Classified (heuristic): %d chapters", len(classified))

    # Count domains
    from collections import Counter
    domain_counts = Counter(cc.domain for cc in classified)
    logger.info("Classified: %s", dict(domain_counts))

    # ── Step 3: Group by domain ──
    groups = group_by_domain(classified)
    if not groups:
        return _empty_package(
            source_filename,
            f"No rule-relevant domains found in {len(chapters)} chapters "
            f"(domains: {dict(domain_counts)})"
        )

    # ── Step 4: Extract rules per domain ──
    extractor = LLMRuleExtractor(api_url=llm_url, api_key=llm_key, model=llm_model)
    builder = RulePackageBuilder()

    has_llm = bool(llm_url and llm_key)
    all_rules: list = []

    for group in groups:
        body = group.merged_body
        if len(body) < 50:
            logger.info("Skipping domain %s: too short (%d chars)", group.domain, len(body))
            continue

        logger.info("Extracting rules from %s (%d chars, %d chapters)",
                     group.domain, len(body), len(group.chapters))

        if has_llm:
            extracted = await extractor.extract(body, max_chars=8000)
            all_rules.extend(extracted)
        else:
            # Keyword-only fallback
            from app.engine.extractor import KeywordRuleScanner
            scanner = KeywordRuleScanner()
            keyword_rules = scanner.scan(body)
            all_rules.extend(keyword_rules)
            logger.info("  keyword scan: %d rules from %s", len(keyword_rules), group.domain)

    # Deduplicate by source_text prefix
    seen = set()
    unique_rules = []
    for r in all_rules:
        # ExtractedRule is a dataclass, LLM result is a dict
        if hasattr(r, 'source_text'):
            key = r.source_text[:60]
        elif isinstance(r, dict):
            key = r.get('source_text', '')[:60]
        else:
            key = str(r)[:60]
        if key not in seen:
            seen.add(key)
            unique_rules.append(r)

    logger.info("Total unique rules: %d (from %d raw)", len(unique_rules), len(all_rules))

    if not unique_rules:
        return _empty_package(
            source_filename,
            f"No rules extracted from {len(groups)} domain groups "
            f"({sum(g.char_count for g in groups)} chars total)"
        )

    # ── Step 5: Normalize and write candidates ──
    method = "llm_extract" if has_llm else "keyword_scan"
    normalized_candidates = normalize_extracted_rules(unique_rules, method=method)

    # Write candidates to domains/<domain>/candidates/ directory
    # Use the first available domain or fall back
    first_domain = groups[0].domain if groups else "custom"
    candidates_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "domains", first_domain, "candidates",
    )
    os.makedirs(candidates_dir, exist_ok=True)
    candidate_filename = f"auto-{os.path.basename(source_filename or 'untitled').rsplit('.', 1)[0]}.json"
    candidate_path = os.path.join(candidates_dir, candidate_filename)
    candidate_pkg = {
        "id": f"{first_domain}-candidates-{os.urandom(4).hex()}",
        "name": f"Candidates from {source_filename or 'untitled'}",
        "version": "0.1.0-candidate",
        "domain": first_domain,
        "description": (
            f"Auto-extracted candidate rules from '{source_filename or 'document'}'. "
            f"Method: {method}. Total: {len(normalized_candidates)} rules. "
            f"Not yet audited — move to rules.json after review."
        ),
        "maintainer": "auto-extracted",
        "rules": normalized_candidates,
    }
    with open(candidate_path, 'w', encoding='utf-8') as f:
        json.dump(candidate_pkg, f, ensure_ascii=False, indent=2)
    logger.info("Wrote %d candidate rules to %s", len(normalized_candidates), candidate_path)

    # ── Step 6: Build combined package ──
    pkg = builder.build(
        extracted=unique_rules,
        domain="multi-domain",
        package_name=f"Book: {source_filename or 'untitled'}",
        source_filename=source_filename,
    )

    # Attach per-domain metadata
    pkg["description"] = (
        f"Auto-extracted from '{source_filename or 'document'}' "
        f"({len(chapters)} chapters → {len(groups)} domains: "
        + ", ".join(f"{g.domain}({len(g.chapters)})" for g in groups)
        + f").  Extraction: {'LLM' if has_llm else 'keyword'}. "
        f"Please review before installing."
    )

    # Attach statistics so the caller can show them
    pkg["extraction_stats"] = {
        "total_chapters": len(chapters),
        "domain_distribution": dict(domain_counts),
        "domain_groups": len(groups),
        "total_rules_extracted": len(unique_rules),
        "extraction_method": "llm" if has_llm else "keyword",
        "source_filename": source_filename,
        "groups": [
            {
                "domain": g.domain,
                "label": g.label,
                "chapter_count": len(g.chapters),
                "char_count": g.char_count,
                "chapter_titles": [c.title for c in g.chapters],
            }
            for g in groups
        ],
    }

    return pkg


def _empty_package(source_filename: str, reason: str) -> dict:
    """Return a package with zero rules and an explanation."""
    return {
        "id": f"empty-{source_filename or 'book'}",
        "name": f"Empty: {source_filename or 'book'}",
        "version": "0.0.0",
        "domain": "none",
        "description": reason,
        "maintainer": "auto-extracted",
        "rules": [],
        "extraction_stats": {
            "total_chapters": 0,
            "domain_distribution": {},
            "domain_groups": 0,
            "total_rules_extracted": 0,
            "extraction_method": "none",
            "source_filename": source_filename,
            "groups": [],
        },
    }
