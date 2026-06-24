"""LexicalPrototypeStore — CJK bigram domain classifier.

Domain-agnostic, vocabulary-free, zero-training classifier.  Each domain is
represented as a set of CJK character bigrams extracted from its training
corpus.  Classification uses asymmetric COVERAGE (what fraction of this text's
bigrams appear in the domain prototype?) as the primary metric, which works
well for both short and long texts.

Per-domain thresholds are auto-calibrated from training data via
leave-one-out cross-validation: threshold = mean_coverage - 2 * std_coverage.

Usage:
    store = LexicalPrototypeStore.load("data/lexical_prototypes.json")
    domains = store.classify(text)  # -> ["construction"] or [] if unknown
    store.build_domain("philosophy", corpus_texts)
    store.save("data/lexical_prototypes.json")
"""

from __future__ import annotations

import json
import math
import os
import re as _re

_CJK_CHAR = _re.compile(r"[一-鿿]+")


def _bigrams_of(text: str) -> set[str]:
    cjk_chars = _CJK_CHAR.findall(text)
    cjk_str = "".join(cjk_chars)
    bigrams = set()
    for i in range(len(cjk_str) - 1):
        bigrams.add(cjk_str[i : i + 2])
    return bigrams


class LexicalPrototype:
    __slots__ = ("domain_id", "bigrams", "coverage_threshold", "combined_threshold", "sample_count")

    def __init__(self, domain_id, bigrams=None, coverage_threshold=0.15, combined_threshold=0.001, sample_count=0):
        self.domain_id = domain_id
        self.bigrams = bigrams or set()
        self.coverage_threshold = coverage_threshold
        self.combined_threshold = combined_threshold
        self.sample_count = sample_count

    def coverage(self, text_bigrams):
        if not text_bigrams: return 0.0
        return len(text_bigrams & self.bigrams) / len(text_bigrams)

    def jaccard(self, text_bigrams):
        if not text_bigrams or not self.bigrams: return 0.0
        return len(text_bigrams & self.bigrams) / len(text_bigrams | self.bigrams)

    def score(self, text_bigrams):
        cov = self.coverage(text_bigrams)
        return cov, cov * self.jaccard(text_bigrams)

    def add_sample(self, text):
        self.bigrams |= _bigrams_of(text)
        self.sample_count += 1

    def to_dict(self):
        return dict(domain_id=self.domain_id, bigrams=sorted(self.bigrams),
                    coverage_threshold=round(self.coverage_threshold,4),
                    combined_threshold=round(self.combined_threshold,4),
                    sample_count=self.sample_count)


class LexicalPrototypeStore:
    DOMINANCE_RATIO = 2.0

    def __init__(self):
        self.prototypes = {}

    def classify(self, text):
        text_bigrams = _bigrams_of(text)
        scored = []
        for proto in self.prototypes.values():
            cov, comb = proto.score(text_bigrams)
            scored.append((proto.domain_id, cov, comb, proto.coverage_threshold))
        if not scored: return []
        scored.sort(key=lambda x: -x[1])
        top_dom, top_cov, top_comb, top_th = scored[0]
        sec_cov = scored[1][1] if len(scored) > 1 else 0.0
        above = [(d, c, cb) for d, c, cb, th in scored if c >= th]
        if not above: return []
        if top_cov >= sec_cov * self.DOMINANCE_RATIO and sec_cov > 0:
            return [top_dom]
        if sec_cov == 0:
            return [top_dom] if above else []
        return [d for d, _, _ in above]

    def match_scores(self, text):
        text_bigrams = _bigrams_of(text)
        return {p.domain_id: dict(coverage=round(p.coverage(text_bigrams),4),
                                   combined=round(p.score(text_bigrams)[1],4),
                                   pass_=p.coverage(text_bigrams) >= p.coverage_threshold,
                                   threshold=p.coverage_threshold)
                for p in self.prototypes.values()}

    def build_domain(self, domain_id, corpus_texts):
        if not corpus_texts: raise ValueError("empty corpus")
        all_bigrams = set()
        sample_bigrams = []
        for t in corpus_texts:
            bs = _bigrams_of(t)
            all_bigrams |= bs
            sample_bigrams.append(bs)
        coverages = []
        for i in range(len(corpus_texts)):
            loo = set()
            for j in range(len(corpus_texts)):
                if j != i: loo |= sample_bigrams[j]
            cov = len(sample_bigrams[i] & loo) / max(len(sample_bigrams[i]), 1)
            coverages.append(cov)
        cov_mean = sum(coverages) / len(coverages)
        cov_var = sum((x-cov_mean)**2 for x in coverages) / len(coverages)
        cov_std = math.sqrt(cov_var)
        cov_th = max(0.01, round(cov_mean - 2*cov_std, 4))
        comb_th = max(0.001, round(cov_mean * 0.5 * (cov_mean - 2*cov_std), 4))
        proto = LexicalPrototype(domain_id, all_bigrams, cov_th, comb_th, len(corpus_texts))
        self.prototypes[domain_id] = proto
        return proto

    def add_to_prototype(self, domain_id, text):
        if domain_id not in self.prototypes:
            raise KeyError(domain_id)
        self.prototypes[domain_id].add_sample(text)

    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({did: p.to_dict() for did, p in self.prototypes.items()}, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path):
        store = cls()
        if not os.path.isfile(path): return store
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for domain_id, d in data.items():
            if "domain_id" not in d: d["domain_id"] = domain_id
            if "coverage_threshold" not in d:
                if "thresholds" in d:
                    d["coverage_threshold"] = d["thresholds"].get("coverage_threshold", 0.15)
                    d["combined_threshold"] = d["thresholds"].get("combined_threshold", 0.001)
                else:
                    d["coverage_threshold"] = 0.15
                    d["combined_threshold"] = 0.001
            store.prototypes[domain_id] = LexicalPrototype(
                domain_id=d["domain_id"], bigrams=set(d.get("bigrams", [])),
                coverage_threshold=d.get("coverage_threshold", 0.15),
                combined_threshold=d.get("combined_threshold", 0.001),
                sample_count=d.get("sample_count", 0))
        return store
