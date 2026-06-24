"""AST-aware code checking engine — tree-sitter powered.

Adds condition type `ast_check` to the Citta Engine.
Sees code structure, not just text.
"""

from __future__ import annotations

import logging, re as _re
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

_TS_INIT = False
_JS_LANG = None


def _init():
    global _TS_INIT, _JS_LANG
    if _TS_INIT:
        return bool(_JS_LANG)
    _TS_INIT = True
    try:
        from tree_sitter import Language
        import tree_sitter_javascript as tjs
        _JS_LANG = Language(tjs.language())
    except Exception as e:
        logger.debug("tree-sitter unavailable: %s", e)
    return bool(_JS_LANG)


@dataclass
class ASTHit:
    node_type: str
    node_text: str
    line: int
    column: int
    context: str = ""


def _walk(node, limit: int = 50000):
    nodes = []
    stack = [node]
    while stack and len(nodes) < limit:
        n = stack.pop()
        nodes.append({
            "type": n.type,
            "text": n.text.decode("utf-8", errors="replace"),
            "row": n.start_point[0],
            "col": n.start_point[1],
            "child_count": n.child_count,
        })
        for child in reversed(n.children):
            stack.append(child)
    return nodes


def check_ast(code, language, check):
    """Run an AST check. Returns list of ASTHit."""
    if not _init():
        return []
    if language.lower() not in ("javascript", "js"):
        return []

    from tree_sitter import Parser
    p = Parser(_JS_LANG)
    tree = p.parse(code.encode("utf-8"))
    nodes = _walk(tree.root_node)
    lines = code.split("\n")

    st = check.get("search_type", "forbidden_node")
    ntype = check.get("node_type", "")
    npat = check.get("node_pattern", "")
    min_count = check.get("min_count", 1)

    pat = _re.compile(npat) if npat else None
    hits = []

    def _ctx(row, ctx=2):
        s = max(0, row - ctx)
        e = min(len(lines), row + ctx + 1)
        return "\n".join(f"  {i+1}: {lines[i]}" for i in range(s, e))

    if st == "forbidden_node":
        for n in nodes:
            if n["type"] != ntype:
                continue
            if pat and not pat.search(n["text"]):
                continue
            hits.append(ASTHit(n["type"], n["text"][:200], n["row"] + 1, n["col"] + 1, _ctx(n["row"])))

    elif st == "required_node":
        found = [n for n in nodes if n["type"] == ntype]
        if pat:
            found = [n for n in found if pat.search(n["text"])]
        if len(found) < min_count:
            hits.append(ASTHit(ntype, "", 0, 0, f"Expected {min_count} '{ntype}' node(s), found {len(found)}."))

    elif st == "forbidden_pattern":
        for n in nodes:
            if pat and pat.search(n["text"]):
                hits.append(ASTHit(n["type"], n["text"][:200], n["row"] + 1, n["col"] + 1, _ctx(n["row"])))

    elif st == "required_pattern":
        if pat and not any(pat.search(n["text"]) for n in nodes):
            hits.append(ASTHit("*", "", 0, 0, f"Required pattern '{npat}' not found."))

    return hits
