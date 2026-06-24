"""Book chapter splitter — slices long documents into structured sections.

Detects headings by number patterns, Chinese legal markers, and layout
conventions.  Outputs a list of {title, body} dicts ready for domain
classification and rule extraction.
"""

import re
from dataclasses import dataclass


@dataclass
class Chapter:
    """A single chapter/section extracted from a document."""
    index: int
    title: str = ""
    body: str = ""
    level: int = 1
    heading_raw: str = ""
    char_count: int = 0


HEADING_PATTERNS = [
    (r'^第[一二三四五六七八九十百千\d]+[章节]', "chinese_chapter"),
    (r'^[一二三四五六七八九十]+、', "chinese_numbered"),
    (r'^\d+(?:\.\d+)*\.?\s+', "arabic_numbered"),
    (r'^[IVXLCDM]+\.\s+', "roman_numbered"),
    (r'^(总则|分则|附则|通则|补则|定义|术语)', "chinese_section"),
    (r'^[A-Z][A-Z\s/-]{3,60}$', "all_caps"),
    (r'^#{1,6}\s+', "markdown_atx"),
]


class ChapterSplitter:
    """Split a long document into chapters/sections by heading detection.

    Strategy:
    1. Scan line-by-line looking for heading patterns.
    2. Each heading starts a new chapter; body text accumulates until the
       next heading.
    3. Chinese "第X条" (article) lines are NOT headings — they are body
       content within a chapter.

    Falls back to blank-line paragraph splitting if no headings found.
    """

    def __init__(self, min_chapter_chars: int = 50, max_chapters: int = 60):
        self.min_chapter_chars = min_chapter_chars
        self.max_chapters = max_chapters

    def split(self, text: str) -> list[Chapter]:
        """Split text into chapters. Falls back if no headings found."""
        if not text or not text.strip():
            return []

        lines = text.splitlines()
        raw = self._split_by_headings(lines)
        if not raw or len(raw) <= 1:
            raw = self._split_by_blank_lines(text)

        chapters: list[Chapter] = []
        for i, (title, heading_raw, body, level) in enumerate(raw):
            body = body.strip()
            if len(body) < self.min_chapter_chars:
                continue
            chapters.append(Chapter(
                index=i,
                title=title,
                body=body,
                level=level,
                heading_raw=heading_raw,
                char_count=len(body),
            ))
            if len(chapters) >= self.max_chapters:
                break

        return chapters

    def _split_by_headings(self, lines: list[str]) -> list[tuple[str, str, str, int]]:
        heading_spans = self._find_headings(lines)
        if not heading_spans:
            return []

        result = []
        for idx, (lineno, pattern_type, raw_line, level) in enumerate(heading_spans):
            clean_title = self._clean_title(raw_line, pattern_type)
            body_start = lineno + 1
            body_end = len(lines)

            for j in range(idx + 1, len(heading_spans)):
                next_level = heading_spans[j][3]
                if next_level <= level:
                    body_end = heading_spans[j][0]
                    break

            body_lines = [l for l in lines[body_start:body_end] if l.strip()]
            body = "\n".join(body_lines)
            result.append((clean_title, raw_line, body, level))

        return result

    def _find_headings(self, lines: list[str]) -> list[tuple[int, str, str, int]]:
        found = []
        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                continue
            if len(stripped) > 120:
                continue

            for pattern, ptype in HEADING_PATTERNS:
                m = re.match(pattern, stripped)
                if not m:
                    continue

                level = 1
                if ptype == "markdown_atx":
                    prefix = m.group()
                    level = len(prefix.rstrip())
                elif ptype == "arabic_numbered":
                    dots = m.group().count(".")
                    if dots >= 2:
                        level = 3
                    elif dots == 1:
                        level = 2

                found.append((i, ptype, stripped, level))
                break

        return found

    def _split_by_blank_lines(self, text: str) -> list[tuple[str, str, str, int]]:
        blocks = re.split(r'\n\s*\n', text)
        result = []
        for block in blocks:
            block = block.strip()
            if len(block) < self.min_chapter_chars:
                continue
            lines = block.splitlines()
            if len(lines) == 1:
                title = ""
                body = lines[0]
                heading_raw = ""
            else:
                title = lines[0].strip().rstrip("。；;.,")
                body = "\n".join(lines[1:])
                heading_raw = lines[0]
            result.append((title, heading_raw, body, 1))
        return result

    # Per-type title cleaners
    _CLEANERS = {
        "arabic_numbered": re.compile(r'^\d+(?:\.\d+)*\.?\s+'),
        "chinese_chapter": re.compile(r'^第[一二三四五六七八九十百千\d]+[章节]\s*'),
        "chinese_numbered": re.compile(r'^[一二三四五六七八九十]+、\s*'),
        "roman_numbered": re.compile(r'^[IVXLCDM]+\.\s+'),
        "markdown_atx": re.compile(r'^#{1,6}\s+'),
    }

    @classmethod
    def _clean_title(cls, raw: str, pattern_type: str) -> str:
        cleaner = cls._CLEANERS.get(pattern_type)
        if cleaner:
            cleaned = cleaner.sub('', raw).strip()
        else:
            cleaned = raw.strip()
        return cleaned or raw.strip()
