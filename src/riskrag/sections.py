"""Extract Item 1A (Risk Factors) from a 10-K and split it into individual risks.

10-K filings are HTML (often inline XBRL). Each risk factor usually starts with
a bold or italic heading sentence followed by body paragraphs; groups of risks
sit under short headers such as "Risks Related to Our Business". The parser
works on "leaf" text blocks and uses that structure, falling back to
fixed-size chunks when a filing has no detectable headings.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import asdict, dataclass

from bs4 import BeautifulSoup, NavigableString, Tag, XMLParsedAsHTMLWarning

# 10-K filings are inline-XBRL XHTML; parsing them as HTML is intended.
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

BLOCK_TAGS = {"p", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6", "td"}
START_RE = re.compile(r"^\s*item\s*1a\b", re.I)
END_RE = re.compile(r"^\s*item\s*(1b|1c|2)\b", re.I)
EMPH_STYLE_RE = re.compile(r"font-weight\s*:\s*(bold|[6-9]00)|font-style\s*:\s*italic", re.I)


@dataclass
class Block:
    text: str
    emphasized: bool


@dataclass
class RiskItem:
    id: str
    heading: str
    text: str
    group: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def full_text(self) -> str:
        return f"{self.heading}\n{self.text}".strip()


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace("\xa0", " ")).strip()


def _is_emphasized(node: Tag) -> bool:
    if node.name in {"b", "strong", "i", "em"}:
        return True
    return bool(EMPH_STYLE_RE.search(node.get("style", "")))


def _emphasis_share(block: Tag) -> float:
    total = emph = 0
    for s in block.find_all(string=True):
        if not isinstance(s, NavigableString):
            continue
        n = len(s.strip())
        if not n:
            continue
        total += n
        parent = s.parent
        while parent is not None and parent is not block.parent:
            if isinstance(parent, Tag) and _is_emphasized(parent):
                emph += n
                break
            parent = parent.parent
    return emph / total if total else 0.0


def html_blocks(html: str) -> list[Block]:
    soup = BeautifulSoup(html, "lxml")
    for t in soup(["script", "style", "ix:header"]):
        t.decompose()
    blocks = []
    for el in soup.find_all(BLOCK_TAGS):
        if el.find(BLOCK_TAGS):  # only leaf blocks
            continue
        text = _clean(el.get_text(" "))
        if not text or re.fullmatch(r"\d+|table of contents", text, re.I):
            continue
        blocks.append(Block(text, _emphasis_share(el) >= 0.9))
    return blocks


def risk_section(blocks: list[Block]) -> list[Block]:
    """The Item 1A span; the table of contents also mentions Item 1A, so the
    longest start-to-end span is taken as the real section."""
    starts = [i for i, b in enumerate(blocks) if START_RE.match(b.text)]
    best: tuple[int, int] = (0, 0)
    for s in starts:
        end = next((j for j in range(s + 1, len(blocks)) if END_RE.match(blocks[j].text)), len(blocks))
        span = sum(len(b.text) for b in blocks[s + 1 : end])
        if span > sum(len(b.text) for b in blocks[best[0] + 1 : best[1]]):
            best = (s, end)
    return blocks[best[0] + 1 : best[1]]


def split_risks(section: list[Block], prefix: str, fallback_chars: int = 1500) -> list[RiskItem]:
    def is_heading(b: Block) -> bool:
        return b.emphasized and 20 <= len(b.text) <= 700

    headings = [i for i, b in enumerate(section) if is_heading(b)]
    items: list[RiskItem] = []
    if len(headings) >= 5:
        group = ""
        for k, i in enumerate(headings):
            nxt = headings[k + 1] if k + 1 < len(headings) else len(section)
            body = " ".join(b.text for b in section[i + 1 : nxt])
            if not body:  # a header with no body directly above the next heading
                group = section[i].text
                continue
            items.append(RiskItem(f"{prefix}-{len(items) + 1:03d}", section[i].text, body, group))
    else:
        buf = ""
        for b in section:
            buf = f"{buf} {b.text}".strip()
            if len(buf) >= fallback_chars:
                items.append(RiskItem(f"{prefix}-{len(items) + 1:03d}", buf[:160] + "...", buf))
                buf = ""
        if buf:
            items.append(RiskItem(f"{prefix}-{len(items) + 1:03d}", buf[:160] + "...", buf))
    return items


def extract_risk_items(html: str, prefix: str) -> list[RiskItem]:
    section = risk_section(html_blocks(html))
    if not section:
        raise ValueError("Item 1A (Risk Factors) not found in document")
    return split_risks(section, prefix)
