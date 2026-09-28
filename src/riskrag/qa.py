"""Retrieval-augmented question answering over ingested risk factors.

Retrieval uses TF-IDF vectors (scikit-learn), which works the same whichever
LLM provider is selected (Anthropic offers no embeddings endpoint). The model
must cite the passage ids it used; citations that were not in the retrieved
set are reported as invalid instead of being shown to the user as sources.
"""

from __future__ import annotations

import re

from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .config import Settings
from .llm import LLM
from .store import latest


class Answer(BaseModel):
    answer: str
    citations: list[str]


SYSTEM = """You answer questions about companies' disclosed risk factors using ONLY the
numbered passages provided. Cite passages inline as [ID] using their exact ids.
If the passages do not contain the answer, say so. Keep the answer under 200 words.
Return JSON only: {"answer": "...", "citations": ["ID", ...]}"""


class Index:
    def __init__(self, cfg: Settings, tickers: list[str]):
        self.items = [dict(i, ticker=t.upper()) for t in tickers for i in latest(cfg, t)["items"]]
        if not self.items:
            raise LookupError("No risk items to search")
        self.vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        self.matrix = self.vec.fit_transform(f"{i['heading']} {i['text']}" for i in self.items)

    def search(self, query: str, k: int = 8, per_company: int | None = None) -> list[dict]:
        scores = cosine_similarity(self.vec.transform([query]), self.matrix)[0]
        order = scores.argsort()[::-1]
        out, count = [], {}
        for idx in order:
            it = self.items[idx]
            if per_company and count.get(it["ticker"], 0) >= per_company:
                continue
            count[it["ticker"]] = count.get(it["ticker"], 0) + 1
            out.append(dict(it, score=round(float(scores[idx]), 3)))
            if len(out) == k:
                break
        return out


def ask(cfg: Settings, question: str, tickers: list[str], provider: str, k: int = 8) -> dict:
    index = Index(cfg, tickers)
    # balance evidence across companies so comparisons are not one-sided
    hits = index.search(question, k=k, per_company=max(2, k // max(1, len(tickers))))
    passages = "\n\n".join(f"[{h['id']}] ({h['ticker']}) {h['heading']}\n{h['text'][:1200]}" for h in hits)
    llm = LLM(provider, cfg)
    ans, c = llm.complete_json(SYSTEM, f"Question: {question}\n\nPassages:\n{passages}", Answer, task="ask")
    retrieved = {h["id"] for h in hits}
    inline = set(re.findall(r"\[([A-Z0-9.]+-\d{4}-\d{3})\]", ans.answer))
    cited = set(ans.citations) | inline
    return {
        "question": question,
        "answer": ans.answer,
        "citations": sorted(cited & retrieved),
        "invalid_citations": sorted(cited - retrieved),
        "sources": [{k2: h[k2] for k2 in ("id", "ticker", "heading", "score")} for h in hits if h["id"] in cited],
        "provider": provider,
        "model": llm.model,
        "tokens": {"input": c.input_tokens, "output": c.output_tokens},
    }
