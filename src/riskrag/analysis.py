"""Scoring, peer comparison and year-over-year change detection."""

from __future__ import annotations

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .config import Settings
from .extract import load_extraction
from .store import load_filings
from .taxonomy import CATEGORIES, PROFILES


def category_table(extraction: dict) -> pd.DataFrame:
    """Per category: number of risk items, mean and max severity."""
    df = pd.DataFrame(extraction["risks"])
    g = df.groupby("category")["severity"].agg(items="count", mean_severity="mean", max_severity="max")
    return g.reindex(list(CATEGORIES)).fillna({"items": 0}).reset_index()


def weighted_index(cat: pd.DataFrame, profile: str = "semiconductor", full_coverage: int = 3) -> float:
    """0-100 risk index.

    index = 100 * sum_c w_c * (mean_sev_c / 5) * coverage_c / sum_c w_c
    coverage_c = min(1, items_c / full_coverage): a category disclosed through
    three or more separate risk items counts fully, one item counts a third.
    """
    w = PROFILES[profile]
    num = den = 0.0
    for _, r in cat.iterrows():
        wc = w.get(r["category"], 1.0)
        den += wc
        if r["items"] > 0:
            num += wc * (r["mean_severity"] / 5) * min(1.0, r["items"] / full_coverage)
    return round(100 * num / den, 1) if den else 0.0


def compare(cfg: Settings, tickers: list[str], provider: str, profile: str = "semiconductor") -> dict:
    """Company x category matrices plus the weighted index for each company."""
    items, severity, index = {}, {}, {}
    for t in tickers:
        cat = category_table(load_extraction(cfg, t, provider))
        items[t] = cat.set_index("category")["items"]
        severity[t] = cat.set_index("category")["mean_severity"]
        index[t] = weighted_index(cat, profile)
    return {
        "items": pd.DataFrame(items).fillna(0).astype(int),
        "mean_severity": pd.DataFrame(severity).round(2),
        "weighted_index": pd.Series(index, name=f"index_{profile}").sort_values(ascending=False),
    }


def risk_changes(cfg: Settings, ticker: str, threshold: float = 0.5) -> pd.DataFrame:
    """Risks added to (or dropped from) the latest 10-K versus the prior one.

    Headings are compared with TF-IDF cosine similarity; a heading whose best
    match in the other year is below `threshold` counts as added or removed
    (a substantially rewritten risk shows up as both)."""
    filings = load_filings(cfg, ticker)
    if len(filings) < 2:
        raise LookupError(f"Need two 10-Ks for {ticker}; run: python -m riskrag ingest {ticker} --years 2")
    cur, prev = filings[0]["items"], filings[1]["items"]
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2)).fit(
        [i["heading"] for i in cur + prev])
    sim = cosine_similarity(vec.transform([i["heading"] for i in cur]), vec.transform([i["heading"] for i in prev]))
    rows = []
    for k, c in enumerate(cur):
        if sim[k].max() < threshold:
            rows.append({"change": "added", "id": c["id"], "heading": c["heading"],
                         "best_match_similarity": round(float(sim[k].max()), 2)})
    for k, p in enumerate(prev):
        if sim[:, k].max() < threshold:
            rows.append({"change": "removed", "id": p["id"], "heading": p["heading"],
                         "best_match_similarity": round(float(sim[:, k].max()), 2)})
    return pd.DataFrame(rows, columns=["change", "id", "heading", "best_match_similarity"])
