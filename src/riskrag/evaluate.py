"""Run the same extraction through several providers and compare them.

Reports, per provider: tokens, latency and missing items; per provider pair:
category agreement, Cohen's kappa and mean absolute severity difference.
If a hand-labelled file (id,category,severity) is given, each provider is also
scored against it.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score

from .config import Settings
from .extract import extract_ticker


def compare_providers(cfg: Settings, tickers: list[str], providers: list[str], limit: int | None = None,
                      labels: Path | None = None) -> dict:
    runs: dict[str, pd.DataFrame] = {}
    usage_rows = []
    for p in providers:
        frames = []
        for t in tickers:
            out = extract_ticker(cfg, t, p, limit)
            frames.append(pd.DataFrame(out["risks"]))
            u = out["usage"]
            usage_rows.append({"provider": p, "model": out["model"], "ticker": t, **u})
        df = pd.concat(frames).set_index("id")
        if "missing" not in df:
            df["missing"] = False
        runs[p] = df

    usage = (pd.DataFrame(usage_rows).groupby(["provider", "model"], as_index=False)
             [["calls", "input_tokens", "output_tokens", "latency_s", "cached_calls"]].sum())
    usage["items"] = usage["provider"].map(lambda p: len(runs[p]))
    usage["missing_items"] = usage["provider"].map(lambda p: int(runs[p]["missing"].fillna(False).sum()))
    usage["latency_per_item_s"] = (usage["latency_s"] / usage["items"]).round(3)

    pairs = []
    for a, b in itertools.combinations(providers, 2):
        ids = runs[a].index.intersection(runs[b].index)
        ca, cb = runs[a].loc[ids, "category"], runs[b].loc[ids, "category"]
        pairs.append({
            "pair": f"{a} vs {b}",
            "items": len(ids),
            "category_agreement": round(float((ca == cb).mean()), 3),
            "cohens_kappa": round(float(cohen_kappa_score(ca, cb)), 3) if ca.nunique() > 1 or cb.nunique() > 1 else None,
            "severity_mae": round(float((runs[a].loc[ids, "severity"] - runs[b].loc[ids, "severity"]).abs().mean()), 2),
        })

    human = []
    if labels and Path(labels).exists():
        gold = pd.read_csv(labels).set_index("id")
        for p, df in runs.items():
            ids = gold.index.intersection(df.index)
            if len(ids):
                human.append({
                    "provider": p, "labelled_items": len(ids),
                    "category_accuracy": round(float((df.loc[ids, "category"] == gold.loc[ids, "category"]).mean()), 3),
                    "severity_mae": round(float((df.loc[ids, "severity"] - gold.loc[ids, "severity"]).abs().mean()), 2),
                })

    result = {"usage": usage, "agreement": pd.DataFrame(pairs), "vs_human": pd.DataFrame(human),
              "categories": pd.DataFrame({p: df["category"] for p, df in runs.items()})}
    return result


def save_comparison(result: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in result.items():
        df.to_csv(out_dir / f"provider_{name}.csv")
    md = ["# Provider comparison", ""]
    for name in ("usage", "agreement", "vs_human"):
        if len(result[name]):
            md += [f"## {name}", "", result[name].to_markdown(index=False), ""]
    path = out_dir / "provider_comparison.md"
    path.write_text("\n".join(md))
    return path
