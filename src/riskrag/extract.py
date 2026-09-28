"""LLM extraction: classify each risk item into the taxonomy with a severity."""

from __future__ import annotations

import json
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

from .config import Settings
from .llm import LLM
from .store import latest
from .taxonomy import CATEGORIES, SEVERITY_RUBRIC

Category = Literal[tuple(CATEGORIES)]  # type: ignore[valid-type]


class ExtractedRisk(BaseModel):
    id: str
    category: Category
    secondary_category: Optional[Category] = None
    severity: int = Field(ge=1, le=5)
    emerging: bool = False
    summary: str

    @field_validator("secondary_category", mode="before")
    @classmethod
    def _blank_to_none(cls, v):
        return v or None


class ExtractionBatch(BaseModel):
    risks: list[ExtractedRisk]


SYSTEM = f"""You are a risk analyst classifying risk factors disclosed in SEC 10-K filings.
For every risk item you receive, return one JSON object. Use only these category keys:
{json.dumps(CATEGORIES, indent=1)}

{SEVERITY_RUBRIC}

Mark "emerging": true only if the text describes a new or rapidly growing risk
(new regulation, new technology, a recent event), not a long-standing one.
"summary" is at most 25 words in your own words.
Return JSON only: {{"risks": [{{"id", "category", "secondary_category", "severity", "emerging", "summary"}}]}}
with exactly one entry per input id."""


def _prompt(items: list[dict], max_chars: int = 1800) -> str:
    payload = [{"id": i["id"], "heading": i["heading"], "text": i["text"][:max_chars]} for i in items]
    return "Classify these risk items:\n" + json.dumps(payload, ensure_ascii=False)


def extract_items(llm: LLM, items: list[dict], batch_size: int = 8) -> tuple[list[dict], dict]:
    results: list[dict] = []
    usage = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "latency_s": 0.0, "cached_calls": 0}
    for k in range(0, len(items), batch_size):
        batch = items[k : k + batch_size]
        parsed, c = llm.complete_json(SYSTEM, _prompt(batch), ExtractionBatch, task="extract")
        got = {r.id: r for r in parsed.risks}
        for it in batch:  # keep input order; flag anything the model skipped
            r = got.get(it["id"])
            results.append(r.model_dump() if r else {"id": it["id"], "category": "other", "secondary_category": None,
                                                      "severity": 1, "emerging": False, "summary": "(not returned by model)",
                                                      "missing": True})
        usage["calls"] += 1
        usage["input_tokens"] += c.input_tokens
        usage["output_tokens"] += c.output_tokens
        usage["latency_s"] += c.latency_s
        usage["cached_calls"] += int(c.cached)
    return results, usage


def extract_ticker(cfg: Settings, ticker: str, provider: str, limit: int | None = None) -> dict:
    filing = latest(cfg, ticker)
    items = filing["items"][:limit] if limit else filing["items"]
    llm = LLM(provider, cfg)
    risks, usage = extract_items(llm, items, cfg.batch_size)
    out = {"filing": filing["filing"], "provider": provider, "model": llm.model, "usage": usage, "risks": risks}
    path = cfg.extractions_dir / provider / f"{ticker.upper()}_{filing['filing']['accession']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    return out


def load_extraction(cfg: Settings, ticker: str, provider: str) -> dict:
    filing = latest(cfg, ticker)
    path = cfg.extractions_dir / provider / f"{ticker.upper()}_{filing['filing']['accession']}.json"
    if not path.exists():
        raise LookupError(f"No {provider} extraction for {ticker}; run: python -m riskrag extract {ticker} --provider {provider}")
    return json.loads(path.read_text())
