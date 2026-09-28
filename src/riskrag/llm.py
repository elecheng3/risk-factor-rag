"""One interface over OpenAI, Anthropic and Gemini, plus an offline mock.

Every call goes through `LLM.complete_json`, which
- returns parsed JSON validated against a pydantic model (one repair retry),
- caches responses on disk keyed by (provider, model, prompt), so re-running
  the pipeline never pays twice for the same call,
- appends token counts and latency to data/usage.jsonl.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from .config import Settings

T = TypeVar("T", bound=BaseModel)
PROVIDERS = ("openai", "anthropic", "gemini", "mock")


@dataclass
class Completion:
    text: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    cached: bool = False


class Backend:
    name = "base"

    def __init__(self, model: str, api_key: str | None):
        self.model = model
        self.api_key = api_key

    def complete(self, system: str, user: str, max_tokens: int) -> Completion:
        raise NotImplementedError


class OpenAIBackend(Backend):
    name = "openai"

    def __init__(self, model, api_key):
        super().__init__(model, api_key)
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key)

    def complete(self, system, user, max_tokens):
        t0 = time.perf_counter()
        r = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            response_format={"type": "json_object"},
            max_completion_tokens=max_tokens,
        )
        u = r.usage
        return Completion(r.choices[0].message.content or "", u.prompt_tokens, u.completion_tokens,
                          time.perf_counter() - t0)


class AnthropicBackend(Backend):
    name = "anthropic"

    def __init__(self, model, api_key):
        super().__init__(model, api_key)
        from anthropic import Anthropic

        self.client = Anthropic(api_key=api_key)

    def complete(self, system, user, max_tokens):
        t0 = time.perf_counter()
        r = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
        return Completion(text, r.usage.input_tokens, r.usage.output_tokens, time.perf_counter() - t0)


class GeminiBackend(Backend):
    name = "gemini"

    def __init__(self, model, api_key):
        super().__init__(model, api_key)
        from google import genai

        self.client = genai.Client(api_key=api_key)

    def complete(self, system, user, max_tokens):
        from google.genai import types

        t0 = time.perf_counter()
        r = self.client.models.generate_content(
            model=self.model,
            contents=user,
            config=types.GenerateContentConfig(
                system_instruction=system,
                max_output_tokens=max_tokens,
                response_mime_type="application/json",
            ),
        )
        u = r.usage_metadata
        return Completion(r.text or "", u.prompt_token_count or 0, u.candidates_token_count or 0,
                          time.perf_counter() - t0)


class MockBackend(Backend):
    """Deterministic keyword rules, so the whole pipeline runs offline and in
    tests without an API key. It is a stand-in, not a model."""

    name = "mock"
    RULES = [
        ("geopolitical_trade", r"export control|tariff|sanction|china|taiwan|trade restriction|geopolit"),
        ("cybersecurity", r"cyber|breach|ransomware|hack|information technology system"),
        ("supply_chain", r"supplier|supply chain|single source|foundr|subcontract|raw material|wafer supply"),
        ("customer_concentration", r"customer concentration|significant customer|largest customer|distributor"),
        ("ai_related", r"artificial intelligence|\bai\b|machine learning|generative"),
        ("regulatory_legal", r"regulat|litigation|intellectual property|patent|tax|compliance|lawsuit"),
        ("operational_manufacturing", r"manufactur|fab\b|yield|capacity|natural disaster|earthquake|plant"),
        ("macroeconomic_demand", r"demand|cyclical|economic|inflation|interest rate|currency|recession|inventory"),
        ("technology_competition", r"compet|pricing pressure|technolog|new product|r&d"),
        ("human_capital", r"employee|talent|personnel|workforce|retain"),
        ("financial_liquidity", r"debt|liquidity|impairment|credit|capital|cash"),
        ("esg_climate", r"climate|environmental|emission|sustainab"),
    ]

    def complete(self, system, user, max_tokens):
        if '"answer"' in system:  # question answering
            ids = re.findall(r"\[([A-Z0-9]+-\d{4}-\d{3})\]", user)[:2]
            ans = "Offline mock answer based on retrieved passages " + " ".join(f"[{i}]" for i in ids) + "."
            return Completion(json.dumps({"answer": ans, "citations": ids}), 0, 0, 0.0)
        items = json.loads(user[user.index("["): user.rindex("]") + 1])
        out = []
        for it in items:
            low = f"{it['heading']} {it['text']}".lower()
            hits = [c for c, pat in self.RULES if re.search(pat, low)]
            sev = 2 + min(2, len(re.findall(r"significant|material|substantial|adverse", low)) // 2)
            out.append({
                "id": it["id"],
                "category": hits[0] if hits else "other",
                "secondary_category": hits[1] if len(hits) > 1 else None,
                "severity": sev,
                "emerging": bool(re.search(r"artificial intelligence|new regulation|recent|newly", low)),
                "summary": it["heading"][:140],
            })
        return Completion(json.dumps({"risks": out}), 0, 0, 0.0)


BACKENDS = {"openai": OpenAIBackend, "anthropic": AnthropicBackend, "gemini": GeminiBackend, "mock": MockBackend}


def _parse_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            raise
        return json.loads(m.group(0))


class LLM:
    def __init__(self, provider: str, cfg: Settings):
        if provider not in BACKENDS:
            raise ValueError(f"provider must be one of {PROVIDERS}")
        key = cfg.key_for(provider)
        if provider != "mock" and not key:
            raise RuntimeError(f"No API key for {provider}; set it in .env (see .env.example)")
        self.provider = provider
        self.cfg = cfg
        self.backend = BACKENDS[provider](cfg.model_for(provider), key)
        cfg.cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def model(self) -> str:
        return self.backend.model

    def _cache_path(self, system: str, user: str) -> Path:
        h = hashlib.sha256(f"{self.provider}|{self.model}|{system}|{user}".encode()).hexdigest()[:24]
        return self.cfg.cache_dir / f"{self.provider}_{h}.json"

    def _log(self, task: str, c: Completion) -> None:
        self.cfg.usage_log.parent.mkdir(parents=True, exist_ok=True)
        with self.cfg.usage_log.open("a") as f:
            f.write(json.dumps({
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "task": task, "provider": self.provider,
                "model": self.model, "input_tokens": c.input_tokens, "output_tokens": c.output_tokens,
                "latency_s": round(c.latency_s, 3), "cached": c.cached,
            }) + "\n")

    def complete(self, system: str, user: str, task: str = "") -> Completion:
        path = self._cache_path(system, user)
        if path.exists():
            d = json.loads(path.read_text())
            c = Completion(d["text"], d["input_tokens"], d["output_tokens"], d["latency_s"], cached=True)
        else:
            c = self.backend.complete(system, user, self.cfg.max_tokens)
            path.write_text(json.dumps(c.__dict__))
        self._log(task, c)
        return c

    def complete_json(self, system: str, user: str, schema: type[T], task: str = "") -> tuple[T, Completion]:
        c = self.complete(system, user, task)
        try:
            return schema.model_validate(_parse_json(c.text)), c
        except (json.JSONDecodeError, ValidationError) as e:
            self._cache_path(system, user).unlink(missing_ok=True)  # never cache a bad answer
            repair = (f"{user}\n\nYour previous reply was not valid for the required JSON schema "
                      f"({str(e)[:300]}). Reply again with valid JSON only.")
            c2 = self.complete(system, repair, task + ":repair")
            return schema.model_validate(_parse_json(c2.text)), c2
