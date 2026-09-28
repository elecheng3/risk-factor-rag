"""Offline tests: parser, pipeline on the fictional demo filings, the three real
provider backends (with their SDK clients faked), caching and the API."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from riskrag import llm as llm_mod  # noqa: E402
from riskrag.analysis import category_table, compare, risk_changes, weighted_index  # noqa: E402
from riskrag.config import Settings  # noqa: E402
from riskrag.demo import filings, load_demo  # noqa: E402
from riskrag.extract import ExtractionBatch, extract_items, extract_ticker  # noqa: E402
from riskrag.qa import ask  # noqa: E402
from riskrag.sections import extract_risk_items  # noqa: E402


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    for k in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("RISKRAG_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    c = Settings()
    load_demo(c)
    return c


# ---- parsing -----------------------------------------------------------------

def test_parser_skips_table_of_contents_and_finds_groups():
    meta, html = filings()[0]  # ACME 2024
    items = extract_risk_items(html, "ACME-2024")
    assert len(items) == 11
    assert items[0].group == "Risks Related to Our Industry"
    assert items[0].heading.startswith("Demand for our products is cyclical")
    assert all(i.text for i in items)
    assert not any("Unresolved Staff Comments" in i.text for i in items)


def test_parser_falls_back_to_chunks_without_headings():
    body = " ".join(f"<p>Plain paragraph {n} about supply and demand risks. " * 20 + "</p>" for n in range(6))
    html = f"<html><body><p>Item 1A. Risk Factors</p>{body}<p>Item 2. Properties</p></body></html>"
    items = extract_risk_items(html, "X-2025")
    assert len(items) >= 2 and all(len(i.text) > 0 for i in items)


def test_missing_section_raises():
    with pytest.raises(ValueError):
        extract_risk_items("<html><body><p>No risk section here.</p></body></html>", "X-2025")


# ---- pipeline ------------------------------------------------------------------

def test_extract_compare_and_changes(cfg):
    for t in ("ACME", "BORL"):
        out = extract_ticker(cfg, t, "mock")
        assert len(out["risks"]) == len({r["id"] for r in out["risks"]})
    r = compare(cfg, ["ACME", "BORL"], "mock")
    assert set(r["weighted_index"].index) == {"ACME", "BORL"}
    assert r["items"].loc["geopolitical_trade", "ACME"] >= 1
    ch = risk_changes(cfg, "ACME")
    added = ch[ch.change == "added"].heading.str.cat()
    assert "export control" in added and "artificial intelligence" in added
    assert "intellectual property" in ch[ch.change == "removed"].heading.str.cat()


def test_weighted_index_bounds_and_profile_effect(cfg):
    cat = category_table(extract_ticker(cfg, "ACME", "mock"))
    semi, neutral = weighted_index(cat, "semiconductor"), weighted_index(cat, "neutral")
    assert 0 <= semi <= 100 and 0 <= neutral <= 100
    assert semi != neutral


def test_ask_returns_only_retrieved_citations(cfg):
    a = ask(cfg, "export controls China", ["ACME", "BORL"], "mock")
    assert a["citations"] and not a["invalid_citations"]
    assert a["sources"][0]["id"] == "ACME-2025-001"


def test_missing_items_are_flagged_not_dropped(cfg, monkeypatch):
    class Partial(llm_mod.Backend):
        name = "mock"

        def complete(self, system, user, max_tokens):
            items = json.loads(user[user.index("["): user.rindex("]") + 1])
            first = items[0]
            return llm_mod.Completion(json.dumps({"risks": [{"id": first["id"], "category": "cybersecurity",
                                                             "severity": 3, "summary": "x"}]}), 1, 1, 0.0)

    llm = llm_mod.LLM("mock", cfg)
    llm.backend = Partial("m", None)
    items = [{"id": f"T-2025-00{i}", "heading": "h", "text": "t"} for i in range(1, 4)]
    res, _ = extract_items(llm, items, batch_size=8)
    assert [r["id"] for r in res] == [i["id"] for i in items]
    assert sum(bool(r.get("missing")) for r in res) == 2


# ---- the three real backends, with SDK clients faked ----------------------------------

PAYLOAD = json.dumps({"risks": [{"id": "A-2025-001", "category": "supply_chain", "secondary_category": "",
                                 "severity": 4, "emerging": False, "summary": "Foundry dependence"}]})


def _fake(provider):
    calls = {}
    if provider == "openai":
        def create(**kw):
            calls.update(kw)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=PAYLOAD))],
                                   usage=SimpleNamespace(prompt_tokens=120, completion_tokens=40))
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    elif provider == "anthropic":
        def create(**kw):
            calls.update(kw)
            return SimpleNamespace(content=[SimpleNamespace(type="text", text="```json\n" + PAYLOAD + "\n```")],
                                   usage=SimpleNamespace(input_tokens=130, output_tokens=45))
        client = SimpleNamespace(messages=SimpleNamespace(create=create))
    else:
        def generate_content(**kw):
            calls.update(kw)
            return SimpleNamespace(text=PAYLOAD, usage_metadata=SimpleNamespace(prompt_token_count=110,
                                                                                candidates_token_count=35))
        client = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
    return client, calls


@pytest.mark.parametrize("provider", ["openai", "anthropic", "gemini"])
def test_real_backends_build_requests_and_parse_usage(cfg, monkeypatch, provider):
    monkeypatch.setenv({"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY",
                        "gemini": "GEMINI_API_KEY"}[provider], "test-key")
    llm = llm_mod.LLM(provider, cfg)
    client, calls = _fake(provider)
    llm.backend.client = client
    parsed, c = llm.complete_json("system JSON", "user prompt", ExtractionBatch, task="t")
    assert parsed.risks[0].category == "supply_chain" and parsed.risks[0].secondary_category is None
    assert c.input_tokens > 0 and c.output_tokens > 0
    assert calls["model"] == cfg.model_for(provider)
    if provider == "openai":
        assert calls["response_format"] == {"type": "json_object"} and calls["messages"][0]["role"] == "system"
    if provider == "anthropic":
        assert calls["system"] == "system JSON" and "temperature" not in calls
    if provider == "gemini":
        assert calls["config"].system_instruction == "system JSON"
        assert calls["config"].response_mime_type == "application/json"
    # second identical call is served from the cache, not the API
    calls.clear()
    _, c2 = llm.complete_json("system JSON", "user prompt", ExtractionBatch, task="t")
    assert c2.cached and not calls
    log = [json.loads(line) for line in cfg.usage_log.read_text().splitlines()]
    assert log[-1]["cached"] and log[-2]["provider"] == provider


def test_invalid_json_triggers_one_repair_call(cfg, monkeypatch):
    replies = iter(['{"risks": [{"id": "A-2025-001", "category": "not_a_category", "severity": 9, "summary": "x"}]}',
                    PAYLOAD])

    class Flaky(llm_mod.Backend):
        def complete(self, system, user, max_tokens):
            return llm_mod.Completion(next(replies), 10, 10, 0.1)

    llm = llm_mod.LLM("mock", cfg)
    llm.backend = Flaky("m", None)
    parsed, _ = llm.complete_json("s", "u", ExtractionBatch, task="t")
    assert parsed.risks[0].severity == 4
    assert not any(p.read_text().count("not_a_category") for p in cfg.cache_dir.glob("*.json"))


def test_missing_key_is_a_clear_error(cfg):
    with pytest.raises(RuntimeError, match="No API key for openai"):
        llm_mod.LLM("openai", cfg)


# ---- API -------------------------------------------------------------------------

def test_api_endpoints(cfg):
    from fastapi.testclient import TestClient
    from riskrag.api import app

    c = TestClient(app)
    assert c.get("/health").json()["status"] == "ok"
    assert c.get("/companies").json()["tickers"] == ["ACME", "BORL"]
    assert c.post("/extract", json={"tickers": ["ACME", "BORL"], "provider": "mock"}).status_code == 200
    cmp_ = c.get("/compare", params={"tickers": "ACME,BORL", "provider": "mock"}).json()
    assert set(cmp_["weighted_index"]) == {"ACME", "BORL"}
    assert c.get("/risks/ZZZZ", params={"provider": "mock"}).status_code == 404
    assert c.get("/compare", params={"tickers": "ACME", "profile": "bogus"}).status_code == 400
    a = c.post("/ask", json={"question": "export controls", "tickers": ["ACME"], "provider": "mock"}).json()
    assert a["citations"]


def test_provider_comparison_metrics(cfg, monkeypatch, tmp_path):
    from riskrag import config as config_mod
    from riskrag.evaluate import compare_providers, save_comparison

    class AlwaysOther(llm_mod.MockBackend):
        name = "other"

        def complete(self, system, user, max_tokens):
            c = super().complete(system, user, max_tokens)
            d = json.loads(c.text)
            for r in d["risks"]:
                r["category"] = "other"
            return llm_mod.Completion(json.dumps(d), 50, 20, 0.2)

    monkeypatch.setitem(llm_mod.BACKENDS, "other", AlwaysOther)
    monkeypatch.setitem(config_mod.DEFAULT_MODELS, "other", "always-other")
    monkeypatch.setattr(config_mod.Settings, "key_for", lambda self, p: "test-key")
    labels = tmp_path / "labels.csv"
    labels.write_text("id,category,severity\nACME-2025-001,geopolitical_trade,5\nACME-2025-002,ai_related,4\n")
    res = compare_providers(cfg, ["ACME"], ["mock", "other"], labels=labels)
    pair = res["agreement"].iloc[0]
    assert pair["items"] == 12 and 0 <= pair["category_agreement"] < 0.5
    human = res["vs_human"].set_index("provider")
    assert human.loc["mock", "category_accuracy"] == 1.0 and human.loc["other", "category_accuracy"] == 0.0
    assert res["usage"].set_index("provider").loc["other", "input_tokens"] > 0
    assert "## agreement" in save_comparison(res, tmp_path / "out").read_text()


def test_group_header_glued_to_first_heading_is_split():
    risks = [("Legal and regulatory risks Our operations could be affected by complex laws and regulations.",
              "Body about laws."),
             ("Cybersecurity risks could disrupt our operations and expose confidential information.", "Body."),
             ("We face risks related to our international operations and trade restrictions.", "Body."),
             ("Our debt could affect our operations and financial condition going forward.", "Body."),
             ("Material impairments of our goodwill could adversely affect our results.", "Body.")]
    paras = "".join(f'<p><b>{h}</b></p><p>{b}</p>' for h, b in risks)
    html = f"<html><body><p>Item 1A. Risk Factors</p>{paras}<p>Item 1B. Unresolved Staff Comments</p></body></html>"
    items = extract_risk_items(html, "T-2025")
    assert items[0].group == "Legal and regulatory risks"
    assert items[0].heading.startswith("Our operations could be affected")
    assert items[1].heading.startswith("Cybersecurity risks could")  # a sentence, not a header
    assert items[2].heading.startswith("We face risks related")
    assert all(i.group == "Legal and regulatory risks" for i in items)
