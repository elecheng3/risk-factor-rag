"""Settings, read from environment variables (and a local .env file if present)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # python-dotenv is optional
    pass

DEFAULT_MODELS = {
    # Cheapest general-purpose model from each provider as of Sep 2026.
    # Override with OPENAI_MODEL / ANTHROPIC_MODEL / GEMINI_MODEL.
    "openai": "gpt-6-luna",
    "anthropic": "claude-haiku-4-5-20251001",
    "gemini": "gemini-3.5-flash-lite",
    "mock": "keyword-rules",
}


@dataclass
class Settings:
    provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "mock"))
    sec_user_agent: str = field(default_factory=lambda: os.getenv("SEC_USER_AGENT", ""))
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("RISKRAG_DATA_DIR", "data")))
    max_tokens: int = field(default_factory=lambda: int(os.getenv("LLM_MAX_TOKENS", "4000")))
    batch_size: int = field(default_factory=lambda: int(os.getenv("EXTRACT_BATCH_SIZE", "8")))

    def model_for(self, provider: str) -> str:
        return os.getenv(f"{provider.upper()}_MODEL", DEFAULT_MODELS[provider])

    def key_for(self, provider: str) -> str | None:
        names = {
            "openai": ["OPENAI_API_KEY"],
            "anthropic": ["ANTHROPIC_API_KEY"],
            "gemini": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
        }.get(provider, [])
        for n in names:
            if os.getenv(n):
                return os.getenv(n)
        return None

    def available_providers(self) -> list[str]:
        return [p for p in ("openai", "anthropic", "gemini") if self.key_for(p)]

    # ---- storage layout -------------------------------------------------
    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def filings_dir(self) -> Path:
        return self.data_dir / "filings"

    @property
    def extractions_dir(self) -> Path:
        return self.data_dir / "extractions"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "llm_cache"

    @property
    def usage_log(self) -> Path:
        return self.data_dir / "usage.jsonl"


def settings() -> Settings:
    return Settings()
