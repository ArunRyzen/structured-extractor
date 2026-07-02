"""Configuration loaded from the environment (12-factor style).

Why a dedicated settings object instead of reading os.environ inline: it gives one
typed, validated place for configuration, makes defaults explicit, and keeps secrets
out of code. `pydantic-settings` reads a local `.env` automatically for dev.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration. Values come from env vars or a local `.env` file."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Provider selection: "gemini", "anthropic", or "openai".
    # Gemini is the default because it has a generous free tier — you can try live
    # extraction with just a GEMINI_API_KEY and no billing set up.
    provider: str = Field(default="gemini")

    # Credentials (never hard-code these; they load from the environment).
    # Only the key for the provider you actually use needs to be set.
    gemini_api_key: str | None = Field(default=None)
    anthropic_api_key: str | None = Field(default=None)
    openai_api_key: str | None = Field(default=None)

    # Models. The Anthropic default follows Anthropic's guidance (Opus 4.8); for
    # high-volume, cost-sensitive extraction, switch to "claude-haiku-4-5" — see the
    # cost trade-off discussion in docs/architecture.md.
    gemini_model: str = Field(default="gemini-2.5-flash")
    anthropic_model: str = Field(default="claude-opus-4-8")
    openai_model: str = Field(default="gpt-4o")

    # Generation + reliability knobs.
    max_tokens: int = Field(default=4096)
    max_retries: int = Field(default=2)


def load_settings() -> Settings:
    """Build a Settings instance. Factored out so tests can construct their own."""
    return Settings()
