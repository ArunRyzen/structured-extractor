"""Build a configured provider and Extractor from Settings.

A single composition root keeps wiring decisions in one place. The CLI and the API both
call `build_extractor(settings)` instead of constructing providers themselves, so adding a
provider or changing a default happens here, not scattered across entry points.
"""

from __future__ import annotations

from structured_extractor.config import Settings
from structured_extractor.errors import ProviderError
from structured_extractor.extractor import Extractor
from structured_extractor.providers.base import LLMProvider


def build_provider(settings: Settings) -> LLMProvider:
    """Instantiate the provider named in settings, validating that creds exist.

    This is the ONE place where "which LLM backend?" is decided. Each branch checks the
    matching API key first so a missing key fails fast with a clear message, instead of
    a confusing HTTP 401 from deep inside an SDK.
    """
    if settings.provider == "gemini":
        if not settings.gemini_api_key:
            raise ProviderError("GEMINI_API_KEY is not set.")
        # Imported here (not at the top) so simply having this file on the import path
        # never forces the Gemini SDK to load — only choosing it does.
        from structured_extractor.providers.gemini_provider import GeminiProvider

        return GeminiProvider(
            model=settings.gemini_model,
            max_tokens=settings.max_tokens,
            api_key=settings.gemini_api_key,
        )
    if settings.provider == "anthropic":
        if not settings.anthropic_api_key:
            raise ProviderError("ANTHROPIC_API_KEY is not set.")
        from structured_extractor.providers.anthropic_provider import AnthropicProvider

        return AnthropicProvider(
            model=settings.anthropic_model,
            max_tokens=settings.max_tokens,
            api_key=settings.anthropic_api_key,
        )
    if settings.provider == "openai":
        if not settings.openai_api_key:
            raise ProviderError("OPENAI_API_KEY is not set.")
        from structured_extractor.providers.openai_provider import OpenAIProvider

        return OpenAIProvider(
            model=settings.openai_model,
            max_tokens=settings.max_tokens,
            api_key=settings.openai_api_key,
        )
    raise ProviderError(
        f"Unknown provider '{settings.provider}' (use 'gemini', 'anthropic', or 'openai')."
    )


def build_extractor(settings: Settings) -> Extractor:
    """Compose a ready-to-use Extractor from configuration."""
    return Extractor(build_provider(settings), max_retries=settings.max_retries)
