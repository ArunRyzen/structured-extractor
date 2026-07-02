"""GeminiProvider unit tests — fully offline, mirroring the fake-provider philosophy.

We never talk to Google here. The provider is built with a dummy key, then its client
is swapped for a `MagicMock`, so the tests pin down *our* wiring: what we send in the
request config (JSON mode + schema + temperature 0) and how we translate the SDK's
response — or its failures — into the provider contract.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from structured_extractor.config import Settings
from structured_extractor.errors import ProviderError
from structured_extractor.factory import build_provider
from structured_extractor.providers.gemini_provider import GeminiProvider
from structured_extractor.schemas import ContactInfo


def _provider_with_mock_client(
    response: Any = None, error: Exception | None = None
) -> tuple[GeminiProvider, MagicMock]:
    """Build a real GeminiProvider, then replace its network client with a mock."""
    provider = GeminiProvider(model="gemini-2.5-flash", max_tokens=4096, api_key="test-key")
    mock_client = MagicMock()
    if error is not None:
        mock_client.models.generate_content.side_effect = error
    else:
        mock_client.models.generate_content.return_value = response
    provider._client = mock_client
    return provider, mock_client


def _fake_response(
    parsed: Any,
    prompt_tokens: int | None = 100,
    output_tokens: int | None = 25,
    with_usage: bool = True,
) -> SimpleNamespace:
    """The minimal shape of a google-genai GenerateContentResponse that we rely on."""
    usage = (
        SimpleNamespace(prompt_token_count=prompt_tokens, candidates_token_count=output_tokens)
        if with_usage
        else None
    )
    return SimpleNamespace(parsed=parsed, usage_metadata=usage)


def test_extract_returns_validated_object_and_usage() -> None:
    contact = ContactInfo(name="Ada Lovelace", email="ada@example.com")
    provider, client = _provider_with_mock_client(response=_fake_response(contact))

    result = provider.extract(text="Ada, ada@example.com", schema=ContactInfo, instructions="sys")

    assert result.data is contact
    assert result.usage.model == "gemini-2.5-flash"
    assert result.usage.input_tokens == 100
    assert result.usage.output_tokens == 25

    # Verify the request actually enforced structured output.
    kwargs = client.models.generate_content.call_args.kwargs
    assert kwargs["model"] == "gemini-2.5-flash"
    assert kwargs["contents"] == "Ada, ada@example.com"
    config = kwargs["config"]
    assert config.response_mime_type == "application/json"
    assert config.response_schema is ContactInfo
    assert config.temperature == 0
    assert config.system_instruction == "sys"
    assert config.max_output_tokens == 4096


def test_api_error_becomes_provider_error() -> None:
    from google.genai import errors

    api_error = errors.APIError(429, {"error": {"message": "quota exhausted"}})
    provider, _ = _provider_with_mock_client(error=api_error)

    with pytest.raises(ProviderError, match="Gemini request failed"):
        provider.extract(text="x", schema=ContactInfo, instructions="sys")


def test_missing_parsed_output_raises_provider_error() -> None:
    # `parsed is None` happens when the model's JSON was truncated or unparseable.
    provider, _ = _provider_with_mock_client(response=_fake_response(parsed=None))

    with pytest.raises(ProviderError, match="no parseable structured output"):
        provider.extract(text="x", schema=ContactInfo, instructions="sys")


def test_missing_usage_metadata_defaults_to_zero_tokens() -> None:
    contact = ContactInfo(name="Grace Hopper")
    provider, _ = _provider_with_mock_client(response=_fake_response(contact, with_usage=False))

    result = provider.extract(text="x", schema=ContactInfo, instructions="sys")

    assert result.usage.input_tokens == 0
    assert result.usage.output_tokens == 0


def test_none_token_counts_default_to_zero() -> None:
    # The SDK types token counts as Optional; our accounting must not blow up on None.
    contact = ContactInfo(name="Grace Hopper")
    provider, _ = _provider_with_mock_client(
        response=_fake_response(contact, prompt_tokens=None, output_tokens=None)
    )

    result = provider.extract(text="x", schema=ContactInfo, instructions="sys")

    assert result.usage.total_tokens == 0


def _make_settings(**overrides: Any) -> Settings:
    # `_env_file=None` keeps tests hermetic even if a developer has a local .env.
    # (mypy doesn't know pydantic-settings' special init kwarg, hence the ignore.)
    return Settings(_env_file=None, **overrides)  # type: ignore[call-arg]


def test_factory_builds_gemini_provider() -> None:
    settings = _make_settings(provider="gemini", gemini_api_key="test-key")
    provider = build_provider(settings)
    assert isinstance(provider, GeminiProvider)
    assert provider.name == "gemini"


def test_factory_rejects_missing_gemini_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    settings = _make_settings(provider="gemini", gemini_api_key=None)
    with pytest.raises(ProviderError, match="GEMINI_API_KEY"):
        build_provider(settings)


def test_gemini_is_the_default_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PROVIDER", raising=False)
    settings = _make_settings()
    assert settings.provider == "gemini"
