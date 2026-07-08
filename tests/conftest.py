"""Shared test fixtures: fake providers so the suite needs no API keys or network.

Because every backend implements the same narrow `LLMProvider` protocol, a hand-written
fake is a complete, honest stand-in. The tests exercise *our* logic (retries, wiring,
HTTP mapping) without paying for or depending on a real model call.
"""

from __future__ import annotations

from typing import TypeVar

import pytest
from pydantic import BaseModel

from structured_extractor.errors import ProviderError
from structured_extractor.providers.base import ProviderResponse
from structured_extractor.schemas import ContactInfo
from structured_extractor.usage import TokenUsage

SchemaT = TypeVar("SchemaT", bound=BaseModel)


@pytest.fixture(autouse=True)
def _no_llm_debug(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the suite deterministic even if a developer has LLM_DEBUG exported.

    Tests that exercise the debug output opt back in with `monkeypatch.setenv`.
    """
    monkeypatch.delenv("LLM_DEBUG", raising=False)


class FakeProvider:
    """Always succeeds, returning a fixed ContactInfo and usage."""

    name = "fake"

    def __init__(self, model: str = "claude-opus-4-8") -> None:
        self._model = model
        self.calls = 0

    def extract(self, *, text: str, schema: type[SchemaT], instructions: str) -> ProviderResponse:
        self.calls += 1
        data = ContactInfo(name="Ada Lovelace", email="ada@example.com")
        usage = TokenUsage(model=self._model, input_tokens=120, output_tokens=30)
        return ProviderResponse(data=data, usage=usage)


class FlakyProvider:
    """Raises ProviderError `fail_times` times, then succeeds — to test retries."""

    name = "flaky"

    def __init__(self, fail_times: int) -> None:
        self._fail_times = fail_times
        self.calls = 0

    def extract(self, *, text: str, schema: type[SchemaT], instructions: str) -> ProviderResponse:
        self.calls += 1
        if self.calls <= self._fail_times:
            raise ProviderError(f"transient failure #{self.calls}")
        data = ContactInfo(name="Grace Hopper")
        usage = TokenUsage(model="claude-opus-4-8", input_tokens=100, output_tokens=20)
        return ProviderResponse(data=data, usage=usage)
