"""LLM_DEBUG learning-mode tests — fully offline, like everything else in the suite.

Four things must hold:
- With LLM_DEBUG off, extraction produces NO debug output (silence is the default).
- With LLM_DEBUG=1, every provider call prints `=== AI REQUEST` / `=== AI RESPONSE`
  blocks to stderr (never stdout — the CLI's JSON lives there).
- When the model's output flunks schema validation and a retry fires, a clearly
  labeled `VALIDATION FAILED ... retrying` block appears.
- `LLM_DEBUG` can also come from a `.env` file, but a real environment variable
  always outranks it.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import TypeVar
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from structured_extractor.debuglog import _debug_from_dotenv, debug_enabled, log_block
from structured_extractor.extractor import Extractor
from structured_extractor.providers.base import ProviderResponse
from structured_extractor.providers.gemini_provider import GeminiProvider
from structured_extractor.schemas import ContactInfo
from structured_extractor.usage import TokenUsage

SchemaT = TypeVar("SchemaT", bound=BaseModel)

SECRET_KEY = "sk-test-secret-key"


def _mocked_gemini_provider() -> GeminiProvider:
    """A real GeminiProvider whose network client is a mock — zero network, zero keys."""
    provider = GeminiProvider(model="gemini-2.5-flash", max_tokens=4096, api_key=SECRET_KEY)
    contact = ContactInfo(name="Ada Lovelace", email="ada@example.com")
    usage = SimpleNamespace(prompt_token_count=93, candidates_token_count=55)
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = SimpleNamespace(
        parsed=contact, usage_metadata=usage
    )
    provider._client = mock_client
    return provider


class InvalidThenValidProvider:
    """First call raises a real pydantic ValidationError, second call succeeds."""

    name = "fake"

    def __init__(self) -> None:
        self.calls = 0

    def extract(self, *, text: str, schema: type[SchemaT], instructions: str) -> ProviderResponse:
        self.calls += 1
        if self.calls == 1:
            ContactInfo.model_validate({})  # `name` is required -> ValidationError
        data = ContactInfo(name="Grace Hopper")
        usage = TokenUsage(model="gemini-2.5-flash", input_tokens=10, output_tokens=5)
        return ProviderResponse(data=data, usage=usage)


# --- the on/off switch ---------------------------------------------------------------


def test_debug_disabled_when_unset() -> None:
    # conftest's autouse fixture pins LLM_DEBUG to "0", which outranks any local .env.
    assert debug_enabled() is False


@pytest.mark.parametrize("value", ["0", "false", "FALSE", "False", ""])
def test_debug_disabled_for_falsey_values(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("LLM_DEBUG", value)
    assert debug_enabled() is False


@pytest.mark.parametrize("value", ["1", "true", "yes", "on"])
def test_debug_enabled_for_truthy_values(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("LLM_DEBUG", value)
    assert debug_enabled() is True


# --- the .env fallback ---------------------------------------------------------------


@pytest.fixture
def _fresh_dotenv_cache() -> Iterator[None]:
    """Clear the cached .env lookup before AND after a test that depends on the cwd."""
    _debug_from_dotenv.cache_clear()
    yield
    _debug_from_dotenv.cache_clear()


def test_dotenv_file_enables_debug(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _fresh_dotenv_cache: None
) -> None:
    (tmp_path / ".env").write_text("LLM_DEBUG=1\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("LLM_DEBUG")  # no env var -> the .env file gets its say

    assert debug_enabled() is True


def test_env_var_outranks_dotenv_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _fresh_dotenv_cache: None
) -> None:
    (tmp_path / ".env").write_text("LLM_DEBUG=1\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LLM_DEBUG", "0")  # a real env var wins, even a falsey one

    assert debug_enabled() is False


def test_disabled_with_no_dotenv_and_no_env_var(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _fresh_dotenv_cache: None
) -> None:
    monkeypatch.chdir(tmp_path)  # an empty directory: no .env file here
    monkeypatch.delenv("LLM_DEBUG")

    assert debug_enabled() is False


def test_disabled_when_dotenv_library_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _fresh_dotenv_cache: None
) -> None:
    (tmp_path / ".env").write_text("LLM_DEBUG=1\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("LLM_DEBUG")
    # Setting a module's sys.modules entry to None makes `import dotenv` raise
    # ImportError — simulating an environment without python-dotenv installed.
    monkeypatch.setitem(sys.modules, "dotenv", None)  # type: ignore[arg-type]

    assert debug_enabled() is False


# --- provider call sites -------------------------------------------------------------


def test_no_debug_output_when_unset(capsys: pytest.CaptureFixture[str]) -> None:
    provider = _mocked_gemini_provider()
    provider.extract(text="Ada, ada@example.com", schema=ContactInfo, instructions="sys prompt")

    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out == ""


def test_request_and_response_blocks_when_enabled(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("LLM_DEBUG", "1")
    provider = _mocked_gemini_provider()
    provider.extract(text="Ada, ada@example.com", schema=ContactInfo, instructions="sys prompt")

    captured = capsys.readouterr()
    assert captured.out == ""  # debug chatter must never touch stdout
    assert "=== AI REQUEST (gemini/gemini-2.5-flash) ===" in captured.err
    assert "=== AI RESPONSE (gemini/gemini-2.5-flash) ===" in captured.err
    assert "system: sys prompt" in captured.err
    assert "user: Ada, ada@example.com" in captured.err
    assert "schema: ContactInfo" in captured.err
    assert "ada@example.com" in captured.err  # the model's output is shown
    assert "tokens: in=93 out=55" in captured.err
    assert SECRET_KEY not in captured.err  # the API key is NEVER logged


# --- the retry seatbelt --------------------------------------------------------------


def test_validation_failure_retry_block(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("LLM_DEBUG", "1")
    extractor = Extractor(InvalidThenValidProvider(), max_retries=2)

    result = extractor.extract("text", ContactInfo)

    assert result.attempts == 2
    captured = capsys.readouterr()
    assert "=== VALIDATION FAILED (attempt 1) - retrying ===" in captured.err
    assert "name" in captured.err  # the validation error names the offending field


def test_no_retry_block_when_debug_unset(capsys: pytest.CaptureFixture[str]) -> None:
    extractor = Extractor(InvalidThenValidProvider(), max_retries=2)
    extractor.extract("text", ContactInfo)

    assert "VALIDATION FAILED" not in capsys.readouterr().err


# --- formatting details --------------------------------------------------------------


def test_long_values_are_truncated(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("LLM_DEBUG", "1")
    log_block("AI REQUEST (test/model)", user="x" * 5000)

    err = capsys.readouterr().err
    assert "... [truncated]" in err
    assert "x" * 5000 not in err
