"""The orchestration layer: prompt construction + reliability, provider-agnostic.

This is where "make it production-grade" lives. Providers do one call; the Extractor
owns the cross-cutting concerns: building a good instruction, retrying transient/invalid
responses with bounded backoff-free retries, and surfacing clear errors. Keeping this
separate from the providers means the retry policy is tested once and applies to every backend.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from structured_extractor.debuglog import log_block
from structured_extractor.errors import ProviderError, SchemaValidationError
from structured_extractor.providers.base import LLMProvider
from structured_extractor.usage import TokenUsage

logger = logging.getLogger("structured_extractor")

SchemaT = TypeVar("SchemaT", bound=BaseModel)


@dataclass(frozen=True, slots=True)
class ExtractionResult:
    """The outcome of one extraction: the typed object, what it cost, and how it ran."""

    data: BaseModel
    usage: TokenUsage
    provider: str
    attempts: int


def _default_instructions(schema: type[BaseModel]) -> str:
    """A schema-aware system prompt.

    The model already sees the JSON schema via structured output; these instructions add
    the *behavioral* contract: extract only what's present, never invent, prefer nulls.
    """
    return (
        f"You extract structured data into the '{schema.__name__}' schema from the user's "
        "text. Extract only information that is explicitly present. Do not guess or "
        "fabricate values; when a field is absent, leave it null or empty. Normalize "
        "obvious formats (e.g. trim whitespace) but do not infer beyond the text."
    )


class Extractor:
    """Extract validated Pydantic objects from text using any `LLMProvider`."""

    def __init__(self, provider: LLMProvider, *, max_retries: int = 2) -> None:
        self._provider = provider
        self._max_retries = max_retries

    def extract(
        self,
        text: str,
        schema: type[SchemaT],
        *,
        instructions: str | None = None,
    ) -> ExtractionResult:
        """Run extraction with bounded retries.

        Retries cover two failure classes: a transient provider error (rate limit, 5xx,
        connection blip) and the rare case where output doesn't validate. Both are worth
        one or two more attempts; a persistent failure is raised so callers can react.
        """
        # If the caller didn't write their own instructions, build sensible ones from
        # the schema. Either way, this string becomes the model's "system prompt" —
        # the standing rules it must follow while reading the user's text.
        system = instructions or _default_instructions(schema)
        last_error: Exception | None = None

        # `max_retries=2` means up to 3 total tries: the first attempt + 2 retries.
        # LLM APIs fail transiently all the time (rate limits, brief outages), so a
        # couple of retries turns "flaky" into "reliable" for almost no extra code.
        for attempt in range(1, self._max_retries + 2):  # initial try + retries
            try:
                # The actual model call. Which provider runs here was decided elsewhere
                # (factory.py) — this loop genuinely does not know or care which one.
                response = self._provider.extract(text=text, schema=schema, instructions=system)
                # Success: bundle the validated object with bookkeeping (which provider,
                # how many tries, what it cost) so callers can log and budget.
                return ExtractionResult(
                    data=response.data,
                    usage=response.usage,
                    provider=self._provider.name,
                    attempts=attempt,
                )
            except (ProviderError, ValidationError) as error:
                # Two retryable failure kinds: the API call itself broke (ProviderError),
                # or the model's output didn't match our schema (ValidationError).
                # Remember the error, log it, and let the loop try again.
                last_error = error
                logger.warning(
                    "extraction attempt %d/%d failed: %s",
                    attempt,
                    self._max_retries + 1,
                    error,
                )
                # Learning aid (LLM_DEBUG=1): when the model's output flunks schema
                # validation and we're about to try again, say so loudly — this is the
                # "seatbelt" of the whole project firing in real time.
                if isinstance(error, ValidationError) and attempt <= self._max_retries:
                    log_block(
                        f"VALIDATION FAILED (attempt {attempt}) - retrying",
                        error=str(error),
                        note=(
                            "The model's reply did not match the schema, so the "
                            "extractor is asking it again."
                        ),
                    )

        # Exhausted retries — translate into a clear, typed failure so the CLI/API can
        # tell the user *what kind* of problem it was, not just "something went wrong".
        if isinstance(last_error, ValidationError):
            raise SchemaValidationError(
                f"Output failed schema validation after {self._max_retries + 1} attempts."
            ) from last_error
        raise ProviderError(
            f"Extraction failed after {self._max_retries + 1} attempts: {last_error}"
        ) from last_error
