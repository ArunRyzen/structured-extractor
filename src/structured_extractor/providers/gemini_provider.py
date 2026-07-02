"""Google Gemini backend using JSON-mode structured output.

This is the beginner-friendly default: Gemini has a generous free tier, so a single
`GEMINI_API_KEY` is enough to run live extractions without setting up billing.

How Gemini does structured output (compare with the other two providers):
- We pass `response_mime_type="application/json"` + `response_schema=<Pydantic class>`
  in the request config. The model can only ever produce text; these two settings force
  that text to be valid JSON in exactly our schema's shape.
- The SDK then hands us `response.parsed` — an already-validated Pydantic instance —
  which is the same "parsed object, not free text" guarantee that Anthropic's
  `messages.parse` and OpenAI's `chat.completions.parse` give.
- Like OpenAI (and unlike Claude Opus 4.8), Gemini accepts `temperature`, so we pin it
  to 0 for deterministic extraction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

from pydantic import BaseModel

from structured_extractor.errors import ProviderError
from structured_extractor.providers.base import ProviderResponse
from structured_extractor.usage import TokenUsage

if TYPE_CHECKING:
    from google import genai

SchemaT = TypeVar("SchemaT", bound=BaseModel)


class GeminiProvider:
    """Structured extraction via the Google Gen AI SDK (`google-genai`)."""

    name = "gemini"

    def __init__(self, *, model: str, max_tokens: int, api_key: str | None = None) -> None:
        # Imported lazily so the package (and its tests) don't hard-require the SDK
        # unless this provider is actually used — same pattern as the other providers.
        from google import genai

        # The client is the object that talks to Google's servers. If `api_key` is None
        # the SDK falls back to the GEMINI_API_KEY environment variable on its own.
        self._client: genai.Client = genai.Client(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens

    def extract(
        self,
        *,
        text: str,
        schema: type[SchemaT],
        instructions: str,
    ) -> ProviderResponse:
        from google.genai import errors, types

        try:
            response = self._client.models.generate_content(
                model=self._model,
                # `contents` is the user-facing input — the messy text to extract from.
                contents=text,
                config=types.GenerateContentConfig(
                    # The behavioral rules ("don't invent values", etc.) go in the
                    # system instruction, separate from the text being processed.
                    system_instruction=instructions,
                    # These two lines are the heart of structured output: the model's
                    # reply MUST be JSON, and that JSON MUST match our Pydantic schema.
                    response_mime_type="application/json",
                    response_schema=schema,
                    # temperature=0 → pick the most likely token every time, which is
                    # what you want for extraction (repeatable answers, no creativity).
                    temperature=0,
                    max_output_tokens=self._max_tokens,
                ),
            )
        except errors.APIError as exc:  # auth, rate limit, server, connection
            raise ProviderError(f"Gemini request failed: {exc}") from exc

        # When `response_schema` is a Pydantic class, the SDK parses and validates the
        # JSON for us and exposes the resulting object here. None means the model's
        # output couldn't be parsed (e.g. it was cut off by the token limit).
        parsed = response.parsed
        if parsed is None or not isinstance(parsed, schema):
            raise ProviderError("Gemini returned no parseable structured output.")

        # Token accounting so every extraction reports what it cost. The metadata can
        # be missing on some responses, hence the defensive `or 0` fallbacks.
        meta = response.usage_metadata
        usage = TokenUsage(
            model=self._model,
            input_tokens=(meta.prompt_token_count or 0) if meta else 0,
            output_tokens=(meta.candidates_token_count or 0) if meta else 0,
        )
        return ProviderResponse(data=parsed, usage=usage)
