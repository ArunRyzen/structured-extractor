"""Anthropic backend using schema-constrained structured output.

Design notes worth knowing for interviews:
- We use `client.messages.parse(output_format=Schema)`, which constrains generation to
  the schema and returns a *validated* Pydantic instance — far more reliable than asking
  for "JSON" in prose and hoping.
- We deliberately do NOT pass `temperature`. On Claude Opus 4.8 sampling parameters are
  removed and would raise a 400. Extraction wants determinism, which the model gives here
  without a temperature knob.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

from pydantic import BaseModel

from structured_extractor.debuglog import log_block
from structured_extractor.errors import ProviderError
from structured_extractor.providers.base import ProviderResponse
from structured_extractor.usage import TokenUsage

if TYPE_CHECKING:
    from anthropic import Anthropic

SchemaT = TypeVar("SchemaT", bound=BaseModel)


class AnthropicProvider:
    """Structured extraction via the Anthropic Messages API."""

    name = "anthropic"

    def __init__(self, *, model: str, max_tokens: int, api_key: str | None = None) -> None:
        # Imported lazily so the package (and its tests) don't hard-require the SDK
        # unless this provider is actually used.
        from anthropic import Anthropic

        self._client: Anthropic = Anthropic(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens

    def extract(
        self,
        *,
        text: str,
        schema: type[SchemaT],
        instructions: str,
    ) -> ProviderResponse:
        import anthropic

        # Learning aid: with LLM_DEBUG=1 this shows exactly what we send to the model
        # (never the API key). It's a no-op otherwise.
        log_block(
            f"AI REQUEST ({self.name}/{self._model})",
            system=instructions,
            user=text,
            schema=schema.__name__,
        )

        try:
            response = self._client.messages.parse(
                model=self._model,
                # Hard cap on how much the model may write back — a cost/safety brake.
                max_tokens=self._max_tokens,
                # The behavioral rules ("extract, don't invent") ride along as the
                # system prompt, separate from the text being processed.
                system=instructions,
                messages=[{"role": "user", "content": text}],
                # The key line: the model can only return text, and this forces that
                # text to be valid JSON in exactly our Pydantic schema's shape.
                output_format=schema,
            )
        except anthropic.APIError as exc:  # auth, rate limit, server, connection
            # Translate the SDK-specific error into OUR error type, so the rest of the
            # app never needs to import (or even know about) the Anthropic SDK.
            raise ProviderError(f"Anthropic request failed: {exc}") from exc

        # The model may decline unsafe requests instead of answering; surface that as a
        # clear error rather than a confusing "no output".
        if response.stop_reason == "refusal":
            raise ProviderError("Anthropic declined the request (safety refusal).")

        # `parsed_output` is the already-validated Pydantic object the SDK built for us.
        # None means the output couldn't be parsed (e.g. it was cut off by max_tokens).
        parsed = response.parsed_output
        if parsed is None:
            raise ProviderError("Anthropic returned no parseable structured output.")

        usage = TokenUsage(
            model=self._model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
        # And what came back: the validated JSON plus what it cost in tokens.
        log_block(
            f"AI RESPONSE ({self.name}/{self._model})",
            output=parsed.model_dump_json(),
            tokens=f"in={usage.input_tokens} out={usage.output_tokens}",
        )
        return ProviderResponse(data=parsed, usage=usage)
