"""OpenAI backend using structured-output parsing.

Contrast with the Anthropic provider — a useful thing to understand:
- OpenAI exposes `client.beta.chat.completions.parse(response_format=Schema)`, which is
  the analogue of Anthropic's `messages.parse`.
- Here we DO set `temperature=0` for deterministic extraction. Unlike Claude Opus 4.8
  (which removed sampling params), OpenAI models still accept it. Same goal — determinism —
  reached differently per provider. Hiding that difference behind one interface is the point.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

from pydantic import BaseModel

from structured_extractor.debuglog import log_block
from structured_extractor.errors import ProviderError
from structured_extractor.providers.base import ProviderResponse
from structured_extractor.usage import TokenUsage

if TYPE_CHECKING:
    from openai import OpenAI

SchemaT = TypeVar("SchemaT", bound=BaseModel)


class OpenAIProvider:
    """Structured extraction via the OpenAI Chat Completions parse helper."""

    name = "openai"

    def __init__(self, *, model: str, max_tokens: int, api_key: str | None = None) -> None:
        from openai import OpenAI

        self._client: OpenAI = OpenAI(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens

    def extract(
        self,
        *,
        text: str,
        schema: type[SchemaT],
        instructions: str,
    ) -> ProviderResponse:
        import openai

        # Learning aid: with LLM_DEBUG=1 this shows exactly what we send to the model
        # (never the API key). It's a no-op otherwise.
        log_block(
            f"AI REQUEST ({self.name}/{self._model})",
            system=instructions,
            user=text,
            schema=schema.__name__,
        )

        try:
            completion = self._client.beta.chat.completions.parse(
                model=self._model,
                max_tokens=self._max_tokens,
                # temperature=0 → always pick the most likely next token. Extraction
                # wants the same answer every time, not creative variety.
                temperature=0,
                # OpenAI puts the system prompt in the messages list (role "system"),
                # unlike Anthropic's separate `system=` argument. Same idea, new spelling.
                messages=[
                    {"role": "system", "content": instructions},
                    {"role": "user", "content": text},
                ],
                # Forces the reply to be valid JSON matching our Pydantic schema —
                # OpenAI's equivalent of Anthropic's `output_format`.
                response_format=schema,
            )
        except openai.OpenAIError as exc:
            # Wrap the SDK error in our own type so callers stay provider-agnostic.
            raise ProviderError(f"OpenAI request failed: {exc}") from exc

        # A completion can contain several alternative answers ("choices"); we asked
        # for one, so we take the first.
        message = completion.choices[0].message
        if message.refusal:
            raise ProviderError(f"OpenAI declined the request: {message.refusal}")

        parsed = message.parsed
        if parsed is None:
            raise ProviderError("OpenAI returned no parseable structured output.")

        usage_raw = completion.usage
        usage = TokenUsage(
            model=self._model,
            input_tokens=usage_raw.prompt_tokens if usage_raw else 0,
            output_tokens=usage_raw.completion_tokens if usage_raw else 0,
        )
        # And what came back: the validated JSON plus what it cost in tokens.
        log_block(
            f"AI RESPONSE ({self.name}/{self._model})",
            output=parsed.model_dump_json(),
            tokens=f"in={usage.input_tokens} out={usage.output_tokens}",
        )
        return ProviderResponse(data=parsed, usage=usage)
