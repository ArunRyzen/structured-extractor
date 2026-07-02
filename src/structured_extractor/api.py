"""FastAPI service exposing extraction over HTTP.

Why an API at all: it's the production surface AI features ship behind. The endpoint is
thin on purpose — it validates the request, delegates to the same `Extractor` the CLI
uses, and maps domain errors to HTTP status codes. All the logic lives in the library,
so the transport is swappable and testable.
"""

from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from structured_extractor.config import Settings, load_settings
from structured_extractor.errors import ProviderError
from structured_extractor.factory import build_extractor
from structured_extractor.schemas import SCHEMA_REGISTRY

app = FastAPI(title="structured-extractor", version="0.1.0")


@lru_cache
def _settings() -> Settings:
    # Cached so we don't re-read the environment on every request.
    return load_settings()


class ExtractRequest(BaseModel):
    """The JSON body a client POSTs to /extract.

    FastAPI validates incoming requests against this model automatically — e.g. an
    empty `text` is rejected with a 422 before our code (or any paid model) runs.
    """

    text: str = Field(min_length=1, description="The text to extract from.")
    schema_name: str = Field(description="A registered schema name; see GET /schemas.")
    provider: str | None = Field(default=None, description="Override provider for this call.")


class ExtractResponse(BaseModel):
    """What /extract returns: the extracted data plus full cost/attempt accounting."""

    provider: str
    model: str
    attempts: int
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float
    data: dict


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/schemas")
def list_schemas() -> dict[str, list[str]]:
    """Expose the available schemas and their fields so clients can discover them."""
    return {name: list(model.model_fields) for name, model in SCHEMA_REGISTRY.items()}


@app.post("/extract", response_model=ExtractResponse)
def extract(request: ExtractRequest) -> ExtractResponse:
    # Look up the schema by name; 404 is the natural "that thing doesn't exist" answer.
    schema = SCHEMA_REGISTRY.get(request.schema_name)
    if schema is None:
        raise HTTPException(status_code=404, detail=f"Unknown schema '{request.schema_name}'.")

    # Same override trick as the CLI: clone the settings for this one request rather
    # than mutating the cached global settings (which other requests share).
    settings = _settings()
    if request.provider is not None:
        settings = settings.model_copy(update={"provider": request.provider})

    try:
        result = build_extractor(settings).extract(request.text, schema)
    except ProviderError as exc:
        # Upstream/config problem — 502 signals "the dependency failed," not "you erred."
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return ExtractResponse(
        provider=result.provider,
        model=result.usage.model,
        attempts=result.attempts,
        input_tokens=result.usage.input_tokens,
        output_tokens=result.usage.output_tokens,
        estimated_cost_usd=result.usage.estimated_cost_usd,
        data=result.data.model_dump(),
    )
