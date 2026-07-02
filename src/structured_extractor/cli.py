"""Command-line interface.

`extract run --schema contact "..."` or pipe a file with `--file`. Prints the validated
JSON plus a one-line cost summary to stderr so the JSON on stdout stays pipeable.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

import typer

from structured_extractor.config import load_settings
from structured_extractor.errors import ExtractionError
from structured_extractor.factory import build_extractor
from structured_extractor.schemas import SCHEMA_REGISTRY

app = typer.Typer(help="Extract structured, validated data from messy text.", no_args_is_help=True)


@app.command()
def schemas() -> None:
    """List the built-in schemas available to `--schema`."""
    for name, model in SCHEMA_REGISTRY.items():
        fields = ", ".join(model.model_fields)
        typer.echo(f"{name:10s} -> {model.__name__}({fields})")


@app.command()
def run(
    schema: Annotated[str, typer.Option(help="Schema name; see `extract schemas`.")],
    text: Annotated[str | None, typer.Argument(help="Text to extract from.")] = None,
    file: Annotated[Path | None, typer.Option(help="Read text from a file instead.")] = None,
    provider: Annotated[
        str | None, typer.Option(help="Override provider: gemini | anthropic | openai.")
    ] = None,
) -> None:
    """Extract `--schema` from the given text (positional) or `--file`."""
    # Fail fast on a bad schema name — exit code 2 is the Unix convention for
    # "you called me wrong" (vs 1 for "I tried and failed").
    if schema not in SCHEMA_REGISTRY:
        typer.echo(f"Unknown schema '{schema}'. Run `extract schemas`.", err=True)
        raise typer.Exit(code=2)

    # Three ways in, one variable out: --file wins, then the positional argument,
    # then whatever is piped into stdin (`echo "..." | extract run ...`).
    if file is not None:
        source = file.read_text(encoding="utf-8")
    elif text is not None:
        source = text
    else:
        source = sys.stdin.read()
    if not source.strip():
        typer.echo("No input text provided.", err=True)
        raise typer.Exit(code=2)

    # Settings come from env vars / .env; the --provider flag overrides just for this
    # run without touching your config (`model_copy` clones instead of mutating).
    settings = load_settings()
    if provider is not None:
        settings = settings.model_copy(update={"provider": provider})

    try:
        # build_extractor wires up the right provider from settings; extract() does the
        # model call, retries, and validation. The CLI stays a thin shell around it.
        extractor = build_extractor(settings)
        result = extractor.extract(source, SCHEMA_REGISTRY[schema])
    except ExtractionError as exc:
        typer.echo(f"Extraction failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    # JSON goes to stdout, the cost summary to stderr — so you can pipe the JSON into
    # another tool (`extract run ... | jq .name`) without the summary polluting it.
    typer.echo(result.data.model_dump_json(indent=2))
    usage = result.usage
    typer.echo(
        f"[{result.provider}/{usage.model}] {usage.total_tokens} tokens "
        f"(~${usage.estimated_cost_usd:.4f}), {result.attempts} attempt(s)",
        err=True,
    )


if __name__ == "__main__":
    app()
