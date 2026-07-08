# Code Walkthrough — a plain-English tour

This is the "read the codebase with me" guide. It assumes you know a few basics —
what tokens are, what `temperature` does, what tool calling is, and how Pydantic can
force a model's answer into a fixed JSON shape — and not much else.

## The one-paragraph version of the whole project

You give the program some messy text ("Hi, I'm Ada, email me at ada@example.com") and the
name of a shape you want back (a "schema", like `contact`). The program sends the text to
an LLM **with the schema attached**, so the model isn't allowed to reply with free-form
prose — it must reply with JSON matching your shape. Pydantic then double-checks that JSON
and hands you a real Python object. Around that single model call, the code adds the boring
production stuff that actually matters: retries when the API hiccups, token/cost tracking,
a CLI, and a web API.

## Where to find X

| "Where does … happen?" | File | What to look for |
|---|---|---|
| The provider (Gemini/Anthropic/OpenAI) gets **chosen** | `src/structured_extractor/factory.py` | `build_provider()` — the `if settings.provider == ...` chain |
| The default provider + all settings | `src/structured_extractor/config.py` | `Settings` class (`provider: str = Field(default="gemini")`) |
| **Retries** | `src/structured_extractor/extractor.py` | the `for attempt in range(...)` loop in `Extractor.extract` |
| **Validation** (forcing the model's text into your shape) | each provider file + `schemas.py` | `response_schema=schema` (Gemini), `output_format=schema` (Anthropic), `response_format=schema` (OpenAI) — the SDKs return already-validated Pydantic objects |
| The **tool-use / structured-output loop** | provider files | there is no manual loop! See "Wait, where's the tool-use loop?" below |
| The schemas themselves | `src/structured_extractor/schemas.py` | `ContactInfo`, `Invoice`, `Feedback` + `SCHEMA_REGISTRY` |
| The system prompt | `src/structured_extractor/extractor.py` | `_default_instructions()` |
| Token counting & cost math | `src/structured_extractor/usage.py` | `TokenUsage.estimated_cost_usd` |
| Error types | `src/structured_extractor/errors.py` | `ProviderError`, `SchemaValidationError` |
| The `extract` command | `src/structured_extractor/cli.py` | `run()` |
| The HTTP endpoints | `src/structured_extractor/api.py` | `POST /extract` |
| Offline test fakes | `tests/conftest.py` | `FakeProvider`, `FlakyProvider` |
| **Debug logging** (watch every request/response live) | `src/structured_extractor/debuglog.py` | `debug_enabled()` / `log_block()` — set `LLM_DEBUG=1` and everything prints to stderr |

## Suggested reading order

Read the files in this order — each one only depends on the ones before it:

1. `schemas.py` — the shapes we extract into (pure Pydantic, no LLM stuff)
2. `errors.py` — three tiny exception classes
3. `usage.py` — token/cost arithmetic
4. `providers/base.py` — the contract every provider must follow (the most important file)
5. `providers/gemini_provider.py` — one real provider, start to finish
6. `providers/anthropic_provider.py` and `providers/openai_provider.py` — same idea, different SDK spelling
7. `extractor.py` — retries + prompt building around any provider
8. `config.py` and `factory.py` — how settings turn into a wired-up extractor
9. `cli.py` and `api.py` — the two thin "front doors"
10. `tests/` — how all of it is proven without spending a cent

---

## 1. `schemas.py` — the shapes

A schema is just a Pydantic class. This one says "a contact has a name, and maybe an
email/phone/company/title":

```python
class ContactInfo(BaseModel):
    name: str = Field(description="Full name of the person.")
    email: str | None = Field(default=None, description="Email address, if present.")
```

Two things worth noticing:

- **The `description` strings are for the model, not for you.** They get converted into the
  JSON schema that is sent with every request, so the model reads them as instructions
  about each field. Writing a good description is prompt engineering.
- `str | None = None` means "optional" — it gives the model a legal way to say "the text
  didn't mention this", which is how we discourage it from inventing values.

At the bottom, `SCHEMA_REGISTRY` is a plain dict mapping friendly names (`"contact"`) to
classes (`ContactInfo`) so the CLI flag `--schema contact` can find the right class.

## 2. `errors.py` — three exceptions

```python
class ExtractionError(Exception): ...        # the family name
class ProviderError(ExtractionError): ...    # the API call itself failed
class SchemaValidationError(ExtractionError): ...  # the model's output didn't fit the schema
```

Why bother? So callers can react differently: a `ProviderError` might mean "check your API
key", while a `SchemaValidationError` means "the model misbehaved". The FastAPI layer turns
`ProviderError` into HTTP 502 ("the upstream service failed"), for example.

## 3. `usage.py` — what did that cost?

Every model reply tells us how many tokens went in and came out. `TokenUsage` stores those
two numbers, and `estimated_cost_usd` multiplies them by a small hand-maintained price
table (dollars per million tokens):

```python
return (self.input_tokens * in_rate + self.output_tokens * out_rate) / 1_000_000
```

If the model isn't in the table, the cost is reported as `0.0` instead of crashing —
missing price info shouldn't break an extraction.

## 4. `providers/base.py` — the contract (read this twice)

```python
class LLMProvider(Protocol):
    name: str

    def extract(self, *, text: str, schema: type[SchemaT], instructions: str) -> ProviderResponse: ...
```

This says: *anything* that has a `name` and an `extract` method with this exact signature
counts as a provider. It's a `Protocol`, which is Python's "duck typing, but checked":
providers don't inherit from it — they just match its shape, and `mypy` verifies that.

This one small interface is why the project is "provider-agnostic". The Extractor, the
CLI, the API, and the tests all talk to *this*, never to a specific SDK. Adding Gemini
meant writing one new file that matches this shape — nothing else changed.

`ProviderResponse` is the standard return parcel: the validated Pydantic object plus the
token usage.

## 5. `providers/gemini_provider.py` — one provider, line by line

This is the default provider (free tier — you only need a `GEMINI_API_KEY`). The heart of
it is one call:

```python
response = self._client.models.generate_content(
    model=self._model,                      # e.g. "gemini-2.5-flash"
    contents=text,                          # the messy text to extract from
    config=types.GenerateContentConfig(
        system_instruction=instructions,    # the standing rules ("don't invent values")
        response_mime_type="application/json",
        response_schema=schema,             # ← the magic: output MUST match our Pydantic class
        temperature=0,                      # most-likely token every time = repeatable answers
        max_output_tokens=self._max_tokens, # cost/safety brake
    ),
)
```

Walk through what each piece is doing:

- **`contents=text`** — the user message. The invoice/email/bio you want mined.
- **`system_instruction=instructions`** — the system prompt. Kept separate from the text so
  the model can't confuse "rules about how to behave" with "data to extract from".
- **`response_mime_type` + `response_schema`** — the model can only ever produce text;
  these two settings force that text to be valid JSON in exactly our schema's shape. This
  is the "forced JSON" you drilled, implemented server-side by Google.
- **`temperature=0`** — extraction wants the same answer for the same input, so we turn
  randomness off.

After the call:

```python
parsed = response.parsed          # an already-validated ContactInfo/Invoice/... instance
if parsed is None or not isinstance(parsed, schema):
    raise ProviderError("Gemini returned no parseable structured output.")
```

`response.parsed` is the SDK doing the `json.loads` + Pydantic validation for us. `None`
usually means the output got cut off (e.g. hit `max_output_tokens`). Finally the token
counts are copied out of `response.usage_metadata` into our own `TokenUsage`.

One more pattern to notice: errors are **translated at the boundary** —

```python
except errors.APIError as exc:
    raise ProviderError(f"Gemini request failed: {exc}") from exc
```

The rest of the app never sees a Google-specific exception. Each provider catches its own
SDK's errors and re-raises our `ProviderError`. That's what keeps the SDKs quarantined
inside the provider files.

## 6. The Anthropic and OpenAI providers — same song, different lyrics

Read them side by side with the Gemini one; the mapping is:

| Concept | Gemini | Anthropic | OpenAI |
|---|---|---|---|
| The parse-into-schema call | `generate_content(... response_schema=S)` | `messages.parse(... output_format=S)` | `beta.chat.completions.parse(... response_format=S)` |
| System prompt goes in… | `config.system_instruction` | `system=` argument | a `{"role": "system"}` message |
| Determinism | `temperature=0` | *omitted* — Claude Opus 4.8 removed sampling params entirely (sending `temperature` would be an API error) | `temperature=0` |
| Validated object comes back as | `response.parsed` | `response.parsed_output` | `message.parsed` |

Same goal every time; each vendor spells it differently. Hiding those spellings behind
`LLMProvider` is the entire architectural point of the project.

### Wait, where's the tool-use loop?

If you just learned tool calling, you might expect a loop like: *send request → model asks
to "call a tool" with JSON arguments → we hand the JSON back → repeat*. Early
structured-extraction code really was built that way — you'd define one fake tool whose
input schema was your target shape, force the model to "call" it, and read the arguments.

This project uses the modern replacement: all three vendors now offer **native structured
output** (`response_schema` / `output_format` / `response_format`), which is that same
trick built into the API — constrained to the schema during generation, validated by the
SDK, no loop needed. So the "tool-use loop" collapsed into a single `parse`-style call
inside each provider's `extract` method. If you ever add a provider that lacks native
structured output, its `extract` method is where you'd write the old-style forced
tool-call dance — and nothing outside that file would change.

## 7. `extractor.py` — retries and the system prompt

`Extractor` wraps any provider and owns the cross-cutting concerns.

**The default system prompt** is built from the schema:

```python
f"You extract structured data into the '{schema.__name__}' schema from the user's text. "
"Extract only information that is explicitly present. Do not guess or fabricate values..."
```

The schema (sent separately) controls the *shape*; this prompt controls the *behavior* —
most importantly "don't invent data", which is the number-one failure mode of extraction.

**The retry loop** — the most production-critical few lines in the repo:

```python
for attempt in range(1, self._max_retries + 2):   # initial try + retries
    try:
        response = self._provider.extract(...)
        return ExtractionResult(...)
    except (ProviderError, ValidationError) as error:
        last_error = error
        logger.warning(...)
```

LLM APIs fail transiently all the time — rate limits, brief 500s, network blips. One or
two retries convert most of those into successes. The loop is *bounded* (no infinite
hammering), it logs each failure, and when attempts run out it raises a typed error that
says which kind of failure persisted. Because retries live here — not in each provider —
the policy is written once and every backend gets it for free.

## 8. `config.py` + `factory.py` — from env vars to a working extractor

`config.py` defines a `Settings` class (via `pydantic-settings`) that reads environment
variables or a local `.env` file:

```python
provider: str = Field(default="gemini")      # the default backend
gemini_api_key: str | None = Field(default=None)
gemini_model: str = Field(default="gemini-2.5-flash")
```

Env var names map automatically: `PROVIDER`, `GEMINI_API_KEY`, `GEMINI_MODEL`, etc. This
is why you never see `os.environ[...]` sprinkled around the code — configuration has one
typed home.

`factory.py` is the **composition root** — the single place that turns a `Settings` into a
live provider:

```python
if settings.provider == "gemini":
    if not settings.gemini_api_key:
        raise ProviderError("GEMINI_API_KEY is not set.")
    from structured_extractor.providers.gemini_provider import GeminiProvider
    return GeminiProvider(model=settings.gemini_model, ...)
```

Note the import *inside* the `if`: the Gemini SDK only loads if you actually pick Gemini.
Also note the key check happens **before** any network call, so a missing key fails in
milliseconds with a readable message instead of a cryptic 401 later.

`build_extractor(settings)` just wraps `build_provider` in an `Extractor` with the
configured retry count. Both the CLI and the API call this one function — so "add a
provider" is: new provider file + one `if` branch here. Done.

## 9. `cli.py` and `api.py` — the two front doors

Both are deliberately thin: parse input → call the same `Extractor` → format output.

**CLI** (`extract run --schema contact "some text"`), notable bits:

- Input can come from an argument, `--file`, or stdin (piping).
- `--provider` overrides the configured provider for one run via
  `settings.model_copy(update={"provider": provider})` — a copy, so nothing global mutates.
- The JSON result goes to **stdout** and the cost summary to **stderr**, so
  `extract run ... | jq .name` works cleanly.
- Exit codes follow Unix convention: `2` = you called it wrong, `1` = extraction failed.

**API** (`POST /extract`), notable bits:

- `ExtractRequest` (Pydantic again!) validates the incoming JSON body — an empty `text`
  is rejected with HTTP 422 before any money is spent on a model call.
- Unknown schema → 404. Provider/config failure → 502 ("the dependency failed, not you").
- The response includes provider, model, attempts, token counts, and estimated cost —
  everything a caller needs to log and budget.

## 10. `tests/` — proving it all without API keys

The trick that makes the suite free and instant: because everything depends only on the
`LLMProvider` protocol, tests substitute hand-written fakes:

- `FakeProvider` (in `conftest.py`) always "succeeds" with a fixed `ContactInfo` — used to
  test the happy path through the Extractor and the API.
- `FlakyProvider` fails N times, then succeeds — used to prove the retry loop really
  retries, and really gives up after the limit.
- `test_gemini_provider.py` goes one level deeper: it builds a real `GeminiProvider` and
  swaps its network client for a `MagicMock`, then asserts on what *would have been sent*
  (JSON mode on? schema attached? temperature 0?) and on how responses/errors are
  translated. Still zero network, zero keys.
- `test_api.py` uses FastAPI's `TestClient` with `build_extractor` monkeypatched, checking
  status codes (200/404/422/502) and response shape.

The pattern to internalize: **fake the boundary, test your own logic.** The tests don't try
to verify that Google's model extracts well (that needs live evaluation); they verify that
*our* wiring, retries, validation, and error mapping behave — which is the part we can
break.

## Tracing one request end-to-end

`uv run extract run --schema contact "I'm Ada, ada@example.com"` does, in order:

1. `cli.py:run` — looks up `ContactInfo` in `SCHEMA_REGISTRY`, reads the text.
2. `config.py` — `load_settings()` reads `.env` → provider `gemini`, your key, model.
3. `factory.py:build_provider` — checks the key exists, builds `GeminiProvider`.
4. `extractor.py:Extractor.extract` — builds the system prompt, enters the retry loop.
5. `gemini_provider.py:extract` — one `generate_content` call with
   `response_schema=ContactInfo`; gets back `response.parsed` (a `ContactInfo`).
6. `usage.py` — token counts wrapped in `TokenUsage`, cost estimated.
7. Back in `cli.py` — JSON to stdout, `[gemini/gemini-2.5-flash] ... tokens (~$...)` to stderr.

If you can narrate those seven steps from memory, you understand the codebase.

## Bonus: watch a request live with `LLM_DEBUG`

Set the environment variable `LLM_DEBUG=1` and every provider prints the full request
(system prompt, user text, schema name) and response (raw output, token counts) to stderr,
and the retry loop announces when a validation failure triggers another attempt. All the
printing lives in `debuglog.py` — three tiny functions, called at each LLM call site, that
never touch stdout and never print your API key. It's the fastest way to *see* the
concepts above (system prompt vs. user text, forced JSON, retries) instead of just
reading about them.
