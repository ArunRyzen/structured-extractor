"""Learning aid: print every AI request/response to stderr when LLM_DEBUG is set.

Set the environment variable `LLM_DEBUG=1` (or put `LLM_DEBUG=1` in your `.env` file)
and each provider prints exactly what it sends to the model and exactly what came back,
and the retry loop announces when a validation failure makes it try again. Unset it
(or set "0"/"false") and this module does nothing — normal runs pay zero cost.

Two rules the whole feature obeys:
- Everything goes to **stderr**, never stdout. The CLI prints its JSON result on
  stdout, and debug chatter must never corrupt that (piping to `jq` still works).
- The API key is **never** printed. Only prompts, outputs, and token counts.
"""

from __future__ import annotations

import functools
import os
import sys

# Print long prompts/outputs in full (this is a local learning tool), but clip anything
# beyond this many characters so a huge document can't flood the terminal.
_MAX_FIELD_CHARS = 2000


def _is_truthy(value: str) -> bool:
    """The shared on/off rule: anything except "", "0", or "false" (case-insensitive)."""
    return value.strip().lower() not in {"", "0", "false"}


@functools.lru_cache(maxsize=1)
def _debug_from_dotenv() -> bool:
    """Read LLM_DEBUG from a `.env` file in the current directory (cached).

    We reuse `python-dotenv` (already installed as a dependency of pydantic-settings)
    to parse the file. `dotenv_values` just *reads* the file into a dict — it never
    touches `os.environ`, so this stays a pure lookup. Reading a file on every debug
    check would be wasteful, so the result is cached; tests call `.cache_clear()`
    after changing directories so the cache doesn't serve a stale answer.
    """
    try:
        # Imported lazily so this module still works if python-dotenv is missing.
        from dotenv import dotenv_values
    except ImportError:
        return False  # no dotenv library -> quietly fall back to "disabled"
    value = dotenv_values(".env").get("LLM_DEBUG")
    if value is None:
        return False
    return _is_truthy(value)


def debug_enabled() -> bool:
    """True when LLM_DEBUG is switched on, checking two places in a fixed order.

    Precedence (first match wins):
    1. The real environment variable `LLM_DEBUG` — if it is set at all, it decides.
       That lets you type `$env:LLM_DEBUG="0"` to silence debugging for one session
       even when your `.env` file says `LLM_DEBUG=1`.
    2. Otherwise, `LLM_DEBUG` in a `.env` file in the current working directory —
       handy for keeping debug mode on across sessions while you're learning.

    Both places use the same truthiness rule: any value except "", "0", or "false"
    (case-insensitive) counts as "on".
    """
    value = os.environ.get("LLM_DEBUG")
    if value is not None:
        return _is_truthy(value)
    return _debug_from_dotenv()


def _clip(value: object) -> str:
    """Stringify a field, truncating very long values with a clear marker."""
    text = str(value)
    if len(text) > _MAX_FIELD_CHARS:
        return text[:_MAX_FIELD_CHARS] + "... [truncated]"
    return text


def log_block(title: str, **fields: object) -> None:
    """Print one labeled block to stderr, e.g.::

        === AI REQUEST (gemini/gemini-2.5-flash) ===
        system: <full system instruction>
        user: <full user text>
        schema: ContactInfo
        ============================================

    Does nothing unless `LLM_DEBUG` is enabled, so call sites can log unconditionally.
    Plain ASCII on purpose — fancy box-drawing characters break on some Windows consoles.
    """
    if not debug_enabled():
        return
    header = f"=== {title} ==="
    lines = [header]
    lines.extend(f"{name}: {_clip(value)}" for name, value in fields.items())
    lines.append("=" * len(header))
    print("\n".join(lines), file=sys.stderr, flush=True)
