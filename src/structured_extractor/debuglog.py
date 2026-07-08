"""Learning aid: print every AI request/response to stderr when LLM_DEBUG is set.

Set the environment variable `LLM_DEBUG=1` and each provider prints exactly what it
sends to the model and exactly what came back, and the retry loop announces when a
validation failure makes it try again. Unset it (or set "0"/"false") and this module
does nothing — normal runs pay zero cost.

Two rules the whole feature obeys:
- Everything goes to **stderr**, never stdout. The CLI prints its JSON result on
  stdout, and debug chatter must never corrupt that (piping to `jq` still works).
- The API key is **never** printed. Only prompts, outputs, and token counts.
"""

from __future__ import annotations

import os
import sys

# Print long prompts/outputs in full (this is a local learning tool), but clip anything
# beyond this many characters so a huge document can't flood the terminal.
_MAX_FIELD_CHARS = 2000


def debug_enabled() -> bool:
    """True when LLM_DEBUG is set to anything except "0" or "false" (case-insensitive)."""
    value = os.environ.get("LLM_DEBUG")
    if value is None:
        return False
    return value.strip().lower() not in {"", "0", "false"}


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
