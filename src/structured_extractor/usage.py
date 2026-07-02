"""Token usage and cost accounting.

Cost awareness is a first-class concern for production LLM apps (and a recurring
interview topic). Every extraction reports tokens and an estimated USD cost so callers
can budget, log, and alert on spend.
"""

from __future__ import annotations

from dataclasses import dataclass

# USD per 1M tokens, (input, output). Keep this small and explicit rather than
# pulling a pricing SDK — prices are public and change rarely. Update as needed.
_PRICING_PER_MTOK: dict[str, tuple[float, float]] = {
    # Anthropic (June 2026)
    "claude-opus-4-8": (5.00, 25.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
    # OpenAI (illustrative defaults; override via the table if your model differs)
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    # Google Gemini (paid-tier list prices; the free tier costs $0 up to its quota)
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-pro": (1.25, 10.00),
}


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Immutable record of what one extraction cost, in tokens and dollars."""

    model: str
    input_tokens: int
    output_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def estimated_cost_usd(self) -> float:
        """Best-effort USD estimate. Returns 0.0 for models we have no price for."""
        rate = _PRICING_PER_MTOK.get(self.model)
        if rate is None:
            return 0.0
        in_rate, out_rate = rate
        return (self.input_tokens * in_rate + self.output_tokens * out_rate) / 1_000_000
