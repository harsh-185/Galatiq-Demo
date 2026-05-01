"""Strip prompt-injection patterns from untrusted invoice text before it reaches an LLM."""
from __future__ import annotations

import re

# Anything that looks like a chat-control tag, role marker, or fenced system block.
_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"<\s*system[-_ ]?reminder\s*>.*?<\s*/\s*system[-_ ]?reminder\s*>", re.IGNORECASE | re.DOTALL),
    re.compile(r"<\s*system\s*>.*?<\s*/\s*system\s*>", re.IGNORECASE | re.DOTALL),
    re.compile(r"<\|[^|]+\|>"),  # OpenAI-style special tokens
    re.compile(r"^\s*(system|assistant|developer)\s*:", re.IGNORECASE | re.MULTILINE),
    re.compile(r"```[\s\S]*?```"),  # fenced code blocks (often hide instructions)
    re.compile(r"(?i)ignore (all |the |any |previous |prior )+(instructions|prompts?|messages?)"),
]


def strip_control_tags(text: str) -> str:
    """Remove prompt-injection-style markup. Idempotent and safe on clean text."""
    cleaned = text
    for pat in _PATTERNS:
        cleaned = pat.sub(" ", cleaned)
    return cleaned
