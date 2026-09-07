"""Shared configuration helpers.

Kept in its own module so the weather layer, the LLM layer and the UI all agree
on what counts as a configured value. Three copies of this rule drifted apart
once already: the UI reported a key as present while the service layer treated
the same value as missing.
"""

from __future__ import annotations

from typing import Optional

# Substrings that appear in the placeholders shipped in .env.example. A real
# API key is hex or base64-ish and will not contain them.
_PLACEHOLDER_MARKERS = ("your_", "paste_", "_here", "changeme", "xxxx")


def is_configured(value: Optional[str]) -> bool:
    """True when ``value`` is a real setting rather than blank or a placeholder."""
    if not value:
        return False
    cleaned = value.strip().strip("\"'")
    if not cleaned:
        return False
    lowered = cleaned.lower()
    return not any(marker in lowered for marker in _PLACEHOLDER_MARKERS)
