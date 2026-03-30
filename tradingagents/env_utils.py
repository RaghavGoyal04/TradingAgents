"""Shared helpers for environment / ``.env`` handling."""

from __future__ import annotations

from typing import Optional


def clean_env_value(value: Optional[str]) -> str:
    """Strip whitespace and optional surrounding quotes from ``.env`` values."""
    s = (value or "").strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "'\"":
        s = s[1:-1].strip()
    return s
