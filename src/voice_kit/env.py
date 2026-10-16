"""Small environment-variable readers shared by voice-kit consumers."""

from __future__ import annotations

import os

__all__ = ["env_bool", "env_float", "env_int", "env_list", "env_str"]


def env_bool(name: str, default: bool) -> bool:
    """``1/true/yes/on`` (case-insensitive) are true; blank or unset returns ``default``."""
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw and raw.strip() else default


def env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return float(raw) if raw and raw.strip() else default


def env_str(name: str, default: str = "") -> str:
    """Stripped value, or ``default`` when unset or blank."""
    raw = os.getenv(name)
    return raw.strip() if raw and raw.strip() else default


def env_list(name: str, default: str) -> list[str]:
    """Comma-separated list with blanks dropped."""
    return [p.strip() for p in env_str(name, default).split(",") if p.strip()]
