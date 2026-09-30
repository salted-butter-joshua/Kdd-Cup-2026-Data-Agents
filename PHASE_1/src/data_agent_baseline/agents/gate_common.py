"""Shared gate verdict helpers (架构设计 §13.5): PASS / REJECT / UNDECIDED.

Historically every L3 gate was binary: return a rejection dict or None (pass).
"Cannot decide" (e.g. the SQL could not be parsed) silently fell into the pass
branch — indistinguishable from "checked and clean", and invisible in telemetry.

Gates now have a third state:

- PASS      → return None
- REJECT    → return a rejection dict (ok=False, error, hint, evidence)
- UNDECIDED → return ``undecided_payload(...)``; the caller lets the answer
              through but records telemetry and may surface a soft suggestion
"""

from __future__ import annotations

from typing import Any


def undecided_payload(check: str, reason: str, suggestion: str) -> dict[str, Any]:
    """Mark 'gate wanted to check but could not decide' — NOT a rejection."""
    return {
        "ok": True,
        "undecided": True,
        "check": check,
        "reason": reason,
        "suggestion": suggestion,
    }


def is_undecided(payload: dict[str, Any] | None) -> bool:
    return bool(payload) and payload.get("undecided") is True
