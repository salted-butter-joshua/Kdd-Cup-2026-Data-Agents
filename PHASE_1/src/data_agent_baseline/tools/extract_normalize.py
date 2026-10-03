"""Post-extract format conversion: model declares formats, code runs in a subprocess."""

from __future__ import annotations

import json
import subprocess
import sys
from typing import Any

NORMALIZE_TIMEOUT_SECONDS = 5.0

_RUNNER = r"""
import json
import sys

payload = json.load(sys.stdin)
code = str(payload.get("code") or "")
values = payload.get("values") or []
ns = {}
if code.strip():
    exec(compile(code, "<extract_normalize>", "exec"), ns, ns)
fn = ns.get("convert")
out = []
for raw in values:
    text = "" if raw is None else str(raw)
    if not text or fn is None:
        out.append(text)
        continue
    try:
        converted = fn(text)
        out.append(text if converted is None else str(converted))
    except Exception:
        out.append(text)
json.dump(out, sys.stdout, ensure_ascii=False)
"""

_BUILTIN_CODE = {
    "iso_date": """
def convert(value):
    import re
    text = (value or "").strip()
    match = re.search(r"(\\d{4})[-/.](\\d{1,2})[-/.](\\d{1,2})", text)
    if not match:
        return value
    y, m, d = match.group(1), match.group(2).zfill(2), match.group(3).zfill(2)
    return f"{y}-{m}-{d}"
""",
    "upper": """
def convert(value):
    return (value or "").strip().upper()
""",
    "lower": """
def convert(value):
    return (value or "").strip().lower()
""",
    "title": """
def convert(value):
    return " ".join((value or "").split()).title()
""",
    "strip": """
def convert(value):
    return " ".join((value or "").split())
""",
    "digits": """
def convert(value):
    import re
    digits = re.sub(r"\\D+", "", value or "")
    return digits or value
""",
}


def run_converter_code(
    code: str,
    values: list[str],
    *,
    timeout_seconds: float = NORMALIZE_TIMEOUT_SECONDS,
) -> list[str]:
    """Run ``convert(value)`` in a subprocess. On timeout/error, return originals."""
    originals = [str(v or "") for v in values]
    if not code.strip():
        return originals
    try:
        completed = subprocess.run(
            [sys.executable, "-c", _RUNNER],
            input=json.dumps({"code": code, "values": originals}, ensure_ascii=False),
            capture_output=True,
            text=True,
            timeout=max(0.5, float(timeout_seconds)),
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError):
        return originals
    if completed.returncode != 0:
        return originals
    try:
        parsed = json.loads(completed.stdout or "[]")
    except Exception:
        return originals
    if not isinstance(parsed, list) or len(parsed) != len(originals):
        return originals
    return [originals[i] if parsed[i] is None else str(parsed[i]) for i in range(len(originals))]


def builtin_converter_code(spec: str) -> str:
    key = (spec or "").strip().lower().replace("-", "_")
    return _BUILTIN_CODE.get(key, "")


def apply_column_converters(
    rows: list[dict[str, str]],
    *,
    columns: list[str],
    converters: dict[str, str],
    timeout_seconds: float = NORMALIZE_TIMEOUT_SECONDS,
) -> list[dict[str, str]]:
    """Apply per-column converter code. Failed columns keep original values."""
    if not rows or not converters:
        return rows
    updated = [dict(row) for row in rows]
    for col in columns:
        code = converters.get(col) or ""
        if not code.strip():
            continue
        values = [str(row.get(col, "") or "") for row in updated]
        converted = run_converter_code(code, values, timeout_seconds=timeout_seconds)
        for row, value in zip(updated, converted):
            row[col] = value
    return updated


def parse_format_declarations(payload: Any, columns: list[str]) -> dict[str, str]:
    """Map column -> converter source from a model JSON object."""
    allowed = set(columns)
    raw = payload
    if isinstance(payload, dict) and isinstance(payload.get("formats"), dict):
        raw = payload["formats"]
    if not isinstance(raw, dict):
        return {}
    converters: dict[str, str] = {}
    for key, spec in raw.items():
        name = str(key or "").strip()
        if name not in allowed:
            continue
        code = ""
        if isinstance(spec, dict):
            code = str(spec.get("code") or "").strip()
            if not code:
                code = builtin_converter_code(str(spec.get("format") or spec.get("target") or ""))
        else:
            code = builtin_converter_code(str(spec or ""))
        if code:
            converters[name] = code
    return converters
