"""L1.0 value normalization: align question literals with stored values.

Compares semantic values (parsed + truncated to question grain), not raw strings.
Mechanism-level only — no task_id hardcoding.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from data_agent_baseline.agents.runtime import StepRecord

# H:MM:SS(.mmm) or M:SS(.mmm) or compact times in questions.
_TIME_TOKEN_RE = re.compile(
    r"\b(\d{1,2}):(\d{2})(?::(\d{2})(?:\.(\d+))?|\.(\d+))?\b"
)
_DATE_MONTH_RE = re.compile(r"\b((?:19|20)\d{2})[-/](\d{1,2})\b")
_LIKE_RE = re.compile(
    r"""LIKE\s+'([^']+)'|LIKE\s+"([^"]+)"|starts_with\s*\([^,]+,\s*'([^']+)'\)""",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class TimeLiteral:
    raw: str
    total_seconds: float
    grain: str  # "second" | "millisecond" | "minute"
    canonical_mmss: str  # e.g. "1:54" (no leading hour zero)


@dataclass(frozen=True, slots=True)
class DateMonthLiteral:
    raw: str
    year: int
    month: int


@dataclass(slots=True)
class NormalizePlan:
    time_literals: list[TimeLiteral] = field(default_factory=list)
    date_months: list[DateMonthLiteral] = field(default_factory=list)
    knowledge_hints: list[str] = field(default_factory=list)

    @property
    def has_work(self) -> bool:
        return bool(self.time_literals or self.date_months)


def parse_time_token(raw: str) -> TimeLiteral | None:
    """Parse a time token into seconds + question grain."""
    match = _TIME_TOKEN_RE.fullmatch(raw.strip())
    if match is None:
        # Allow search from a larger string.
        match = _TIME_TOKEN_RE.search(raw.strip())
        if match is None or match.group(0) != raw.strip():
            return None
    a, b, sec, frac_hms, frac_ms = match.groups()
    hours_or_min = int(a)
    minutes_or_sec = int(b)
    if sec is not None:
        # H:MM:SS(.frac)
        hours, minutes, seconds = hours_or_min, minutes_or_sec, int(sec)
        frac = frac_hms or "0"
        grain = "millisecond" if frac_hms else "second"
        total = hours * 3600 + minutes * 60 + seconds + float(f"0.{frac}")
        whole = hours * 3600 + minutes * 60 + seconds
        # Canonical M:SS when hours==0 and total < 1h, else H:MM:SS without frac at grain.
        if hours == 0:
            canonical = f"{minutes}:{seconds:02d}"
        else:
            canonical = f"{hours}:{minutes:02d}:{seconds:02d}"
        return TimeLiteral(
            raw=match.group(0),
            total_seconds=total,
            grain=grain,
            canonical_mmss=canonical if hours == 0 else f"{minutes + hours * 60}:{seconds:02d}",
        )
    # M:SS(.frac)  — first component is minutes
    minutes, seconds = hours_or_min, minutes_or_sec
    frac = frac_ms or "0"
    grain = "millisecond" if frac_ms else "second"
    total = minutes * 60 + seconds + float(f"0.{frac}")
    canonical = f"{minutes}:{seconds:02d}"
    return TimeLiteral(
        raw=match.group(0),
        total_seconds=total,
        grain=grain,
        canonical_mmss=canonical,
    )


def extract_time_literals(text: str) -> list[TimeLiteral]:
    found: list[TimeLiteral] = []
    seen: set[str] = set()
    for match in _TIME_TOKEN_RE.finditer(text or ""):
        token = match.group(0)
        if token in seen:
            continue
        parsed = parse_time_token(token)
        if parsed is None:
            continue
        seen.add(token)
        found.append(parsed)
    return found


def extract_date_months(text: str) -> list[DateMonthLiteral]:
    out: list[DateMonthLiteral] = []
    seen: set[str] = set()
    for match in _DATE_MONTH_RE.finditer(text or ""):
        raw = match.group(0)
        if raw in seen:
            continue
        seen.add(raw)
        out.append(
            DateMonthLiteral(
                raw=raw,
                year=int(match.group(1)),
                month=int(match.group(2)),
            )
        )
    return out


def build_normalize_plan(
    *,
    question: str,
    knowledge_text: str = "",
) -> NormalizePlan:
    plan = NormalizePlan(
        time_literals=extract_time_literals(question or ""),
        date_months=extract_date_months(question or ""),
    )
    knowledge = knowledge_text or ""
    hints: list[str] = []
    if re.search(r"MM:SS\.mmm|M:SS\.mmm|H:MM:SS", knowledge, flags=re.IGNORECASE):
        hints.append(
            "knowledge declares a time unit/format — parse stored values with that "
            "format, then truncate to the question's grain before comparing."
        )
    if plan.time_literals and re.search(r"qualifying|q1|q2|q3|lap", knowledge, flags=re.IGNORECASE):
        hints.append(
            "qualifying / lap times are often M:SS.mmm while the question may use H:MM:SS."
        )
    plan.knowledge_hints = hints
    return plan


def format_normalize_plan(plan: NormalizePlan) -> str:
    if not plan.has_work:
        return ""
    lines = [
        "Value normalization (compare semantic values, not raw strings):",
    ]
    for lit in plan.time_literals[:4]:
        lines.append(
            f"- Question time '{lit.raw}' → {lit.total_seconds:g}s; "
            f"grain={lit.grain}; canonical minute:second form '{lit.canonical_mmss}'."
        )
        if lit.grain in {"second", "minute"}:
            lines.append(
                f"  Stored values may be finer (e.g. M:SS.mmm). Truncate / prefix-match "
                f"to '{lit.canonical_mmss}' (also accept leading zeros / hour=0 forms). "
                f"Do NOT require col = '{lit.raw}'."
            )
            lines.append(
                f"  Suggested predicate: col LIKE '{lit.canonical_mmss}%' "
                f"OR floor(parsed_seconds(col)) = {int(lit.total_seconds)}."
            )
        lines.append(
            "  If a probe at this grain returns rows, run the SAME predicate with "
            "final=true and keep ALL ties — do not retreat to exact string equality."
        )
    for dm in plan.date_months[:3]:
        lines.append(
            f"- Question month '{dm.raw}' → year={dm.year} month={dm.month}. "
            f"Truncate stored dates to month before '='."
        )
    for hint in plan.knowledge_hints[:3]:
        lines.append(f"- {hint}")
    return "\n".join(lines)


def sql_uses_exact_question_literal(sql: str, plan: NormalizePlan) -> bool:
    """True when SQL equality-compares a column to a raw question time/date literal."""
    if not plan.has_work:
        return False
    text = sql or ""
    raws = {lit.raw for lit in plan.time_literals} | {d.raw for d in plan.date_months}
    for match in re.finditer(r"""=\s*'([^']+)'|=\s*"([^"]+)\"""", text):
        value = match.group(1) or match.group(2) or ""
        if value in raws:
            return True
        for lit in plan.time_literals:
            if _same_hms_spelling(value, lit.raw):
                return True
    return False


def _same_hms_spelling(a: str, b: str) -> bool:
    """True if two strings are the same H:MM:SS value with optional leading zeros."""
    pa, pb = parse_time_token(a), parse_time_token(b)
    if pa is None or pb is None:
        return False
    if pa.grain == "millisecond" or pb.grain == "millisecond":
        return abs(pa.total_seconds - pb.total_seconds) < 1e-9
    # Same whole seconds and both written with two ':' (H:MM:SS family) OR identical raw.
    if a == b:
        return True
    return (
        int(pa.total_seconds) == int(pb.total_seconds)
        and a.count(":") == 2
        and b.count(":") == 2
    )


def sql_uses_coarse_time_match(sql: str, plan: NormalizePlan) -> bool:
    """True when SQL matches at question grain (prefix / LIKE / floor seconds)."""
    if not plan.time_literals:
        return False
    text = sql or ""
    for lit in plan.time_literals:
        canon = lit.canonical_mmss
        # LIKE '1:54%' or LIKE '%1:54%' or LIKE '1:54.%'
        for pattern in _LIKE_RE.finditer(text):
            like = pattern.group(1) or pattern.group(2) or pattern.group(3) or ""
            if canon in like.replace("%", ""):
                return True
            if like.startswith(canon) or f"%{canon}" in like or like.startswith(f"{canon}."):
                return True
        if f"LIKE '{canon}%'" in text or f'LIKE "{canon}%"' in text:
            return True
        if f"LIKE '%{canon}%'" in text or f"LIKE '{canon}.%" in text:
            return True
        # floor / cast seconds equality to the question's whole seconds
        whole = str(int(lit.total_seconds))
        if re.search(rf"(?:floor|trunc).*{whole}|=\s*{whole}\b", text, flags=re.IGNORECASE):
            if re.search(r"second|epoch|to_seconds|interval", text, flags=re.IGNORECASE):
                return True
    return False


def sql_is_unrelated_scan(sql: str, plan: NormalizePlan) -> bool:
    """Heuristic: DISTINCT / schema dumps that should not clear exact-empty streaks."""
    text = (sql or "").upper()
    if "DISTINCT" in text and not sql_uses_coarse_time_match(sql, plan):
        return True
    if re.search(r"\bGROUP\s+BY\b", text) and "=" not in text and "LIKE" not in text:
        return True
    if "SQLITE_MASTER" in text or "DUCKDB_TABLES" in text or "DESCRIBE" in text:
        return True
    return False


def find_promotable_probe(
    *,
    question: str,
    steps: list[StepRecord],
    plan: NormalizePlan | None,
) -> tuple[str, str] | None:
    """Pick a non-final nonempty probe SQL related to normalized question values.

    Returns (sql, reason) or None. Prefers coarse time matches and narrower SELECTs.
    Trailing LIMIT is stripped so promotion can be a full-scan final.
    """
    if plan is None or not plan.has_work:
        return None
    best: tuple[float, int, str, str] | None = None
    for index, step in enumerate(steps):
        if step.action != "run_sql" or not step.ok:
            continue
        if not isinstance(step.action_input, dict):
            continue
        if bool(step.action_input.get("final")):
            continue
        sql = step.action_input.get("sql")
        if not isinstance(sql, str) or not sql.strip():
            continue
        content = step.observation.get("content") if isinstance(step.observation, dict) else None
        if not isinstance(content, dict):
            continue
        count = content.get("row_count")
        if not isinstance(count, int) or count <= 0:
            continue
        if not sql_uses_coarse_time_match(sql, plan) and not _probe_mentions_canon(sql, plan):
            continue
        score = 3.0 + min(count, 20) * 0.1 + index * 0.05
        upper = sql.upper()
        if upper.strip().startswith("SELECT *"):
            score -= 1.5
        if "JOIN" in upper:
            score += 0.5
        if re.search(r"\bLIMIT\s+\d+\b", upper):
            score -= 0.2
        reason = f"coarse_probe_rows={count}"
        if best is None or score > best[0]:
            best = (score, index, sql, reason)
    if best is None:
        return None
    cleaned = re.sub(r"\s+LIMIT\s+\d+\s*$", "", best[2].strip(), flags=re.IGNORECASE)
    return cleaned, best[3]


def _probe_mentions_canon(sql: str, plan: NormalizePlan) -> bool:
    text = sql or ""
    for lit in plan.time_literals:
        if lit.canonical_mmss in text:
            return True
    return False
