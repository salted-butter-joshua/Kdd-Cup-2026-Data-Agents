"""Dual-grain aggregation gate for extremum questions (and averages with a SUM formula).

Forces the agent to probe both a fine (row-level MIN/MAX/AVG) path and a coarse
(SUM / GROUP BY aggregate) path before answer. Bare average/mean does not trigger
this gate — member-expansion still does, via wants_member_expansion_check.
Does not rewrite SQL or hardcode task answers — when the two paths differ,
prompts tell the model to prefer fine unless knowledge says otherwise.
"""

from __future__ import annotations

import re
from typing import Any

from data_agent_baseline.agents.gate_common import undecided_payload
from data_agent_baseline.agents.runtime import StepRecord
from data_agent_baseline.agents.sql_ir import normalize_sql

# Extremum: row-level vs SUM-then-compare (task_25 class).
_EXTREMUM_Q_RE = re.compile(
    r"(?:"
    r"\b(?:lowest|highest|cheapest|smallest|largest|fewest|minimum|maximum|least)\b|"
    r"\bmost\s+(?:expensive|costly|cheap)\b"
    r")",
    flags=re.IGNORECASE,
)
_AVERAGE_Q_RE = re.compile(r"\b(?:average|mean)\b|\bavg\b", flags=re.IGNORECASE)
_SUM_FORMULA_RE = re.compile(r"\bSUM\s*\(", flags=re.IGNORECASE)

_SUM_RE = re.compile(r"\bSUM\s*\(", flags=re.IGNORECASE)
_MIN_MAX_AVG_RE = re.compile(r"\b(?:MIN|MAX|AVG)\s*\(", flags=re.IGNORECASE)
_GROUP_BY_RE = re.compile(r"\bGROUP\s+BY\b", flags=re.IGNORECASE)


def wants_dual_grain_check(
    question: str, knowledge_text: str | None = None
) -> bool:
    """True when row-level vs entity-total is a real competing pair.

    Extremum questions always qualify. Bare average/mean does not — unless
    knowledge defines a SUM(...) coarse formula. Member-expansion uses
    ``wants_member_expansion_check`` (still includes average).
    """
    text = question or ""
    if _EXTREMUM_Q_RE.search(text):
        return True
    if _AVERAGE_Q_RE.search(text) and _SUM_FORMULA_RE.search(knowledge_text or ""):
        return True
    return False


def wants_member_expansion_check(question: str) -> bool:
    """Filter-then-expand questions: extremum or average (task_199 class)."""
    text = question or ""
    return bool(_EXTREMUM_Q_RE.search(text) or _AVERAGE_Q_RE.search(text))


def grains_in_sql(sql: str) -> set[str]:
    """Classify a SQL string as fine and/or coarse aggregate grain.

    - fine: MIN/MAX/AVG without SUM in the same statement (row-level extremum / mean)
    - coarse: SUM(...) and/or GROUP BY with aggregates (entity totals then compare)

    Consumes the canonical IR (§13.1): quoted identifiers cannot bypass detection.
    """
    text = normalize_sql(sql)
    grains: set[str] = set()
    has_sum = bool(_SUM_RE.search(text))
    has_minmaxavg = bool(_MIN_MAX_AVG_RE.search(text))
    has_group = bool(_GROUP_BY_RE.search(text))
    if has_sum or (has_group and has_minmaxavg):
        grains.add("coarse")
    if has_minmaxavg and not has_sum:
        grains.add("fine")
    return grains


def _step_sql(step: StepRecord) -> str:
    if isinstance(step.action_input, dict):
        sql = step.action_input.get("sql")
        if isinstance(sql, str) and sql.strip():
            return sql
    content = step.observation.get("content") if isinstance(step.observation, dict) else None
    if isinstance(content, dict):
        sql = content.get("sql")
        if isinstance(sql, str):
            return sql
    return ""


def _step_grain_tag(step: StepRecord) -> str | None:
    """Optional explicit tag from action_input.grain: fine | coarse."""
    if not isinstance(step.action_input, dict):
        return None
    tag = step.action_input.get("grain")
    if isinstance(tag, str):
        normalized = tag.strip().lower()
        if normalized in {"fine", "coarse"}:
            return normalized
    content = step.observation.get("content") if isinstance(step.observation, dict) else None
    if isinstance(content, dict):
        tag = content.get("grain")
        if isinstance(tag, str) and tag.strip().lower() in {"fine", "coarse"}:
            return tag.strip().lower()
    return None


# Aggregation words and question stopwords never anchor a measure column.
_ANCHOR_STOPWORDS = frozenset(
    {
        "the", "an", "of", "in", "on", "for", "to", "is", "are", "was", "were",
        "has", "have", "had", "which", "what", "who", "whom", "whose", "when",
        "where", "how", "why", "and", "or", "by", "at", "from", "with", "that",
        "this", "these", "those", "their", "its", "as", "be", "been", "do",
        "does", "did", "not", "no", "all", "any", "each", "per", "than", "then",
        "there", "into", "over", "under", "between", "among", "within", "during",
        "before", "after", "most", "least", "lowest", "highest", "average",
        "mean", "avg", "minimum", "maximum", "fewest", "smallest", "largest",
        "cheapest", "expensive", "costly", "many", "much", "total", "number",
        "count", "list", "give", "show", "find", "get", "me", "us", "it",
    }
)

_MIN_MAX_AVG_ARG_RE = re.compile(
    r"\b(?:MIN|MAX|AVG)\s*\(\s*(?:DISTINCT\s+)?"
    r"(?:[A-Za-z_][A-Za-z0-9_]*\s*\.\s*)?([A-Za-z_][A-Za-z0-9_]*)",
    flags=re.IGNORECASE,
)


def _question_anchor_tokens(question: str) -> set[str]:
    """Content tokens of the question that may anchor a measure column."""
    tokens = {
        m.group(0).casefold()
        for m in re.finditer(r"[A-Za-z0-9_]{2,}", question or "")
    }
    return tokens - _ANCHOR_STOPWORDS


def _fine_anchored(sql: str, question_tokens: set[str]) -> bool:
    """True when a MIN/MAX/AVG targets a column the question itself names.

    手段三 (§13.3): a profiling probe like MIN(spent)/MAX(spent) only inspects a
    column's range — it is NOT fine-grain evidence for a question about "cost".
    Fine evidence requires the aggregate to act on a column anchored in the
    question text (exact token or underscore-part match).
    """
    if not question_tokens:
        return True  # no anchor information → keep legacy behavior
    for match in _MIN_MAX_AVG_ARG_RE.finditer(normalize_sql(sql)):
        column = match.group(1).casefold()
        if column in question_tokens:
            return True
        parts = {p for p in re.split(r"[_\s]+", column) if len(p) >= 2}
        if parts & question_tokens:
            return True
    return False


def collect_aggregate_grains(
    steps: list[StepRecord],
    question: str | None = None,
) -> set[str]:
    """Union of grains seen on successful run_sql steps (probe or final).

    When ``question`` is given, a probe only counts as *fine* evidence when its
    MIN/MAX/AVG targets a question-anchored column (§13.3). Explicit
    ``grain`` tags are trusted as-is.
    """
    tokens = _question_anchor_tokens(question) if question is not None else set()
    found: set[str] = set()
    for step in steps:
        if step.action != "run_sql" or not step.ok:
            continue
        tag = _step_grain_tag(step)
        if tag is not None:
            found.add(tag)
        sql = _step_sql(step)
        grains = grains_in_sql(sql)
        if (
            "fine" in grains
            and question is not None
            and not _fine_anchored(sql, tokens)
        ):
            grains = grains - {"fine"}
        found |= grains
    return found


def dual_grain_satisfied(
    question: str,
    steps: list[StepRecord],
    knowledge_text: str | None = None,
) -> bool:
    """If the question needs dual grain, require both fine and coarse probes."""
    if not wants_dual_grain_check(question, knowledge_text):
        return True
    grains = collect_aggregate_grains(steps, question)
    return "fine" in grains and "coarse" in grains


def dual_grain_rejected_observation(
    question: str,
    steps: list[StepRecord],
    knowledge_text: str | None = None,
) -> dict[str, Any]:
    grains = sorted(collect_aggregate_grains(steps, question))
    missing = [g for g in ("fine", "coarse") if g not in set(grains)]
    return {
        "ok": False,
        "error": (
            "answer rejected: this question needs both aggregation grains before submit. "
            f"seen={grains or ['none']}; missing={missing}."
        ),
        "hint": (
            "Probe TWO versions with run_sql (final=false is enough for the missing grain): "
            "(1) fine: row-level MIN/MAX/AVG on the measure column named like the question "
            "(e.g. MIN(cost)), optionally tag action_input.grain=\"fine\"; "
            "(2) coarse: SUM / GROUP BY entity totals then MIN/MAX/AVG "
            "(e.g. SUM(spent) per event), tag grain=\"coarse\". "
            "Compare the two result sets. If they differ, prefer the fine path unless "
            "knowledge.md explicitly defines the coarse formula. Then run_sql final=true "
            "with the chosen query and call answer."
        ),
        "grain_check": {
            "required": True,
            "seen": grains,
            "missing": missing,
            "question_flagged": wants_dual_grain_check(question, knowledge_text),
        },
    }


_SUM_MEASURE_RE = re.compile(
    r"\bSUM\s*\(\s*(?:DISTINCT\s+)?"
    r"(?:[A-Za-z_][A-Za-z0-9_]*\s*\.\s*)?([A-Za-z_][A-Za-z0-9_]*)\s*\)",
    flags=re.IGNORECASE,
)


def _extract_coarse_measure(sql: str) -> str | None:
    """Return the column inside the first SUM(...) call, if any.

    Runs on the canonical IR, so quoted identifiers (SUM(e."cost")) cannot
    bypass extraction (task_25 hole, §13.1).
    """
    match = _SUM_MEASURE_RE.search(normalize_sql(sql))
    return match.group(1) if match else None


def _whole_word(text: str, word: str) -> bool:
    """Case-insensitive whole-word search."""
    pattern = r"\b" + re.escape(word) + r"\b"
    return bool(re.search(pattern, text, flags=re.IGNORECASE))


def knowledge_defines_coarse(knowledge_text: str, measure_col: str | None) -> bool:
    """True when knowledge explicitly defines a coarse aggregate for this measure.

    Looks for a sentence/line that mentions the measure column together with a
    coarse aggregate indicator (SUM, GROUP BY, or Total).
    """
    if not knowledge_text or not measure_col:
        return False
    # Split on sentence boundaries and newlines.
    parts = re.split(r"[.!?\n]", knowledge_text)
    for part in parts:
        if (
            _whole_word(part, measure_col)
            and (
                _whole_word(part, "SUM")
                or _whole_word(part, "GROUP BY")
                or _whole_word(part, "Total")
            )
        ):
            return True
    return False


def submit_grain_rejection(
    question: str,
    steps: list[StepRecord],
    sql: str | None,
    knowledge_text: str | None,
) -> dict[str, Any] | None:
    """Reject a coarse final SQL when knowledge does not define the coarse formula.

    When both fine and coarse grains were probed and the final query is coarse,
    require knowledge.md to explicitly define a coarse aggregate for the measure
    column. Otherwise the agent must submit the fine-grain query.
    """
    if not wants_dual_grain_check(question, knowledge_text):
        return None
    grains = collect_aggregate_grains(steps, question)
    if "fine" not in grains or "coarse" not in grains:
        return None
    final_grains = grains_in_sql(sql or "")
    if "coarse" not in final_grains:
        return None
    measure_col = _extract_coarse_measure(sql or "")
    if measure_col is None:
        # Coarse final (e.g. GROUP BY + AVG, no SUM) whose measure we cannot
        # extract deterministically. §13.5: this is UNDECIDED, not a silent
        # pass — the caller lets it through but records telemetry.
        return undecided_payload(
            "grain",
            "final SQL is coarse but no SUM(measure) could be extracted, so the "
            "coarse-formula check cannot run.",
            "Rewrite the aggregate in canonical form (e.g. SUM(table.column) "
            "without quoted identifiers), or submit the fine-grain query.",
        )
    if knowledge_defines_coarse(knowledge_text or "", measure_col):
        return None
    return {
        "ok": False,
        "error": (
            "answer rejected: final SQL uses the coarse grain, but knowledge.md does not "
            f"define a coarse (SUM/GROUP BY/Total) formula for measure '{measure_col}'."
        ),
        "hint": (
            "Both grains were probed. Since knowledge.md does not explicitly define the "
            "coarse formula for this measure, prefer the fine-grain query. "
            "Run SQL final=true with the fine-grain query and call answer."
        ),
        "grain_check": {
            "required": True,
            "seen": sorted(grains),
            "final_grain": "coarse",
            "measure_col": measure_col,
            "knowledge_defines_coarse": False,
        },
    }


def pre_submit_grain_notes(
    question: str,
    steps: list[StepRecord],
    sql: str | None,
    knowledge_text: str | None,
) -> list[str]:
    """Soft warnings attached to a successful run_sql final=true observation.

    §13.5: surface gate verdicts BEFORE the model calls answer, so it can fix
    the SQL without spending a rejection round-trip. These are notes, not
    rejections — the answer-time gate re-runs deterministically either way.
    """
    notes: list[str] = []
    if not sql or not wants_dual_grain_check(question, knowledge_text):
        return notes
    if "coarse" not in grains_in_sql(sql):
        return notes
    measure_col = _extract_coarse_measure(sql)
    if measure_col is None:
        notes.append(
            "gate note: this final SQL is coarse-grained, but its measure column "
            "could not be verified (no extractable SUM(table.column)). At answer "
            "time this is recorded as UNDECIDED. Prefer canonical SQL without "
            "quoted identifiers, or submit the fine-grain query."
        )
        return notes
    grains = collect_aggregate_grains(steps, question)
    if "fine" in grains and not knowledge_defines_coarse(
        knowledge_text or "", measure_col
    ):
        notes.append(
            f"gate note: this final SQL is coarse-grained and knowledge.md does "
            f"not define a coarse formula for '{measure_col}'. Submitting it will "
            "be REJECTED. Re-run final=true with the fine-grain query (row-level "
            f"MIN/MAX/AVG on {measure_col}) instead."
        )
    return notes
