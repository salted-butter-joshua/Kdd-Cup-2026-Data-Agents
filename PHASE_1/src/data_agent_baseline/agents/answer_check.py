"""Final-answer guards: full-table scan requirement + conservative column prune.

Prune policy (mechanism-level):
- Proposed SQL columns are the only physical columns.
- Inferencer votes keep/drop on those names (or lists names that must align back).
- Drop only with confidence; uncertainty → keep original columns.
- Single-column tighten requires question AND table-structure dual conditions.
- Never invent a column by picking candidates[-1] when no metric match exists.
"""

from __future__ import annotations

import json
import re
from typing import Any, Literal

from data_agent_baseline.agents.model import ModelAdapter, ModelMessage
from data_agent_baseline.agents.answer_contract import drop_unasked_sidecar_columns
from data_agent_baseline.agents.runtime import StepRecord
from data_agent_baseline.benchmark.schema import AnswerTable

Vote = Literal["keep", "drop", "unknown"]


def _is_final_run_sql(step: StepRecord) -> bool:
    if step.action != "run_sql" or not step.ok:
        return False
    content = step.observation.get("content") if isinstance(step.observation, dict) else None
    return isinstance(content, dict) and bool(content.get("full_scan"))


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", flags=re.IGNORECASE | re.DOTALL)

# Candidate single-scalar question shapes (not sufficient alone to hard-prune).
_SINGLE_METRIC_RE = re.compile(
    r"(?:"
    r"what(?:'s|\s+is)\s+the\s+percentage|"
    r"what\s+percentage|"
    r"percentage\s+of|"
    r"\bpercent(?:age)?\b|"
    r"\bhow\s+many\b|"
    r"\bhow\s+much\b|"
    r"what(?:'s|\s+is)\s+the\s+(?:total|average|avg|mean|count|number|ratio|share)"
    r")",
    flags=re.IGNORECASE,
)
_MULTI_OUTPUT_RE = re.compile(
    r"(?:"
    r"\baverage\b.+\band\b.+\b(?:average|avg|mean)\b|"
    r"\b(?:avg|mean)\b.+\band\b.+\b(?:average|avg|mean)\b|"
    r"\bfull\s+name\b.+\band\b|"
    r"\band\b.+\btotal\s+cost\b|"
    r"\bthe\s+[a-z][\w\s]{0,40}\band\s+the\s+[a-z]"
    r")",
    flags=re.IGNORECASE | re.DOTALL,
)
_PERCENT_HINT_RE = re.compile(
    r"percent|percentage|pct|\bratio\b|\bshare\b|proportion",
    flags=re.IGNORECASE,
)
_PERCENT_COL_RE = re.compile(
    r"percent|pct|ratio|share|proportion|rate",
    flags=re.IGNORECASE,
)
_COUNT_COL_RE = re.compile(
    r"(?:^|_)(?:n|cnt|count|total|num|number)(?:_|$)|count$|total$",
    flags=re.IGNORECASE,
)
_AVG_COL_RE = re.compile(r"(?:^|_)(?:avg|average|mean)(?:_|$)|(?:avg|average|mean)", flags=re.IGNORECASE)
_METRIC_COL_RE = re.compile(
    r"percent|pct|ratio|share|proportion|rate|"
    r"(?:^|_)(?:n|cnt|count|total|num|number|sum|min|max|avg|average|mean)(?:_|$)|"
    r"(?:count|total|sum|avg|average|mean)$",
    flags=re.IGNORECASE,
)
# High-confidence helpers. `total_*` is helper only when a percent/ratio column
# also exists in the same table (denominator pattern) — never alone as the metric.
_HELPER_ALWAYS_RE = re.compile(
    r"(?:^|_)(?:denom|denominator|numer|numerator|helper|debug)(?:_|$)|"
    r"(?:^|_)(?:post_count|row_count)(?:_|$)|"
    r"(?:customerid|userid|postid|creationdate)$",
    flags=re.IGNORECASE,
)
_TOTAL_PREFIX_RE = re.compile(r"(?:^|_)total(?:_|$)|^total", flags=re.IGNORECASE)
_SOFT_EXTRA_COL_RE = re.compile(
    r"(?:^|_)(?:date|time|amount|balance|score|label|text|description|created|updated)"
    r"(?:_|$)|(?:date|time|amount|balance|score)$",
    flags=re.IGNORECASE,
)

# Synonym clusters: an inferred label may expand to several proposed columns.
_NAME_PAIR_GROUPS: tuple[tuple[str, ...], ...] = (
    ("first_name", "last_name"),
    ("forename", "surname"),
    ("given_name", "family_name"),
    ("firstname", "lastname"),
)
_NAME_LABELS = frozenset(
    {
        "name",
        "full_name",
        "fullname",
        "full name",
        "member_name",
        "person_name",
        "driver_name",
    }
)

INFER_COLUMNS_PROMPT = """
You vote which submitted columns to keep for the final answer table.

Question:
{question}

Columns the agent is about to submit (authoritative physical names):
{proposed}

Return exactly one JSON object and nothing else. Prefer votes form:
{{"votes": {{"<exact proposed name>": "keep"|"drop"|"unknown", ...}}}}

You may instead return:
{{"columns": ["<exact proposed name>", ...]}}
as the keep-set (names MUST be copied from the proposed list).

Rules:
- Only refer to names that appear in the proposed list. Never invent new names
  (do not invent full_name when proposed has first_name/last_name — vote keep on both).
- Default to "keep" or "unknown" when unsure. Use "drop" only for clear helpers:
  denominators/numerators beside a percentage, join ids, debug counts.
- Percentage / ratio questions: keep the percentage/ratio column; drop total_*/count
  helpers used only to build it — but only if that percentage column is in proposed.
- Multi-metric questions (e.g. average A and average B; full name and total cost):
  keep every asked metric / identity column.
- "How many" / single count: usually one column when proposed already has it.
- If nothing clearly matches, vote keep on all proposed columns (or return them all
  in "columns").
""".strip()


def has_full_sql_scan(steps: list[StepRecord]) -> bool:
    """True if a prior successful run_sql used final=true (untruncated warehouse query)."""
    return any(_is_final_run_sql(step) for step in steps)


def has_full_table_scan(steps: list[StepRecord]) -> bool:
    """Compatibility alias: full scan now means a final run_sql."""
    return has_full_sql_scan(steps)


def used_preview_without_python(steps: list[StepRecord]) -> bool:
    """Deprecated: NL2SQL path has no CSV preview tools. Always False."""
    return False


def preview_answer_rejected_observation() -> dict[str, Any]:
    return sql_answer_rejected_observation()


def sql_answer_rejected_observation() -> dict[str, Any]:
    return {
        "ok": False,
        "error": (
            "answer rejected: final rows must come from run_sql with final=true "
            "(full table, no probe LIMIT). knowledge.md is already in the task prompt."
        ),
        "hint": "Call list_tables, probe with run_sql, then run_sql final=true, then answer.",
        "scan_check": {"full_scan": False},
    }


def wants_single_metric_column(question: str) -> bool:
    """True when the question looks like a single-scalar ask (candidate only)."""
    return bool(_SINGLE_METRIC_RE.search(question or ""))


def question_looks_like_multi_output(question: str) -> bool:
    """True when the question likely asks for more than one output field."""
    return bool(_MULTI_OUTPUT_RE.search(question or ""))


def percent_like_columns(columns: list[str]) -> list[str]:
    return [name for name in columns if _PERCENT_COL_RE.search(name)]


def metric_like_columns(columns: list[str]) -> list[str]:
    return [name for name in columns if _METRIC_COL_RE.search(name)]


def avg_like_columns(columns: list[str]) -> list[str]:
    return [name for name in columns if _AVG_COL_RE.search(name)]


def structural_multi_output(columns: list[str]) -> bool:
    """Table already carries multiple parallel metrics (e.g. two avg_* columns)."""
    return len(avg_like_columns(columns)) >= 2


def is_helper_column(name: str, *, proposed: list[str] | None = None) -> bool:
    """True for clear helpers. total_* counts only beside a percent/ratio column."""
    text = name or ""
    if _HELPER_ALWAYS_RE.search(text):
        return True
    if (
        proposed
        and percent_like_columns(proposed)
        and _TOTAL_PREFIX_RE.search(text)
        and not _PERCENT_COL_RE.search(text)
    ):
        return True
    return False


def is_soft_extra_column(name: str) -> bool:
    return bool(_SOFT_EXTRA_COL_RE.search(name or ""))


def can_enforce_single_metric(question: str, columns: list[str]) -> bool:
    """Dual gate: scalar-looking question AND exactly one matching metric column."""
    if len(columns) <= 1:
        return False
    if question_looks_like_multi_output(question) or structural_multi_output(columns):
        return False
    if not wants_single_metric_column(question):
        return False
    if _PERCENT_HINT_RE.search(question or ""):
        return len(percent_like_columns(columns)) == 1
    return len(metric_like_columns(columns)) == 1


def _pick_single_metric_column(question: str, candidates: list[str]) -> str | None:
    """Return the unique matching metric column, or None when unsafe to choose."""
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    if _PERCENT_HINT_RE.search(question or ""):
        pct = percent_like_columns(candidates)
        if len(pct) == 1:
            return pct[0]
        return None
    mets = metric_like_columns(candidates)
    if len(mets) == 1:
        return mets[0]
    # Prefer an explicit count column only when it is the sole metric-like hit.
    counts = [
        name
        for name in candidates
        if _COUNT_COL_RE.search(name) and not _PERCENT_COL_RE.search(name)
    ]
    if len(counts) == 1 and len(mets) <= 1:
        return counts[0]
    return None


def enforce_single_metric_column(question: str, columns: list[str]) -> list[str]:
    """Tighten to one column only when dual conditions hold; else no-op."""
    if not can_enforce_single_metric(question, columns):
        return list(columns)
    picked = _pick_single_metric_column(question, columns)
    if picked is None:
        return list(columns)
    return [picked]


def _strip_fence(text: str) -> str:
    match = _FENCE_RE.search(text.strip())
    if match is not None:
        return match.group(1).strip()
    return text.strip()


def parse_inferred_columns(raw: str) -> list[str]:
    """Backward-compatible parser for {\"columns\": [...]} responses."""
    payload = json.loads(_strip_fence(raw))
    if isinstance(payload, list):
        columns = payload
    elif isinstance(payload, dict):
        columns = payload.get("columns", [])
    else:
        raise ValueError("inferred columns must be a JSON object or list")
    if not isinstance(columns, list) or not all(isinstance(item, str) for item in columns):
        raise ValueError("columns must be a list of strings")
    return [item.strip() for item in columns if item.strip()]


def parse_column_votes(raw: str) -> dict[str, Vote] | None:
    """Parse {\"votes\": {name: keep|drop|unknown}}. None if payload has no votes."""
    payload = json.loads(_strip_fence(raw))
    if not isinstance(payload, dict):
        return None
    votes_raw = payload.get("votes")
    if not isinstance(votes_raw, dict):
        return None
    parsed: dict[str, Vote] = {}
    for key, value in votes_raw.items():
        if not isinstance(key, str) or not key.strip():
            continue
        if not isinstance(value, str):
            continue
        normalized = value.strip().lower()
        if normalized in {"keep", "drop", "unknown"}:
            parsed[key.strip()] = normalized  # type: ignore[assignment]
    return parsed


def _tokenize(text: str) -> set[str]:
    return {tok for tok in re.split(r"[^a-z0-9]+", (text or "").casefold()) if tok}


def _score_name_match(inferred: str, proposed: str) -> float:
    if inferred.casefold() == proposed.casefold():
        return 1.0
    left = _tokenize(inferred)
    right = _tokenize(proposed)
    if not left or not right:
        return 0.0
    overlap = left & right
    if not overlap:
        return 0.0
    return len(overlap) / max(len(left), len(right))


def align_name_to_proposed(inferred: str, proposed: list[str]) -> list[str]:
    """Map one inferred label onto zero or more proposed columns.

    Unaligned labels yield [] and must be ignored (never used to delete columns).
    """
    label = (inferred or "").strip()
    if not label or not proposed:
        return []

    proposed_fold = {name.casefold(): name for name in proposed}
    if label.casefold() in proposed_fold:
        return [proposed_fold[label.casefold()]]

    # Name / full_name → split identity pairs present in proposed.
    if label.casefold().replace(" ", "_") in _NAME_LABELS or label.casefold() in _NAME_LABELS:
        fold_set = set(proposed_fold)
        for group in _NAME_PAIR_GROUPS:
            if all(part in fold_set for part in group):
                return [proposed_fold[part] for part in group]

    best_name: str | None = None
    best_score = 0.0
    for name in proposed:
        score = _score_name_match(label, name)
        if score > best_score:
            best_score = score
            best_name = name
    if best_name is not None and best_score >= 0.5:
        return [best_name]
    return []


def align_inferred_to_proposed(proposed: list[str], inferred: list[str]) -> list[str]:
    """Align inferred names onto proposed; preserve proposed order; skip failures."""
    kept: list[str] = []
    seen: set[str] = set()
    for label in inferred:
        for name in align_name_to_proposed(label, proposed):
            if name not in seen:
                kept.append(name)
                seen.add(name)
    return kept


def intersect_columns(proposed: list[str], inferred: list[str]) -> list[str]:
    """Align then intersect. Empty alignment → keep original proposed (conservative)."""
    aligned = align_inferred_to_proposed(proposed, inferred)
    return aligned if aligned else list(proposed)


def merge_column_decisions(
    question: str,
    proposed: list[str],
    *,
    votes: dict[str, Vote] | None = None,
    keep_names: list[str] | None = None,
) -> list[str]:
    """Conservative merge: prefer keep/unknown; drop only with confidence."""
    if not proposed:
        return []

    multi = question_looks_like_multi_output(question) or structural_multi_output(proposed)

    # Start optimistic: everything kept.
    decision: dict[str, Vote] = {name: "keep" for name in proposed}

    if votes:
        fold_map = {name.casefold(): name for name in proposed}
        for raw_name, vote in votes.items():
            aligned = align_name_to_proposed(raw_name, proposed)
            targets = aligned or (
                [fold_map[raw_name.casefold()]] if raw_name.casefold() in fold_map else []
            )
            for name in targets:
                if vote == "keep":
                    decision[name] = "keep"
                elif vote == "drop":
                    # Only honor drop on helpers/soft extras; never on multi-metric peers.
                    if multi and name in avg_like_columns(proposed):
                        decision[name] = "keep"
                    elif is_helper_column(name, proposed=proposed) or is_soft_extra_column(
                        name
                    ):
                        decision[name] = "drop"
                    # else ignore aggressive drop → stay keep
                # unknown → leave as keep

    if keep_names is not None:
        aligned_keep = align_inferred_to_proposed(proposed, keep_names)
        if not aligned_keep:
            # Free-form names that do not align: do not delete anything.
            return list(proposed)
        keep_set = set(aligned_keep)
        for name in proposed:
            if name in keep_set:
                decision[name] = "keep"
            elif multi and name in avg_like_columns(proposed):
                decision[name] = "keep"
            elif is_helper_column(name, proposed=proposed) or is_soft_extra_column(name):
                decision[name] = "drop"
            else:
                # Non-helper omitted from keep-set: still drop when the model gave an
                # aligned non-empty keep-set (classic extra-column prune), unless multi.
                decision[name] = "keep" if multi else "drop"

    kept = [name for name in proposed if decision.get(name, "keep") != "drop"]
    return kept if kept else list(proposed)


def project_answer(answer: AnswerTable, columns: list[str]) -> AnswerTable:
    index = {name: i for i, name in enumerate(answer.columns)}
    positions = [index[name] for name in columns]
    rows = [[row[pos] for pos in positions] for row in answer.rows]
    return AnswerTable(columns=list(columns), rows=rows)


def _infer_raw(
    model: ModelAdapter,
    *,
    question: str,
    proposed: list[str],
) -> str:
    return model.complete(
        [
            ModelMessage(
                role="user",
                content=INFER_COLUMNS_PROMPT.format(
                    question=question,
                    proposed=json.dumps(proposed, ensure_ascii=False),
                ),
            )
        ]
    )


def infer_question_columns(
    model: ModelAdapter,
    *,
    question: str,
    proposed: list[str],
) -> list[str]:
    """Ask the model; return a keep-list in proposed order (votes expanded to keeps)."""
    raw = _infer_raw(model, question=question, proposed=proposed)
    votes = parse_column_votes(raw)
    if votes is not None:
        vote_fold = {key.casefold(): value for key, value in votes.items()}
        resolved = [
            name
            for name in proposed
            if vote_fold.get(name.casefold(), "unknown") != "drop"
        ]
        return resolved if resolved else list(proposed)
    return parse_inferred_columns(raw)


def prune_answer_columns(
    model: ModelAdapter,
    *,
    question: str,
    answer: AnswerTable,
) -> tuple[AnswerTable, dict[str, Any]]:
    """Conservatively prune extra columns; never hard-pick when unsure."""
    proposed = list(answer.columns)
    single_candidate = wants_single_metric_column(question)
    check: dict[str, Any] = {
        "proposed": proposed,
        "inferred": [],
        "votes": {},
        "kept": proposed,
        "dropped": [],
        "used_original": True,
        "single_metric": single_candidate,
        "single_metric_enforced": False,
        "multi_output": question_looks_like_multi_output(question)
        or structural_multi_output(proposed),
        "error": None,
    }
    try:
        raw = _infer_raw(model, question=question, proposed=proposed)
        votes = parse_column_votes(raw)
        keep_names: list[str] | None = None
        if votes is not None:
            check["votes"] = dict(votes)
            check["inferred"] = [
                name
                for name, vote in votes.items()
                if vote == "keep"
            ]
        else:
            keep_names = parse_inferred_columns(raw)
            check["inferred"] = list(keep_names)

        kept = merge_column_decisions(
            question,
            proposed,
            votes=votes,
            keep_names=keep_names if votes is None else None,
        )
        before_single = list(kept)
        kept = enforce_single_metric_column(question, kept)
        check["single_metric_enforced"] = kept != before_single and can_enforce_single_metric(
            question, before_single
        )
        after_sidecar = drop_unasked_sidecar_columns(question, kept)
        if after_sidecar != kept:
            check["sidecar_dropped"] = [name for name in kept if name not in after_sidecar]
            kept = after_sidecar

        used_original = kept == proposed
        pruned = answer if used_original else project_answer(answer, kept)
        check.update(
            {
                "kept": list(pruned.columns),
                "dropped": [name for name in proposed if name not in pruned.columns],
                "used_original": used_original,
            }
        )
        return pruned, check
    except Exception as exc:  # noqa: BLE001
        # Inference failed → keep original columns (no aggressive hard prune).
        # Dual-condition single-metric may still apply when structure is unambiguous.
        try:
            kept = list(proposed)
            if can_enforce_single_metric(question, proposed):
                kept = enforce_single_metric_column(question, proposed)
            kept = drop_unasked_sidecar_columns(question, kept)
            if kept != proposed:
                pruned = project_answer(answer, kept)
                check.update(
                    {
                        "kept": list(pruned.columns),
                        "dropped": [
                            name for name in proposed if name not in pruned.columns
                        ],
                        "used_original": False,
                        "single_metric_enforced": can_enforce_single_metric(
                            question, proposed
                        )
                        and len(kept) == 1,
                        "error": str(exc),
                    }
                )
                return pruned, check
        except Exception:
            pass
        check["error"] = str(exc)
        return answer, check
