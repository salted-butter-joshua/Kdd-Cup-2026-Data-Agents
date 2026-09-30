"""Schema linking: map question + knowledge tokens to candidate tables/columns.

Does NOT classify question types. It only builds a soft candidate set so the
agent can probe the right scope (step 2–3 of the pipeline).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]{2,}")
_SPLIT_RE = re.compile(r"[_\s]+")

# Synonym bags expand question terms toward common warehouse names.
_SYNONYM_BAGS: list[frozenset[str]] = [
    frozenset({"position", "rank", "order", "place", "standing", "track"}),
    frozenset({"number", "num", "no", "code"}),
    frozenset({"round", "race", "heat", "stage"}),
    frozenset({"time", "duration", "finish", "q1", "q2", "q3", "milliseconds"}),
    frozenset({"consumption", "amount", "cost", "spent", "value", "total"}),
    frozenset({"type", "category", "kind", "class", "label"}),
    frozenset({"name", "title", "forename", "surname", "fullname"}),
    frozenset({"element", "atom", "bond"}),
    frozenset({"sex", "gender"}),
    frozenset({"date", "year", "month", "birthday"}),
]


@dataclass(frozen=True, slots=True)
class ColumnCandidate:
    table: str
    column: str
    score: float
    reasons: tuple[str, ...] = ()
    # Column role (§13.3): measure | dimension | identifier.
    role: str = "dimension"


@dataclass(frozen=True, slots=True)
class AmbiguityGroup:
    """Columns that could answer the same question term."""

    term: str
    columns: tuple[ColumnCandidate, ...]


@dataclass(slots=True)
class SchemaLinkPlan:
    question_tokens: set[str] = field(default_factory=set)
    candidate_tables: list[tuple[str, float]] = field(default_factory=list)
    candidate_columns: list[ColumnCandidate] = field(default_factory=list)
    ambiguity_groups: list[AmbiguityGroup] = field(default_factory=list)
    # Soft project-vs-filter hints for same-named columns (not a submit gate).
    homonym_projections: list[str] = field(default_factory=list)

    def format_for_prompt(self, *, max_columns: int = 24) -> str:
        if not self.candidate_tables and not self.candidate_columns:
            return ""
        lines = [
            "Schema linking (candidates only — verify with probes before trusting):",
        ]
        if self.candidate_tables:
            tables = ", ".join(
                f"{name}({score:.2f})" for name, score in self.candidate_tables[:12]
            )
            lines.append(f"- Candidate tables: {tables}")
        if self.candidate_columns:
            cols = ", ".join(
                f"{c.table}.{c.column}({c.score:.2f},{c.role})"
                for c in self.candidate_columns[:max_columns]
            )
            lines.append(f"- Candidate columns: {cols}")
        if self.homonym_projections:
            lines.append(
                "- Same-named columns: project the entity-table copy; use the "
                "event table only as a filter (still probe both):"
            )
            for hint in self.homonym_projections[:6]:
                lines.append(f"  · {hint}")
        if self.ambiguity_groups:
            lines.append(
                "- Ambiguous terms (probe EACH column with the same filter before choosing):"
            )
            for group in self.ambiguity_groups[:6]:
                opts = ", ".join(f"{c.table}.{c.column}" for c in group.columns)
                lines.append(f"  · '{group.term}' → {opts}")
        return "\n".join(lines)


def _tokens(text: str) -> set[str]:
    return {m.group(0).casefold() for m in _TOKEN_RE.finditer(text or "")}


_ID_SNAKE_RE = re.compile(r"(?:^|_)id$", flags=re.IGNORECASE)
_ID_CAMEL_RE = re.compile(r"[a-z0-9]I[dD]$")
_NUMERIC_TYPE_RE = re.compile(
    r"INT|DOUBLE|FLOAT|REAL|DECIMAL|NUMERIC|HUGEINT", flags=re.IGNORECASE
)


def infer_column_role(name: str, dtype: str | None) -> str:
    """Classify a column as measure / dimension / identifier (§13.3).

    - identifier: primary/foreign-key-like names (id, bond_id, CustomerID, …);
    - measure: numeric-typed columns that are not identifiers (aggregatable);
    - dimension: everything else (grouping / filtering attributes).

    Roles feed the schema-linking prompt block and downstream evidence
    classification; inference failure simply yields "dimension" (legacy path).
    """
    text = name or ""
    fold = text.casefold()
    if fold in {"id", "uuid", "guid"} or _ID_SNAKE_RE.search(text) or _ID_CAMEL_RE.search(text):
        return "identifier"
    if dtype and _NUMERIC_TYPE_RE.search(str(dtype)):
        return "measure"
    return "dimension"


def _expand_with_synonyms(tokens: set[str]) -> set[str]:
    expanded = set(tokens)
    for bag in _SYNONYM_BAGS:
        if tokens & bag:
            expanded |= set(bag)
    return expanded


def _parts(name: str) -> set[str]:
    return {p for p in _SPLIT_RE.split(name.casefold()) if len(p) >= 2}


_CAMEL_BOUNDARY_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")

# Columns that mark a table as *filter/event* context when the question names them.
# Used only for table event-fit. Shared leaves are still reranked; tied event-fit
# fail-opens (the asked field may itself be an event column).
_EVENT_COL_BAG = frozenset(
    {
        "q1",
        "q2",
        "q3",
        "race",
        "round",
        "date",
        "year",
        "month",
        "session",
        "status",
        "heat",
        "stage",
    }
)
_EVENT_MARGIN = 1.5
_ENTITY_ATTR_BONUS = 2.5
_FILTER_TABLE_PENALTY = 2.0
_GENERIC_PK_LEAVES = frozenset({"id", "uuid", "guid"})
_KNOWLEDGE_PLACE_BONUS = 2.5
_Q_RANK_RE = re.compile(r"\brank(?:ed|s)?\b", flags=re.IGNORECASE)
_Q_FINISH_POS_RE = re.compile(
    r"\bfinish(?:ed|ing)\s+position\b|"
    r"\bfinishing\s+place\b|"
    r"\bposition\s+order\b|"
    r"\bgrid\s+order\b",
    flags=re.IGNORECASE,
)
_KNOWLEDGE_POSITION_ORDER_RE = re.compile(
    r"position[_\s]*order", flags=re.IGNORECASE
)


def _group_parts(name: str) -> set[str]:
    """Parts used ONLY for ambiguity-group hits: also splits camelCase.

    ``positionOrder`` must hit the position/rank/order synonym bag, otherwise
    the disambiguation gate silently never fires on camelCase schemas (the
    task_86 class). Scoring keeps using ``_parts`` so candidate ranking is
    unchanged.
    """
    parts: set[str] = set()
    for piece in _SPLIT_RE.split(name or ""):
        if len(piece) >= 2:
            parts.add(piece.casefold())
        for sub in _CAMEL_BOUNDARY_RE.split(piece):
            if len(sub) >= 2:
                parts.add(sub.casefold())
    return parts


def _score_name(name: str, tokens: set[str]) -> tuple[float, list[str]]:
    fold = name.casefold()
    parts = _parts(name)
    score = 0.0
    reasons: list[str] = []
    if fold in tokens:
        score += 3.0
        reasons.append("exact")
    overlap = parts & tokens
    if overlap:
        score += 1.5 * len(overlap)
        reasons.append("parts:" + ",".join(sorted(overlap)[:4]))
    for tok in tokens:
        if len(tok) >= 4 and (tok in fold or fold in tok):
            score += 0.5
            reasons.append(f"substr:{tok}")
            break
    return score, reasons


def _column_event_overlap(cname: str, q_tokens: set[str]) -> float:
    """How strongly this column is a question-named event/filter field."""
    fold = cname.casefold()
    parts = _group_parts(cname) | {fold}
    if fold not in _EVENT_COL_BAG and not (parts & _EVENT_COL_BAG):
        return 0.0
    score, _ = _score_name(cname, q_tokens)
    if fold in q_tokens or (parts & q_tokens):
        return max(score, 3.0)
    return score


def _table_event_fit(
    table: dict[str, Any],
    q_tokens: set[str],
    exclude_leaf: str,
) -> tuple[float, list[str]]:
    """Event/filter fit from columns other than the homonym leaf (raw question tokens)."""
    skip = (exclude_leaf or "").casefold()
    total = 0.0
    hits: list[str] = []
    tname = str(table.get("name") or "")
    name_parts = _group_parts(tname) | {tname.casefold()}
    if name_parts & _EVENT_COL_BAG:
        n_score, _ = _score_name(tname, q_tokens)
        if n_score > 0:
            total += n_score
            hits.append(tname)
    for col in table.get("columns") or []:
        cname = str(col.get("name") or "") if isinstance(col, dict) else str(col)
        if not cname or cname.casefold() == skip:
            continue
        overlap = _column_event_overlap(cname, q_tokens)
        if overlap <= 0:
            continue
        total += overlap
        hits.append(cname)
    return total, hits[:4]


def _unique_event_table(
    tables: list[dict[str, Any]],
    homonym_tables: set[str],
    q_tokens: set[str],
    leaf: str,
) -> tuple[str | None, list[str]]:
    """The unique highest event-fit table that also has this homonym, or None."""
    scored: list[tuple[str, float, list[str]]] = []
    for table in tables:
        tname = str(table.get("name") or "")
        if tname.casefold() not in homonym_tables:
            continue
        fit, hits = _table_event_fit(table, q_tokens, leaf)
        scored.append((tname, fit, hits))
    scored.sort(key=lambda item: item[1], reverse=True)
    if not scored or scored[0][1] <= 0:
        return None, []
    if len(scored) > 1 and scored[0][1] - scored[1][1] < _EVENT_MARGIN:
        return None, []
    return scored[0][0], scored[0][2]


def _rerank_homonym_columns(
    *,
    question: str,
    tables: list[dict[str, Any]],
    columns: list[ColumnCandidate],
) -> tuple[list[ColumnCandidate], list[str]]:
    """Boost entity-table copies of a shared leaf; demote the filter-table copy.

    Fail-open when event-fit is tied. Generic PK leaves are not reranked.
    """
    q_raw = _tokens(question)
    buckets: dict[str, list[ColumnCandidate]] = {}
    for candidate in columns:
        buckets.setdefault(candidate.column.casefold(), []).append(candidate)

    adjustments: dict[str, tuple[float, str]] = {}
    hints: list[str] = []
    for leaf, hits in buckets.items():
        tables_with = {c.table.casefold() for c in hits}
        if len(tables_with) < 2 or leaf not in q_raw:
            continue
        if leaf in _GENERIC_PK_LEAVES:
            continue
        event_table, event_cols = _unique_event_table(
            tables, tables_with, q_raw, leaf
        )
        if event_table is None:
            continue
        entity_hits = [
            c for c in hits if c.table.casefold() != event_table.casefold()
        ]
        if len({c.table.casefold() for c in entity_hits}) != 1:
            continue
        entity = entity_hits[0]
        event_key = next(
            c for c in hits if c.table.casefold() == event_table.casefold()
        )
        adjustments[_candidate_key(entity)] = (_ENTITY_ATTR_BONUS, "entity-attr")
        adjustments[_candidate_key(event_key)] = (
            -_FILTER_TABLE_PENALTY,
            "filter-table",
        )
        filt = ", ".join(event_cols) if event_cols else event_table
        hints.append(
            f"{leaf}: project {entity.table}.{entity.column}; "
            f"filter on {event_table} ({filt})"
        )

    if not adjustments:
        return columns, []
    out: list[ColumnCandidate] = []
    for candidate in columns:
        key = _candidate_key(candidate)
        delta_reason = adjustments.get(key)
        if delta_reason is None:
            out.append(candidate)
            continue
        delta, reason = delta_reason
        out.append(
            ColumnCandidate(
                table=candidate.table,
                column=candidate.column,
                score=round(candidate.score + delta, 3),
                reasons=tuple((*candidate.reasons, reason)[:6]),
                role=candidate.role,
            )
        )
    return out, hints


def _knowledge_defines_rank_vs_position_order(knowledge_text: str) -> bool:
    """True when knowledge names both rank and positionOrder as competing fields."""
    text = knowledge_text or ""
    if not _KNOWLEDGE_POSITION_ORDER_RE.search(text):
        return False
    return bool(re.search(r"\brank\b", text, flags=re.IGNORECASE))


def _apply_knowledge_place_boost(
    question: str,
    knowledge_text: str,
    columns: list[ColumnCandidate],
) -> list[ColumnCandidate]:
    """Soft-boost rank vs positionOrder when knowledge lists both and the question picks one side.

    Fail-open when both sides (or neither) appear in the question.
    """
    if not columns or not _knowledge_defines_rank_vs_position_order(knowledge_text):
        return columns
    wants_rank = bool(_Q_RANK_RE.search(question or ""))
    wants_finish = bool(_Q_FINISH_POS_RE.search(question or ""))
    if wants_rank == wants_finish:
        return columns
    target = "rank" if wants_rank else "positionorder"
    reason = "knowledge-rank" if wants_rank else "knowledge-positionOrder"
    out: list[ColumnCandidate] = []
    for candidate in columns:
        if candidate.column.casefold() != target:
            out.append(candidate)
            continue
        out.append(
            ColumnCandidate(
                table=candidate.table,
                column=candidate.column,
                score=round(candidate.score + _KNOWLEDGE_PLACE_BONUS, 3),
                reasons=tuple((*candidate.reasons, reason)[:6]),
                role=candidate.role,
            )
        )
    return out


def link_schema(
    *,
    question: str,
    knowledge_text: str,
    tables: list[dict[str, Any]],
) -> SchemaLinkPlan:
    """Build candidate tables/columns from question + knowledge + warehouse schema."""
    q_tokens = _expand_with_synonyms(_tokens(question) | _tokens(knowledge_text[:4000]))
    plan = SchemaLinkPlan(question_tokens=set(q_tokens))
    if not tables or not q_tokens:
        return plan

    table_scores: list[tuple[str, float]] = []
    columns: list[ColumnCandidate] = []
    for table in tables:
        tname = str(table.get("name") or "")
        if not tname:
            continue
        t_score, _ = _score_name(tname, q_tokens)
        # Light prior: larger tables slightly preferred when name matches weakly.
        n_rows = int(table.get("n_rows") or 0)
        if t_score > 0 and n_rows > 0:
            t_score += min(0.3, n_rows / 1_000_000)
        if t_score > 0:
            table_scores.append((tname, round(t_score, 3)))
        for col in table.get("columns") or []:
            cname = str(col.get("name") or "") if isinstance(col, dict) else str(col)
            if not cname:
                continue
            c_score, reasons = _score_name(cname, q_tokens)
            if t_score > 0:
                c_score += 0.25 * min(t_score, 2.0)
            if c_score <= 0:
                continue
            cdtype = col.get("type") if isinstance(col, dict) else None
            columns.append(
                ColumnCandidate(
                    table=tname,
                    column=cname,
                    score=round(c_score, 3),
                    reasons=tuple(reasons[:4]),
                    role=infer_column_role(cname, cdtype),
                )
            )

    columns = _apply_knowledge_place_boost(question, knowledge_text, columns)
    columns, homonym_hints = _rerank_homonym_columns(
        question=question,
        tables=tables,
        columns=columns,
    )
    table_scores.sort(key=lambda item: item[1], reverse=True)
    columns.sort(key=lambda c: c.score, reverse=True)
    plan.candidate_tables = table_scores[:16]
    plan.candidate_columns = columns[:40]
    plan.homonym_projections = homonym_hints
    plan.ambiguity_groups = _build_ambiguity_groups(question, columns)
    return plan


def _candidate_key(column: ColumnCandidate) -> str:
    return f"{column.table}.{column.column}".casefold()


def _dedupe_columns(hits: list[ColumnCandidate]) -> list[ColumnCandidate]:
    best: dict[str, ColumnCandidate] = {}
    for candidate in hits:
        key = _candidate_key(candidate)
        prev = best.get(key)
        if prev is None or candidate.score > prev.score:
            best[key] = candidate
    return sorted(best.values(), key=lambda item: item.score, reverse=True)


def _build_ambiguity_groups(
    question: str, columns: list[ColumnCandidate]
) -> list[AmbiguityGroup]:
    q_fold = (question or "").casefold()
    q_tokens = _tokens(question)
    groups: list[AmbiguityGroup] = []
    seen_group_keys: set[frozenset[str]] = set()

    def _append(term: str, hits: list[ColumnCandidate]) -> None:
        uniq = _dedupe_columns(hits)
        if len(uniq) < 2:
            return
        key = frozenset(_candidate_key(item) for item in uniq[:6])
        if key in seen_group_keys:
            return
        seen_group_keys.add(key)
        groups.append(AmbiguityGroup(term=term, columns=tuple(uniq[:6])))

    for bag in _SYNONYM_BAGS:
        # Only surface a group when the question mentions something in the bag.
        mentioned = [
            tok
            for tok in bag
            if tok in q_fold or any(tok in t for t in q_tokens)
        ]
        if not mentioned:
            continue
        hits = [
            c
            for c in columns
            if c.column.casefold() in bag or any(p in bag for p in _group_parts(c.column))
        ]
        _append(mentioned[0], hits)

    # Homonym groups: same leaf name on ≥2 tables, and the question uses that name.
    # Identity is table.column — this is the task_80 class (drivers.number vs
    # qualifying.number). Do not synonym-expand the mention check: "No.903"
    # must not pull every `code` column into a homonym group.
    buckets: dict[str, list[ColumnCandidate]] = {}
    for candidate in columns:
        buckets.setdefault(candidate.column.casefold(), []).append(candidate)
    for col_fold, hits in buckets.items():
        if col_fold not in q_tokens:
            continue
        # Generic PKs share a leaf on every table; not the number/time/code class.
        if col_fold in _GENERIC_PK_LEAVES:
            continue
        tables = {item.table.casefold() for item in hits}
        if len(tables) < 2:
            continue
        _append(col_fold, hits)
    return groups
