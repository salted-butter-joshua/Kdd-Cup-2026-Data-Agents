"""Task-local plan / evidence blackboard: fingerprint bans, confirmed probes, replan.

No cross-task memory. Illegal SQL classes never execute. Repeated fingerprints
are rejected before DuckDB. Replan is capped.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from data_agent_baseline.agents.runtime import StepRecord
from data_agent_baseline.agents.schema_link import SchemaLinkPlan
from data_agent_baseline.agents.hypothesis import Hypothesis

_WS_RE = re.compile(r"\s+")
_LIMIT_RE = re.compile(r"\s+LIMIT\s+\d+\s*$", flags=re.IGNORECASE)
_ILLEGAL_SQL_RE = re.compile(
    r"\battach\b|\bglob\s*\(|\bduckdb_|\bsqlite_master\b|\binformation_schema\b|"
    r"\bread_csv|\bread_json|\bread_parquet|\bsqlite_attach\b|\bpragma\b|"
    r"\bshow\s+tables\b|\bcopy\s+",
    flags=re.IGNORECASE,
)
MAX_REPLAN = 2
MAX_CATALOG_WHEN_EMPTY = 2


def fingerprint_sql(sql: str) -> str:
    text = (sql or "").strip().rstrip(";")
    text = _LIMIT_RE.sub("", text)
    text = _WS_RE.sub(" ", text).strip().lower()
    return text


def illegal_sql_reason(sql: str) -> str | None:
    if _ILLEGAL_SQL_RE.search(sql or ""):
        return (
            "Banned SQL class (ATTACH/glob/catalog/file-scan). "
            "Use warehouse tables from the schema; do not open files."
        )
    return None


@dataclass
class ConfirmedProbe:
    sql: str
    fingerprint: str
    row_count: int
    columns: list[str]


@dataclass
class EpisodeState:
    plan_text: str = ""
    plan_version: int = 1
    banned: dict[str, str] = field(default_factory=dict)
    confirmed: list[ConfirmedProbe] = field(default_factory=list)
    replan_count: int = 0
    parse_errors: int = 0
    catalog_attempts: int = 0
    empty_warehouse: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def best_probe(self) -> ConfirmedProbe | None:
        nonempty = [item for item in self.confirmed if item.row_count > 0]
        if nonempty:
            return nonempty[-1]
        return self.confirmed[-1] if self.confirmed else None

    def best_probe_for(self, question: str) -> ConfirmedProbe | None:
        """Promotion candidate: scalar questions never get a multi-row name list."""
        from data_agent_baseline.agents.answer_contract import question_wants_scalar_value

        nonempty = [item for item in self.confirmed if item.row_count > 0]
        if not nonempty:
            return self.confirmed[-1] if self.confirmed else None
        if not question_wants_scalar_value(question):
            return nonempty[-1]
        scalarish = [
            item
            for item in nonempty
            if item.row_count == 1 and len(item.columns) <= 2
        ]
        return scalarish[-1] if scalarish else None

    def observe_list_tables(self, table_count: int) -> None:
        self.empty_warehouse = table_count <= 0
        if self.empty_warehouse:
            self.catalog_attempts += 1

    def should_stop_empty_warehouse(self) -> bool:
        return self.empty_warehouse and self.catalog_attempts >= MAX_CATALOG_WHEN_EMPTY

    def gate_sql(self, sql: str) -> str | None:
        illegal = illegal_sql_reason(sql)
        if illegal:
            fp = fingerprint_sql(sql)
            self.banned[fp] = illegal
            return illegal
        fp = fingerprint_sql(sql)
        if not fp:
            return "Empty SQL."
        if fp in self.banned:
            return f"Already explored: {self.banned[fp]}"
        return None

    def record_sql_result(
        self,
        *,
        sql: str,
        ok: bool,
        row_count: int | None,
        columns: list[str] | None,
        error: str | None,
    ) -> None:
        fp = fingerprint_sql(sql)
        if not fp:
            return
        if not ok:
            reason = (error or "sql_error")[:180]
            self.banned[fp] = reason
            return
        count = int(row_count or 0)
        cols = list(columns or [])
        if count <= 0:
            self.banned[fp] = "empty result"
            return
        self.confirmed.append(
            ConfirmedProbe(sql=sql, fingerprint=fp, row_count=count, columns=cols)
        )

    def record_parse_error(self) -> None:
        self.parse_errors += 1

    def reset_parse_errors(self) -> None:
        self.parse_errors = 0

    def maybe_replan(self, *, empty_streak: int) -> bool:
        if self.replan_count >= MAX_REPLAN:
            return False
        conflict = self._grain_conflict()
        stuck = empty_streak >= 4 and len(self.confirmed) >= 1
        if not conflict and not stuck:
            return False
        self.replan_count += 1
        self.plan_version += 1
        if conflict:
            self.plan_text = (
                f"{self.plan_text}\n"
                f"[replan {self.replan_count}] Two nonempty probes disagree. "
                "Prefer knowledge.md, then the entity table named in the question. "
                "Do not oscillate. Next SQL must change table or grain."
            ).strip()
        else:
            self.plan_text = (
                f"{self.plan_text}\n"
                f"[replan {self.replan_count}] Repeated empty probes. "
                "Change column/predicate; banned fingerprints stay banned."
            ).strip()
        self.notes.append(f"replan_{self.replan_count}")
        return True

    def _grain_conflict(self) -> bool:
        if len(self.confirmed) < 2:
            return False
        last, prev = self.confirmed[-1], self.confirmed[-2]
        if last.row_count <= 0 or prev.row_count <= 0:
            return False
        if last.columns and prev.columns and last.columns != prev.columns:
            ratio = max(last.row_count, prev.row_count) / max(
                1, min(last.row_count, prev.row_count)
            )
            return ratio >= 8.0
        if last.row_count > 0 and prev.row_count > 0:
            ratio = max(last.row_count, prev.row_count) / max(
                1, min(last.row_count, prev.row_count)
            )
            return ratio >= 50.0
        return False

    def format_progress(
        self,
        *,
        remaining_seconds: float | None,
        remaining_steps: int | None,
        table_count: int | None,
        extract_incomplete: bool = False,
        dual_block: str = "",
    ) -> str:
        best = self.best_probe
        best_line = "none"
        if best is not None:
            snippet = best.sql.replace("\n", " ")
            if len(snippet) > 180:
                snippet = snippet[:180] + "…"
            best_line = f"rows={best.row_count} sql={snippet}"
        banned_n = len(self.banned)
        reasons = sorted({reason.split(":")[0][:40] for reason in self.banned.values()})[:6]
        lines = [
            "PROGRESS (authoritative; do not retry banned SQL):",
            f"plan_v{self.plan_version}: {self.plan_text or '(use schema + knowledge)'}",
            f"best_probe: {best_line}",
            f"banned_fingerprints: {banned_n}"
            + (f" ({', '.join(reasons)})" if reasons else ""),
        ]
        if remaining_seconds is not None:
            lines.append(f"remaining_seconds: {max(0, int(remaining_seconds))}")
        if remaining_steps is not None:
            lines.append(f"remaining_steps: {remaining_steps}")
        if table_count is not None:
            lines.append(f"warehouse_tables: {table_count}")
        if extract_incomplete:
            lines.append(
                "Document extract is incomplete. Call search_docs for missing "
                "measure/join keys; do not submit 0 from empty columns. Query "
                "CSV/JSON/SQLite only. Do not glob files."
            )
        if dual_block:
            lines.append(dual_block)
        if self.empty_warehouse:
            lines.append("Warehouse has zero tables. Stop catalog exploration.")
        if self.replan_count >= MAX_REPLAN:
            lines.append("Replan cap reached. Submit best_probe with final=true then answer.")
        return "\n".join(lines)


def build_plan0(
    *,
    question: str,
    hypotheses: list[Hypothesis],
    link: SchemaLinkPlan | None,
    table_count: int,
    extract_incomplete: bool,
) -> str:
    bits: list[str] = []
    if extract_incomplete:
        bits.append("Docs not fully extracted; solve from structured tables.")
    if table_count <= 0:
        bits.append("No tables. Do not glob/ATTACH. Stop after confirming list_tables.")
        return " ".join(bits)
    if link is not None and link.candidate_columns:
        names = [
            f"{cand.table}.{cand.column}"
            for cand in link.candidate_columns[:6]
            if cand.table and cand.column
        ]
        if names:
            bits.append("Focus tables/cols: " + ", ".join(names))
    if hypotheses:
        bits.append("Hypotheses: " + "; ".join(h.title for h in hypotheses[:4]))
    q = (question or "").strip().replace("\n", " ")
    if len(q) > 160:
        q = q[:160] + "…"
    bits.append(f"Answer: {q}")
    bits.append("Probe encodings/joins, then final=true SELECT, then answer. No file scans.")
    return " ".join(bits)


def pack_step_window(
    steps: list[StepRecord],
    *,
    keep_full: int = 3,
    skip_fold_below: int = 6,
) -> tuple[list[StepRecord], list[StepRecord]]:
    """Return (folded_prefix, recent_full). Prefix is summarized by the caller."""
    usable = [step for step in steps if step.step_index > 0]
    if len(usable) < skip_fold_below:
        return [], usable
    if len(usable) <= keep_full:
        return [], usable
    return usable[:-keep_full], usable[-keep_full:]
