from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from data_agent_baseline.benchmark.schema import AnswerTable, PublicTask
from data_agent_baseline.tools.warehouse import WarehouseSession, describe_tables, execute_sql

PROBE_LIMIT_DEFAULT = 50
PROBE_SHOW_ROWS = 20


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ToolExecutionResult:
    ok: bool
    content: dict[str, Any]
    is_terminal: bool = False
    answer: AnswerTable | None = None


ToolHandler = Callable[[PublicTask, dict[str, Any]], ToolExecutionResult]


def _preview_rows(rows: list[list[Any]], *, max_rows: int = PROBE_SHOW_ROWS) -> list[list[Any]]:
    return [list(row) for row in rows[:max_rows]]


@dataclass(slots=True)
class ToolRegistry:
    specs: dict[str, ToolSpec]
    handlers: dict[str, ToolHandler]
    session: WarehouseSession | None = None

    def describe_for_prompt(self) -> str:
        lines = []
        for name in sorted(self.specs):
            spec = self.specs[name]
            lines.append(f"- {spec.name}: {spec.description}")
            lines.append(f"  input_schema: {spec.input_schema}")
        return "\n".join(lines)

    def execute(self, task: PublicTask, action: str, action_input: dict[str, Any]) -> ToolExecutionResult:
        if action not in self.handlers:
            raise KeyError(f"Unknown tool: {action}")
        return self.handlers[action](task, action_input)


def create_default_tool_registry(model: Any | None = None) -> ToolRegistry:
    session = WarehouseSession(model=model)

    def _list_tables(task: PublicTask, _action_input: dict[str, Any]) -> ToolExecutionResult:
        state = session.for_task(task)
        tables = describe_tables(state)
        return ToolExecutionResult(
            ok=True,
            content={
                "tables": tables,
                "table_count": len(tables),
                "note": (
                    "These tables belong only to this task's in-memory DuckDB. "
                    "Official SQLite under context/db/*.db is included. "
                    "knowledge.md is copied verbatim into the task prompt and is not a table. "
                    "Narrative docs are extracted as tables named after the file stem."
                ),
            },
        )

    def _run_sql(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
        sql = str(action_input.get("sql", "")).strip()
        if not sql:
            return ToolExecutionResult(ok=False, content={"error": "run_sql.sql is required."})
        final = bool(action_input.get("final", False))
        limit = int(action_input.get("limit", PROBE_LIMIT_DEFAULT))
        state = session.for_task(task)
        try:
            result = execute_sql(state, sql, final=final, limit=limit)
        except Exception as exc:  # noqa: BLE001
            msg = str(exc)[:600]
            content: dict[str, Any] = {"error": msg, "sql": sql}
            lower = msg.lower()
            if lower.startswith("sql_parse:") or "parser error" in lower:
                content["sql_parse_check"] = {"sql": sql, "error": msg}
                content["hint"] = (
                    "Fix SQL syntax against the schema in the task message "
                    "(table/column names, quotes, commas)."
                )
            elif lower.startswith("sql_explain:") or "binder error" in lower or "catalog error" in lower:
                content["sql_explain_check"] = {"sql": sql, "error": msg}
                content["hint"] = (
                    "EXPLAIN failed: check that every table/column exists in this "
                    "task's warehouse and types/casts are valid. Call list_tables if unsure."
                )
            return ToolExecutionResult(ok=False, content=content)

        answer = result.pop("answer")
        if answer is not None:
            session.remember_final(sql, answer)
        shown = _preview_rows(result["rows"])
        grain = action_input.get("grain")
        grain_note = None
        if isinstance(grain, str) and grain.strip().lower() in {"fine", "coarse"}:
            grain_note = grain.strip().lower()
        content = {
            "columns": result["columns"],
            "rows": shown,
            "row_count": result["row_count"],
            "shown_rows": len(shown),
            "truncated": result["truncated"] or len(result["rows"]) > len(shown),
            "full_scan": result["full_scan"],
            "sql": result["sql"],
            "note": (
                "Probe results may be LIMIT-capped. Set final=true for the full answer table."
                if not result["full_scan"]
                else "Full-table result stored; call answer to submit it."
            ),
        }
        if grain_note is not None:
            content["grain"] = grain_note
        return ToolExecutionResult(ok=True, content=content)

    def _answer(_task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
        if session.last_final is None:
            return ToolExecutionResult(
                ok=False,
                content={
                    "error": (
                        "answer rejected: rows must come from a full-table run_sql "
                        "(final=true, no probe LIMIT). Do not hand-type rows."
                    ),
                    "hint": "Call run_sql with final=true, then call answer.",
                },
            )
        answer = session.last_final
        extra_columns = action_input.get("columns")
        if isinstance(extra_columns, list) and extra_columns and all(isinstance(item, str) for item in extra_columns):
            # Optional: caller may pass names; still submit the SQL table (prune happens later).
            pass
        return ToolExecutionResult(
            ok=True,
            content={
                "status": "submitted",
                "column_count": len(answer.columns),
                "row_count": len(answer.rows),
                "used_sql_result": True,
                "sql": session.last_final_sql,
            },
            is_terminal=True,
            answer=answer,
        )

    specs = {
        "list_tables": ToolSpec(
            name="list_tables",
            description=(
                "List DuckDB tables for this task only (name, columns, types, row counts). "
                "Includes CSV/JSON/SQLite and extracted doc tables. "
                "knowledge.md is already in the task prompt."
            ),
            input_schema={},
        ),
        "run_sql": ToolSpec(
            name="run_sql",
            description=(
                "Read-only SQL on this task's in-memory warehouse (SELECT/WITH/DESCRIBE). "
                "File reads cannot leave this task's context/. "
                "Probe queries are LIMIT-capped. Set final=true for the untruncated "
                "result that answer will submit. DuckDB dialect: double-quoted identifiers."
            ),
            input_schema={
                "sql": "SELECT ...",
                "final": False,
                "limit": PROBE_LIMIT_DEFAULT,
                "grain": "optional: fine|coarse for lowest/highest/average dual-grain probes",
            },
        ),
        "answer": ToolSpec(
            name="answer",
            description=(
                "Submit the last final=true run_sql result as prediction.csv. "
                "Do not paste rows. Extra columns are pruned at submit time. "
                "For lowest/highest/average questions, both fine and coarse aggregate "
                "probes must appear in the trace or answer is rejected."
            ),
            input_schema={},
        ),
    }
    handlers = {
        "list_tables": _list_tables,
        "run_sql": _run_sql,
        "answer": _answer,
    }
    return ToolRegistry(specs=specs, handlers=handlers, session=session)
