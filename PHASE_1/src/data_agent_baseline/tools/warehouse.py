"""Per-task in-memory DuckDB warehouse: CSV / JSON / SQLite → tables.

Official SQLite lives under ``context/db/*.db``. The agent warehouse is an
in-memory DuckDB (``:memory:``) with ``memory_limit`` + ``temp_directory`` so
large joins/aggregations spill into ``context/db/warehouse_tmp`` instead of
OOM'ing. Legacy on-disk ``warehouse.duckdb`` leftovers are deleted on open/close.
"""

from __future__ import annotations

import gc
import json
import math
import os
import re
import shutil
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from data_agent_baseline.benchmark.schema import AnswerTable, PublicTask

_IDENT_RE = re.compile(r"[^0-9A-Za-z_]+")
_ALLOWED_PREFIXES = ("select", "with", "pragma", "describe", "desc", "show", "explain")
_META_PREFIXES = ("pragma", "describe", "desc", "show", "explain")
_COMMENT_RE = re.compile(r"/\*.*?\*/", flags=re.DOTALL)
_LINE_COMMENT_RE = re.compile(r"--[^\n]*")
_BACKTICK_RE = re.compile(r"`([^`]+)`")
_BRACKET_RE = re.compile(r"\[([^\]]+)\]")
_FILE_FN_RE = re.compile(
    r"\b(?:glob|read_csv_auto|read_csv|read_json_auto|read_json|read_parquet|"
    r"read_xlsx|sqlite_scan)\s*\(",
    flags=re.IGNORECASE,
)

# Agent-owned DuckDB artifacts under context/db/; never ingest these as source tables.
WAREHOUSE_DB_NAME = "warehouse.duckdb"  # legacy filename; no longer the live database
WAREHOUSE_TMP_DIRNAME = "warehouse_tmp"
WAREHOUSE_MEMORY_LIMIT = (
    os.environ.get("DATA_AGENT_WAREHOUSE_MEMORY", "2GB") or "2GB"
).strip()
WAREHOUSE_MAX_TEMP_SIZE = (
    os.environ.get("DATA_AGENT_WAREHOUSE_MAX_TEMP", "50GB") or "50GB"
).strip()
_SKIP_NAMES = {
    WAREHOUSE_DB_NAME,
    f"{WAREHOUSE_DB_NAME}.wal",
    "manifest.json",
}


def stem_to_ident(stem: str) -> str:
    ident = _IDENT_RE.sub("_", stem).strip("_").lower()
    if not ident:
        ident = "table"
    if ident[0].isdigit():
        ident = f"t_{ident}"
    return ident


def quote_ident(ident: str) -> str:
    return '"' + ident.replace('"', '""') + '"'


def sql_str(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def duckdb_path_literal(path: Path) -> str:
    return sql_str(path.resolve().as_posix())


def normalize_quote_dialect(sql: str) -> str:
    def _to_dq(match: re.Match[str]) -> str:
        return quote_ident(match.group(1))

    sql = _BACKTICK_RE.sub(_to_dq, sql)
    return _BRACKET_RE.sub(_to_dq, sql)


def _first_keyword(sql: str) -> str:
    stripped = _COMMENT_RE.sub("", _LINE_COMMENT_RE.sub("", sql)).strip()
    if not stripped:
        return ""
    return stripped.split(None, 1)[0].lower()


def assert_read_only_sql(sql: str) -> None:
    keyword = _first_keyword(sql)
    if keyword not in _ALLOWED_PREFIXES:
        raise ValueError(
            f"Only read-only SQL is allowed (SELECT/WITH/DESCRIBE/SHOW/PRAGMA), got {keyword or 'empty'}."
        )


def is_meta_sql(sql: str) -> bool:
    return _first_keyword(sql) in _META_PREFIXES


def is_warehouse_artifact(path: Path, context_dir: Path) -> bool:
    """Skip DuckDB leftovers and doc-extract JSON caches; keep official context/db/*.db."""
    name = path.name.lower()
    if name in _SKIP_NAMES or name.endswith(".duckdb") or name.endswith(".duckdb.wal"):
        return True
    parts = {part.lower() for part in path.relative_to(context_dir).parts}
    if WAREHOUSE_TMP_DIRNAME in parts:
        return True
    if "extract" in parts and name.endswith(".json"):
        return True
    return False


def warehouse_db_path(context_dir: Path) -> Path:
    return context_dir / "db" / WAREHOUSE_DB_NAME


def warehouse_tmp_dir(context_dir: Path) -> Path:
    return context_dir / "db" / WAREHOUSE_TMP_DIRNAME


def clear_warehouse_files(context_dir: Path) -> None:
    """Delete legacy on-disk warehouse DB/WAL and the spill temp dir.

    Safe to call repeatedly. Does not touch official ``*.db`` sources or extract
    caches under ``db/extract/``.
    """
    db_path = warehouse_db_path(context_dir)
    candidates = [
        db_path,
        Path(str(db_path) + ".wal"),
        Path(str(db_path) + ".tmp"),
        db_path.with_suffix(db_path.suffix + ".wal"),
    ]
    for path in candidates:
        try:
            if path.is_file():
                path.unlink()
        except Exception:
            pass
    tmp_dir = warehouse_tmp_dir(context_dir)
    if tmp_dir.is_dir():
        shutil.rmtree(tmp_dir, ignore_errors=True)


def assert_sql_in_context(sql: str, context_dir: Path) -> None:
    normalized = sql.replace("\\", "/")
    if "/../" in f"/{normalized}/" or normalized.strip().startswith("../"):
        raise ValueError("SQL file access cannot leave this task's context/")
    if _FILE_FN_RE.search(sql):
        raise ValueError(
            "File scans (glob/read_csv/read_json) are blocked. "
            "Query warehouse tables from list_tables; they already live in this task's DuckDB."
        )


def _cell(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return value
    if isinstance(value, float) and math.isnan(value):
        return ""
    if isinstance(value, datetime):
        if value.hour == 0 and value.minute == 0 and value.second == 0 and value.microsecond == 0:
            return value.date().isoformat()
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, (bytes, bytearray, memoryview)):
        return f"<binary len={len(value)}>"
    if isinstance(value, (int, float, str)):
        return value
    # DuckDB / pandas: Decimal, UUID, numpy scalars, etc.
    if hasattr(value, "item") and not isinstance(value, (bytes, bytearray, memoryview)):
        try:
            return _cell(value.item())
        except Exception:
            pass
    return str(value)


def result_to_answer(columns: list[str], rows: list[tuple[Any, ...]]) -> AnswerTable:
    return AnswerTable(
        columns=list(columns),
        rows=[[_cell(value) for value in row] for row in rows],
    )


@dataclass
class RegisteredTable:
    canonical: str
    aliases: list[str]
    source_type: str
    source_rel: str
    sqlite_table: str | None = None
    report: dict[str, Any] | None = None


@dataclass
class WarehouseState:
    conn: duckdb.DuckDBPyConnection
    context_dir: Path
    tables: list[RegisteredTable] = field(default_factory=list)
    db_path: Path | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def close(self) -> None:
        # Close the in-memory connection, then drop spill/legacy files. Either step
        # can hang on Windows; callers should wrap with a timeout (see runner).
        with self._lock:
            conn = self.conn
            try:
                conn.close()
            except Exception:
                pass
        try:
            clear_warehouse_files(self.context_dir)
        except Exception:
            pass
        try:
            gc.collect()
        except Exception:
            pass


def _used_names(state: WarehouseState) -> set[str]:
    used: set[str] = set()
    for table in state.tables:
        used.add(table.canonical)
        used.update(table.aliases)
    return used


def _build_aliases(canonical: str, *original_names: str, used: set[str]) -> list[str]:
    canonical_cf = canonical.casefold()
    used_cf = {name.casefold() for name in used}
    candidates: list[str] = [f"df_{canonical}"]
    for name in original_names:
        if not name or name.casefold() == canonical_cf:
            continue
        candidates.append(name)
        low = name.lower()
        if low != name:
            candidates.append(low)

    aliases: list[str] = []
    seen_cf: set[str] = set()
    for candidate in candidates:
        key = candidate.casefold()
        if key == canonical_cf or key in used_cf or key in seen_cf:
            continue
        seen_cf.add(key)
        aliases.append(candidate)
    used.add(canonical)
    used.update(aliases)
    return aliases


def _relative(context_dir: Path, path: Path) -> str:
    try:
        return str(path.relative_to(context_dir)).replace("\\", "/")
    except ValueError:
        return path.name


def _create_aliases(state: WarehouseState, canonical: str, aliases: list[str]) -> None:
    for alias in aliases:
        state.conn.execute(
            f"CREATE OR REPLACE VIEW {quote_ident(alias)} AS SELECT * FROM {quote_ident(canonical)}"
        )


def _pick_canonical(stem: str, used: set[str], *, prefix: str | None = None) -> str:
    used_cf = {name.casefold() for name in used}
    bare = stem_to_ident(stem)
    if bare.casefold() not in used_cf:
        return bare
    if prefix:
        prefixed = stem_to_ident(f"{prefix}__{stem}")
        if prefixed.casefold() not in used_cf:
            return prefixed
    index = 2
    while True:
        candidate = f"{bare}_{index}"
        if candidate.casefold() not in used_cf:
            return candidate
        index += 1


def _materialize_frame(state: WarehouseState, canonical: str, frame: pd.DataFrame) -> None:
    tmp = f"_tmp_{canonical}"
    state.conn.register(tmp, frame)
    try:
        state.conn.execute(
            f"CREATE OR REPLACE TABLE {quote_ident(canonical)} AS SELECT * FROM {quote_ident(tmp)}"
        )
    finally:
        try:
            state.conn.unregister(tmp)
        except Exception:
            pass


def _register_csv(state: WarehouseState, path: Path) -> None:
    used = _used_names(state)
    canonical = _pick_canonical(path.stem, used, prefix=path.parent.name)
    aliases = _build_aliases(canonical, path.stem, used=used)
    state.conn.execute(
        f"CREATE OR REPLACE TABLE {quote_ident(canonical)} AS "
        f"SELECT * FROM read_csv_auto({duckdb_path_literal(path)})"
    )
    _create_aliases(state, canonical, aliases)
    state.tables.append(
        RegisteredTable(
            canonical=canonical,
            aliases=aliases,
            source_type="csv",
            source_rel=_relative(state.context_dir, path),
        )
    )


def _records_from_json(payload: Any) -> tuple[str | None, list[dict[str, Any]] | None]:
    if isinstance(payload, dict) and isinstance(payload.get("records"), list):
        records = payload["records"]
        if all(isinstance(item, dict) for item in records):
            table = payload.get("table")
            name = str(table) if table else None
            return name, records
    if isinstance(payload, list) and payload and all(isinstance(item, dict) for item in payload):
        return None, payload
    return None, None


def _register_json(state: WarehouseState, path: Path) -> None:
    # Prefer DuckDB file scan so large JSON never fully enters Python RAM.
    try:
        peek = path.read_text(encoding="utf-8", errors="replace")[:2048].lstrip()
    except Exception:
        peek = ""
    if peek.startswith("["):
        try:
            used = _used_names(state)
            stem = path.stem
            canonical = _pick_canonical(stem, used, prefix=path.parent.name)
            aliases = _build_aliases(canonical, stem, used=used)
            state.conn.execute(
                f"CREATE OR REPLACE TABLE {quote_ident(canonical)} AS "
                f"SELECT * FROM read_json_auto({duckdb_path_literal(path)})"
            )
            _create_aliases(state, canonical, aliases)
            state.tables.append(
                RegisteredTable(
                    canonical=canonical,
                    aliases=aliases,
                    source_type="json",
                    source_rel=_relative(state.context_dir, path),
                )
            )
            return
        except Exception:
            # Fall through to the Python JSON path below.
            pass

    payload = json.loads(path.read_text(encoding="utf-8"))
    table_name, records = _records_from_json(payload)
    if records is None:
        return
    used = _used_names(state)
    stem = table_name or path.stem
    canonical = _pick_canonical(stem, used, prefix=path.parent.name)
    aliases = _build_aliases(canonical, stem, path.stem, used=used)
    _materialize_frame(state, canonical, pd.DataFrame(records))
    del records
    del payload
    _create_aliases(state, canonical, aliases)
    state.tables.append(
        RegisteredTable(
            canonical=canonical,
            aliases=aliases,
            source_type="json",
            source_rel=_relative(state.context_dir, path),
        )
    )


def _sqlite_table_names(path: Path) -> list[str]:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
        return [str(row[0]) for row in rows]
    finally:
        conn.close()


def _register_sqlite(state: WarehouseState, path: Path) -> None:
    sqlite_tables = _sqlite_table_names(path)
    if not sqlite_tables:
        return
    db_stem = stem_to_ident(path.stem)
    used = _used_names(state)
    sqlite_conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        for table_name in sqlite_tables:
            canonical = _pick_canonical(table_name, used, prefix=db_stem)
            aliases = _build_aliases(canonical, table_name, used=used)
            quoted = '"' + table_name.replace('"', '""') + '"'
            frame = pd.read_sql_query(f"SELECT * FROM {quoted}", sqlite_conn)
            _materialize_frame(state, canonical, frame)
            _create_aliases(state, canonical, aliases)
            state.tables.append(
                RegisteredTable(
                    canonical=canonical,
                    aliases=aliases,
                    source_type="sqlite",
                    source_rel=_relative(state.context_dir, path),
                    sqlite_table=table_name,
                )
            )
            used = _used_names(state)
    finally:
        sqlite_conn.close()


def _register_extracted(state: WarehouseState, extracted: Any) -> None:
    if extracted is None or not extracted.rows:
        return
    used = _used_names(state)
    canonical = _pick_canonical(extracted.stem, used)
    aliases = _build_aliases(canonical, extracted.stem, used=used)
    frame = pd.DataFrame(extracted.rows, columns=extracted.columns)
    _materialize_frame(state, canonical, frame)
    _create_aliases(state, canonical, aliases)
    report_payload: dict[str, Any] | None = None
    report_obj = getattr(extracted, "report", None)
    if report_obj is not None:
        report_payload = {
            "stem": getattr(report_obj, "stem", canonical),
            "row_count": getattr(report_obj, "row_count", len(extracted.rows)),
            "key_scannable": getattr(report_obj, "key_scannable", False),
            "missing_keys": list(getattr(report_obj, "missing_keys", [])),
            "empty_field_cells": getattr(report_obj, "empty_field_cells", 0),
            "field_fill_rate": dict(getattr(report_obj, "field_fill_rate", {})),
        }
    state.tables.append(
        RegisteredTable(
            canonical=canonical,
            aliases=aliases,
            source_type="doc",
            source_rel=extracted.source_rel,
            report=report_payload,
        )
    )


def _collect_files(context_dir: Path, pattern: str) -> list[Path]:
    found: list[Path] = []
    seen: set[str] = set()
    for path in sorted(context_dir.rglob(pattern)):
        if not path.is_file() or is_warehouse_artifact(path, context_dir):
            continue
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        found.append(path)
    return found


def _open_warehouse_connection(
    context_dir: Path,
) -> tuple[duckdb.DuckDBPyConnection, Path | None]:
    """Open an in-memory DuckDB that spills to ``warehouse_tmp`` under memory pressure."""
    clear_warehouse_files(context_dir)
    db_dir = context_dir / "db"
    db_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = warehouse_tmp_dir(context_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    conn = duckdb.connect(":memory:")
    # Cap RAM so large CREATE TABLE / joins spill into warehouse_tmp instead of OOM.
    try:
        conn.execute(f"SET memory_limit='{WAREHOUSE_MEMORY_LIMIT}'")
    except Exception:
        pass
    try:
        conn.execute(f"SET temp_directory={duckdb_path_literal(tmp_dir)}")
    except Exception:
        pass
    try:
        conn.execute(f"SET max_temp_directory_size='{WAREHOUSE_MAX_TEMP_SIZE}'")
    except Exception:
        pass
    return conn, None


def build_warehouse(
    context_dir: Path,
    *,
    model: Any | None = None,
    force_rebuild: bool = True,
) -> WarehouseState:
    """Scan one task's context/ into a fresh in-memory DuckDB warehouse.

    Clears legacy ``warehouse.duckdb`` leftovers and ``warehouse_tmp`` before
    opening so each run/retry starts clean. Spill files live only under
    ``warehouse_tmp`` while the connection is open.
    """
    from data_agent_baseline.run.progress import mark as _mark

    del force_rebuild  # rebuild is always fresh; kept for call-site compatibility
    if not context_dir.is_dir():
        conn = duckdb.connect(":memory:")
        return WarehouseState(conn=conn, context_dir=context_dir, db_path=None)

    _mark(
        "warehouse_open",
        context=str(context_dir),
        mode="memory",
        memory_limit=WAREHOUSE_MEMORY_LIMIT,
    )
    conn, db_path = _open_warehouse_connection(context_dir)
    state = WarehouseState(conn=conn, context_dir=context_dir, db_path=db_path)
    _mark(
        "warehouse_opened",
        mode="memory",
        temp_dir=str(warehouse_tmp_dir(context_dir)),
        memory_limit=WAREHOUSE_MEMORY_LIMIT,
    )

    csv_paths = _collect_files(context_dir, "*.csv")
    for path in csv_paths:
        try:
            _mark("warehouse_register_csv", file=path.name, bytes=path.stat().st_size)
            _register_csv(state, path)
        except Exception:
            # Match json/sqlite: one bad/huge CSV must not abort the whole warehouse.
            continue
    json_paths = _collect_files(context_dir, "*.json")
    for path in json_paths:
        try:
            _mark("warehouse_register_json", file=path.name, bytes=path.stat().st_size)
            _register_json(state, path)
        except Exception:
            continue
    sqlite_paths = _collect_files(context_dir, "*.db") + _collect_files(context_dir, "*.sqlite")
    for path in sqlite_paths:
        try:
            _mark("warehouse_register_sqlite", file=path.name, bytes=path.stat().st_size)
            _register_sqlite(state, path)
        except Exception:
            continue
    try:
        from data_agent_baseline.tools.doc_extract import (
            ExtractionBudget,
            extract_all_documents,
        )

        _mark("warehouse_extract_docs")
        from data_agent_baseline.run.task_budget import get_current_budget

        tb = get_current_budget()
        if tb is not None:
            extract_budget = ExtractionBudget.from_env(seconds=tb.extract_seconds())
        else:
            extract_budget = ExtractionBudget.from_env()
        for extracted in extract_all_documents(
            context_dir, model, budget=extract_budget
        ):
            _register_extracted(state, extracted)
        _mark(
            "warehouse_extract_docs_done",
            budget_remaining=round(extract_budget.remaining, 1),
        )
    except Exception as exc:
        _mark("warehouse_extract_docs_failed", error=str(exc)[:300])
    # Encourage releasing peak Python allocations from pandas/json loads.
    gc.collect()
    _mark("warehouse_ready", table_count=len(state.tables))
    return state


def describe_tables(state: WarehouseState) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    with state._lock:
        for table in state.tables:
            info = state.conn.execute(f"DESCRIBE {quote_ident(table.canonical)}").fetchall()
            count_row = state.conn.execute(
                f"SELECT COUNT(*) FROM {quote_ident(table.canonical)}"
            ).fetchone()
            payload.append(
                {
                    "name": table.canonical,
                    "aliases": list(table.aliases),
                    "source_type": table.source_type,
                    "source": table.source_rel,
                    "n_rows": int(count_row[0]) if count_row else 0,
                    "columns": [
                        {"name": row[0], "type": row[1]} for row in info
                    ],
                    "report": table.report,
                }
            )
    return payload


def format_table_schema(state: WarehouseState, *, sample_rows: int = 2, max_chars: int = 12000) -> str:
    """Schema text from the warehouse: name, columns, types, row count, sample rows."""
    blocks: list[str] = []
    for table in describe_tables(state):
        columns = table["columns"]
        col_line = ", ".join(f"{col['name']}:{col['type']}" for col in columns)
        lines = [
            f"[TABLE] {table['name']}",
            f"  source: {table['source']}",
            f"  rows: {table['n_rows']}",
            f"  columns ({len(columns)}): {col_line}",
        ]
        report = table.get("report")
        if report:
            if report.get("missing_keys"):
                lines.append(
                    f"  EXTRACTION WARNING: {len(report['missing_keys'])} source keys "
                    f"missing from extracted table (e.g. {report['missing_keys'][0]}). "
                    "Aggregates over this table are lower bounds."
                )
            elif not report.get("key_scannable"):
                lines.append(
                    "  EXTRACTION NOTE: primary key could not be scanned in source text; "
                    "rows may be incomplete."
                )
        try:
            with state._lock:
                relation = state.conn.execute(
                    f"SELECT * FROM {quote_ident(table['name'])} LIMIT {int(sample_rows)}"
                )
                names = [item[0] for item in relation.description] if relation.description else []
                fetched = relation.fetchall()
        except Exception:
            fetched = []
            names = []
        for index, row in enumerate(fetched, 1):
            cells = {
                name: _clip_cell(value)
                for name, value in zip(names, row)
            }
            lines.append(f"  sample{index}: {json.dumps(cells, ensure_ascii=False, default=str)}")
        blocks.append("\n".join(lines))
    text = "\n\n".join(blocks)
    if len(text) <= max_chars:
        return text
    trimmed: list[str] = []
    total = 0
    for block in blocks:
        header = "\n".join(line for line in block.splitlines() if not line.startswith("  sample"))
        extra = len(header) + (2 if trimmed else 0)
        if total + extra > max_chars:
            break
        trimmed.append(header)
        total += extra
    return "\n\n".join(trimmed)


def _clip_cell(value: Any, limit: int = 80) -> Any:
    cell = _cell(value)
    if isinstance(cell, str) and len(cell) > limit:
        return cell[:limit] + "…"
    return cell


def explain_sql_plan(state: WarehouseState, sql: str) -> None:
    """Run DuckDB EXPLAIN before execute. Raises ValueError with sql_parse/sql_explain prefix."""
    executed = normalize_quote_dialect(sql.strip().rstrip(";"))
    if not executed or is_meta_sql(executed):
        return
    try:
        with state._lock:
            state.conn.execute(f"EXPLAIN {executed}").fetchall()
    except Exception as exc:  # noqa: BLE001
        msg = str(exc).strip() or type(exc).__name__
        lower = msg.lower()
        if "parser error" in lower or "syntax error" in lower:
            raise ValueError(f"sql_parse: {msg}") from exc
        raise ValueError(f"sql_explain: {msg}") from exc


def execute_sql(
    state: WarehouseState,
    sql: str,
    *,
    final: bool = False,
    limit: int = 50,
) -> dict[str, Any]:
    """Run read-only SQL. Probe wraps LIMIT; final returns the full result as AnswerTable."""
    assert_read_only_sql(sql)
    assert_sql_in_context(sql, state.context_dir)
    # B1: plan check before any fetch (syntax / binder / missing objects).
    explain_sql_plan(state, sql)
    executed = normalize_quote_dialect(sql.strip().rstrip(";"))
    wrapped = False
    if not final and not is_meta_sql(executed):
        cap = max(1, min(int(limit), 500))
        executed = f"SELECT * FROM ({executed}) AS _probe LIMIT {cap}"
        wrapped = True

    with state._lock:
        relation = state.conn.execute(executed)
        columns = [item[0] for item in relation.description] if relation.description else []
        rows = relation.fetchall()

    answer = result_to_answer(columns, rows)
    return {
        "columns": answer.columns,
        "rows": answer.rows,
        "row_count": len(answer.rows),
        "truncated": wrapped,
        "full_scan": bool(final) and not is_meta_sql(sql),
        "sql": executed,
        "answer": answer if final and not is_meta_sql(sql) else None,
    }


@dataclass
class WarehouseSession:
    """One in-memory DuckDB per task (disk spill under memory pressure)."""

    model: Any | None = None
    task_id: str | None = None
    state: WarehouseState | None = None
    last_final: AnswerTable | None = None
    last_final_sql: str | None = None

    def for_task(self, task: PublicTask, *, force_rebuild: bool = False) -> WarehouseState:
        if force_rebuild or self.task_id != task.task_id or self.state is None:
            self.close()
            self.state = build_warehouse(task.context_dir, model=self.model, force_rebuild=True)
            self.task_id = task.task_id
            self.last_final = None
            self.last_final_sql = None
        return self.state

    def remember_final(self, sql: str, answer: AnswerTable) -> None:
        self.last_final = answer
        self.last_final_sql = sql

    def close(self) -> None:
        if self.state is not None:
            self.state.close()
        self.state = None
        self.task_id = None
        self.last_final = None
        self.last_final_sql = None
