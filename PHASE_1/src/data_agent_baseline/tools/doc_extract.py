"""Extract narrative context docs into tabular rows for the per-task DuckDB warehouse.

knowledge.md stays in the prompt and is never a table. Other .md/.txt files go
through a document-level plan (record universe + schema + segments), then one
isolated worker per segment. Rows merge on primary key into a wide table.
Format conversion runs after extract in a subprocess. Table name = file stem.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from data_agent_baseline.agents.model import ModelAdapter, strip_think_content

_IDENT_RE = re.compile(r"[^0-9A-Za-z_]+")
SKIP_DOC_NAMES = {"knowledge.md"}
DOC_SUFFIXES = {".md", ".txt"}
MAX_KNOWLEDGE_CHARS = 8000
MAX_PARA_CHARS = 2000
MIN_PARA_CHARS = 60
MAX_SAMPLE_PARAS = 6
MAX_PLAN_CHARS = 24000
MAX_SEGMENTS = 40
MAX_SEGMENT_CHARS = 8000
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", flags=re.DOTALL)
_BLANK_RE = re.compile(r"\n\s*\n+")

# Small parallel batch extract. Cap at 4 — higher tends to 429 storms on MiniMax.
EXTRACT_CONCURRENCY = max(
    1, min(4, int(os.environ.get("DATA_AGENT_EXTRACT_CONCURRENCY", "2") or "2"))
)
# Optional healthy-path pacing. Default 0: only sleep on 429 backoff.
EXTRACT_CALL_DELAY_SECONDS = max(
    0.0, float(os.environ.get("DATA_AGENT_EXTRACT_DELAY", "0") or "0")
)
# Wall-clock budget for cold LLM extraction inside one task (seconds).
EXTRACT_BUDGET_SECONDS = max(
    30.0, float(os.environ.get("DATA_AGENT_EXTRACT_BUDGET", "180") or "180")
)
# Soft floor for paragraphs/batch (used when packing under time pressure / recovery).
EXTRACT_BATCH_SIZE = max(
    1, min(60, int(os.environ.get("DATA_AGENT_EXTRACT_BATCH_SIZE", "24") or "24"))
)
# Hard caps for char-aware packing (target ~4–8k chars of paragraph text / call).
EXTRACT_BATCH_MAX_PARAS = max(
    1, min(60, int(os.environ.get("DATA_AGENT_EXTRACT_BATCH_MAX_PARAS", "48") or "48"))
)
EXTRACT_BATCH_MAX_CHARS = max(
    1000,
    min(12000, int(os.environ.get("DATA_AGENT_EXTRACT_BATCH_MAX_CHARS", "6000") or "6000")),
)
# Conservative seconds/call used only for "can we finish?" preflight.
EXTRACT_SEC_PER_CALL = max(
    1.0, float(os.environ.get("DATA_AGENT_EXTRACT_SEC_PER_CALL", "8") or "8")
)
EXTRACT_429_BACKOFF_SECONDS = max(
    0.5, float(os.environ.get("DATA_AGENT_EXTRACT_429_BACKOFF", "3") or "3")
)
# Serialize 429 backoff so concurrent workers do not stampede the API together.
_RATE_LIMIT_LOCK = threading.Lock()

# Extraction cache version. Bump ONLY when LLM-side extraction logic changes
# (prompts, schema inference). Deterministic post-processing (e.g. Registry
# official-name arbitration) never needs a bump: it is re-applied at load time.
_EXTRACT_VERSION = 6
# Oldest reusable cache version. Accepted caches are upgraded deterministically
# at load and re-saved at the current version, without any LLM call.
# v1/v2 are included so scored runs never re-extract just because of an old
# version stamp (that re-extract is the main cause of 300s warehouse timeouts).
_MIN_CACHE_VERSION = 1

_RATE_LIMIT_RE = re.compile(
    r"\b429\b|rate[\s_-]?limit|insufficient[_\s]?quota|quota[\s_-]?exceeded|"
    r"usage[\s_-]?limit|too many requests|tokens?\s+(?:exhausted|exceeded)|"
    r"billing|credit|余额不足|用量超",
    flags=re.IGNORECASE,
)
_TRANSIENT_RE = re.compile(
    r"connection error|connect(?:ion)? timed? ?out|request timed out|"
    r"read timed? ?out|timed? ?out|temporarily unavailable|"
    r"reset by peer|connection reset|broken pipe|api connection|"
    r"remote disconnected|server disconnected|ssl|eof occurred",
    flags=re.IGNORECASE,
)


class DocumentExtractionError(RuntimeError):
    """Cold extraction could not finish completely inside the budget / after 429."""


@dataclass
class ExtractionBudget:
    """Shared deadline for all cold-extract LLM calls in one warehouse build."""

    deadline: float
    started_at: float

    @classmethod
    def from_env(cls, *, seconds: float | None = None) -> "ExtractionBudget":
        now = time.perf_counter()
        if seconds is None:
            limit = max(1.0, float(EXTRACT_BUDGET_SECONDS))
        else:
            limit = max(0.0, float(seconds))
        return cls(deadline=now + limit, started_at=now)

    @property
    def remaining(self) -> float:
        return self.deadline - time.perf_counter()

    def check(self, where: str) -> None:
        if self.remaining <= 0:
            elapsed = round(time.perf_counter() - self.started_at, 1)
            raise DocumentExtractionError(
                f"extract_timeout after {elapsed}s at {where}"
            )


def _is_rate_limit_error(text: str) -> bool:
    return bool(_RATE_LIMIT_RE.search(text or ""))


def _is_transient_error(text: str) -> bool:
    """API blips that are worth one retry (not a hard 429 / quota fail)."""
    return bool(_TRANSIENT_RE.search(text or ""))


def stem_to_ident(stem: str) -> str:
    ident = _IDENT_RE.sub("_", stem).strip("_").lower()
    if not ident:
        ident = "table"
    if ident[0].isdigit():
        ident = f"t_{ident}"
    return ident


@dataclass
class ExtractionReport:
    stem: str
    row_count: int
    key_scannable: bool
    missing_keys: list[str]
    empty_field_cells: int
    field_fill_rate: dict[str, float]


@dataclass
class ExtractedDoc:
    stem: str
    source_rel: str
    columns: list[str]
    primary_key: list[str]
    rows: list[dict[str, str]]
    report: ExtractionReport | None = None


@dataclass
class DocumentPlan:
    columns: list[str]
    primary_key: list[str]
    record_keys: list[str]
    segments: list[tuple[int, int]]


def extract_cache_dir(context_dir: Path) -> Path:
    # Keep JSON extract cache beside official SQLite, but warehouse skips this folder.
    return context_dir / "db" / "extract"


def collect_doc_paths(context_dir: Path) -> list[Path]:
    db_root = (context_dir / "db").resolve()
    found: list[Path] = []
    seen: set[str] = set()
    for path in sorted(context_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in DOC_SUFFIXES:
            continue
        if path.name.lower() in SKIP_DOC_NAMES:
            continue
        resolved = path.resolve()
        try:
            resolved.relative_to(db_root)
            continue
        except ValueError:
            pass
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        found.append(path)
    return found


def split_paragraphs(text: str) -> list[str]:
    chunks = _BLANK_RE.split(text.replace("\r\n", "\n").strip())
    paras: list[str] = []
    for chunk in chunks:
        para = " ".join(chunk.split())
        if len(para) < MIN_PARA_CHARS:
            continue
        paras.append(para[:MAX_PARA_CHARS])
    return paras


def parse_json_object(text: str) -> dict[str, Any]:
    cleaned = strip_think_content(text or "").strip()
    if not cleaned:
        raise ValueError("empty model response")
    fenced = _JSON_FENCE_RE.search(cleaned)
    if fenced:
        cleaned = fenced.group(1)
    else:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("no JSON object in model response")
        cleaned = cleaned[start : end + 1]
    payload = json.loads(cleaned)
    if not isinstance(payload, dict):
        raise ValueError("JSON payload is not an object")
    return payload


def _knowledge_excerpt(context_dir: Path) -> str:
    path = context_dir / "knowledge.md"
    if not path.is_file():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[:MAX_KNOWLEDGE_CHARS]


def _sample_paragraphs(paragraphs: list[str], limit: int = MAX_SAMPLE_PARAS) -> list[str]:
    if len(paragraphs) <= limit:
        return list(paragraphs)
    picks = {0, len(paragraphs) // 2, len(paragraphs) - 1}
    step = max(1, len(paragraphs) // limit)
    for index in range(0, len(paragraphs), step):
        picks.add(index)
    return [paragraphs[index] for index in sorted(picks)[:limit]]


def _hash_material(path: Path, context_dir: Path) -> tuple[bytes, bytes | None]:
    knowledge = context_dir / "knowledge.md"
    knowledge_bytes = knowledge.read_bytes() if knowledge.is_file() else None
    return path.read_bytes(), knowledge_bytes


def _source_hash_from_material(
    doc_bytes: bytes, knowledge_bytes: bytes | None, version: int
) -> str:
    digest = hashlib.sha256()
    digest.update(f"extract:{version}".encode("utf-8"))
    digest.update(b"\0")
    digest.update(doc_bytes)
    if knowledge_bytes is not None:
        digest.update(b"\0knowledge\0")
        digest.update(knowledge_bytes)
    return digest.hexdigest()


def _source_hash(path: Path, context_dir: Path) -> str:
    doc_bytes, knowledge_bytes = _hash_material(path, context_dir)
    return _source_hash_from_material(doc_bytes, knowledge_bytes, _EXTRACT_VERSION)


def _accepted_source_hashes(path: Path, context_dir: Path) -> set[str]:
    """Source hashes of every reusable cache version for this document."""
    doc_bytes, knowledge_bytes = _hash_material(path, context_dir)
    return {
        _source_hash_from_material(doc_bytes, knowledge_bytes, version)
        for version in range(_MIN_CACHE_VERSION, _EXTRACT_VERSION + 1)
    }


def _cache_path(context_dir: Path, source_rel: str) -> Path:
    ident = stem_to_ident(source_rel.replace("\\", "/").replace("/", "_"))
    return extract_cache_dir(context_dir) / f"{ident}.json"


def _sanitize_columns(raw_columns: Any) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    if not isinstance(raw_columns, list):
        return names
    for item in raw_columns:
        if isinstance(item, dict):
            raw = str(item.get("name") or "").strip()
        else:
            raw = str(item).strip()
        if not raw:
            continue
        name = stem_to_ident(raw)
        if name in seen:
            continue
        seen.add(name)
        names.append(name)
    return names


def _sanitize_pk(raw_pk: Any, columns: list[str]) -> list[str]:
    allowed = set(columns)
    keys: list[str] = []
    if not isinstance(raw_pk, list):
        return keys
    for item in raw_pk:
        name = stem_to_ident(str(item).strip())
        if name in allowed and name not in keys:
            keys.append(name)
    return keys


_CHATTER_COL_RE = re.compile(
    r"^(memo|note|notes|comment|comments|toc|aside|chatter|boilerplate|"
    r"narrative|remark|remarks|annotation)$",
    flags=re.IGNORECASE,
)
_SUPERSEDED_COL_RE = re.compile(
    r"^(original|initial|previous|formerly|listed_as|old|alias|nickname)_",
    flags=re.IGNORECASE,
)


def _plan_document_text(paragraphs: list[str], *, limit: int = MAX_PLAN_CHARS) -> str:
    numbered = [f"[{i + 1}] {para}" for i, para in enumerate(paragraphs)]
    full = "\n\n".join(numbered)
    if len(full) <= limit:
        return full
    # Keep every paragraph index (first 120 chars) plus full text of samples.
    index_lines = [f"[{i + 1}] {para[:120]}" for i, para in enumerate(paragraphs)]
    samples = _sample_paragraphs(paragraphs, limit=min(12, max(6, MAX_SAMPLE_PARAS * 2)))
    sample_block = "\n\n".join(f"[full sample]\n{para}" for para in samples)
    body = "Paragraph index (truncated):\n" + "\n".join(index_lines) + "\n\n" + sample_block
    return body[:limit]


def _coalesce_segments(segments: list[tuple[int, int]], limit: int = MAX_SEGMENTS) -> list[tuple[int, int]]:
    if len(segments) <= limit:
        return segments
    total = segments[-1][1] - segments[0][0]
    target = max(1, (total + limit - 1) // limit)
    merged: list[tuple[int, int]] = []
    start = segments[0][0]
    end = start
    for seg_start, seg_end in segments:
        if end == start:
            start, end = seg_start, seg_end
            continue
        if (seg_end - start) <= target:
            end = seg_end
        else:
            merged.append((start, end))
            start, end = seg_start, seg_end
    merged.append((start, end))
    return merged[:limit] if len(merged) > limit else merged


def _fallback_segments(paragraphs: list[str]) -> list[tuple[int, int]]:
    packed = _pack_paragraph_batches(
        paragraphs,
        max_paras=max(4, EXTRACT_BATCH_SIZE // 2),
        max_chars=MAX_SEGMENT_CHARS,
    )
    return packed[:MAX_SEGMENTS] or ([(0, len(paragraphs))] if paragraphs else [])


def parse_document_plan(payload: dict[str, Any], paragraph_count: int) -> DocumentPlan:
    columns = _sanitize_columns(payload.get("columns"))
    if not columns:
        raise ValueError("plan has no columns")
    primary_key = _sanitize_pk(payload.get("primary_key"), columns)
    records_raw = payload.get("records") or payload.get("record_keys") or []
    record_keys: list[str] = []
    seen: set[str] = set()
    if isinstance(records_raw, list):
        for item in records_raw:
            if isinstance(item, dict):
                value = ""
                if primary_key:
                    value = str(item.get(primary_key[0], "") or "").strip()
                if not value:
                    value = str(next(iter(item.values()), "") or "").strip()
            else:
                value = str(item or "").strip()
            if value and value not in seen:
                seen.add(value)
                record_keys.append(value)
    segments: list[tuple[int, int]] = []
    raw_segments = payload.get("segments")
    if isinstance(raw_segments, list):
        for item in raw_segments:
            if isinstance(item, dict):
                try:
                    start = int(item.get("start") or item.get("para_start") or 0)
                    end = int(item.get("end") or item.get("para_end") or 0)
                except (TypeError, ValueError):
                    continue
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                try:
                    start, end = int(item[0]), int(item[1])
                except (TypeError, ValueError):
                    continue
            else:
                continue
            if start >= 1:
                start0 = start - 1
                end0 = end if end >= start else start
            else:
                start0 = max(0, start)
                end0 = end
            if end0 <= start0:
                end0 = start0 + 1
            segments.append((start0, end0))
    # Fix 1-based inclusive end: if planner said start=1,end=8 then start=0, end=8 (exclusive ok)
    cleaned: list[tuple[int, int]] = []
    for start, end in segments:
        start = max(0, min(start, paragraph_count - 1))
        end = max(start + 1, min(end, paragraph_count))
        cleaned.append((start, end))
    if not cleaned:
        raise ValueError("plan has no segments")
    cleaned.sort(key=lambda item: item[0])
    return DocumentPlan(
        columns=columns,
        primary_key=primary_key,
        record_keys=record_keys,
        segments=_coalesce_segments(cleaned),
    )


def _complete_json_validated(
    model: ModelAdapter,
    prompt: str,
    *,
    parse,
    budget: ExtractionBudget | None = None,
    where: str = "llm_call",
):
    """Call the model, validate, retry once on validation/parse failure."""
    last_error: Exception | None = None
    for attempt in (1, 2):
        label = where if attempt == 1 else f"{where}:retry"
        try:
            payload = _complete_json(model, prompt, budget=budget, where=label)
            return parse(payload)
        except DocumentExtractionError:
            raise
        except Exception as exc:
            last_error = exc
            continue
    raise ValueError(str(last_error) if last_error else f"{where} validation failed")


def plan_document(
    model: ModelAdapter,
    *,
    stem: str,
    paragraphs: list[str],
    knowledge: str,
    budget: ExtractionBudget | None = None,
) -> DocumentPlan:
    knowledge_block = knowledge.strip() or "(no knowledge.md)"
    body = _plan_document_text(paragraphs)
    prompt = f"""Read this document once and plan extraction for table `{stem}`.

Return one JSON object:
{{
  "columns": [{{"name": "snake_case_col"}}],
  "primary_key": ["id_column"],
  "records": ["primary_key_value", ...],
  "segments": [{{"start": 1, "end": 8}}, ...]
}}

Rules:
- Prefer field names from knowledge.md.
- records = the complete set of entity primary keys in the document (not samples).
- segments are 1-based inclusive paragraph indices covering the document.
- One segment should be a coherent slice (entity group or section), not the whole file.
- Do not invent metrics. 4–20 columns.
- knowledge.md excerpt:
{knowledge_block}

Numbered paragraphs:
{body}
"""

    def _parse(payload: dict[str, Any]) -> DocumentPlan:
        return parse_document_plan(payload, len(paragraphs))

    return _complete_json_validated(
        model, prompt, parse=_parse, budget=budget, where="extract_plan"
    )


def prune_extract_columns(
    columns: list[str],
    rows: list[dict[str, str]],
    *,
    primary_key: list[str],
) -> tuple[list[str], list[dict[str, str]]]:
    """Drop chatter columns and originals superseded by a corrected column."""
    kept: list[str] = []
    pk = set(primary_key)
    for col in columns:
        if col in pk:
            kept.append(col)
            continue
        if _CHATTER_COL_RE.match(col):
            continue
        if _SUPERSEDED_COL_RE.match(col):
            continue
        kept.append(col)
    if not kept:
        kept = list(columns)
    pruned_rows = [{col: str(row.get(col, "") or "") for col in kept} for row in rows]
    return kept, pruned_rows


def _stub_missing_records(
    rows: list[dict[str, str]],
    *,
    columns: list[str],
    primary_key: list[str],
    record_keys: list[str],
) -> list[dict[str, str]]:
    if not primary_key or not record_keys:
        return rows
    pk = primary_key[0]
    seen = {str(row.get(pk, "") or "").strip() for row in rows}
    extra: list[dict[str, str]] = []
    for key in record_keys:
        if key in seen:
            continue
        stub = {col: "" for col in columns}
        stub[pk] = key
        extra.append(stub)
        seen.add(key)
    return rows + extra


def extract_segment_table(
    model: ModelAdapter,
    *,
    columns: list[str],
    primary_key: list[str],
    paragraphs: list[str],
    knowledge: str,
    budget: ExtractionBudget | None = None,
    where: str = "extract_segment",
) -> list[dict[str, str]]:
    """Extract one segment in isolation. Parse failure retries once, then keeps blanks."""
    if not paragraphs:
        return []
    col_list = ", ".join(columns)
    pk_list = ", ".join(primary_key) or "(none)"
    knowledge_block = knowledge[:3000] if knowledge else "(no knowledge.md)"
    numbered = "\n\n".join(f"[{index + 1}]\n{para}" for index, para in enumerate(paragraphs))
    prompt = f"""Extract rows from this document segment into JSON.

Return exactly:
{{"rows": [{{"<col>": "<string>"}}, ...]}}

Rules:
- Keys must be a subset of: {col_list}
- Primary key column(s): {pk_list}
- Missing fields → ""
- One row per entity mentioned. Same entity may appear again later; only fill fields this segment states.
- If text says initially X then corrected/finalized Y, store Y.
- Registry IDs / official names beat nicknames. Do not prepend general/former/foundational.
- Do not invent facts. Ignore titles, TOC, memos with no entity.
- knowledge.md excerpt:
{knowledge_block}

Segment:
{numbered}
"""

    def _parse(payload: dict[str, Any]) -> list[dict[str, str]]:
        raw_rows = payload.get("rows")
        if raw_rows is None and isinstance(payload.get("records"), list):
            raw_rows = []
            for item in payload["records"]:
                if isinstance(item, dict) and isinstance(item.get("values"), dict):
                    raw_rows.append(item["values"])
                elif isinstance(item, dict):
                    raw_rows.append(item)
        if not isinstance(raw_rows, list):
            return []
        parsed: list[dict[str, str]] = []
        for item in raw_rows:
            if not isinstance(item, dict):
                continue
            values = item.get("values") if isinstance(item.get("values"), dict) else item
            row = _row_from_values(values, columns)
            if row:
                parsed.append(row)
        return parsed

    try:
        return _complete_json_validated(
            model, prompt, parse=_parse, budget=budget, where=where
        )
    except DocumentExtractionError:
        raise
    except Exception:
        return []


def _run_segment_workers(
    model: ModelAdapter,
    *,
    paragraphs: list[str],
    plan: DocumentPlan,
    knowledge: str,
    budget: ExtractionBudget,
    source_rel: str,
) -> tuple[list[dict[str, str]], bool]:
    """Run one worker per segment. Returns (rows, all_segments_finished)."""
    from data_agent_baseline.run.progress import mark as _mark

    collected: list[dict[str, str]] = []
    total = len(plan.segments)
    if total == 0:
        return [], False
    workers = max(1, min(EXTRACT_CONCURRENCY, total))
    _mark(
        "extract_segments_start",
        file=source_rel,
        segments=total,
        concurrency=workers,
    )

    def _one(index: int, start: int, end: int) -> list[dict[str, str]]:
        chunk = paragraphs[start:end]
        chars = sum(len(p) for p in chunk)
        _mark(
            "extract_segment",
            file=source_rel,
            segment=index,
            total_segments=total,
            paragraphs=len(chunk),
            chars=chars,
            budget_remaining=round(budget.remaining, 1),
        )
        return extract_segment_table(
            model,
            columns=plan.columns,
            primary_key=plan.primary_key,
            paragraphs=chunk,
            knowledge=knowledge,
            budget=budget,
            where=f"extract_segment_{index}/{total}",
        )

    finished = 0
    indexed = list(enumerate(plan.segments, start=1))
    for wave_start in range(0, len(indexed), workers):
        if budget.remaining <= 0:
            return collected, False
        wave = indexed[wave_start : wave_start + workers]
        if workers == 1 or len(wave) == 1:
            index, (start, end) = wave[0]
            try:
                collected.extend(_one(index, start, end))
                finished += 1
            except DocumentExtractionError:
                return collected, False
            continue
        errors: list[BaseException] = []
        with ThreadPoolExecutor(max_workers=len(wave), thread_name_prefix="extract") as pool:
            futures = {
                pool.submit(_one, index, start, end): index
                for index, (start, end) in wave
            }
            for fut in as_completed(futures):
                try:
                    collected.extend(fut.result())
                    finished += 1
                except DocumentExtractionError as exc:
                    errors.append(exc)
                except Exception as exc:
                    errors.append(exc)
        if any(isinstance(exc, DocumentExtractionError) for exc in errors):
            return collected, False
    return collected, finished == total


def declare_and_apply_formats(
    model: ModelAdapter,
    *,
    columns: list[str],
    rows: list[dict[str, str]],
    budget: ExtractionBudget | None = None,
) -> list[dict[str, str]]:
    from data_agent_baseline.tools.extract_normalize import (
        apply_column_converters,
        parse_format_declarations,
    )

    if not rows or not columns:
        return rows
    samples: dict[str, list[str]] = {}
    for col in columns:
        seen: list[str] = []
        for row in rows:
            value = str(row.get(col, "") or "").strip()
            if value and value not in seen:
                seen.append(value)
            if len(seen) >= 8:
                break
        samples[col] = seen
    sample_block = json.dumps(samples, ensure_ascii=False)
    prompt = f"""Declare target string formats for extracted columns. Do not convert the values.

Return JSON:
{{"formats": {{"<col>": {{"format": "iso_date|upper|lower|title|strip|digits", "code": "optional def convert(value): ..."}}}}}}

Rules:
- Only include columns that need normalization.
- code if present must define convert(value) -> str.
- knowledge of actual values:
{sample_block}
Columns: {", ".join(columns)}
"""
    try:
        payload = _complete_json(
            model, prompt, budget=budget, where="extract_normalize_declare"
        )
        converters = parse_format_declarations(payload, columns)
    except DocumentExtractionError:
        return rows
    except Exception:
        return rows
    if not converters:
        return rows
    return apply_column_converters(rows, columns=columns, converters=converters)


def _complete_json(
    model: ModelAdapter,
    prompt: str,
    *,
    budget: ExtractionBudget | None = None,
    where: str = "llm_call",
) -> dict[str, Any]:
    from data_agent_baseline.agents.model import ModelMessage

    def _once(*, pace: bool) -> str:
        if budget is not None:
            budget.check(where)
        if pace and EXTRACT_CALL_DELAY_SECONDS > 0:
            time.sleep(EXTRACT_CALL_DELAY_SECONDS)
            if budget is not None:
                budget.check(where)
        return model.complete([ModelMessage(role="user", content=prompt)])

    try:
        raw = _once(pace=True)
    except Exception as exc:
        err = str(exc)
        rate_limit = _is_rate_limit_error(err)
        transient = _is_transient_error(err)
        if not rate_limit and not transient:
            raise
        # One serialized backoff retry — concurrent workers share this lock so a
        # 429 / connection blip does not fan out into a stampede.
        label = "429_backoff" if rate_limit else "transient_backoff"
        with _RATE_LIMIT_LOCK:
            if budget is not None:
                budget.check(f"{where}:{label}")
            time.sleep(EXTRACT_429_BACKOFF_SECONDS)
            try:
                raw = _once(pace=False)
            except Exception as retry_exc:
                retry_err = str(retry_exc)
                if _is_rate_limit_error(retry_err):
                    raise DocumentExtractionError(
                        f"rate_limit after one retry at {where}: {retry_err[:200]}"
                    ) from retry_exc
                if _is_transient_error(retry_err):
                    raise DocumentExtractionError(
                        f"transient after one retry at {where}: {retry_err[:200]}"
                    ) from retry_exc
                raise
    return parse_json_object(raw)


def infer_schema(
    model: ModelAdapter,
    *,
    stem: str,
    knowledge: str,
    samples: list[str],
    budget: ExtractionBudget | None = None,
) -> tuple[list[str], list[str]]:
    sample_block = "\n\n".join(f"[sample {i + 1}]\n{para}" for i, para in enumerate(samples))
    knowledge_block = knowledge.strip() or "(no knowledge.md)"
    prompt = f"""Infer a SQL table schema from narrative document paragraphs.

The table name will be `{stem}` (file stem). Prefer field names from knowledge.md when they match this document.

Return one JSON object:
{{
  "columns": [{{"name": "snake_case_col", "type": "VARCHAR"}}],
  "primary_key": ["id_or_name_column"]
}}

Rules:
- One entity per paragraph. Ignore memo headers, summaries, and boilerplate.
- Prefer identifiers (Registry ID, molecule id, patient id) as primary key.
- Include human-readable name/title columns from knowledge when present.
- Do not invent metrics. 6–20 columns is enough.
- knowledge.md excerpt:
{knowledge_block}

Sample paragraphs:
{sample_block}
"""
    payload = _complete_json(model, prompt, budget=budget, where="infer_schema")
    columns = _sanitize_columns(payload.get("columns"))
    if not columns:
        raise ValueError("schema inference returned no columns")
    primary_key = _sanitize_pk(payload.get("primary_key"), columns)
    return columns, primary_key


def _row_from_values(values: Any, columns: list[str]) -> dict[str, str] | None:
    if not isinstance(values, dict):
        return None
    row = {col: "" for col in columns}
    nonempty = False
    for col in columns:
        raw = values.get(col)
        if raw is None:
            continue
        text = str(raw).strip()
        if text.lower() in {
            "nan",
            "none",
            "null",
            "n/a",
            "na",
            "not available",
            "not specified",
            "pending",
        }:
            text = ""
        if text:
            row[col] = text
            nonempty = True
    return row if nonempty else None


def extract_paragraph(
    model: ModelAdapter,
    *,
    columns: list[str],
    paragraph: str,
    knowledge: str,
    budget: ExtractionBudget | None = None,
) -> dict[str, Any] | None:
    col_list = ", ".join(columns)
    knowledge_block = knowledge[:3000] if knowledge else "(no knowledge.md)"
    prompt = f"""Extract one record from this paragraph into JSON.

Return exactly:
{{"skip": false, "values": {{"<col>": "<string>"}}}}

Rules:
- Keys must be a subset of: {col_list}
- Missing fields → ""
- If the paragraph is a title, memo, TOC, or has no entity, return {{"skip": true, "values": {{}}}}
- If the text says initially / previously / listed as X and later corrected / finalized / now Y, store Y.
- Registry IDs and official names beat informal nicknames.
- When the paragraph uses "Name (Registry ID: …)" or "program for Name (Registry ID: …)", store Name exactly — do NOT prepend adjectives such as general / former / foundational / basic from nearby descriptive phrases ("the general Name program").
- Never store "general X" / "former X" when a Registry phrase already names the entity as X.
- Extract every field that the paragraph provides. Do not leave a field empty if the paragraph states its value.
- When the paragraph states an entity's publisher, label, status, or other categorical identifier, store it in the matching column.
- Values are strings. Do not invent facts absent from the paragraph.

knowledge.md excerpt:
{knowledge_block}

Paragraph:
{paragraph}
"""
    try:
        payload = _complete_json(
            model, prompt, budget=budget, where="extract_paragraph"
        )
    except DocumentExtractionError:
        raise
    except Exception:
        return None
    if bool(payload.get("skip")):
        return None
    return _row_from_values(payload.get("values"), columns)


def extract_paragraphs_batch(
    model: ModelAdapter,
    *,
    columns: list[str],
    paragraphs: list[str],
    knowledge: str,
    budget: ExtractionBudget | None = None,
    batch_label: str = "extract_batch",
) -> list[dict[str, str] | None]:
    """Extract many paragraphs in one LLM call. Returns one slot per paragraph."""
    if not paragraphs:
        return []
    col_list = ", ".join(columns)
    knowledge_block = knowledge[:3000] if knowledge else "(no knowledge.md)"
    numbered = "\n\n".join(
        f"[{index + 1}]\n{para}" for index, para in enumerate(paragraphs)
    )
    prompt = f"""Extract one record per numbered paragraph into JSON.

Return exactly:
{{"records": [{{"index": 1, "skip": false, "values": {{"<col>": "<string>"}}}}, ...]}}

Rules:
- Include every paragraph index from 1 to {len(paragraphs)} exactly once.
- Keys in values must be a subset of: {col_list}
- Missing fields → ""
- If a paragraph is a title, memo, TOC, or has no entity, set skip=true and values={{}}
- If the text says initially / previously / listed as X and later corrected / finalized / now Y, store Y.
- Registry IDs and official names beat informal nicknames.
- When the paragraph uses "Name (Registry ID: …)" or "program for Name (Registry ID: …)", store Name exactly — do NOT prepend adjectives such as general / former / foundational / basic from nearby descriptive phrases.
- Never store "general X" / "former X" when a Registry phrase already names the entity as X.
- Extract every field that the paragraph provides. Do not invent facts absent from the paragraph.
- Values are strings.

knowledge.md excerpt:
{knowledge_block}

Paragraphs:
{numbered}
"""
    try:
        payload = _complete_json(model, prompt, budget=budget, where=batch_label)
    except DocumentExtractionError:
        raise
    except Exception:
        return [None] * len(paragraphs)

    ordered: list[dict[str, str] | None] = [None] * len(paragraphs)
    records = payload.get("records")
    if not isinstance(records, list):
        return ordered
    for item in records:
        if not isinstance(item, dict):
            continue
        try:
            index = int(item.get("index"))
        except (TypeError, ValueError):
            continue
        if index < 1 or index > len(paragraphs):
            continue
        if bool(item.get("skip")):
            ordered[index - 1] = None
            continue
        ordered[index - 1] = _row_from_values(item.get("values"), columns)
    return ordered


def _pack_paragraph_batches(
    paragraphs: list[str],
    *,
    max_paras: int | None = None,
    max_chars: int | None = None,
) -> list[tuple[int, int]]:
    """Pack paragraphs into (start, end) slices by char budget and para cap.

    Each batch aims to fill ``max_chars`` of paragraph text (plus light numbering
    overhead) without exceeding ``max_paras`` items. A single oversized paragraph
    always gets its own batch.
    """
    if not paragraphs:
        return []
    para_cap = max(1, int(max_paras if max_paras is not None else EXTRACT_BATCH_MAX_PARAS))
    char_cap = max(1, int(max_chars if max_chars is not None else EXTRACT_BATCH_MAX_CHARS))
    batches: list[tuple[int, int]] = []
    index = 0
    total = len(paragraphs)
    while index < total:
        end = index
        used_chars = 0
        while end < total and (end - index) < para_cap:
            # Numbering wrapper "[n]\\n" ≈ 4–6 chars; keep a small fixed overhead.
            piece = len(paragraphs[end]) + 6
            if end > index and used_chars + piece > char_cap:
                break
            used_chars += piece
            end += 1
        if end == index:
            end = index + 1
        batches.append((index, end))
        index = end
    return batches


def _preflight_can_finish(
    batch_count: int, remaining_seconds: float, *, concurrency: int = 1
) -> bool:
    """Estimate wall-clock finishability (schema + parallel batch waves)."""
    workers = max(1, concurrency)
    wall_batches = (max(0, batch_count) + workers - 1) // workers
    calls = 1 + wall_batches
    return calls * EXTRACT_SEC_PER_CALL <= remaining_seconds


def _plan_extract_batches(
    paragraphs: list[str],
    remaining_seconds: float,
    *,
    concurrency: int,
) -> tuple[list[tuple[int, int]], int, int]:
    """Pack paragraphs, widening char/para caps under time pressure if needed."""
    candidates: list[tuple[int, int]] = [
        (EXTRACT_BATCH_MAX_PARAS, EXTRACT_BATCH_MAX_CHARS),
        (EXTRACT_BATCH_MAX_PARAS, 8000),
        (56, 10000),
        (60, 12000),
    ]
    seen: set[tuple[int, int]] = set()
    best_ranges: list[tuple[int, int]] = []
    best_limits = candidates[0]
    for max_paras, max_chars in candidates:
        key = (max_paras, max_chars)
        if key in seen:
            continue
        seen.add(key)
        ranges = _pack_paragraph_batches(
            paragraphs, max_paras=max_paras, max_chars=max_chars
        )
        best_ranges = ranges
        best_limits = (max_paras, max_chars)
        if _preflight_can_finish(
            len(ranges), remaining_seconds, concurrency=concurrency
        ):
            return ranges, max_paras, max_chars
    return best_ranges, best_limits[0], best_limits[1]


def _choose_batch_size(paragraph_count: int, remaining_seconds: float) -> int:
    """Para cap under remaining budget (recovery path / simple callers)."""
    if paragraph_count <= 0:
        return EXTRACT_BATCH_SIZE
    workers = 1
    usable = max(0.0, remaining_seconds - EXTRACT_SEC_PER_CALL)
    wall_slots = max(1, int(usable // EXTRACT_SEC_PER_CALL))
    needed = (paragraph_count + wall_slots - 1) // wall_slots
    return max(1, min(60, max(EXTRACT_BATCH_SIZE, needed)))


def _run_first_pass_batches(
    model: ModelAdapter,
    *,
    paragraphs: list[str],
    batch_ranges: list[tuple[int, int]],
    columns: list[str],
    knowledge: str,
    budget: ExtractionBudget,
    source_rel: str,
) -> tuple[list[dict[str, str] | None], str | None]:
    """Extract packed batches. Returns (rows, incomplete_reason).

    Incomplete means the first pass did not cover every paragraph. The caller
    must discard these rows — never register or cache a partial document table.
    Waves stop when the budget is gone so later batches are not submitted.
    """
    from data_agent_baseline.run.progress import mark as _mark

    ordered: list[dict[str, str] | None] = [None] * len(paragraphs)
    total_batches = len(batch_ranges)
    if total_batches == 0:
        return ordered, None

    def _one(batch_index: int, start: int, end: int) -> tuple[int, list[dict[str, str] | None]]:
        if budget.remaining <= 0:
            raise DocumentExtractionError(
                f"extract_timeout at extract_batch_{batch_index}/{total_batches}"
            )
        chunk = paragraphs[start:end]
        _mark(
            "extract_batch",
            file=source_rel,
            batch=batch_index,
            total_batches=total_batches,
            paragraphs=len(chunk),
            chars=sum(len(p) for p in chunk),
            budget_remaining=round(budget.remaining, 1),
        )
        rows = extract_paragraphs_batch(
            model,
            columns=columns,
            paragraphs=chunk,
            knowledge=knowledge,
            budget=budget,
            batch_label=f"extract_batch_{batch_index}/{total_batches}",
        )
        return start, rows

    workers = max(1, min(EXTRACT_CONCURRENCY, total_batches))
    _mark(
        "extract_batches_start",
        file=source_rel,
        total_batches=total_batches,
        concurrency=workers,
    )
    incomplete: str | None = None
    indexed = list(enumerate(batch_ranges, start=1))
    for wave_start in range(0, len(indexed), workers):
        if budget.remaining <= 0:
            incomplete = (
                f"extract_timeout after {round(time.perf_counter() - budget.started_at, 1)}s "
                f"before extract_batch_{indexed[wave_start][0]}/{total_batches}"
            )
            break
        wave = indexed[wave_start : wave_start + workers]
        if workers == 1 or len(wave) == 1:
            batch_index, (start, end) = wave[0]
            try:
                start_idx, rows = _one(batch_index, start, end)
            except DocumentExtractionError as exc:
                incomplete = str(exc)
                break
            for offset, row in enumerate(rows):
                ordered[start_idx + offset] = row
            continue

        errors: list[BaseException] = []
        with ThreadPoolExecutor(max_workers=len(wave), thread_name_prefix="extract") as pool:
            futures = [
                pool.submit(_one, batch_index, start, end)
                for batch_index, (start, end) in wave
            ]
            for fut in as_completed(futures):
                try:
                    start_idx, rows = fut.result()
                except DocumentExtractionError as exc:
                    errors.append(exc)
                    continue
                except Exception as exc:
                    errors.append(exc)
                    continue
                for offset, row in enumerate(rows):
                    ordered[start_idx + offset] = row
        if errors:
            timeout_exc = next(
                (exc for exc in errors if isinstance(exc, DocumentExtractionError)),
                None,
            )
            if timeout_exc is not None:
                incomplete = str(timeout_exc)
                break
            # Non-timeout batch errors already mapped to empty slots; keep going.
    return ordered, incomplete


def merge_rows(
    rows: list[dict[str, str]],
    *,
    columns: list[str],
    primary_key: list[str],
) -> list[dict[str, str]]:
    if not rows:
        return []
    if not primary_key:
        return list(rows)
    grouped: dict[tuple[str, ...], dict[str, str]] = {}
    order: list[tuple[str, ...]] = []
    for row in rows:
        key = tuple(str(row.get(col, "") or "").strip() for col in primary_key)
        if not any(key):
            continue
        if key not in grouped:
            merged = {col: "" for col in columns}
            for col, value in zip(primary_key, key):
                merged[col] = value
            grouped[key] = merged
            order.append(key)
        target = grouped[key]
        for col in columns:
            value = str(row.get(col, "") or "").strip()
            # Only fill empty cells. Do not overwrite an already-populated value
            # with a later value, to avoid aliases/descriptions replacing official
            # names (e.g. "Business" -> "General Business").
            if value and not target[col]:
                target[col] = value
    return [grouped[key] for key in order]


_REGISTRY_PAIR_RE = re.compile(
    r"([^\n\r(]{1,120}?)\s*\(\s*Registry\s+ID\s*:\s*([A-Za-z0-9_\-]+)\s*\)",
    flags=re.IGNORECASE,
)
_REGISTRY_LEADING_NOISE_RE = re.compile(
    r"^(?:the\s+)?(?:program\s+for\s+|entry\s+for\s+|record\s+for\s+|item\s+for\s+)",
    flags=re.IGNORECASE,
)
_REGISTRY_TRAILING_NOISE_RE = re.compile(r"\s+program$", flags=re.IGNORECASE)
_NAME_MODIFIER_RE = re.compile(
    r"^(?:general|former|foundational|basic|overall|main)\s+",
    flags=re.IGNORECASE,
)
_NAME_COL_RE = re.compile(r"name|title|program|label|entity", flags=re.IGNORECASE)
_ID_COL_RE = re.compile(r"registry|record_id|(?:^|_)id$", flags=re.IGNORECASE)
_REGISTRY_FILLER = frozenset(
    {
        "later",
        "then",
        "and",
        "but",
        "for",
        "of",
        "in",
        "on",
        "at",
        "the",
        "a",
        "an",
        "this",
        "that",
        "their",
        "its",
    }
)


def _clean_registry_phrase(raw: str) -> str:
    """Normalize a Registry-phrase left-hand name without stripping proper nouns."""
    text = " ".join((raw or "").strip().split())
    text = _REGISTRY_LEADING_NOISE_RE.sub("", text)
    text = _REGISTRY_TRAILING_NOISE_RE.sub("", text).strip().strip("\"'")
    parts = text.split()
    while parts and parts[0].casefold() in _REGISTRY_FILLER:
        parts.pop(0)
    return " ".join(parts)


def _pick_official_name(names: list[str]) -> str:
    """Prefer a name that does not start with adjective modifiers when both exist."""
    bare = [n for n in names if n and not _NAME_MODIFIER_RE.match(n)]
    pool = bare or [n for n in names if n]
    if not pool:
        return ""
    return min(pool, key=lambda n: (len(n), n.casefold()))


def apply_registry_official_names(
    paragraphs: list[str],
    rows: list[dict[str, str]],
    *,
    columns: list[str],
) -> list[dict[str, str]]:
    """Prefer Registry-phrase official names over adjective-prefixed aliases.

    Example: 'Business (Registry ID: …)' wins over 'the general Business program'.
    Never strips modifiers unless a bare alternative for the same Registry ID exists
    (so 'General Motors' alone is kept).
    """
    if not rows or not paragraphs:
        return rows

    id_to_names: dict[str, list[str]] = {}
    for para in paragraphs:
        for match in _REGISTRY_PAIR_RE.finditer(para):
            name = _clean_registry_phrase(match.group(1))
            rid = match.group(2).strip()
            if name and rid:
                id_to_names.setdefault(rid, []).append(name)

    if not id_to_names:
        return rows

    id_to_official = {
        rid: _pick_official_name(names) for rid, names in id_to_names.items()
    }
    id_to_official = {rid: name for rid, name in id_to_official.items() if name}
    if not id_to_official:
        return rows

    official_fold = {name.casefold(): name for name in id_to_official.values()}

    id_cols = [c for c in columns if _ID_COL_RE.search(c)]
    name_cols = [c for c in columns if _NAME_COL_RE.search(c)]
    if not name_cols:
        return rows

    for row in rows:
        for id_col in id_cols:
            rid = str(row.get(id_col, "") or "").strip()
            official = id_to_official.get(rid)
            if not official:
                continue
            for name_col in name_cols:
                current = str(row.get(name_col, "") or "").strip()
                if not current:
                    row[name_col] = official
                    continue
                if current.casefold() == official.casefold():
                    continue
                stripped = _NAME_MODIFIER_RE.sub("", current).strip()
                if stripped.casefold() == official.casefold():
                    row[name_col] = official
            break
        else:
            for name_col in name_cols:
                current = str(row.get(name_col, "") or "").strip()
                if not current:
                    continue
                stripped = _NAME_MODIFIER_RE.sub("", current).strip()
                if stripped.casefold() == current.casefold():
                    continue
                official = official_fold.get(stripped.casefold())
                if official is not None:
                    row[name_col] = official
    return rows


def _extract_key_pattern(rows: list[dict[str, str]], primary_key: list[str]) -> re.Pattern[str] | None:
    """Build a regex that matches primary-key values already seen in rows.

    If all values share a common prefix+number format (e.g. TR391, TR483), generalize
    the pattern so missing keys like TR450 are also found in the raw text.
    """
    values: set[str] = set()
    for row in rows:
        for col in primary_key:
            value = str(row.get(col, "") or "").strip()
            if value:
                values.add(value)
    if not values:
        return None
    sorted_values = sorted(values, key=len, reverse=True)
    # Detect prefix + digits pattern.
    prefix_match = re.match(r"^([A-Za-z]+)(\d+)$", sorted_values[0])
    if prefix_match and all(re.match(rf"^{re.escape(prefix_match.group(1))}\d+$", v) for v in sorted_values):
        prefix = re.escape(prefix_match.group(1))
        return re.compile(rf"\b({prefix}\d+)\b")
    # Detect numeric IDs only when they are long enough to be specific.
    if all(v.isdigit() for v in sorted_values):
        if all(len(v) >= 3 for v in sorted_values):
            return re.compile(r"\b(\d{3,})\b")
        # Short numeric IDs are too ambiguous for text scanning.
        return None
    escaped = [re.escape(v) for v in sorted_values]
    return re.compile(r"\b(" + "|".join(escaped) + r")\b")


def _missing_key_rows(
    paragraphs: list[str],
    rows: list[dict[str, str]],
    columns: list[str],
    primary_key: list[str],
) -> list[dict[str, str]]:
    """Create stub rows for primary-key values found in text but not in extracted rows."""
    pattern = _extract_key_pattern(rows, primary_key)
    if pattern is None:
        return []
    seen: set[tuple[str, ...]] = set()
    for row in rows:
        seen.add(tuple(str(row.get(col, "") or "").strip() for col in primary_key))
    stubs: list[dict[str, str]] = []
    for para in paragraphs:
        for match in pattern.finditer(para):
            key_value = match.group(1)
            key_tuple = tuple(key_value if col == primary_key[0] else "" for col in primary_key)
            if key_tuple in seen:
                continue
            seen.add(key_tuple)
            stub = {col: "" for col in columns}
            for col, value in zip(primary_key, key_tuple):
                stub[col] = value
            stubs.append(stub)
    return stubs


def _extract_missing_key_paragraphs(
    model: ModelAdapter,
    paragraphs: list[str],
    rows: list[dict[str, str]],
    *,
    columns: list[str],
    primary_key: list[str],
    knowledge: str,
    budget: ExtractionBudget | None = None,
) -> list[dict[str, str]]:
    """Re-extract paragraphs whose primary key is missing from rows.

    This catches entities that the first per-paragraph pass failed to extract.
    Stops early when the shared extract budget is exhausted (no partial failure:
    the first pass already covered every paragraph).
    """
    pattern = _extract_key_pattern(rows, primary_key)
    if pattern is None:
        return []
    existing_keys = {
        tuple(str(row.get(col, "") or "").strip() for col in primary_key)
        for row in rows
    }
    pending_paras: list[str] = []
    pending_keys: list[str] = []
    for para in paragraphs:
        for match in pattern.finditer(para):
            key_value = match.group(1)
            key_tuple = tuple(key_value if col == primary_key[0] else "" for col in primary_key)
            if key_tuple in existing_keys:
                continue
            existing_keys.add(key_tuple)
            pending_paras.append(para)
            pending_keys.append(key_value)
    if not pending_paras:
        return []

    extra: list[dict[str, str]] = []
    remaining = budget.remaining if budget is not None else EXTRACT_BUDGET_SECONDS
    batch_ranges, _max_paras, _max_chars = _plan_extract_batches(
        pending_paras, remaining, concurrency=1
    )
    for batch_index, (start, end) in enumerate(batch_ranges, start=1):
        if budget is not None and budget.remaining < EXTRACT_SEC_PER_CALL:
            break
        chunk = pending_paras[start:end]
        key_chunk = pending_keys[start:end]
        try:
            batch_rows = extract_paragraphs_batch(
                model,
                columns=columns,
                paragraphs=chunk,
                knowledge=knowledge,
                budget=budget,
                batch_label=f"extract_recover_{batch_index}",
            )
        except DocumentExtractionError:
            # Recovery is best-effort after a complete first pass.
            break
        for row, key_value in zip(batch_rows, key_chunk):
            if row is None:
                continue
            if primary_key:
                row[primary_key[0]] = key_value
            extra.append(row)
    return extra


def _rows_with_empty_fields(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if any(not str(row.get(col, "") or "").strip() for col in row)]


def _build_extraction_report(
    stem: str,
    paragraphs: list[str],
    rows: list[dict[str, str]],
    *,
    columns: list[str],
    primary_key: list[str],
) -> ExtractionReport:
    """Best-effort completeness report for an extracted doc table."""
    pattern = _extract_key_pattern(rows, primary_key)
    key_scannable = pattern is not None
    missing_keys: list[str] = []
    seen_keys: set[tuple[str, ...]] = set()
    for row in rows:
        seen_keys.add(tuple(str(row.get(col, "") or "").strip() for col in primary_key))
    if pattern is not None and primary_key:
        seen_in_text: set[str] = set()
        for para in paragraphs:
            for match in pattern.finditer(para):
                seen_in_text.add(match.group(1))
        for key_value in sorted(seen_in_text):
            key_tuple = tuple(
                key_value if col == primary_key[0] else "" for col in primary_key
            )
            if key_tuple not in seen_keys:
                missing_keys.append(key_value)
    empty_cells = 0
    fill_rate: dict[str, float] = {}
    for col in columns:
        nonempty = sum(
            1 for row in rows if str(row.get(col, "") or "").strip()
        )
        fill_rate[col] = round(nonempty / max(len(rows), 1), 3)
        empty_cells += len(rows) - nonempty
    return ExtractionReport(
        stem=stem,
        row_count=len(rows),
        key_scannable=key_scannable,
        missing_keys=missing_keys,
        empty_field_cells=empty_cells,
        field_fill_rate=fill_rate,
    )


def _fill_empty_fields(
    model: ModelAdapter,
    paragraphs: list[str],
    rows: list[dict[str, str]],
    *,
    columns: list[str],
    primary_key: list[str],
    knowledge: str,
) -> list[dict[str, str]]:
    """Re-extract paragraphs for rows that still have empty fields.

    Some per-paragraph extractions miss fields that appear in later paragraphs about
    the same entity. Re-extract those paragraphs and merge the values in.
    """
    rows_need = _rows_with_empty_fields(rows)
    if not rows_need:
        return rows
    pattern = _extract_key_pattern(rows, primary_key)
    if pattern is None:
        return rows
    row_by_key: dict[tuple[str, ...], dict[str, str]] = {}
    for row in rows:
        key = tuple(str(row.get(col, "") or "").strip() for col in primary_key)
        if any(key):
            row_by_key[key] = row
    for para in paragraphs:
        for match in pattern.finditer(para):
            key_value = match.group(1)
            key_tuple = tuple(key_value if col == primary_key[0] else "" for col in primary_key)
            row = row_by_key.get(key_tuple)
            if row is None:
                continue
            if not any(not str(row.get(col, "") or "").strip() for col in row):
                continue
            extracted = extract_paragraph(
                model,
                columns=columns,
                paragraph=para,
                knowledge=knowledge,
            )
            if extracted is not None:
                for col in columns:
                    value = str(extracted.get(col, "") or "").strip()
                    if value and not str(row.get(col, "") or "").strip():
                        row[col] = value
    return list(row_by_key.values())


def _load_cache(
    path: Path, accepted_hashes: set[str] | str
) -> tuple[ExtractedDoc, int] | None:
    """Load a cached extraction. Returns (doc, cache_version) or None.

    Any version in [_MIN_CACHE_VERSION, _EXTRACT_VERSION] with a matching
    source hash is accepted; the caller upgrades older docs deterministically.
    """
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    version = int(payload.get("version") or 0)
    if version < _MIN_CACHE_VERSION or version > _EXTRACT_VERSION:
        return None
    if isinstance(accepted_hashes, str):
        accepted_hashes = {accepted_hashes}
    if payload.get("source_hash") not in accepted_hashes:
        return None
    columns = [str(item) for item in payload.get("columns") or [] if str(item).strip()]
    rows_raw = payload.get("rows")
    if not columns or not isinstance(rows_raw, list):
        return None
    rows: list[dict[str, str]] = []
    for item in rows_raw:
        if not isinstance(item, dict):
            continue
        rows.append({col: str(item.get(col, "") or "") for col in columns})
    report_payload = payload.get("report")
    report: ExtractionReport | None = None
    if isinstance(report_payload, dict):
        report = ExtractionReport(
            stem=str(report_payload.get("stem") or payload.get("stem") or path.stem),
            row_count=int(report_payload.get("row_count") or len(rows)),
            key_scannable=bool(report_payload.get("key_scannable", False)),
            missing_keys=[str(k) for k in report_payload.get("missing_keys") or []],
            empty_field_cells=int(report_payload.get("empty_field_cells") or 0),
            field_fill_rate={
                str(k): float(v)
                for k, v in (report_payload.get("field_fill_rate") or {}).items()
            },
        )
    return (
        ExtractedDoc(
            stem=str(payload.get("stem") or path.stem),
            source_rel=str(payload.get("source_rel") or path.stem),
            columns=columns,
            primary_key=[str(item) for item in payload.get("primary_key") or []],
            rows=rows,
            report=report,
        ),
        version,
    )


def _upgrade_cached_doc(
    doc: ExtractedDoc,
    path: Path,
    cache_file: Path,
    context_dir: Path,
) -> ExtractedDoc:
    """Re-apply deterministic post-processing to an older cached extraction.

    Row fixes that do not need the LLM (e.g. Registry official-name
    arbitration) are re-run here, so behavior improvements never force a
    re-extraction. The upgraded doc is re-saved at the current cache version.
    """
    try:
        paragraphs = split_paragraphs(path.read_text(encoding="utf-8", errors="replace"))
        rows = apply_registry_official_names(
            paragraphs, [dict(row) for row in doc.rows], columns=doc.columns
        )
        columns, rows = prune_extract_columns(
            doc.columns, rows, primary_key=doc.primary_key
        )
        primary_key = [col for col in doc.primary_key if col in columns]
        upgraded = ExtractedDoc(
            stem=doc.stem,
            source_rel=doc.source_rel,
            columns=columns,
            primary_key=primary_key,
            rows=rows,
            report=doc.report,
        )
        _save_cache(cache_file, upgraded, _source_hash(path, context_dir))
        return upgraded
    except Exception:
        return doc


def _save_cache(path: Path, doc: ExtractedDoc, source_hash: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    report_payload: dict[str, Any] | None = None
    if doc.report is not None:
        report_payload = {
            "stem": doc.report.stem,
            "row_count": doc.report.row_count,
            "key_scannable": doc.report.key_scannable,
            "missing_keys": doc.report.missing_keys,
            "empty_field_cells": doc.report.empty_field_cells,
            "field_fill_rate": doc.report.field_fill_rate,
        }
    path.write_text(
        json.dumps(
            {
                "version": _EXTRACT_VERSION,
                "source_hash": source_hash,
                "stem": doc.stem,
                "source_rel": doc.source_rel,
                "columns": doc.columns,
                "primary_key": doc.primary_key,
                "rows": doc.rows,
                "report": report_payload,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def extract_document(
    path: Path,
    context_dir: Path,
    model: ModelAdapter | None,
    *,
    budget: ExtractionBudget | None = None,
) -> ExtractedDoc | None:
    from data_agent_baseline.run.progress import mark as _mark

    source_rel = str(path.relative_to(context_dir)).replace("\\", "/")
    cache_file = _cache_path(context_dir, source_rel)
    cached = _load_cache(cache_file, _accepted_source_hashes(path, context_dir))
    if cached is not None:
        doc, cache_version = cached
        if cache_version < _EXTRACT_VERSION:
            doc = _upgrade_cached_doc(doc, path, cache_file, context_dir)
        _mark("extract_cache_hit", file=source_rel, rows=len(doc.rows))
        return doc
    from data_agent_baseline.run.task_budget import get_current_budget

    tb = get_current_budget()
    if tb is not None and tb.should_stop_extract():
        if tb.tier == "medium":
            tb.maybe_upgrade_to_hard()
            extra = tb.extract_seconds()
            if extra > EXTRACT_SEC_PER_CALL and budget is not None:
                budget.deadline = time.perf_counter() + extra
        if tb.should_stop_extract():
            _mark(
                "extract_doc_abandoned",
                file=source_rel,
                error="solve reserve reached; skip cold extract",
                stage="plan",
            )
            return None
    if model is None:
        return None
    paragraphs = split_paragraphs(path.read_text(encoding="utf-8", errors="replace"))
    if not paragraphs:
        return None

    active_budget = budget or ExtractionBudget.from_env()
    knowledge = _knowledge_excerpt(context_dir)
    _mark(
        "extract_plan",
        file=source_rel,
        paragraphs=len(paragraphs),
        budget_remaining=round(active_budget.remaining, 1),
    )
    try:
        plan = plan_document(
            model,
            stem=path.stem,
            paragraphs=paragraphs,
            knowledge=knowledge,
            budget=active_budget,
        )
    except Exception as exc:
        _mark("extract_doc_abandoned", file=source_rel, error=str(exc)[:300], stage="plan")
        return None
    if not plan.segments:
        plan.segments = _fallback_segments(paragraphs)
    _mark(
        "extract_plan_done",
        file=source_rel,
        columns=len(plan.columns),
        records=len(plan.record_keys),
        segments=len(plan.segments),
    )

    try:
        segment_rows, complete = _run_segment_workers(
            model,
            paragraphs=paragraphs,
            plan=plan,
            knowledge=knowledge,
            budget=active_budget,
            source_rel=source_rel,
        )
    except Exception as exc:
        _mark("extract_doc_abandoned", file=source_rel, error=str(exc)[:300], stage="segments")
        return None
    if not complete:
        _mark(
            "extract_doc_abandoned",
            file=source_rel,
            error="segments incomplete; will full-retry next run",
            stage="segments",
        )
        return None

    rows = merge_rows(segment_rows, columns=plan.columns, primary_key=plan.primary_key)
    rows = _stub_missing_records(
        rows,
        columns=plan.columns,
        primary_key=plan.primary_key,
        record_keys=plan.record_keys,
    )
    if not rows:
        _mark("extract_doc_abandoned", file=source_rel, error="no rows after merge", stage="merge")
        return None

    try:
        rows = apply_registry_official_names(
            paragraphs, rows, columns=plan.columns
        )
    except Exception:
        pass

    try:
        _mark("extract_normalize", file=source_rel)
        rows = declare_and_apply_formats(
            model, columns=plan.columns, rows=rows, budget=active_budget
        )
    except Exception:
        pass

    columns, rows = prune_extract_columns(
        plan.columns, rows, primary_key=plan.primary_key
    )
    primary_key = [col for col in plan.primary_key if col in columns]

    report = _build_extraction_report(
        stem=path.stem,
        paragraphs=paragraphs,
        rows=rows,
        columns=columns,
        primary_key=primary_key,
    )
    doc = ExtractedDoc(
        stem=path.stem,
        source_rel=source_rel,
        columns=columns,
        primary_key=primary_key,
        rows=rows,
        report=report,
    )
    source_hash = _source_hash(path, context_dir)
    _save_cache(_cache_path(context_dir, source_rel), doc, source_hash)
    _mark(
        "extract_saved",
        file=source_rel,
        rows=len(rows),
        segments=len(plan.segments),
        columns=len(columns),
    )
    return doc


def extract_all_documents(
    context_dir: Path,
    model: ModelAdapter | None,
    *,
    budget: ExtractionBudget | None = None,
) -> list[ExtractedDoc]:
    from data_agent_baseline.run.progress import mark as _mark

    active_budget = budget or ExtractionBudget.from_env()
    docs: list[ExtractedDoc] = []
    for path in collect_doc_paths(context_dir):
        try:
            extracted = extract_document(
                path, context_dir, model, budget=active_budget
            )
        except DocumentExtractionError as exc:
            _mark("extract_doc_abandoned", file=path.name, error=str(exc)[:300])
            continue
        except Exception as exc:
            _mark("extract_doc_abandoned", file=path.name, error=str(exc)[:300])
            continue
        if extracted is not None and extracted.rows:
            docs.append(extracted)
    return docs
