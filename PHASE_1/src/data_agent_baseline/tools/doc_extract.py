"""Extract narrative context docs into tabular rows for the per-task DuckDB warehouse.

knowledge.md stays in the prompt and is never a table. Other .md/.txt files are
split into paragraphs, schema-inferred from knowledge + samples, then extracted
with the same chat model. Last / corrected values win. Table name = file stem.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
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
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", flags=re.DOTALL)
_BLANK_RE = re.compile(r"\n\s*\n+")

# Extraction is serial + paced by default: concurrent bursts trip API rate
# limits, and the resulting 429 backoff storm costs far more wall-clock than
# paced serial calls. Override via env when the quota allows more.
EXTRACT_CONCURRENCY = max(1, int(os.environ.get("DATA_AGENT_EXTRACT_CONCURRENCY", "1") or "1"))
EXTRACT_CALL_DELAY_SECONDS = max(
    0.0, float(os.environ.get("DATA_AGENT_EXTRACT_DELAY", "0.5") or "0.5")
)

# Extraction cache version. Bump ONLY when LLM-side extraction logic changes
# (prompts, schema inference). Deterministic post-processing (e.g. Registry
# official-name arbitration) never needs a bump: it is re-applied at load time.
_EXTRACT_VERSION = 4
# Oldest reusable cache version. Accepted caches are upgraded deterministically
# at load and re-saved at the current version, without any LLM call.
# v1/v2 are included so scored runs never re-extract just because of an old
# version stamp (that re-extract is the main cause of 300s warehouse timeouts).
_MIN_CACHE_VERSION = 1


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


def _complete_json(model: ModelAdapter, prompt: str) -> dict[str, Any]:
    from data_agent_baseline.agents.model import ModelMessage

    if EXTRACT_CALL_DELAY_SECONDS > 0:
        # Pace every extraction LLM call (schema inference, per-paragraph, and
        # recovery passes all funnel through here) to stay under rate limits.
        time.sleep(EXTRACT_CALL_DELAY_SECONDS)
    raw = model.complete([ModelMessage(role="user", content=prompt)])
    return parse_json_object(raw)


def infer_schema(
    model: ModelAdapter,
    *,
    stem: str,
    knowledge: str,
    samples: list[str],
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
    payload = _complete_json(model, prompt)
    columns = _sanitize_columns(payload.get("columns"))
    if not columns:
        raise ValueError("schema inference returned no columns")
    primary_key = _sanitize_pk(payload.get("primary_key"), columns)
    return columns, primary_key


def extract_paragraph(
    model: ModelAdapter,
    *,
    columns: list[str],
    paragraph: str,
    knowledge: str,
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
        payload = _complete_json(model, prompt)
    except Exception:
        return None
    if bool(payload.get("skip")):
        return None
    values = payload.get("values")
    if not isinstance(values, dict):
        return None
    row = {col: "" for col in columns}
    nonempty = False
    for col in columns:
        raw = values.get(col)
        if raw is None:
            continue
        text = str(raw).strip()
        if text.lower() in {"nan", "none", "null", "n/a", "na", "not available", "not specified", "pending"}:
            text = ""
        if text:
            row[col] = text
            nonempty = True
    return row if nonempty else None


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
) -> list[dict[str, str]]:
    """Re-extract paragraphs whose primary key is missing from rows.

    This catches entities that the first per-paragraph pass failed to extract.
    """
    pattern = _extract_key_pattern(rows, primary_key)
    if pattern is None:
        return []
    existing_keys = {
        tuple(str(row.get(col, "") or "").strip() for col in primary_key)
        for row in rows
    }
    extra: list[dict[str, str]] = []
    for para in paragraphs:
        for match in pattern.finditer(para):
            key_value = match.group(1)
            key_tuple = tuple(key_value if col == primary_key[0] else "" for col in primary_key)
            if key_tuple in existing_keys:
                continue
            existing_keys.add(key_tuple)
            row = extract_paragraph(
                model,
                columns=columns,
                paragraph=para,
                knowledge=knowledge,
            )
            if row is not None:
                # Force the primary key value from the text.
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
        upgraded = ExtractedDoc(
            stem=doc.stem,
            source_rel=doc.source_rel,
            columns=doc.columns,
            primary_key=doc.primary_key,
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
) -> ExtractedDoc | None:
    source_rel = str(path.relative_to(context_dir)).replace("\\", "/")
    cache_file = _cache_path(context_dir, source_rel)
    cached = _load_cache(cache_file, _accepted_source_hashes(path, context_dir))
    if cached is not None:
        doc, cache_version = cached
        if cache_version < _EXTRACT_VERSION:
            doc = _upgrade_cached_doc(doc, path, cache_file, context_dir)
        return doc
    if model is None:
        return None
    paragraphs = split_paragraphs(path.read_text(encoding="utf-8", errors="replace"))
    if not paragraphs:
        return None
    knowledge = _knowledge_excerpt(context_dir)
    columns, primary_key = infer_schema(
        model,
        stem=path.stem,
        knowledge=knowledge,
        samples=_sample_paragraphs(paragraphs),
    )
    ordered: list[dict[str, str] | None] = [None] * len(paragraphs)
    workers = min(EXTRACT_CONCURRENCY, max(1, len(paragraphs)))
    if workers == 1:
        # Serial + paced: under rate limits this is faster than a concurrent
        # burst that triggers 429 backoff, and it cannot burst-trip the limit.
        for index, paragraph in enumerate(paragraphs):
            try:
                ordered[index] = extract_paragraph(
                    model,
                    columns=columns,
                    paragraph=paragraph,
                    knowledge=knowledge,
                )
            except Exception:
                ordered[index] = None
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    extract_paragraph,
                    model,
                    columns=columns,
                    paragraph=paragraph,
                    knowledge=knowledge,
                ): index
                for index, paragraph in enumerate(paragraphs)
            }
            for future in as_completed(futures):
                try:
                    ordered[futures[future]] = future.result()
                except Exception:
                    ordered[futures[future]] = None
    extracted = [row for row in ordered if row]
    rows = merge_rows(extracted, columns=columns, primary_key=primary_key)
    if not rows:
        return None

    # Second pass: recover entities that the independent per-paragraph extraction missed.
    try:
        missing_rows = _extract_missing_key_paragraphs(
            model,
            paragraphs,
            rows,
            columns=columns,
            primary_key=primary_key,
            knowledge=knowledge,
        )
        if missing_rows:
            rows = merge_rows(rows + missing_rows, columns=columns, primary_key=primary_key)
    except Exception:
        pass

    # Third pass (DISABLED): filling empty fields from later paragraphs risks
    # overwriting official values with aliases or descriptive text (e.g.
    # "Business" -> "General Business"). Keep the function for reference but do
    # not invoke it. Missing entities are still recovered by the second pass.
    # try:
    #     rows = _fill_empty_fields(...)
    # except Exception:
    #     pass

    # C1: Registry-phrase official names beat adjective-prefixed aliases.
    try:
        rows = apply_registry_official_names(
            paragraphs, rows, columns=columns
        )
    except Exception:
        pass

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
    _save_cache(_cache_path(context_dir, source_rel), doc, source_hash)
    return doc


def extract_all_documents(
    context_dir: Path,
    model: ModelAdapter | None,
) -> list[ExtractedDoc]:
    docs: list[ExtractedDoc] = []
    for path in collect_doc_paths(context_dir):
        try:
            extracted = extract_document(path, context_dir, model)
        except Exception:
            continue
        if extracted is not None and extracted.rows:
            docs.append(extracted)
    return docs
