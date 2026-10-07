"""Keyword / paragraph retrieval over narrative docs (P2, CHESS-lite).

Does not extract tables and does not scan CSV/JSON/SQLite. knowledge.md is
already in the task prompt and is skipped here.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from data_agent_baseline.benchmark.schema import PublicTask

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]{2,}")
_SKIP_NAMES = {"knowledge.md"}
_SUFFIXES = {".md", ".txt"}
_STOP = frozenset(
    {
        "the",
        "and",
        "for",
        "that",
        "with",
        "from",
        "this",
        "what",
        "which",
        "were",
        "was",
        "are",
        "how",
        "many",
        "into",
        "their",
    }
)
MAX_FILE_BYTES = 2_000_000
MAX_HITS = 4
SNIPPET_CHARS = 900


def _tokens(text: str) -> set[str]:
    return {
        tok
        for tok in (m.group(0).casefold() for m in _TOKEN_RE.finditer(text or ""))
        if tok not in _STOP and len(tok) > 2
    }


def iter_doc_files(context_dir: Path) -> list[Path]:
    found: list[Path] = []
    seen: set[Path] = set()
    roots = [context_dir, context_dir / "doc"]
    for root in roots:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() not in _SUFFIXES:
                continue
            if path.name.lower() in _SKIP_NAMES:
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            found.append(path)
    return found


def _paragraphs(text: str) -> list[str]:
    chunks = re.split(r"\n\s*\n+", text)
    return [chunk.strip() for chunk in chunks if len(chunk.strip()) >= 40]


def search_docs(task: PublicTask, query: str, *, max_hits: int = MAX_HITS) -> dict[str, Any]:
    q_tok = _tokens(query) | _tokens(task.question)
    files = iter_doc_files(task.context_dir)
    scored: list[tuple[float, str, str]] = []
    for path in files:
        try:
            if path.stat().st_size > MAX_FILE_BYTES:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        rel = path.relative_to(task.context_dir).as_posix()
        for para in _paragraphs(text):
            overlap = len(q_tok & _tokens(para))
            if overlap <= 0:
                continue
            scored.append((float(overlap), rel, para))
    scored.sort(key=lambda item: item[0], reverse=True)
    hits = []
    seen_para: set[str] = set()
    for score, rel, para in scored:
        key = para[:120]
        if key in seen_para:
            continue
        seen_para.add(key)
        snippet = para if len(para) <= SNIPPET_CHARS else para[: SNIPPET_CHARS - 1] + "…"
        hits.append({"path": rel, "score": score, "snippet": snippet})
        if len(hits) >= max(1, min(max_hits, 8)):
            break
    return {
        "query": query,
        "hits": hits,
        "n_files": len(files),
        "note": (
            "Snippets only. They are not warehouse rows. If a measure lives in "
            "the doc and the table column is empty, do not submit 0; the extract "
            "step must populate the table. Do not glob CSV."
        ),
    }
