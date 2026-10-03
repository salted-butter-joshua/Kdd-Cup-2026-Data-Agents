"""Per-task phase heartbeat written to disk so timeouts retain hang location.

The parent kills the worker after ``task_timeout_seconds``. In-memory steps are
lost; this file is flushed after every phase mark so the failure trace can still
say which stage was running.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

_progress_path: Path | None = None
_started_at: float | None = None
_phases: list[dict[str, Any]] = []
_current: dict[str, Any] | None = None
_lock = threading.Lock()


def reset_progress(path: Path | str | None) -> None:
    """Bind this process to a progress file (or disable when path is None)."""
    global _progress_path, _started_at, _phases, _current
    with _lock:
        _phases = []
        _current = None
        _started_at = time.perf_counter()
        if path is None:
            _progress_path = None
            return
        _progress_path = Path(path)
        _progress_path.parent.mkdir(parents=True, exist_ok=True)
    mark("progress_open", pid=os.getpid())


def clear_progress() -> None:
    global _progress_path, _started_at, _phases, _current
    with _lock:
        _progress_path = None
        _started_at = None
        _phases = []
        _current = None


def mark(phase: str, **detail: Any) -> None:
    """Record that ``phase`` just started (or a milestone just completed)."""
    global _current
    with _lock:
        if _progress_path is None or _started_at is None:
            return
        event: dict[str, Any] = {
            "phase": phase,
            "elapsed_seconds": round(time.perf_counter() - _started_at, 3),
            "ts_unix": round(time.time(), 3),
        }
        if detail:
            # Keep the file small and JSON-safe.
            clean: dict[str, Any] = {}
            for key, value in detail.items():
                if value is None:
                    continue
                if isinstance(value, (str, int, float, bool)):
                    clean[key] = value
                else:
                    clean[key] = str(value)[:500]
            event["detail"] = clean
        _current = event
        _phases.append(event)
        _flush_unlocked()


def load_progress(path: Path | str | None) -> dict[str, Any] | None:
    if path is None:
        return None
    candidate = Path(path)
    if not candidate.is_file():
        return None
    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def snapshot() -> dict[str, Any] | None:
    with _lock:
        if _progress_path is None:
            return None
        return {
            "path": str(_progress_path),
            "current": _current,
            "phases": list(_phases),
            "phase_count": len(_phases),
        }


def _flush_unlocked() -> None:
    if _progress_path is None:
        return
    payload = {
        "current": _current,
        "phases": _phases,
        "phase_count": len(_phases),
        "pid": os.getpid(),
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    tmp = _progress_path.with_suffix(_progress_path.suffix + ".tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(_progress_path)
    except Exception:
        try:
            _progress_path.write_text(text, encoding="utf-8")
        except Exception:
            pass
