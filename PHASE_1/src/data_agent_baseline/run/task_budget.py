"""Per-task wall-clock policy: tiered max time, extract vs solve split.

Easy (no narrative docs): 3 min. Medium (docs): 7 min. Hard (large / multi-file
or extract stall): 10 min. Extract never consumes the solve reserve. Parent
process kill is T_easy when there are no docs, otherwise T_cap so a mid-extract
upgrade to Hard can finish.
"""

from __future__ import annotations

import os
import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path

T_CAP = max(60.0, float(os.environ.get("DATA_AGENT_T_CAP", "600") or "600"))
T_EASY = max(30.0, float(os.environ.get("DATA_AGENT_T_EASY", "180") or "180"))
T_MED = max(T_EASY, float(os.environ.get("DATA_AGENT_T_MED", "420") or "420"))
T_HARD = max(T_MED, min(T_CAP, float(os.environ.get("DATA_AGENT_T_HARD", "600") or "600")))

EXTRACT_EASY = 0.0
EXTRACT_MED = max(0.0, float(os.environ.get("DATA_AGENT_EXTRACT_MED", "180") or "180"))
EXTRACT_HARD = max(EXTRACT_MED, float(os.environ.get("DATA_AGENT_EXTRACT_HARD", "300") or "300"))

SOLVE_EASY = T_EASY
SOLVE_MED = max(60.0, float(os.environ.get("DATA_AGENT_SOLVE_MED", "180") or "180"))
SOLVE_HARD = max(SOLVE_MED, float(os.environ.get("DATA_AGENT_SOLVE_HARD", "240") or "240"))

STEPS_EASY = max(4, int(os.environ.get("DATA_AGENT_STEPS_EASY", "16") or "16"))
STEPS_MED = max(STEPS_EASY, int(os.environ.get("DATA_AGENT_STEPS_MED", "24") or "24"))
STEPS_HARD = max(STEPS_MED, int(os.environ.get("DATA_AGENT_STEPS_HARD", "32") or "32"))

HARD_DOC_CHARS = max(
    20_000, int(os.environ.get("DATA_AGENT_HARD_DOC_CHARS", "80000") or "80000")
)
EXTRACT_SEC_PER_CALL = max(1.0, float(os.environ.get("DATA_AGENT_EXTRACT_SEC_PER_CALL", "8") or "8"))

_current: ContextVar["TaskBudget | None"] = ContextVar("task_budget", default=None)


@dataclass
class DocInventory:
    paths: list[Path]
    n_files: int
    n_chars: int
    cache_complete: bool


@dataclass
class TaskBudget:
    """Live clocks for one task. ``started_at`` is ``perf_counter``."""

    tier: str
    task_timeout: float
    extract_max: float
    solve_reserve: float
    max_steps: int
    n_doc_files: int = 0
    n_doc_chars: int = 0
    cache_complete: bool = False
    started_at: float = field(default_factory=time.perf_counter)
    upgraded: bool = False

    @property
    def task_deadline(self) -> float:
        return self.started_at + self.task_timeout

    @property
    def extract_deadline(self) -> float:
        reserved = self.task_deadline - self.solve_reserve
        planned = self.started_at + self.extract_max
        return min(planned, reserved)

    def remaining_task(self) -> float:
        return self.task_deadline - time.perf_counter()

    def remaining_extract(self) -> float:
        return self.extract_deadline - time.perf_counter()

    def should_stop_extract(self) -> bool:
        if self.extract_max <= 0:
            return True
        if self.remaining_extract() < EXTRACT_SEC_PER_CALL:
            return True
        if self.remaining_task() <= self.solve_reserve:
            return True
        return False

    def extract_seconds(self) -> float:
        return max(0.0, self.remaining_extract())

    def maybe_upgrade_to_hard(self) -> bool:
        """If Medium extract is exhausted before docs finish, grow to Hard (≤ T_cap)."""
        if self.tier != "medium":
            return False
        self.tier = "hard"
        self.task_timeout = min(T_CAP, T_HARD)
        self.extract_max = EXTRACT_HARD
        self.solve_reserve = min(SOLVE_HARD, self.task_timeout * 0.4)
        self.max_steps = STEPS_HARD
        self.upgraded = True
        return True

    def to_mark(self) -> dict[str, object]:
        return {
            "tier": self.tier,
            "task_timeout": round(self.task_timeout, 1),
            "extract_max": round(self.extract_max, 1),
            "solve_reserve": round(self.solve_reserve, 1),
            "max_steps": self.max_steps,
            "n_doc_files": self.n_doc_files,
            "n_doc_chars": self.n_doc_chars,
            "cache_complete": self.cache_complete,
            "upgraded": self.upgraded,
            "remaining_task": round(self.remaining_task(), 1),
        }


def inventory_docs(context_dir: Path) -> DocInventory:
    from data_agent_baseline.tools.doc_extract import (
        _accepted_source_hashes,
        _cache_path,
        _load_cache,
        collect_doc_paths,
    )

    paths = collect_doc_paths(context_dir) if context_dir.is_dir() else []
    n_chars = 0
    cache_ok = True
    if not paths:
        cache_ok = False
    for path in paths:
        try:
            n_chars += path.stat().st_size
        except OSError:
            pass
        try:
            rel = str(path.relative_to(context_dir)).replace("\\", "/")
            cached = _load_cache(
                _cache_path(context_dir, rel),
                _accepted_source_hashes(path, context_dir),
            )
        except Exception:
            cached = None
        if cached is None:
            cache_ok = False
    if not paths:
        cache_ok = False
    return DocInventory(
        paths=paths,
        n_files=len(paths),
        n_chars=n_chars,
        cache_complete=bool(paths) and cache_ok,
    )


def classify_task_budget(context_dir: Path, *, now: float | None = None) -> TaskBudget:
    inv = inventory_docs(context_dir)
    if inv.n_files == 0:
        tier = "easy"
        task_timeout = min(T_CAP, T_EASY)
        extract_max = EXTRACT_EASY
        solve_reserve = min(SOLVE_EASY, task_timeout)
        max_steps = STEPS_EASY
    elif inv.n_files >= 2 or inv.n_chars >= HARD_DOC_CHARS:
        tier = "hard"
        task_timeout = min(T_CAP, T_HARD)
        extract_max = EXTRACT_HARD
        solve_reserve = min(SOLVE_HARD, task_timeout * 0.5)
        max_steps = STEPS_HARD
    else:
        tier = "medium"
        task_timeout = min(T_CAP, T_MED)
        extract_max = 0.0 if inv.cache_complete else EXTRACT_MED
        solve_reserve = min(SOLVE_MED, task_timeout * 0.5)
        max_steps = STEPS_MED
    started = time.perf_counter() if now is None else now
    return TaskBudget(
        tier=tier,
        task_timeout=task_timeout,
        extract_max=extract_max,
        solve_reserve=solve_reserve,
        max_steps=max_steps,
        n_doc_files=inv.n_files,
        n_doc_chars=inv.n_chars,
        cache_complete=inv.cache_complete,
        started_at=started,
    )


def process_kill_timeout(budget: TaskBudget) -> float:
    """Parent subprocess kill. Docs get T_cap so Medium→Hard upgrade can finish."""
    if budget.n_doc_files <= 0:
        return min(T_CAP, budget.task_timeout)
    return T_CAP


def set_current_budget(budget: TaskBudget | None) -> None:
    _current.set(budget)


def get_current_budget() -> TaskBudget | None:
    return _current.get()
