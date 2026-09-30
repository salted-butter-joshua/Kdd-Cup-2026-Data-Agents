"""Official DataAgent-Bench column-signature scoring (KDD Cup 2026 spec 6.2–6.5).

Rules:
  * Build a column signature by normalizing values then sorting them per column.
  * Ignore column names and row order; match purely on column content.
  * Duplicate signatures match as a multiset.
  * Score = max(0, Recall - lambda * (Extra / Predicted))
      Recall  = Matched / GoldCols
      Extra   = PredictedCols - Matched
  * Missing prediction.csv -> score 0.

Normalization (spec 6.5):
  * Nulls ("", "null", "none", "nan", "nat", "<na>", case-insensitive) -> ""
  * Numeric -> Decimal rounded to 2 decimals, ROUND_HALF_UP
  * Date -> ISO YYYY-MM-DD
  * DateTime -> UTC ISO with Z if tz present, else raw ISO
  * String -> strip whitespace, case-sensitive
"""

from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from typing import Any

LAMBDA_EXTRA = 0.1
NULL_TOKENS = {"", "null", "none", "nan", "nat", "<na>"}
_DATE_RE = re.compile(r"^\d{4}-\d{1,2}-\d{1,2}$")
_DATETIME_HINT_RE = re.compile(r"^\d{4}-\d{1,2}-\d{1,2}[ T]")
_NUMERIC_RE = re.compile(r"^[\-+]?[\d,]*\.?\d+([eE][\-+]?\d+)?$")


def _is_null(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float):
        try:
            if math.isnan(value):
                return True
        except Exception:
            pass
    return str(value).strip().lower() in NULL_TOKENS


def _normalize_numeric(s: str) -> str | None:
    try:
        d = Decimal(s.replace(",", ""))
    except (InvalidOperation, ValueError):
        return None
    if not d.is_finite():
        return None
    try:
        q = d.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    except InvalidOperation:
        return None
    if q == 0:
        q = q.copy_abs()
    return format(q, "f")


def _normalize_date(s: str) -> str | None:
    if not _DATE_RE.match(s):
        return None
    try:
        dt = datetime.strptime(s, "%Y-%m-%d")
    except ValueError:
        parts = s.split("-")
        try:
            dt = datetime(int(parts[0]), int(parts[1]), int(parts[2]))
        except Exception:
            return None
    return dt.strftime("%Y-%m-%d")


def _normalize_datetime(s: str) -> str | None:
    if not _DATETIME_HINT_RE.match(s):
        return None
    candidate = s.replace(" ", "T")
    iso = candidate[:-1] + "+00:00" if candidate.endswith("Z") else candidate
    try:
        dt = datetime.fromisoformat(iso)
    except ValueError:
        return None
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc)
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return dt.strftime("%Y-%m-%dT%H:%M:%S")


def normalize_value(value: object) -> str:
    if _is_null(value):
        return ""
    s = str(value).replace("\r", "").replace("\n", "").strip()
    if s == "":
        return ""
    num = _normalize_numeric(s)
    if num is not None and _NUMERIC_RE.match(s):
        return num
    date_value = _normalize_date(s)
    if date_value is not None:
        return date_value
    datetime_value = _normalize_datetime(s)
    if datetime_value is not None:
        return datetime_value
    return s


def column_signature(values: list[object]) -> tuple[str, ...]:
    return tuple(sorted(normalize_value(v) for v in values))


def _read_csv_columns(path: Path) -> list[list[str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        rows = list(reader)
    if not rows:
        return []
    header_len = len(rows[0])
    columns: list[list[str]] = [[] for _ in range(header_len)]
    for row in rows[1:]:
        padded = list(row) + [""] * (header_len - len(row))
        for index in range(header_len):
            columns[index].append(padded[index])
    return columns


def compare_csv(
    gold_csv: Path,
    pred_csv: Path,
    lambda_extra: float = LAMBDA_EXTRA,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "gold_csv": str(gold_csv),
        "pred_csv": str(pred_csv),
        "score": 0.0,
        "recall": 0.0,
        "matched_cols": 0,
        "gold_cols": 0,
        "pred_cols": 0,
        "extra_cols": 0,
        "error": None,
    }
    if not gold_csv.exists():
        out["error"] = "gold missing"
        return out
    if not pred_csv.exists():
        out["error"] = "prediction missing"
        return out

    try:
        gold_columns = _read_csv_columns(gold_csv)
        pred_columns = _read_csv_columns(pred_csv)
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"read error: {exc}"
        return out

    gold_sigs = Counter(column_signature(col) for col in gold_columns)
    pred_sigs = Counter(column_signature(col) for col in pred_columns)
    matched = sum((gold_sigs & pred_sigs).values())
    n_gold = sum(gold_sigs.values())
    n_pred = sum(pred_sigs.values())
    extra = max(0, n_pred - matched)
    recall = matched / n_gold if n_gold else 0.0
    penalty = lambda_extra * (extra / n_pred) if n_pred else 0.0
    score = max(0.0, recall - penalty)
    out.update(
        {
            "score": round(score, 4),
            "recall": round(recall, 4),
            "matched_cols": matched,
            "gold_cols": n_gold,
            "pred_cols": n_pred,
            "extra_cols": extra,
        }
    )
    return out


def _task_prediction_csv(run_dir: Path, task_id: str) -> Path:
    flat = run_dir / task_id / "prediction.csv"
    nested = run_dir / task_id / "workdir" / "prediction.csv"
    if nested.exists():
        return nested
    return flat


def score_task(
    task_id: str,
    gold_root: Path,
    run_dir: Path,
    lambda_extra: float = LAMBDA_EXTRA,
    *,
    write_score: bool = True,
) -> dict[str, Any]:
    gold_csv = gold_root / task_id / "gold.csv"
    pred_csv = _task_prediction_csv(run_dir, task_id)
    result = compare_csv(gold_csv, pred_csv, lambda_extra=lambda_extra)
    result["task_id"] = task_id
    if write_score:
        out_dir = run_dir / task_id
        if out_dir.exists():
            (out_dir / "score.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
    return result


def _task_sort_key(task_id: str) -> tuple[int, str]:
    prefix, _, suffix = task_id.partition("_")
    if prefix == "task" and suffix.isdigit():
        return (int(suffix), task_id)
    return (10**9, task_id)


def score_run(
    gold_root: Path,
    run_dir: Path,
    *,
    task_ids: list[str] | None = None,
    lambda_extra: float = LAMBDA_EXTRA,
    write_files: bool = True,
) -> dict[str, Any]:
    if task_ids is None:
        task_ids = [
            path.name
            for path in run_dir.iterdir()
            if path.is_dir() and path.name.startswith("task_")
        ]
    task_ids = sorted(task_ids, key=_task_sort_key)

    rows = [
        score_task(task_id, gold_root, run_dir, lambda_extra=lambda_extra, write_score=write_files)
        for task_id in task_ids
    ]
    has_pred = [row for row in rows if row.get("error") != "prediction missing"]
    mean_score = sum(row["score"] for row in rows) / len(rows) if rows else 0.0
    mean_recall = sum(row["recall"] for row in rows) / len(rows) if rows else 0.0
    mean_score_with_pred = (
        sum(row["score"] for row in has_pred) / len(has_pred) if has_pred else 0.0
    )
    summary: dict[str, Any] = {
        "run_dir": str(run_dir),
        "gold_root": str(gold_root),
        "lambda_extra": lambda_extra,
        "tasks_scored": len(rows),
        "mean_score": round(mean_score, 4),
        "mean_recall": round(mean_recall, 4),
        "tasks_with_prediction": len(has_pred),
        "mean_score_with_prediction": round(mean_score_with_pred, 4),
        "missing_predictions": [row["task_id"] for row in rows if row.get("error") == "prediction missing"],
        "missing_gold": [row["task_id"] for row in rows if row.get("error") == "gold missing"],
        "scoring_formula": {
            "per_task_recall": "matched_cols / gold_cols",
            "per_task_penalty": "lambda_extra * (extra_cols / pred_cols)",
            "per_task_score": "max(0, recall - lambda_extra * (extra_cols / pred_cols))",
            "final_score": "mean(per_task_score over tasks in this run)",
            "notes": [
                "matched_cols = column-signature multiset intersection",
                "extra_cols = max(0, pred_cols - matched_cols)",
                "column names and row order are ignored",
                "missing prediction.csv -> per_task_score = 0",
            ],
        },
        "tasks": rows,
    }
    if write_files and run_dir.exists():
        (run_dir / "scores_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        scores_csv = run_dir / "scores.csv"
        with scores_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "task_id",
                    "score",
                    "recall",
                    "matched_cols",
                    "gold_cols",
                    "pred_cols",
                    "extra_cols",
                    "error",
                ],
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        "task_id": row["task_id"],
                        "score": row["score"],
                        "recall": row["recall"],
                        "matched_cols": row["matched_cols"],
                        "gold_cols": row["gold_cols"],
                        "pred_cols": row["pred_cols"],
                        "extra_cols": row["extra_cols"],
                        "error": row.get("error") or "",
                    }
                )
        summary["scores_csv"] = str(scores_csv)
        summary["scores_summary_json"] = str(run_dir / "scores_summary.json")
    return summary
