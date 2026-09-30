"""Compare two benchmark runs task-by-task and report score deltas + triggered checks."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from data_agent_baseline.eval.scoring import score_run

_CHECK_KEY_RE = re.compile(r"^([a-z_]+)_check$")


def _task_sort_key(task_id: str) -> tuple[int, int]:
    match = re.match(r"task_(\d+)", task_id)
    if match:
        return (0, int(match.group(1)))
    return (1, 0)


def _collect_triggered_checks(trace_path: Path) -> list[str]:
    """Scan a trace.json for submit-time check names that fired."""
    if not trace_path.is_file():
        return []
    try:
        payload = json.loads(trace_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    steps = payload.get("steps") or []
    checks: set[str] = set()
    for step in steps:
        observation = step.get("observation") if isinstance(step, dict) else None
        if not isinstance(observation, dict):
            continue
        content = observation.get("content")
        if not isinstance(content, dict):
            continue
        for key in content:
            m = _CHECK_KEY_RE.match(key)
            if m:
                checks.add(m.group(1))
    return sorted(checks)


def diff_runs(
    gold_root: Path,
    baseline_dir: Path,
    current_dir: Path,
    *,
    output: Path | None = None,
) -> dict[str, Any]:
    """Return a structured diff between two run directories.

    The result contains per-task score changes and the checks triggered in the
    current run. If ``output`` is provided, a Markdown report is written there.
    """
    baseline_summary = score_run(gold_root, baseline_dir, write_files=False)
    current_summary = score_run(gold_root, current_dir, write_files=False)

    baseline_by_task = {row["task_id"]: row for row in baseline_summary["tasks"]}
    current_by_task = {row["task_id"]: row for row in current_summary["tasks"]}
    task_ids = sorted(set(baseline_by_task) | set(current_by_task), key=_task_sort_key)

    rows: list[dict[str, Any]] = []
    changed: list[dict[str, Any]] = []
    for task_id in task_ids:
        baseline_row = baseline_by_task.get(task_id)
        current_row = current_by_task.get(task_id)
        baseline_score = baseline_row["score"] if baseline_row else 0.0
        current_score = current_row["score"] if current_row else 0.0
        delta = round(current_score - baseline_score, 4)
        checks = _collect_triggered_checks(current_dir / task_id / "trace.json")
        row = {
            "task_id": task_id,
            "baseline_score": baseline_score,
            "current_score": current_score,
            "delta": delta,
            "checks": checks,
        }
        rows.append(row)
        if delta != 0 or checks:
            changed.append(row)

    mean_baseline = baseline_summary["mean_score"]
    mean_current = current_summary["mean_score"]
    overall_delta = round(mean_current - mean_baseline, 4)

    result = {
        "baseline_run": str(baseline_dir),
        "current_run": str(current_dir),
        "baseline_mean": mean_baseline,
        "current_mean": mean_current,
        "delta": overall_delta,
        "tasks": rows,
    }

    if output is not None:
        lines: list[str] = [
            "# Run Diff Report",
            "",
            f"- Baseline: `{baseline_dir}` (mean={mean_baseline})",
            f"- Current: `{current_dir}` (mean={mean_current})",
            f"- Overall delta: **{overall_delta:+.4f}**",
            "",
            "## Per-task changes",
            "",
            "| task_id | baseline | current | delta | checks_triggered |",
            "|---|---|---|---|---|",
        ]
        for row in rows:
            checks = ", ".join(row["checks"]) if row["checks"] else "-"
            lines.append(
                f"| {row['task_id']} | {row['baseline_score']} | {row['current_score']} | "
                f"{row['delta']:+.4f} | {checks} |"
            )
        lines.extend([
            "",
            "## Notable changes",
            "",
        ])
        improved = [r for r in changed if r["delta"] > 0]
        regressed = [r for r in changed if r["delta"] < 0]
        if improved:
            lines.append("### Improved")
            for r in improved:
                lines.append(f"- {r['task_id']}: {r['baseline_score']} → {r['current_score']} ({r['delta']:+.4f})")
            lines.append("")
        if regressed:
            lines.append("### Regressed")
            for r in regressed:
                lines.append(f"- {r['task_id']}: {r['baseline_score']} → {r['current_score']} ({r['delta']:+.4f})")
            lines.append("")
        if not changed:
            lines.append("No score changes between runs.")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("\n".join(lines) + "\n", encoding="utf-8")
        result["report_path"] = str(output)

    return result


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Compare two benchmark runs.")
    parser.add_argument("gold_root", type=Path, help="Path to gold answer root directory.")
    parser.add_argument("baseline_run", type=Path, help="Baseline run output directory.")
    parser.add_argument("current_run", type=Path, help="Current run output directory.")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Write Markdown report to this path.",
    )
    args = parser.parse_args(argv)

    result = diff_runs(
        gold_root=args.gold_root,
        baseline_dir=args.baseline_run,
        current_dir=args.current_run,
        output=args.output,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
