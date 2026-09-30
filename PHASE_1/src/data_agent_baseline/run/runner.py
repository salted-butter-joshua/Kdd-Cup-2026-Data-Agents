from __future__ import annotations

import csv
import gc
import json
import multiprocessing
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

from collections import Counter

from data_agent_baseline.agents.model import OpenAIModelAdapter
from data_agent_baseline.agents.react import ReActAgent, ReActAgentConfig
from data_agent_baseline.benchmark.dataset import DABenchPublicDataset
from data_agent_baseline.eval.scoring import column_signature
from data_agent_baseline.config import AppConfig
from data_agent_baseline.tools.registry import ToolRegistry, create_default_tool_registry


@dataclass(frozen=True, slots=True)
class TaskRunArtifacts:
    task_id: str
    task_output_dir: Path
    prediction_csv_path: Path | None
    trace_path: Path
    succeeded: bool
    failure_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_output_dir": str(self.task_output_dir),
            "prediction_csv_path": str(self.prediction_csv_path) if self.prediction_csv_path else None,
            "trace_path": str(self.trace_path),
            "succeeded": self.succeeded,
            "failure_reason": self.failure_reason,
        }


def create_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def resolve_run_id(run_id: str | None = None) -> str:
    if run_id is None:
        return create_run_id()

    normalized = run_id.strip()
    if not normalized:
        raise ValueError("run_id must not be empty.")
    if normalized in {".", ".."} or "/" in normalized or "\\" in normalized:
        raise ValueError("run_id must be a single directory name, not a path.")
    return normalized


def create_run_output_dir(output_root: Path, *, run_id: str | None = None) -> tuple[str, Path]:
    effective_run_id = resolve_run_id(run_id)
    run_output_dir = output_root / effective_run_id
    run_output_dir.mkdir(parents=True, exist_ok=False)
    return effective_run_id, run_output_dir


def build_model_adapter(config: AppConfig):
    return OpenAIModelAdapter(
        provider=config.agent.provider,
        model=config.agent.model,
        api_base=config.agent.api_base,
        api_key=config.agent.api_key,
        temperature=config.agent.temperature,
        strip_think=config.agent.strip_think,
        reasoning_split=config.agent.reasoning_split,
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    # Explicit utf-8: Windows Chinese locales default to gbk for text IO.
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, columns: list[str], rows: list[list[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Must use utf-8: answers often contain non-GBK characters; default locale
    # encoding on Chinese Windows raises UnicodeEncodeError mid-benchmark.
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for row in rows:
            writer.writerow(row)


def _has_answer(run_result: dict[str, Any]) -> bool:
    answer = run_result.get("answer")
    return isinstance(answer, dict) and isinstance(answer.get("columns"), list)


def _answer_key(answer: dict[str, Any]) -> tuple[tuple[str, ...], ...]:
    rows = answer.get("rows") or []
    width = len(answer.get("columns") or [])
    columns: list[list[object]] = [[] for _ in range(width)]
    for row in rows:
        padded = list(row) + [None] * (width - len(row))
        for index in range(width):
            columns[index].append(padded[index])
    return tuple(sorted(column_signature(column) for column in columns))


def select_majority_answer(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Pick the answer table that appears most often. Do not union rows.

    Ties keep the earliest submission. Results with no answer are ignored
    unless every result is empty, in which case the last one is returned.
    """
    keyed = [( _answer_key(result["answer"]), result) for result in results if _has_answer(result)]
    if not keyed:
        return results[-1]
    counts = Counter(key for key, _result in keyed)
    best = counts.most_common(1)[0][1]
    for key, result in keyed:
        if counts[key] == best:
            return result
    return keyed[0][1]


def _format_exception(exc: BaseException) -> str:
    """Prefer a non-empty error string. ``MemoryError()`` has ``str(exc) == ''``."""
    message = str(exc).strip()
    name = type(exc).__name__
    if message:
        return f"{name}: {message}"
    return name


def _failure_run_result_payload(
    task_id: str,
    failure_reason: str,
    *,
    steps: list[Any] | None = None,
    traceback_text: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "task_id": task_id,
        "answer": None,
        "steps": list(steps or []),
        "failure_reason": failure_reason,
        "succeeded": False,
    }
    if traceback_text:
        payload["traceback"] = traceback_text
    return payload


def _run_single_task_core(
    *,
    task_id: str,
    config: AppConfig,
    model=None,
    tools: ToolRegistry | None = None,
) -> dict[str, Any]:
    public_dataset = DABenchPublicDataset(config.dataset.root_path)
    task = public_dataset.get_task(task_id)

    resolved_model = model or build_model_adapter(config)
    owns_tools = tools is None
    resolved_tools = tools or create_default_tool_registry(model=resolved_model)
    agent = ReActAgent(
        model=resolved_model,
        tools=resolved_tools,
        config=ReActAgentConfig(max_steps=config.agent.max_steps),
    )
    try:
        run_result = agent.run(task)
        return run_result.to_dict()
    finally:
        # Always drop on-disk warehouse (+ WAL/tmp) after the task or retry ends.
        session = getattr(resolved_tools, "session", None)
        if session is not None:
            try:
                session.close()
            except Exception:
                pass
        if owns_tools:
            del resolved_tools
        gc.collect()


def _run_single_task_in_subprocess(task_id: str, config: AppConfig, queue: multiprocessing.Queue[Any]) -> None:
    try:
        queue.put(
            {
                "ok": True,
                "run_result": _run_single_task_core(task_id=task_id, config=config),
            }
        )
    except BaseException as exc:  # noqa: BLE001
        import traceback
        queue.put(
            {
                "ok": False,
                "error": _format_exception(exc),
                "traceback": traceback.format_exc(),
            }
        )


def _drain_queue(queue: multiprocessing.Queue[Any], *, timeout: float = 5.0) -> Any | None:
    """Fetch one item from a multiprocessing queue with a hard timeout.

    ``Queue.empty()`` is unreliable across processes; never gate on it.
    """
    try:
        return queue.get(timeout=timeout)
    except Exception:
        return None


def _cleanup_process_queue(
    process: multiprocessing.Process,
    queue: multiprocessing.Queue[Any],
) -> None:
    """Ensure the worker process and queue feeder threads cannot keep the parent alive."""
    if process.is_alive():
        process.terminate()
        process.join(timeout=1.0)
        if process.is_alive():
            process.kill()
            process.join(timeout=1.0)
    try:
        queue.close()
    except Exception:
        pass
    try:
        queue.join_thread()
    except Exception:
        pass


def _run_single_task_with_timeout(*, task_id: str, config: AppConfig) -> dict[str, Any]:
    timeout_seconds = config.run.task_timeout_seconds
    if timeout_seconds <= 0:
        return _run_single_task_core(task_id=task_id, config=config)

    queue: multiprocessing.Queue[Any] = multiprocessing.Queue()
    process = multiprocessing.Process(
        target=_run_single_task_in_subprocess,
        args=(task_id, config, queue),
    )
    process.start()
    process.join(timeout_seconds)

    if process.is_alive():
        _cleanup_process_queue(process, queue)
        return _failure_run_result_payload(task_id, f"Task timed out after {timeout_seconds} seconds.")

    # Child exited. Pull the result with a short timeout so a stuck queue feeder
    # cannot hang the parent forever after the task is already done.
    result = _drain_queue(queue, timeout=5.0)
    _cleanup_process_queue(process, queue)

    if result is None:
        exit_code = process.exitcode
        if exit_code not in (None, 0):
            return _failure_run_result_payload(
                task_id,
                f"Task exited unexpectedly with exit code {exit_code}.",
            )
        return _failure_run_result_payload(task_id, "Task exited without returning a result.")

    if result.get("ok"):
        return dict(result["run_result"])
    error_msg = result.get("error") or "<empty error>"
    traceback_txt = result.get("traceback") or ""
    return _failure_run_result_payload(
        task_id,
        f"Task failed with uncaught error: {error_msg}",
        traceback_text=traceback_txt or None,
    )


def _write_task_outputs(task_id: str, run_output_dir: Path, run_result: dict[str, Any]) -> TaskRunArtifacts:
    task_output_dir = run_output_dir / task_id
    task_output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = task_output_dir / "trace.json"
    _write_json(trace_path, run_result)

    prediction_csv_path: Path | None = None
    answer = run_result.get("answer")
    if isinstance(answer, dict):
        prediction_csv_path = task_output_dir / "prediction.csv"
        _write_csv(
            prediction_csv_path,
            list(answer.get("columns", [])),
            [list(row) for row in answer.get("rows", [])],
        )

    return TaskRunArtifacts(
        task_id=task_id,
        task_output_dir=task_output_dir,
        prediction_csv_path=prediction_csv_path,
        trace_path=trace_path,
        succeeded=bool(run_result.get("succeeded")),
        failure_reason=run_result.get("failure_reason"),
    )


def _execute_task_once(
    *,
    task_id: str,
    config: AppConfig,
    model=None,
    tools: ToolRegistry | None = None,
) -> dict[str, Any]:
    if model is None and tools is None:
        return _run_single_task_with_timeout(task_id=task_id, config=config)
    return _run_single_task_core(task_id=task_id, config=config, model=model, tools=tools)


def run_single_task(
    *,
    task_id: str,
    config: AppConfig,
    run_output_dir: Path,
    model=None,
    tools: ToolRegistry | None = None,
    allow_empty_retry: bool = True,
) -> TaskRunArtifacts:
    import traceback

    started_at = perf_counter()
    attempts: list[dict[str, Any]] = []
    try:
        attempts.append(_execute_task_once(task_id=task_id, config=config, model=model, tools=tools))
    except BaseException as exc:  # noqa: BLE001 — keep the rest of the benchmark running
        attempts.append(
            _failure_run_result_payload(
                task_id,
                f"Task failed with uncaught error: {_format_exception(exc)}",
                traceback_text=traceback.format_exc(),
            )
        )

    retries_left = config.run.empty_retries if allow_empty_retry else 0
    while retries_left > 0 and not _has_answer(attempts[-1]):
        retries_left -= 1
        try:
            retry = _execute_task_once(task_id=task_id, config=config, model=model, tools=tools)
        except BaseException as exc:  # noqa: BLE001
            retry = _failure_run_result_payload(
                task_id,
                f"Task failed with uncaught error: {_format_exception(exc)}",
                traceback_text=traceback.format_exc(),
            )
        retry["empty_retry"] = True
        attempts.append(retry)

    run_result = select_majority_answer(attempts)
    if len(attempts) > 1:
        run_result = dict(run_result)
        run_result["attempt_count"] = len(attempts)
        run_result["empty_retry"] = any(item.get("empty_retry") for item in attempts)
    run_result["e2e_elapsed_seconds"] = round(perf_counter() - started_at, 3)
    return _write_task_outputs(task_id, run_output_dir, run_result)


def run_benchmark(
    *,
    config: AppConfig,
    model=None,
    tools: ToolRegistry | None = None,
    limit: int | None = None,
    progress_callback: Callable[[TaskRunArtifacts], None] | None = None,
    status_callback: Callable[[str], None] | None = None,
) -> tuple[Path, list[TaskRunArtifacts]]:
    effective_run_id, run_output_dir = create_run_output_dir(config.run.output_dir, run_id=config.run.run_id)

    dataset = DABenchPublicDataset(config.dataset.root_path)
    tasks = dataset.iter_tasks()
    if limit is not None:
        tasks = tasks[:limit]

    effective_workers = config.run.max_workers
    if effective_workers < 1:
        raise ValueError("max_workers must be at least 1.")
    if model is not None or tools is not None:
        effective_workers = 1

    task_ids = [task.task_id for task in tasks]

    # Always go through the timeout-wrapped path for benchmark runs.
    # Passing a shared in-process model/tools previously skipped the subprocess
    # timeout, so a hung API call or warehouse build could keep the parent alive
    # forever while the progress bar looked "almost done".
    task_artifacts: list[TaskRunArtifacts]
    if effective_workers == 1 and (model is not None or tools is not None):
        # Explicit shared objects: caller opted into in-process execution (tests).
        task_artifacts = []
        for task_id in task_ids:
            artifact = run_single_task(
                task_id=task_id,
                config=config,
                run_output_dir=run_output_dir,
                model=model,
                tools=tools,
                allow_empty_retry=False,
            )
            task_artifacts.append(artifact)
            if progress_callback is not None:
                progress_callback(artifact)
    elif effective_workers == 1:
        task_artifacts = []
        for task_id in task_ids:
            artifact = run_single_task(
                task_id=task_id,
                config=config,
                run_output_dir=run_output_dir,
                allow_empty_retry=False,
            )
            task_artifacts.append(artifact)
            if progress_callback is not None:
                progress_callback(artifact)
    else:
        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            future_to_index = {
                executor.submit(
                    run_single_task,
                    task_id=task_id,
                    config=config,
                    run_output_dir=run_output_dir,
                    allow_empty_retry=False,
                ): index
                for index, task_id in enumerate(task_ids)
            }
            indexed_artifacts: list[TaskRunArtifacts | None] = [None] * len(task_ids)
            for future in as_completed(future_to_index):
                artifact = future.result()
                indexed_artifacts[future_to_index[future]] = artifact
                if progress_callback is not None:
                    progress_callback(artifact)
            task_artifacts = [artifact for artifact in indexed_artifacts if artifact is not None]

    empty_retry_ids: list[str] = []
    if config.run.empty_retries > 0:
        # Re-runs happen AFTER the main progress bar already reached 100%.
        # This is a common source of "all tasks done but process still running".
        empty_retry_ids = [
            artifact.task_id
            for artifact in task_artifacts
            if artifact.prediction_csv_path is None
        ]
        if empty_retry_ids:
            if status_callback is not None:
                status_callback(
                    f"Retrying {len(empty_retry_ids)} empty/failed task(s): "
                    + ", ".join(empty_retry_ids)
                )
            retried: list[TaskRunArtifacts] = []
            for artifact in task_artifacts:
                if artifact.prediction_csv_path is not None:
                    retried.append(artifact)
                    continue
                if status_callback is not None:
                    status_callback(f"Empty-retry: {artifact.task_id}")
                retried.append(
                    run_single_task(
                        task_id=artifact.task_id,
                        config=config,
                        run_output_dir=run_output_dir,
                        allow_empty_retry=False,
                    )
                )
            task_artifacts = retried

    summary_path = run_output_dir / "summary.json"
    _write_json(
        summary_path,
        {
            "run_id": effective_run_id,
            "task_count": len(task_artifacts),
            "succeeded_task_count": sum(1 for artifact in task_artifacts if artifact.succeeded),
            "max_workers": effective_workers,
            "empty_retry_task_ids": empty_retry_ids,
            "tasks": [artifact.to_dict() for artifact in task_artifacts],
        },
    )
    return run_output_dir, task_artifacts
