from pathlib import Path
from time import perf_counter

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from data_agent_baseline.benchmark.dataset import DABenchPublicDataset
from data_agent_baseline.config import AppConfig, load_app_config
from data_agent_baseline.eval.scoring import score_run
from data_agent_baseline.run.runner import TaskRunArtifacts, build_model_adapter, create_run_output_dir, run_benchmark, run_single_task
from data_agent_baseline.tools.filesystem import list_context_tree

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
ARTIFACT_RUNS_DIR = ARTIFACTS_DIR / "runs"

app = typer.Typer(add_completion=False, no_args_is_help=False)
console = Console()


def _status_value(path: Path) -> str:
    return "present" if path.exists() else "missing"


def _format_compact_rate(completed_count: int, elapsed_seconds: float) -> str:
    if completed_count <= 0 or elapsed_seconds <= 0:
        return "rate=0.0 task/min"
    return f"rate={(completed_count / elapsed_seconds) * 60:.1f} task/min"


def _format_last_task(artifact: TaskRunArtifacts | None) -> str:
    if artifact is None:
        return "last=-"
    status = "ok" if artifact.succeeded else "fail"
    return f"last={artifact.task_id} ({status})"


def _build_compact_progress_fields(
    *,
    completed_count: int,
    succeeded_count: int,
    failed_count: int,
    task_total: int,
    max_workers: int,
    elapsed_seconds: float,
    last_artifact: TaskRunArtifacts | None,
) -> dict[str, str]:
    remaining_count = max(task_total - completed_count, 0)
    running_count = min(max_workers, remaining_count)
    queued_count = max(remaining_count - running_count, 0)
    return {
        "ok": str(succeeded_count),
        "fail": str(failed_count),
        "run": str(running_count),
        "queue": str(queued_count),
        "speed": _format_compact_rate(completed_count, elapsed_seconds),
        "last": _format_last_task(last_artifact),
    }


@app.callback()
def cli() -> None:
    """Utilities for working with the local DABench baseline project."""


@app.command()
def status(
    config: Path = typer.Option(..., exists=True, dir_okay=False, help="YAML config path."),
) -> None:
    """Show the local project layout and public dataset presence."""
    app_config = load_app_config(config)
    config_path = config.resolve()
    public_dataset = DABenchPublicDataset(app_config.dataset.root_path)

    table = Table(title="DABench Baseline Status")
    table.add_column("Item")
    table.add_column("Path")
    table.add_column("State")

    table.add_row("project_root", str(PROJECT_ROOT), "ready")
    table.add_row("data_dir", str(DATA_DIR), _status_value(DATA_DIR))
    table.add_row("configs_dir", str(CONFIGS_DIR), _status_value(CONFIGS_DIR))
    table.add_row("artifacts_dir", str(ARTIFACTS_DIR), _status_value(ARTIFACTS_DIR))
    table.add_row("runs_dir", str(ARTIFACT_RUNS_DIR), _status_value(ARTIFACT_RUNS_DIR))
    table.add_row("dataset_root", str(app_config.dataset.root_path), _status_value(app_config.dataset.root_path))
    table.add_row("gold_root", str(app_config.dataset.gold_path), _status_value(app_config.dataset.gold_path))
    table.add_row("config_path", str(config_path), _status_value(config_path))

    console.print(table)

    if public_dataset.exists:
        console.print(f"Public tasks: {len(public_dataset.list_task_ids())}")
        counts = public_dataset.task_counts()
        if counts:
            rendered_counts = ", ".join(
                f"{difficulty}={count}" for difficulty, count in sorted(counts.items())
            )
            console.print(f"Public task counts: {rendered_counts}")


@app.command("inspect-task")
def inspect_task(
    task_id: str,
    config: Path = typer.Option(..., exists=True, dir_okay=False, help="YAML config path."),
) -> None:
    """Show task metadata and available context files."""
    app_config = load_app_config(config)
    dataset = DABenchPublicDataset(app_config.dataset.root_path)
    task = dataset.get_task(task_id)
    console.print(f"Task: {task.task_id}")
    console.print(f"Difficulty: {task.difficulty}")
    console.print(f"Question: {task.question}")
    context_listing = list_context_tree(task)
    table = Table(title=f"Context Files for {task.task_id}")
    table.add_column("Path")
    table.add_column("Kind")
    table.add_column("Size")
    for entry in context_listing["entries"]:
        table.add_row(str(entry["path"]), str(entry["kind"]), str(entry["size"] or ""))
    console.print(table)


def _raise_run_id_bad_parameter(exc: Exception) -> None:
    """Map create_run_output_dir failures to a clear run.run_id error.

    Do not wrap unrelated ValueError subclasses (e.g. UnicodeEncodeError on
    Chinese Windows), or the real failure is hidden behind 'Invalid value for
    run.run_id'.
    """
    if isinstance(exc, FileExistsError):
        raise typer.BadParameter(str(exc), param_hint="run.run_id") from exc
    if isinstance(exc, ValueError) and not isinstance(exc, UnicodeError):
        message = str(exc)
        if any(
            token in message
            for token in ("run_id", "directory name", "must not be empty", "already exists")
        ):
            raise typer.BadParameter(message, param_hint="run.run_id") from exc
    raise exc


@app.command("run-task")
def run_task_command(
    task_id: str,
    config: Path = typer.Option(..., exists=True, dir_okay=False, help="YAML config path."),
) -> None:
    """Run the ReAct baseline on one task."""
    app_config = load_app_config(config)
    try:
        _, run_output_dir = create_run_output_dir(app_config.run.output_dir, run_id=app_config.run.run_id)
    except (ValueError, FileExistsError) as exc:
        _raise_run_id_bad_parameter(exc)
    artifacts = run_single_task(task_id=task_id, config=app_config, run_output_dir=run_output_dir)

    console.print(f"Run output: {run_output_dir}")
    console.print(f"Task output: {artifacts.task_output_dir}")
    if artifacts.prediction_csv_path is not None:
        console.print(f"Prediction CSV: {artifacts.prediction_csv_path}")
    else:
        console.print("Prediction CSV: not generated")
    if artifacts.failure_reason is not None:
        console.print(f"Failure: {artifacts.failure_reason}")
    _maybe_score_and_print(
        app_config,
        run_output_dir,
        task_ids=[task_id],
    )


@app.command("run-benchmark")
def run_benchmark_command(
    config: Path = typer.Option(..., exists=True, dir_okay=False, help="YAML config path."),
    limit: int | None = typer.Option(None, min=1, help="Maximum number of tasks to run."),
) -> None:
    """Run the ReAct baseline on multiple tasks from the config selection."""
    app_config = load_app_config(config)
    dataset = DABenchPublicDataset(app_config.dataset.root_path)
    task_total = len(dataset.iter_tasks())
    if limit is not None:
        task_total = min(task_total, limit)
    effective_workers = app_config.run.max_workers

    progress_columns = [
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("[dim]|[/dim]"),
        TextColumn("[green]ok={task.fields[ok]}[/green]"),
        TextColumn("[red]fail={task.fields[fail]}[/red]"),
        TextColumn("[cyan]run={task.fields[run]}[/cyan]"),
        TextColumn("[yellow]queue={task.fields[queue]}[/yellow]"),
        TextColumn("[dim]|[/dim]"),
        TextColumn("{task.fields[speed]}"),
        TextColumn("[dim]| elapsed[/dim]"),
        TimeElapsedColumn(),
        TextColumn("[dim]| eta[/dim]"),
        TimeRemainingColumn(),
        TextColumn("[dim]|[/dim]"),
        TextColumn("{task.fields[last]}"),
    ]
    with Progress(*progress_columns, console=console) as progress:
        progress_task_id = progress.add_task(
            "Benchmark",
            total=task_total,
            completed=0,
            **_build_compact_progress_fields(
                completed_count=0,
                succeeded_count=0,
                failed_count=0,
                task_total=task_total,
                max_workers=effective_workers,
                elapsed_seconds=0.0,
                last_artifact=None,
            ),
        )

        completion_count = 0
        succeeded_count = 0
        failed_count = 0
        start_time = perf_counter()

        def on_task_complete(artifact) -> None:
            nonlocal completion_count, succeeded_count, failed_count
            completion_count += 1
            if artifact.succeeded:
                succeeded_count += 1
            else:
                failed_count += 1
            progress.update(
                progress_task_id,
                completed=completion_count,
                description="Benchmark",
                refresh=True,
                **_build_compact_progress_fields(
                    completed_count=completion_count,
                    succeeded_count=succeeded_count,
                    failed_count=failed_count,
                    task_total=task_total,
                    max_workers=effective_workers,
                    elapsed_seconds=perf_counter() - start_time,
                    last_artifact=artifact,
                ),
            )

        def on_status(message: str) -> None:
            # empty_retries runs after the bar already hit 100%; surface it so
            # the process does not look hung.
            progress.update(
                progress_task_id,
                description=f"Benchmark ({message})",
                refresh=True,
            )
            console.print(f"[yellow]{message}[/yellow]")

        try:
            run_output_dir, artifacts = run_benchmark(
                config=app_config,
                limit=limit,
                progress_callback=on_task_complete,
                status_callback=on_status,
            )
        except (ValueError, FileExistsError) as exc:
            _raise_run_id_bad_parameter(exc)
        progress.update(
            progress_task_id,
            completed=task_total,
            description="Benchmark",
            refresh=True,
            **_build_compact_progress_fields(
                completed_count=task_total,
                succeeded_count=succeeded_count,
                failed_count=failed_count,
                task_total=task_total,
                max_workers=effective_workers,
                elapsed_seconds=perf_counter() - start_time,
                last_artifact=artifacts[-1] if artifacts else None,
            ),
        )
    console.print(f"Run output: {run_output_dir}")
    console.print(f"Tasks attempted: {len(artifacts)}")
    console.print(f"Succeeded tasks: {sum(1 for item in artifacts if item.succeeded)}")
    _maybe_score_and_print(
        app_config,
        run_output_dir,
        task_ids=[item.task_id for item in artifacts],
    )


@app.command("warm-extract-cache")
def warm_extract_cache_command(
    config: Path = typer.Option(..., exists=True, dir_okay=False, help="YAML config path."),
    limit: int | None = typer.Option(None, min=1, help="Maximum number of tasks to warm."),
) -> None:
    """Pre-fill document extraction caches, serial and paced (rate-limit friendly).

    Run this BEFORE run-benchmark. Benchmark tasks then hit warm caches and
    warehouse build becomes local IO only, so 429 throttling during the scored
    run can no longer starve warehouse construction.
    """
    from data_agent_baseline.tools.doc_extract import (
        collect_doc_paths,
        extract_all_documents,
        extract_cache_dir,
    )

    app_config = load_app_config(config)
    dataset = DABenchPublicDataset(app_config.dataset.root_path)
    tasks = dataset.iter_tasks()
    if limit is not None:
        tasks = tasks[:limit]
    model = build_model_adapter(app_config)

    def cache_mtimes(cache_dir: Path) -> dict[str, float]:
        if not cache_dir.is_dir():
            return {}
        return {path.name: path.stat().st_mtime for path in cache_dir.glob("*.json")}

    total_written = 0
    total_reused = 0
    started_at = perf_counter()
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        bar_id = progress.add_task("Warm extract cache", total=len(tasks))
        for task in tasks:
            context_dir = task.assets.context_dir
            doc_paths = collect_doc_paths(context_dir)
            if not doc_paths:
                progress.advance(bar_id)
                continue
            before = cache_mtimes(extract_cache_dir(context_dir))
            task_start = perf_counter()
            docs = extract_all_documents(context_dir, model)
            after = cache_mtimes(extract_cache_dir(context_dir))
            written = sum(1 for name, mtime in after.items() if before.get(name) != mtime)
            reused = len(after) - written
            total_written += written
            total_reused += reused
            row_total = sum(len(doc.rows) for doc in docs)
            progress.advance(bar_id)
            console.print(
                f"{task.task_id}: docs={len(doc_paths)} tables={len(docs)} rows={row_total} "
                f"written={written} reused={reused} elapsed={perf_counter() - task_start:.1f}s"
            )
    console.print(
        f"Done. cache files written={total_written} reused={total_reused} "
        f"elapsed={perf_counter() - started_at:.1f}s"
    )


@app.command("score")
def score_command(
    run_dir: Path = typer.Argument(..., exists=True, file_okay=False, help="Run directory containing task_*/prediction.csv."),
    config: Path = typer.Option(..., exists=True, dir_okay=False, help="YAML config path."),
) -> None:
    """Score an existing run against gold.csv using the official column-signature metric."""
    app_config = load_app_config(config)
    _maybe_score_and_print(app_config, run_dir.resolve(), task_ids=None, force=True)


def _maybe_score_and_print(
    app_config: AppConfig,
    run_dir: Path,
    *,
    task_ids: list[str] | None,
    force: bool = False,
) -> None:
    if not force and not app_config.eval.enabled:
        console.print("Scoring skipped (eval.enabled is false).")
        return
    gold_root = app_config.dataset.gold_path
    if not gold_root.exists():
        console.print(f"Scoring skipped: gold directory missing: {gold_root}")
        return
    summary = score_run(
        gold_root,
        run_dir,
        task_ids=task_ids,
        lambda_extra=app_config.eval.lambda_extra,
    )
    table = Table(title=f"Official column-signature scores (λ={app_config.eval.lambda_extra:.2f})")
    table.add_column("task_id")
    table.add_column("score", justify="right")
    table.add_column("recall", justify="right")
    table.add_column("matched", justify="right")
    table.add_column("gold", justify="right")
    table.add_column("pred", justify="right")
    table.add_column("extra", justify="right")
    table.add_column("error")
    for row in summary["tasks"]:
        table.add_row(
            str(row["task_id"]),
            f"{row['score']:.4f}",
            f"{row['recall']:.4f}",
            str(row["matched_cols"]),
            str(row["gold_cols"]),
            str(row["pred_cols"]),
            str(row["extra_cols"]),
            str(row.get("error") or ""),
        )
    console.print(table)
    console.print(
        f"mean score: {summary['mean_score']:.4f}  "
        f"(tasks={summary['tasks_scored']}, "
        f"with prediction={summary['tasks_with_prediction']})"
    )
    if summary.get("scores_csv"):
        console.print(f"Scores CSV: {summary['scores_csv']}")
        console.print(f"Scores summary: {summary['scores_summary_json']}")


def main() -> None:
    app()
