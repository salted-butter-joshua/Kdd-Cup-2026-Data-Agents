<div align="center">

# DataAgent-Bench Starter Kit

English | [中文](README.zh.md)

[![Official Website](https://img.shields.io/badge/Official%20Website-Visit%20dataagent.top-0ea5e9?style=for-the-badge&logo=googlechrome&logoColor=white&labelColor=0f172a)](https://dataagent.top)
[![Demo Dataset](https://img.shields.io/badge/Demo%20Dataset-Download%20Phase%201-f59e0b?style=for-the-badge&logo=googledrive&logoColor=white&labelColor=0f172a)](https://drive.google.com/file/d/1c6u5WlFw4KV7CBRyXh5BvFYbKqxhBSbL/view)
[![Discord](https://img.shields.io/badge/Discord-Join%20Community-5865F2?style=for-the-badge&logo=discord&logoColor=white&labelColor=0f172a)](https://discord.com/invite/7eFwJQN3Fx)

</div>

> Official starter kit for the KDD Cup 2026 DataAgent-Bench challenge. The repository reads tasks from `data/public/input/` and writes predictions for downstream evaluation.

## Overview

| Item | Value |
| --- | --- |
| Dataset input | `data/public/input/` |
| Public demo ground truth | `data/public/output/task_<id>/gold.csv` |
| Hidden test data | `input/` only, no `output/` |
| Entry command | `uv run dabench <command> --config PATH` |
| Default run output | `artifacts/runs/` |
| Latest 50-task mean | **0.7933** (run `20260929T103145Z`) |
| Score history / changelog | [RESULTS.md](RESULTS.md) |
| Architecture notes | [架构设计.md](架构设计.md) (Chinese) |

## Benchmark Results

Recent **50-task** public-demo runs (official column-signature metric, `λ=0.1`):

| Run ID | Mean | With pred. | Mean@pred | Missing |
| --- | ---: | ---: | ---: | ---: |
| `20260929T103145Z` | **0.7933** | 45 | 0.8814 | 5 |
| `20260929T092951Z` | 0.6832 | 41 | 0.8331 | 9 |
| `20260929T072431Z` | 0.5935 | 37 | 0.8021 | 13 |
| `20260924T095853Z` | 0.7570 | 46 | 0.8228 | 4 |

Full curve, per-task failure notes, and mechanism changelog: **[RESULTS.md](RESULTS.md)**.

## Quick Start

1. Install `uv` by following the official guide:
   - https://docs.astral.sh/uv/getting-started/installation/
2. On macOS and Linux, the standalone installer is:

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. Install project dependencies:

   ```bash
   uv sync
   ```

4. Confirm the dataset root is visible:

   ```bash
   uv run dabench status --config configs/react_baseline.example.yaml
   ```

5. Copy the example config and fill in credentials (local configs are gitignored):

   ```bash
   cp configs/react_baseline.example.yaml configs/react_baseline.yaml
   ```

6. (Recommended) Warm document-extract caches before scoring to avoid 429 / warehouse timeouts:

   ```bash
   uv run dabench warm-extract-cache --config configs/react_baseline.yaml
   ```

7. Run the baseline:

   ```bash
   uv run dabench run-benchmark --config configs/react_baseline.yaml
   ```

## Dataset

The public demo dataset lives under `data/public/input/`. Each task directory follows this structure:

```text
data/public/input/task_<id>/
├── task.json
└── context/
```

The corresponding public demo answers live separately under `data/public/output/task_<id>/gold.csv`.
Hidden test sets only include `input/`, so there is no `output/` directory there.

`task.json` contains:

- `task_id`
- `difficulty`
- `question`

The `context/` directory may contain one or more of:

- CSV files
- JSON files
- SQLite / DB files
- Text documents

## Configuration

An example config file lives at `configs/react_baseline.example.yaml`.

```yaml
dataset:
  root_path: data/public/input

agent:
  model: YOUR_MODEL_NAME
  api_base: YOUR_API_BASE_URL
  api_key: YOUR_API_KEY
  max_steps: 16
  temperature: 0.0

run:
  output_dir: artifacts/runs
  run_id:
  max_workers: 4
  task_timeout_seconds: 600
```

Config fields:

| Field | Meaning |
| --- | --- |
| `dataset.root_path` | Root directory of the public demo `input/` dataset. Relative paths are resolved from the project root. |
| `agent.model` | Model name. |
| `agent.api_base` | OpenAI-compatible API base URL. |
| `agent.api_key` | API key, read directly from the config file. |
| `agent.max_steps` | Maximum ReAct steps per task. |
| `agent.temperature` | Sampling temperature. |
| `run.output_dir` | Output directory for run artifacts. |
| `run.run_id` | Optional run directory name. Defaults to a UTC timestamp if omitted. Must be a single directory name; existing run directories are rejected. |
| `run.max_workers` | Parallel worker count for `run-benchmark`. |
| `run.task_timeout_seconds` | Maximum wall-clock time per task. Set to `0` or a negative value to disable the task-level timeout. |

## CLI

```bash
uv run dabench <command> --config PATH [options]
```

| Command | Purpose | Example |
| --- | --- | --- |
| `status` | Show project paths, config path, dataset root, and public task counts. | `uv run dabench status --config configs/react_baseline.example.yaml` |
| `inspect-task` | Show task metadata and list accessible files under `context/`. | `uv run dabench inspect-task task_11 --config configs/react_baseline.yaml` |
| `warm-extract-cache` | Serially warm document extract caches (rate-limit friendly; run before scoring). | `uv run dabench warm-extract-cache --config configs/react_baseline.yaml` |
| `run-task` | Run the baseline on one task and write outputs. | `uv run dabench run-task task_11 --config configs/react_baseline.yaml` |
| `run-benchmark` | Run the baseline across the public dataset. | `uv run dabench run-benchmark --config configs/react_baseline.yaml` |
| `score` | Score an existing run directory. | `uv run dabench score artifacts/runs/<run_id> --config configs/react_baseline.yaml` |

`run-benchmark` / `warm-extract-cache` also support `--limit N`.

## Tools

ReAct tools operate on a **per-task DuckDB warehouse** (not raw context files):

| Tool | Purpose | Inputs |
| --- | --- | --- |
| `list_tables` | List tables, columns, and row counts in the task warehouse. | (none) |
| `run_sql` | Read-only SQL; `final=false` probes, `final=true` stores the answer table. | `sql`, `final`, optional `grain` |
| `answer` | Submit the last `final=true` result and end the task. | `{}` |

SQL is parse/`EXPLAIN`-checked before execution; submit-time gates cover empty results, ties, grain, relation DISTINCT, shape, and ID-column suppression. See [架构设计.md](架构设计.md).

## Outputs

Each successful task run may produce:

- `trace.json`
- `prediction.csv`

Per-task outputs are written to:

```text
artifacts/runs/<run_id>/<task_id>/
├── trace.json
└── prediction.csv
```

Benchmark runs also write:

```text
artifacts/runs/<run_id>/
├── summary.json
├── scores_summary.json   # after dabench score
└── task_*/…
```

## Notable changes (summary)

| Theme | What changed |
| --- | --- |
| Extract cache | Versioned caches; ≥v1 deterministic upgrade; `warm-extract-cache` |
| Rate limits | Serial paced extract; hard LLM timeout/retries; 429 hard-stop |
| Submit gates | Empty / tie / grain / relation DISTINCT / scalar shape / ID suppress |
| Fallback submit | Auto-submit last `final=true` through gates when max_steps is hit |

Full metric history and changelog: [RESULTS.md](RESULTS.md).

## Contact

- Open issues: https://github.com/HKUSTDial/kddcup2026-data-agents-starter-kit/issues
- Official website: https://dataagent.top
- Discord: https://discord.com/invite/7eFwJQN3Fx
- WeChat official account: `数据智能与分析实验室 DIAL`

<div align="center">
  <table>
    <tr>
      <td align="center">
        <a href="https://dataagent.top">
          <img
            src="https://api.qrserver.com/v1/create-qr-code/?size=144x144&data=https://dataagent.top&bgcolor=ffffff&color=111827&margin=8"
            alt="Official website QR code"
            width="144"
          />
        </a>
        <br />
        Official Website
      </td>
      <td align="center">
        <a href="https://discord.com/invite/7eFwJQN3Fx">
          <img
            src="https://api.qrserver.com/v1/create-qr-code/?size=144x144&data=https://discord.com/invite/7eFwJQN3Fx&bgcolor=ffffff&color=111827&margin=8"
            alt="Discord QR code"
            width="144"
          />
        </a>
        <br />
        Discord
      </td>
      <td align="center">
        <img
          src="https://dataagent.top/HKUSTGZ_DIAL.jpg"
          alt="WeChat official account QR code"
          width="144"
        />
        <br />
        WeChat Official Account
      </td>
    </tr>
  </table>
</div>

## Main Modules

| Module | Responsibility |
| --- | --- |
| `src/data_agent_baseline/benchmark/dataset.py` | Public dataset loader |
| `src/data_agent_baseline/tools/filesystem.py` | `list_context`, `read_csv`, `read_json`, `read_doc` |
| `src/data_agent_baseline/tools/python_exec.py` | `execute_python` |
| `src/data_agent_baseline/tools/sqlite.py` | `inspect_sqlite_schema`, `execute_context_sql` |
| `src/data_agent_baseline/tools/registry.py` | Tool registration and terminal `answer` |
| `src/data_agent_baseline/agents/prompt.py` | System prompt, task prompt, observation prompt |
| `src/data_agent_baseline/agents/react.py` | ReAct runtime with JSON action protocol |
| `src/data_agent_baseline/run/runner.py` | Single-task and benchmark execution |
