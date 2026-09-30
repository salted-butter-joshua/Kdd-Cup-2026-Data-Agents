# Benchmark Results & Changelog

This document tracks **full public demo (50 tasks)** runs under `artifacts/runs/`,
using the official column-signature metric (`λ_extra = 0.1`).

Scoring: `score = max(0, recall − λ · extra_cols / pred_cols)`; missing
`prediction.csv` → `0`. Mean is over all scored tasks in the run.

---

## Summary (50-task runs)

| Run ID | Mean | Predictions | Mean@pred | Missing | Workers | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `20260930T065125Z` | **0.8245** | 49 | 0.8413 | 1 | 1 | §13 五手段首轮；199/243 修复确认；180/249 为 SQL authoring 方差回退（非门禁误伤） |
| `20260930T034747Z` | 0.8133 | 47 | 0.8652 | 3 | 1 | 25/199/243 回退；触发 §13 架构级优化 |
| `20260929T103145Z` | 0.7933 | 45 | 0.8814 | 5 | 1 | recoveries on 173/196/199/249/250/180 |
| `20260929T092951Z` | 0.6832 | 41 | 0.8331 | 9 | 1 | Extract cache A/B/C/D; warehouse timeouts ↓ |
| `20260929T072431Z` | 0.5935 | 37 | 0.8021 | 13 | 1 | `_EXTRACT_VERSION=4` cache bust → 429 / timeouts |
| `20260929T055612Z` | 0.5955 | 38 | 0.7836 | 12 | 1 | Mid-iteration gates / prompts |
| `20260929T021745Z` | 0.6947 | 43 | 0.8078 | 7 | 1 | Pre–version-bump baseline of the day |
| `20260928T092609Z` | 0.7337 | 46 | 0.7975 | 4 | 1 | Stable mid-week run |
| `20260924T095853Z` | 0.7570 | 46 | 0.8228 | 4 | 1 | Strong early full-set reference |
| `20260923T011119Z` | 0.5867 | 45 | 0.6519 | 5 | 8 | Early parallel; lower quality |
| `20260922T083552Z` | 0.7577 | 46 | 0.8236 | 4 | 8 | Early full-set |
| `20260922T012037Z` | 0.7240 | 47 | 0.7702 | 3 | 8 | Early full-set |

Single-task smoke runs (`mean=1.0`, 1 prediction) are omitted from the table.

---

## Latest run (`20260930T065125Z`)

| Metric | Value |
| --- | --- |
| Mean score | **0.8245** |
| Tasks with prediction | 49 / 50 |
| Mean among predictions | 0.8413 |
| Missing | `task_352` |

### vs `20260930T034747Z` (Δ = +0.011)

| Direction | Tasks | Mechanism (trace-verified) |
| --- | --- | --- |
| Fixed | 199: 0→1.0 | This run wrote school-level `AvgScrMath > 400` (6 rows) from the start. Dual-grain + `type` ambiguity gates fired and added probes, then the same SQL passed. Member-expansion did **not** fire — the previous run's district-HAVING expansion path was never taken |
| Fixed | 243: 0→1.0 | Ratio arbitration policy explicitly cited in thought (knowledge > direct FK > fewest joins) → 3/8=0.375 |
| Fixed | 250: 0.95→1.0 | — |
| Still 0 | 25 | **Measure substitution**: anchor rule worked (`MIN(cost)` profiling = anchored fine), but final aggregates `spent` — knowledge authorizes `SUM(spent)`, so gate legitimately passes; gold wants row-level `MIN(cost)` |
| Regression | 180: 0.9333→0 | Model dropped `Amount > 0` zero-division guard → `Price/0 → inf > 29` → wrong customer set. SQL-authoring variance; no gate fired |
| Regression | 249: 1.0→0.45 | Model added hygiene filter `WHERE Age IS NOT NULL` → changed `AVG(UpVotes)` population (avg_age still matched). SQL-authoring variance; advisory UNDECIDED note only |
| Now submits | 80, 344: missing→0 | Submit but wrong content |
| Still missing | 352 | max_steps (unchanged) |

No gate misfire caused a regression; both regressions are run-to-run SQL authoring variance.

### Missing (no prediction)

| Task | Failure | Root cause (short) |
| --- | --- | --- |
| 352 | max_steps | Budget↔event join not converged; no final SQL |

### Wrong answers (prediction present, score 0)

| Task | Issue |
| --- | --- |
| 25 | Coarse `SUM(spent)` per event (12 rows) vs gold row-level `MIN(cost)`; knowledge-authorized measure substitution |
| 80 | Submits but wrong (time-format normalization path still off) |
| 163 | Grouped by budget category instead of event `type` (won't fix per discipline) |
| 169 | `SUM/12` vs gold `AVG/12` (won't fix per discipline) |
| 180 | Missing `Amount > 0` guard → inf unit price → wrong customer set |
| 344 | Submits but wrong (WBC/FG ranges absent from knowledge) |
| 418 | Still 0 |

---

## Code changelog (mechanism-level)

Changes are ordered newest-first. Paths are under `src/data_agent_baseline/`.

### 2026-09-30 — §13 architecture means (IR / grain contract / roles / arbitration / tri-state)

| ID | Change | Files |
| --- | --- | --- |
| ① IR | `normalize_sql` / `alias_map` / `strip_subqueries`; gates consume normalized judgment copy only | `agents/sql_ir.py` (new) |
| ② Grain contract | Pre-submit `submit_grain_contract_pre` + post-answer `submit_grain_contract_post` (membership + member-expansion) | `agents/grain_contract.py` (new), `agents/react.py` |
| ②b | Member-expansion gate: dual-grain Q + ≤10-row grouped probe sharing tables with a much larger final → reject | `agents/submit_validation.py` |
| ③ Roles | `infer_column_role` (measure/dimension/identifier); role shown in schema-link candidates; fine evidence needs question-anchor | `agents/schema_link.py`, `agents/aggregate_grain.py` |
| ③b | Ambiguity-group matching fixed for camelCase columns (`positionOrder` → {position, order}) | `agents/schema_link.py` |
| ④ Arbitration | `ratio_direction` hypothesis + fixed arbitration order (knowledge > direct FK > fewest joins) in prompt rule 27 | `agents/hypothesis.py`, `agents/prompt.py` |
| ⑤ Tri-state | Gates return PASS/REJECT/UNDECIDED; `gate_telemetry` in trace; `gate_notes` pre-submit warning on `final=true` | `agents/gate_common.py` (new), `agents/aggregate_grain.py`, `agents/react.py` |

Tests: `_test_sql_ir.py`, `_run_arch_tests.py`, extended `_test_aggregate_grain.py` / `_test_submit_validation.py` / `_test_schema_hypothesis_vote.py`.

### 2026-09-30 — Fallback submit + relation gate + cache reuse

| ID | Change | Files |
| --- | --- | --- |
| Warm / A′ | Accept extract caches from version ≥1; upgrade with deterministic Registry post-process (no LLM) | `tools/doc_extract.py` |
| B | Serial paced extraction (`EXTRACT_CONCURRENCY=1`, delay 0.5s) | `tools/doc_extract.py` |
| C | LLM client `timeout=60`, `max_retries=1` | `agents/model.py` |
| D | CLI `warm-extract-cache` for offline extract warming | `cli.py` |
| Fallback | On max_steps without answer, submit `session.last_final` through L3 gates | `agents/react.py` |
| P0 | Relation gate allows `COUNT(DISTINCT rel_id)` / CASE forms | `agents/submit_validation.py` |

Tests: `_test_fallback_submit.py`, `_test_extract_cache_upgrade.py`, `_test_submit_validation.py`.

### 2026-09-29 — Shape / rate-limit / Registry names

| ID | Change | Files |
| --- | --- | --- |
| A1 | Defaults: `max_workers=4`, `task_timeout=300`, `empty_retries=0` | `config.py` |
| A2 | Hard-stop on 429 / quota exhaustion | `agents/react.py` |
| B1 | Parse + `EXPLAIN` before SQL execute | `tools/warehouse.py`, `tools/registry.py` |
| B2 | Scalar shape lock (block detail overwrite of 1×1) | `agents/submit_validation.py` |
| B3 | Suppress entity ID columns on status/metric answers | `agents/submit_validation.py` |
| B4 | Parse/explain failures enter repair counts | `agents/react.py` |
| C1 | Registry-phrase official names over adjective aliases | `tools/doc_extract.py` |
| C2 | Ambiguous-column / shape hints in prompt | `agents/prompt.py` |

### Earlier — Submit gates & warehouse

| Theme | Change |
| --- | --- |
| L3 gates | Empty / tie / membership / relation / sanity / projection |
| Dual grain | Fine vs coarse aggregation probes before submit |
| Warehouse | Per-task DuckDB; CSV/JSON/SQLite + narrative extract tables |
| Knowledge | `knowledge.md` injected into prompt (not a table) |

---

## Architecture (layers)

```text
L0  Warehouse build   → CSV/SQLite/JSON + doc extract (versioned cache)
L1  Probe             → list_tables / run_sql (LIMIT)
L2  ReAct loop        → JSON {thought, action, action_input}, ≤ max_steps
L3  Submit gates      → deterministic reject + hints (soft→hard escalation)
L4  Repair            → repair_counts + upgraded hints
L5  Submit            → prediction.csv (+ fallback from last_final)
L6  Eval              → column-signature mean score
```

Design notes: `架构设计.md` (Chinese, mechanism-level).

---

## How to reproduce a scored run

```bash
# From PHASE_1/
uv sync
cp configs/react_baseline.example.yaml configs/react_baseline.yaml
# edit api_key / model / provider

# Optional but recommended before scoring (rate-limit friendly)
uv run dabench warm-extract-cache --config configs/react_baseline.yaml

uv run dabench run-benchmark --config configs/react_baseline.yaml
uv run dabench score artifacts/runs/<run_id> --config configs/react_baseline.yaml
```

Artifacts (local only, gitignored):

```text
artifacts/runs/<run_id>/
├── summary.json
├── scores_summary.json
└── task_*/{trace.json,prediction.csv,score.json}
```

---

## Environment knobs

| Variable | Default | Meaning |
| --- | --- | --- |
| `DATA_AGENT_EXTRACT_CONCURRENCY` | `1` | Paragraph extract parallelism |
| `DATA_AGENT_EXTRACT_DELAY` | `0.5` | Seconds between extract LLM calls |
| `DATA_AGENT_LLM_TIMEOUT` | `60` | Per-request HTTP timeout |
| `DATA_AGENT_LLM_MAX_RETRIES` | `1` | SDK retries (keep low under 429) |
