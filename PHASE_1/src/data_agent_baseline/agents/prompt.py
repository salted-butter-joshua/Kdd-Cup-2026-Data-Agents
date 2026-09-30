from __future__ import annotations

import json
import re
from datetime import date, datetime

from data_agent_baseline.benchmark.schema import PublicTask

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]{2,}")


# Whole file goes into the task prompt. Longer files keep whole sections, headings included.
KNOWLEDGE_FULL_MAX_CHARS = 16000

_HEADING_RE = re.compile(r"^#{1,3}\s", flags=re.MULTILINE)


def load_knowledge_md(task: PublicTask) -> str | None:
    """Load full ``context/knowledge.md``."""
    path = task.context_dir / "knowledge.md"
    if not path.is_file():
        return None
    return path.read_text(encoding="utf-8", errors="replace")


def _tokens(text: str) -> set[str]:
    return {m.group(0).casefold() for m in _TOKEN_RE.finditer(text or "")}


def _split_sections(text: str) -> list[str]:
    """Split on markdown headings. Each section keeps its heading and body."""
    matches = list(_HEADING_RE.finditer(text))
    if not matches:
        stripped = text.strip()
        return [stripped] if stripped else []
    sections: list[str] = []
    if matches[0].start() > 0:
        preamble = text[: matches[0].start()].strip()
        if preamble:
            sections.append(preamble)
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        chunk = text[match.start() : end].strip()
        if chunk:
            sections.append(chunk)
    return sections


def retrieve_knowledge_snippets(
    knowledge: str,
    question: str,
    *,
    max_chars: int = KNOWLEDGE_FULL_MAX_CHARS,
    max_chunks: int = 12,
) -> str:
    """Copy whole knowledge sections by token overlap. No paraphrase, no mid-section cut."""
    text = (knowledge or "").strip()
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    q_tok = _tokens(question)
    sections = _split_sections(text)
    if not sections:
        return text[:max_chars]

    scored: list[tuple[float, int, str]] = []
    for index, chunk in enumerate(sections):
        if len(chunk) < 40 and not chunk.startswith("#"):
            continue
        c_tok = _tokens(chunk)
        overlap = len(q_tok & c_tok) if q_tok and c_tok else 0
        if q_tok and overlap == 0:
            continue
        score = overlap + overlap / max(len(c_tok), 1)
        scored.append((score, index, chunk))

    if not scored:
        scored = [(0.0, index, chunk) for index, chunk in enumerate(sections)]

    scored.sort(key=lambda item: item[0], reverse=True)
    chosen: list[tuple[int, str]] = []
    total = 0
    for _, index, chunk in scored[:max_chunks]:
        extra = len(chunk) + (2 if chosen else 0)
        if total + extra > max_chars:
            continue
        chosen.append((index, chunk))
        total += extra
    if not chosen:
        first = scored[0][2]
        return first[:max_chars]
    chosen.sort(key=lambda item: item[0])
    return "\n\n".join(chunk for _, chunk in chosen)


def prepare_knowledge_text(knowledge: str, question: str) -> str:
    """Full knowledge when it fits; otherwise verbatim sections that overlap the question."""
    return retrieve_knowledge_snippets(knowledge, question, max_chars=KNOWLEDGE_FULL_MAX_CHARS)


REACT_SYSTEM_PROMPT = """
You are a ReAct-style data agent that answers with SQL on a per-task DuckDB warehouse.

Each task has its own in-memory DuckDB. Tables are this task's CSV/JSON/SQLite (`context/db/*.db` are official sources) plus extracted `doc/*.md` / narrative `.txt` (file stem = table name). `knowledge.md` is copied into the task message verbatim (full file, or whole sections if it is long). Do not glob files outside this task's `context/`. Do not rewrite the Question text. Do not rewrite knowledge into a shorter prior.

Protocol:
1. Table schema (name, columns, types, row count, two sample rows) is already in the task message. Call `list_tables` only if you need it again.
2. Probe with `run_sql` before the final query: distinct values of encoding columns named by the question, exact-key join match counts, and how many rows an extremum hits (`COUNT` of rows equal to MIN/MAX). Leave those numbers in the observation.
3. When the SELECT is the answer, call `run_sql` with `final=true` (no probe LIMIT).
4. Then call `answer` with empty action_input. The last final SQL result is submitted. Do not paste rows. A 0-row final is rejected. An extremum query whose LIMIT hides tied rows is rejected.
5. Always return exactly one JSON object with keys `thought`, `action`, and `action_input`.
6. Always wrap that JSON object in exactly one fenced code block that starts with ```json and ends with ```.
7. Do not output any text before or after the fenced JSON block.

Analysis constraints:
8. Knowledge text in the task message is copied from `knowledge.md`, not a paraphrase. Treat it as authoritative for field encodings, thresholds, metric formulas, and name/field conventions. Map question terms to those values exactly. Do not broaden a term (for example "severe" is not "most severe", and is not a range of nearby codes).
9. Exemplar SQL in knowledge files is a pattern, not a query to paste. Check `list_tables`. If an example selects a column that table does not have, join the table that actually has it. Ignore tables mentioned in knowledge that are missing from this warehouse.
10. Join tables on exact key equality only. A sparse join (few matching keys) is expected; unmatched rows are out of scope and must be dropped. Do not invent fuzzy key matching (nearest ID, numeric difference thresholds, similar strings) unless the question or knowledge explicitly requires it.
11. Put entity attributes (id, sex, diagnosis, name, number) on the entity table, and event/measure filters (admission, dates, labs, session times) on the corresponding fact table. If both tables share a field name, they are different columns: probe each table.column, then use the table the question refers to (the named entity), not the table you already filtered.
12. Probe with COUNT, DISTINCT, DESCRIBE, samples, and LIMIT. Do not SELECT entire key columns until the final query.
13. As soon as the final SELECT answers the question, call `run_sql` with final=true, then `answer`. Do not keep investigating because the row count "looks too small".
14. SELECT only columns the question asks for. Extra columns are penalized and pruned at submit. Do not SELECT helper/debug fields (filter ids, dates, counts, scores) unless the question names them. For percentage / how many / single-metric questions, SELECT exactly one output column (the metric itself)—never total_* / count helpers alongside it.
15. When the question asks for "full name" or "name" and the schema has `first_name` + `last_name` (or equivalent splits), output those columns separately. Never merge into a single `full_name` column.
16. Metric / aggregation disambiguation: for lowest / highest / average, map to knowledge formulas first. Before `answer`, you MUST probe BOTH grains with run_sql: (fine) row-level MIN/MAX/AVG on the measure column; (coarse) SUM / GROUP BY entity totals then MIN/MAX/AVG. Optional: set action_input.grain to "fine" or "coarse". If the two result sets differ, prefer fine unless knowledge explicitly defines the coarse formula. Prefer the documented operator (MIN / MAX / AVG / SUM) when knowledge states it.
17. Extremum ties: return every entity achieving the extremum. Prefer `WHERE col = (SELECT MIN(col) FROM ...)` (or MAX). Do not use ORDER BY … LIMIT 1 unless the question asks for a single unique winner. Submit is rejected when a trailing LIMIT returns fewer rows than the same query without LIMIT.
17b. Aggregate filter members: when a group is kept only because its aggregate (average, sum, count) passes a threshold, the final result must be the rows that actually participated in that aggregate. Do not drop the aggregate table and re-expand to every row that shares the group key.
17c. Relationship rows with swapped keys: if a relationship table stores both directions of the same fact (e.g., both atom_id→atom_id2 and atom_id2→atom_id for the same bond_id), count the relationship by its own identifier using COUNT(DISTINCT id_col), not by counting rows.
18. Homonymous ranking fields: rank / position / order / place must follow knowledge. Confirm table and column before SELECT.
18b. Ambiguous attribute columns: when a question term could match multiple schema columns — including the SAME name on two tables (e.g. drivers.number vs qualifying.number) as well as different names (position / round / number) — probe EACH table.column with the same filter via run_sql and compare hit counts / row sets before choosing. A probe on one table does not cover the same column name on another table. Prefer: (1) the column knowledge explicitly maps; (2) the identifier on the entity table the question names; (3) fewest joins / direct FK path. When schema linking tags a same-named column as "project …; filter on …", project the entity-table copy and keep the event table in WHERE only. Never pick by English similarity alone. Schema-link candidates and the evidence plan in the task message are soft anchors — verify with probes; do not treat them as the answer.
18c. Hypothesis loop: treat competing interpretations (column choice, grain, time precision, list vs scalar) as hypotheses. Empty probes mean the predicate or column is wrong — change it; do not retry the same literal. When two numeric probes differ by orders of magnitude, re-check knowledge formula vs row-level AVG before final.
19. Column already read: SELECT and GROUP BY the column on the row you already chose. If the chosen row has a content column (`text`, `type`, `name`, `description`) and an id, keep the content column. Submitting only the id is rejected.
20. Thresholds and denominators: use encodings and ranges only from knowledge or schema. Do not invent clinical "normal ranges".
21. Final rows must come from `run_sql` with `final=true`. Probe LIMIT is not the answer. DuckDB dialect: double-quoted identifiers, single-quoted strings, TRY_CAST if needed.
22. Before `answer`, keep only columns the question asks for. Extra columns are dropped at submit time by intersecting with columns inferred from the question.
23. Document tables are already extracted. Query them with SQL using the file-stem name. Corrections in the source text (initially / previously / corrected) are already resolved to the last official value. Registry-phrase names beat adjective-prefixed aliases.
24. Time/date / value normalization: compare semantic values, not raw strings. Parse the question literal and knowledge unit (e.g. H:MM:SS vs M:SS.mmm); truncate stored values to the question's grain; treat 0:01:54 ≡ 01:54 ≡ 1:54. Prefer prefix / floor-seconds predicates over col = 'question text'. If a grain-matched probe already returned rows, call final=true with that predicate and keep ties — do not retreat to exact string equality.
25. Scalar shape: for how-many / calculate / total / percentage questions, the final SELECT must be the metric itself (usually one column, one row). Do not overwrite a successful COUNT with entity-level detail rows.
26. Status/metric questions: do not SELECT entity id columns alongside the status/metric unless the question asks for the id.
27. Interpretation arbitration: when two readings of the question are both self-consistent (ratio numerator/denominator direction, same column name on two tables, time precision, list vs scalar, attribute column), probe each cheaply, then choose by this fixed order: (a) the reading knowledge.md defines; (b) the identifier / attribute on the entity table the question names; (c) the reading using a direct foreign-key path / fewer joins. State the chosen reading in `thought` and do not oscillate between readings.

28. Predicate scope (population): WHERE/HAVING decides which rows enter the answer. Probe a wide population (question filters only) and a narrow one (extra IS NOT NULL, or a/b without a non-zero denominator). Use the narrow predicate only when (a) knowledge defines that population, (b) the question licenses 'known/have/complete' for that attribute, or (c) the extra predicate is required for the arithmetic to be defined (guard b <> 0 before a/b > threshold). Do not promote a probe CAST/empty-string error into a final IS NOT NULL that shrinks other metrics sharing the FROM. For two+ scalar averages, default to wide — each AVG skips its own nulls.
29. Measure identity: when lowest/highest/average could bind to more than one numeric column (cost vs spent, amount vs consumption), probe the same question on EACH measure and compare entity lists. Prefer the column whose name appears in the question. A knowledge formula for a neighboring KPI (e.g. Total Expenditure = SUM(spent)) does not replace the named measure unless that sentence explicitly binds the question word to that column.

Keep reasoning concise and grounded in the observed data.
""".strip()

RESPONSE_EXAMPLES = """
Example: inspect tables
```json
{"thought":"See which views this task registered.","action":"list_tables","action_input":{}}
```

Example: probe
```json
{"thought":"Count matching dues rows before submitting.","action":"run_sql","action_input":{"sql":"SELECT COUNT(*) AS n FROM income","final":false}}
```

Example: full-table SQL (then call answer on the next step)
```json
{"thought":"This SELECT is the answer table.","action":"run_sql","action_input":{"sql":"SELECT date_received FROM income JOIN member ON income.link_to_member = member.member_id WHERE first_name = 'Connor' AND last_name = 'Hilton' AND source = 'Dues'","final":true}}
```

Example: submit the stored SQL result
```json
{"thought":"Submit the last final SQL result.","action":"answer","action_input":{}}
```
""".strip()


def build_system_prompt(tool_descriptions: str, system_prompt: str | None = None) -> str:
    base_prompt = system_prompt or REACT_SYSTEM_PROMPT
    return (
        f"{base_prompt}\n\n"
        "Available tools:\n"
        f"{tool_descriptions}\n\n"
        f"{RESPONSE_EXAMPLES}\n\n"
        "You must always return a single ```json fenced block containing one JSON object "
        "with keys `thought`, `action`, and `action_input`, and no extra text."
    )


def build_task_prompt(
    task: PublicTask,
    *,
    knowledge_text: str | None = None,
    schema_text: str | None = None,
    schema_link_text: str | None = None,
    hypothesis_text: str | None = None,
    normalize_text: str | None = None,
) -> str:
    """Build the user task message.

    ``task.question`` is always the authoritative Question line (never rewritten).
    ``knowledge_text`` is verbatim knowledge; None → load the file (full, or whole sections).
    ``schema_text`` is the programmatic table schema dump.
    ``schema_link_text`` / ``hypothesis_text`` / ``normalize_text`` are soft anchors.
    """
    parts = [
        f"Question: {task.question}",
        "This task has its own DuckDB warehouse (list_tables / run_sql / answer). "
        "Join on exact keys only; drop unmatched rows. "
        "SELECT only columns the question asks for; for percentage / how many / "
        "single-metric questions SELECT exactly one output column; split full name into "
        "first_name and last_name when those fields exist. "
        "Resolve aggregations (MIN/MAX/AVG/SUM), ties, rank fields, and "
        "thresholds from the knowledge text below before writing SQL. "
        "Workflow: (1) use schema-link candidates to narrow scope, "
        "(2) normalize question literals to stored grain before comparing, "
        "(3) probe to verify / falsify competing hypotheses, "
        "(4) write final SQL only after evidence, "
        "(5) call answer. "
        "Before the final query, probe encoding distinct values, exact join match counts, "
        "and extremum hit counts. "
        "For lowest/highest/average: probe BOTH fine (row-level MIN/MAX/AVG) and "
        "coarse (SUM/GROUP BY then aggregate); if they differ prefer fine unless "
        "knowledge says otherwise. "
        "Call run_sql with final=true for the answer table, then call answer "
        "(empty action_input). Do not paste rows. A 0-row final is rejected.",
    ]

    schema = (schema_text or "").strip()
    if schema:
        parts.append(
            "Table schema for this task (generated from the files: name, columns, types, "
            "row count, two sample rows). This is not a rewrite of knowledge:\n\n"
            f"{schema}"
        )

    link = (schema_link_text or "").strip()
    if link:
        parts.append(link)

    norm = (normalize_text or "").strip()
    if norm:
        parts.append(norm)

    hyp = (hypothesis_text or "").strip()
    if hyp:
        parts.append(hyp)

    if knowledge_text is not None:
        knowledge = knowledge_text.strip() if knowledge_text.strip() else None
    else:
        full = load_knowledge_md(task)
        knowledge = prepare_knowledge_text(full or "", task.question) if full else None

    if knowledge:
        parts.append(
            "`context/knowledge.md` copied verbatim "
            "(full file, or whole sections when the file is long). "
            "Not a rewrite of the Question. Authoritative for encodings, thresholds, "
            "metric formulas, and naming:\n\n"
            f"{knowledge}"
        )
    else:
        parts.append(
            "No `context/knowledge.md` for this task. "
            "Infer encodings only from the schema and from queries you run. "
            "Do not invent normal/abnormal numeric ranges."
        )
    return "\n\n".join(parts)


def build_observation_prompt(
    observation: dict[str, object],
    *,
    remaining_steps: int | None = None,
    evidence_notes: str | None = None,
) -> str:
    def _default(value: object) -> str:
        if isinstance(value, (bytes, bytearray)):
            return f"<binary len={len(value)}>"
        if isinstance(value, (date, datetime)):
            return value.isoformat()
        return str(value)

    try:
        rendered = json.dumps(observation, ensure_ascii=False, indent=2, default=_default)
    except TypeError:
        # Last resort: never crash the whole benchmark on a weird tool payload.
        rendered = json.dumps({"ok": False, "error": "observation not JSON-serializable"}, indent=2)
    text = f"Observation:\n{rendered}"
    notes = (evidence_notes or "").strip()
    if notes:
        text += f"\n\nEvidence guidance:\n{notes}"
    if remaining_steps is not None and remaining_steps <= 2:
        text += (
            "\n\nYou have "
            f"{remaining_steps} step(s) left. If you already have the result table, "
            "call `answer` now (after a final=true run_sql). Do not keep dumping keys. "
            "Before `answer`: (1) drop columns the question did not ask for "
            "(only the asked fields survive submit-time intersection); "
            "split person names into first_name/last_name when those fields exist; "
            "(2) self-check metrics—which column, which aggregate (MIN/MAX/AVG/SUM), "
            "any ties to keep, bounds/NULL for ratios; for lowest/highest, if row-level "
            "MIN/MAX and per-entity SUM disagree, prefer the knowledge formula; "
            "(3) do not invent thresholds absent from knowledge."
        )
    return text
