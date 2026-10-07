"""P2 question-type soft routing. Labels only; never task_id templates."""

from __future__ import annotations

import re
from typing import Literal

QuestionKind = Literal["ratio", "agg", "list", "extremum", "existence", "clinical"]

_RATIO_RE = re.compile(
    r"\b(?:how\s+many\s+times|ratio|percentage|percent|compared\s+to|"
    r"divide[sd]?|times\s+(?:as|larger|greater|more)|relative\s+to|per)\b",
    flags=re.IGNORECASE,
)
_AGG_RE = re.compile(
    r"\b(?:average|mean|avg|total|sum|how\s+many|count|calculate)\b",
    flags=re.IGNORECASE,
)
_LIST_RE = re.compile(
    r"\b(?:which|what)\b.+\b(?:races?|names?|ids?|schools?|elements?|types?|items?)\b|"
    r"\blist\b|\btally\b|\ball\s+the\b",
    flags=re.IGNORECASE,
)
_EXTREMUM_RE = re.compile(
    r"\b(?:lowest|highest|cheapest|smallest|largest|fewest|fastest|slowest|"
    r"minimum|maximum|least|best|worst)\b",
    flags=re.IGNORECASE,
)
_EXIST_RE = re.compile(r"\b(?:is there|are there|does any|exist)\b", flags=re.IGNORECASE)
_CLINICAL_RE = re.compile(
    r"\b(?:patient|laboratory|wbc|fibrinogen|\bfg\b|\bldh\b|creatinine|"
    r"normal\s+range|abnormal)\b",
    flags=re.IGNORECASE,
)


def classify_question(question: str) -> frozenset[QuestionKind]:
    text = question or ""
    kinds: set[QuestionKind] = set()
    if _RATIO_RE.search(text):
        kinds.add("ratio")
    if _AGG_RE.search(text):
        kinds.add("agg")
    if _LIST_RE.search(text):
        kinds.add("list")
    if _EXTREMUM_RE.search(text):
        kinds.add("extremum")
    if _EXIST_RE.search(text):
        kinds.add("existence")
    if _CLINICAL_RE.search(text):
        kinds.add("clinical")
    return frozenset(kinds)
