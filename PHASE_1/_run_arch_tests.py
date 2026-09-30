"""Local runner when Agent Shell is stubbed: python _run_arch_tests.py

Covers the §13 architecture work: sql_ir / aggregate_grain / submit_validation /
schema_link+hypothesis. Run from PHASE_1 with PYTHONPATH=src.
"""

from __future__ import annotations

import sys
import traceback

MODULES = (
    "_test_sql_ir",
    "_test_aggregate_grain",
    "_test_submit_validation",
    "_test_schema_hypothesis_vote",
    "_test_predicate_scope",
    "_test_measure_identity",
)


def main() -> int:
    failed = 0
    for mod_name in MODULES:
        try:
            mod = __import__(mod_name)
        except Exception:
            failed += 1
            print(f"FAIL import {mod_name}")
            traceback.print_exc()
            continue
        for name in sorted(dir(mod)):
            if not name.startswith("test_"):
                continue
            try:
                getattr(mod, name)()
                print(f"PASS {mod_name}.{name}")
            except Exception:
                failed += 1
                print(f"FAIL {mod_name}.{name}")
                traceback.print_exc()
    print(f"done failed={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
