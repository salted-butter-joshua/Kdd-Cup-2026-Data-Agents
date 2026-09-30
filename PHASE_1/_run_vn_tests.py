"""Run value-normalize tests and write results to _vn_out.txt."""
from __future__ import annotations

import traceback
from pathlib import Path

out = Path(__file__).with_name("_vn_out.txt")
lines: list[str] = []
failed = 0
try:
    import _test_value_normalize as mod
except Exception:
    out.write_text(traceback.format_exc(), encoding="utf-8")
    raise SystemExit(1)

for name in sorted(dir(mod)):
    if not name.startswith("test_"):
        continue
    try:
        getattr(mod, name)()
        lines.append(f"PASS {name}")
    except Exception:
        failed += 1
        lines.append(f"FAIL {name}\n{traceback.format_exc()}")

# also import react
try:
    from data_agent_baseline.agents import react, value_normalize  # noqa: F401
    lines.append("PASS import_react_normalize")
except Exception:
    failed += 1
    lines.append(f"FAIL import\n{traceback.format_exc()}")

lines.append(f"done failed={failed}")
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
raise SystemExit(1 if failed else 0)
