"""Local runner when Agent Shell is stubbed: python _run_new_tests.py"""

from __future__ import annotations

import sys
import traceback


def main() -> int:
    failed = 0
    mod_name = "_test_schema_hypothesis_vote"
    try:
        import _test_schema_hypothesis_vote as mod
    except Exception:
        traceback.print_exc()
        return 1
    for name in sorted(dir(mod)):
        if not name.startswith("test_"):
            continue
        fn = getattr(mod, name)
        try:
            fn()
            print(f"PASS {name}")
        except Exception:
            failed += 1
            print(f"FAIL {name}")
            traceback.print_exc()
    # fallback smoke
    try:
        import _test_fallback_submit as fb

        for name in sorted(dir(fb)):
            if not name.startswith("test_"):
                continue
            try:
                getattr(fb, name)()
                print(f"PASS {name}")
            except Exception:
                failed += 1
                print(f"FAIL {name}")
                traceback.print_exc()
    except Exception:
        traceback.print_exc()
        failed += 1
    print(f"done failed={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
