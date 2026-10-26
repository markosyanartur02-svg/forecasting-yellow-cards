"""The "football-data is behind" warning must fire on stale data and only then.

    .venv/bin/python cards/test_topup.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import topup

PASS = FAIL = 0


def check(name: str, got, want) -> None:
    global PASS, FAIL
    if got == want:
        PASS += 1
        print(f"  [PASS] {name}: {got}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}: got {got!r}, expected {want!r}")


def main() -> int:
    raw = tempfile.mkdtemp()
    topup.RAW = raw
    with open(os.path.join(raw, "2627_E1.csv"), "w", encoding="latin-1") as f:
        f.write("ï»¿Div,Date,HomeTeam,AwayTeam\n"   # a UTF-8 BOM read as latin-1, as football-data serves it
                "E1,17/10/2026,Leeds,Hull\n")
    fixtures = ("ï»¿Div,Date,Time,HomeTeam,AwayTeam\n"
                "E1,17/10/2026,15:00,Leeds,Hull\n"        # result already in
                "E1,23/10/2026,20:00,Bristol City,Wrexham\n"  # played, no result
                "E1,24/10/2026,15:00,Stoke,Derby\n"       # played, no result
                "E1,25/10/2026,15:00,Swansea,QPR\n"       # yesterday: too soon
                "SC0,23/10/2026,15:00,Celtic,Hearts\n")   # not a division we keep
    today = datetime(2026, 10, 26, 9, 30)
    print("\nfixtures played two or more days ago with no result are reported")
    late = topup.unpublished("2627", fixtures, today)
    check("the two late matches", [(d, h) for _, d, h, _ in late],
          [("2026-10-23", "Bristol City"), ("2026-10-24", "Stoke")])
    print("\nnothing is reported once the results are in")
    with open(os.path.join(raw, "2627_E1.csv"), "a", encoding="latin-1") as f:
        f.write("E1,23/10/2026,Bristol City,Wrexham\nE1,24/10/2026,Stoke,Derby\n")
    check("no warning", topup.unpublished("2627", fixtures, today), [])

    print("\n" + "-" * 62)
    print(f"{PASS}/{PASS + FAIL} checks passed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
