"""Check the betting arithmetic in the weekly sheet actually computes.

    .venv/bin/python cards/test_week_sheet.py

These formulas decide what to stake and how the profit is counted, so
"the references look right" is not good enough. This builds the real sheet
with run_week_sheet.write(), fills the yellow columns the way you would,
evaluates the workbook with a spreadsheet engine, and checks the numbers
against ones worked out by hand.

The case that matters most is the third one. The model predicts yellow cards;
a market settling on total cards is a different quantity, and priced against
it the model shows a phantom OVER edge on nearly every fixture. The guard has
to refuse a bet there even when the edge looks big, so that is what is tested
-- a guard nobody has watched reject something is not a guard.
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
from openpyxl import Workbook, load_workbook

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import run_week_sheet as rws

PASS = FAIL = 0


def check(name: str, got, want, tol: float | None = None) -> None:
    global PASS, FAIL
    if tol is not None and isinstance(got, (int, float)) \
            and isinstance(want, (int, float)):
        ok = abs(float(got) - float(want)) <= tol
    else:
        ok = got == want
    if ok:
        PASS += 1
        print(f"  [PASS] {name}: {got}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}: got {got!r}, expected {want!r}")


def fake_pred(div, home, away, mu, baseline, over35):
    """A prediction with a chosen P(over 3.5); the other lines are filler."""
    pmf = np.zeros(9)
    pmf[3] = 1.0
    over = [0.9, 0.8, over35, 0.2, 0.1, 0.05]
    return {"division": div, "date": "2026-09-26", "home": home, "away": away,
            "referee": None, "mu": mu, "baseline": baseline, "pmf": pmf,
            "over": over, "ref_n": 0, "ref_rel": None, "mode": 3, "iv": "2-5"}


def main() -> int:
    preds = [
        fake_pred("E3", "A", "B", 3.3, 3.9, 0.4151),   # UNDER, wins
        fake_pred("E3", "C", "D", 3.9, 3.9, 0.8000),   # big edge, wrong units
        fake_pred("E3", "E", "F", 3.1, 3.9, 0.3741),   # settlement left blank
        fake_pred("E3", "G", "H", 3.4, 3.9, 0.4476),   # edge under threshold
        fake_pred("E3", "I", "J", 4.1, 3.9, 0.5832),   # OVER, wins
    ]
    tmp = os.path.join(tempfile.mkdtemp(), "week_test.xlsx")
    Workbook().save(tmp)                    # write() expects a workbook to exist
    rws.write(tmp, "Week 26 Sep", preds, "testfingerprint")

    wb = load_workbook(tmp)
    ws = wb["Week 26 Sep"]
    h = {ws.cell(1, c).value: c for c in range(1, ws.max_column + 1)}

    def put(r, name, v):
        ws.cell(r, h[name], v)

    for r, settles, line, over, under, actual in (
            (2, "yellow cards", 3.5, 1.80, 2.05, 2),
            (3, "total cards",  3.5, 1.80, 2.05, 6),
            (4, None,           3.5, 1.80, 2.05, 4),
            (5, "yellow cards", 3.5, 2.20, 1.65, 4),
            (6, "Yellow Cards", 3.5, 2.10, 1.72, 5)):
        if settles:
            put(r, "Settles on", settles)
        put(r, "Card line", line)
        put(r, "Over price", over)
        put(r, "Under price", under)
        put(r, "ACTUAL yellows", actual)
    wb.save(tmp)

    import formulas
    xl = formulas.ExcelModel().loads(tmp).finish()
    sol = xl.calculate()
    base = f"'[{os.path.basename(tmp)}]WEEK 26 SEP'!"

    def g(col, row):
        try:
            v = sol[f"{base}{col}{row}"].value[0, 0]
        except KeyError:
            return "<missing>"
        return float(v) if isinstance(v, (int, float)) \
            and not isinstance(v, bool) else v

    print("\nrow 2 -- model 41.5% against a 53.2% market: bet UNDER, and it wins")
    # de-vig: (1/1.80) / ((1/1.80)+(1/2.05)) = 0.53253
    check("implied P(over) is de-vigged", g("AV", 2), 0.53253, 1e-4)
    check("edge", g("AW", 2), 0.4151 - 0.53253, 1e-4)
    check("verdict", g("AX", 2), "UNDER")
    check("EV on 100 EUR", g("AY", 2), 100 * (0.5849 * 2.05 - 1), 0.05)
    check("2 yellows is under 3.5", g("BA", 2), "WON")
    check("profit", g("BB", 2), 100 * (2.05 - 1), 1e-6)

    print("\nrow 3 -- THE ONE THAT MATTERS: 27-point edge, but wrong units")
    check("edge really is large", g("AW", 3), 0.8 - 0.53253, 1e-4)
    check("refuses to advise a bet", g("AX", 3), "UNITS?")
    check("no EV is offered", g("AY", 3), "")
    check("no profit is counted", g("BB", 3), "")

    print("\nrow 4 -- settlement column left empty")
    check("also refuses", g("AX", 4), "units?")

    print("\nrow 5 -- edge of 1.9 points, below the 5-point threshold")
    check("skips", g("AX", 5), "skip")
    check("no profit counted on a skip", g("BB", 5), "")

    print("\nrow 6 -- model 58.3% against a 45.0% market: bet OVER, and it wins")
    check("verdict", g("AX", 6), "OVER")
    check("profit", g("BB", 6), 100 * (2.10 - 1), 1e-6)
    check("running total is 105 + 110", g("BC", 6), 215.0, 1e-6)

    print("\nsummary block")
    s = 2 + len(preds) + 1          # first row + rows + blank
    check("bets placed counts only the two real bets", g("E", s + 1), 2.0)
    check("bets won", g("E", s + 2), 2.0)
    check("total staked ignores skipped rows", g("E", s + 3), 200.0)
    check("profit and loss", g("E", s + 4), 215.0, 1e-6)
    check("return on stake", g("E", s + 5), 1.075, 1e-6)

    print("\n" + "-" * 62)
    print(f"{PASS}/{PASS + FAIL} checks passed")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
