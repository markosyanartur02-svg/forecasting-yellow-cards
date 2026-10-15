"""Settle the forward test's labels against football-data, once it has them.

    .venv/bin/python cards/reconcile_labels.py            # report readiness
    .venv/bin/python cards/reconcile_labels.py --write    # settle the labels

The ACTUAL column was filled from BBC, days before football-data had the
matches. BBC is the fast source; football-data is the one the model was
TRAINED on. Those are different conventions, and the label a model is scored
against has to be in the convention it learned -- a truer number from
elsewhere is the wrong label.

So once football-data has a match and its card count is out of the
confirmation lag, football-data's number wins.

WHAT IS NOT OVERWRITTEN

A value entered by hand from an official match report. Those rows are
identified by their Notes: anything this project filled automatically says
"actual from BBC", and anything else was typed by a person who had looked at
a source football-data did not.

That rule is not theoretical. AFC Wimbledon v MK Dons finished with 3 yellows
according to the club's own match report. football-data recorded 0, is now
well past its confirmation lag, and still records 0 -- it simply never filled
the row in. Overwriting the 3 with that 0 would corrupt the label while
looking like diligence.

READINESS

Re-scoring costs a holdout look, so it is worth spending once rather than
weekly on a number that is still moving. This reports READY only when every
labelled fixture is out of the lag, and --write refuses otherwise unless
forced.
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd
from openpyxl import load_workbook

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import loader

BOOK = os.path.join(os.path.dirname(HERE), "forward_test_2026-09-18.xlsx")
MACHINE_FILLED = "actual from BBC"


def survey(path: str, sheet: str = "Predictions", asof=None) -> dict:
    m = loader.flag_suspicious_cards(loader.flag_unconfirmed_cards(
        loader.load_matches(unlock_holdout=True), asof=asof))
    frontier = m.date.max()
    ws = load_workbook(path, data_only=True)[sheet]
    h = {ws.cell(1, c).value: c for c in range(1, ws.max_column + 1)}

    agree, change, protected, pending, missing, review = [], [], [], [], [], []
    for r in range(2, ws.max_row + 1):
        home = ws.cell(r, h["Home"]).value
        away = ws.cell(r, h["Away"]).value
        date = ws.cell(r, h["Date"]).value
        actual = ws.cell(r, h["ACTUAL yellows"]).value
        notes = ws.cell(r, h["Notes"]).value or ""
        if not home or home == "Example FC" or not away or actual is None:
            continue
        hit = m[(m.HomeTeam == home) & (m.AwayTeam == away)
                & (m.date >= pd.Timestamp(str(date)) - pd.Timedelta(days=1))
                & (m.date <= pd.Timestamp(str(date)) + pd.Timedelta(days=2))]
        row = {"row": r, "fixture": f"{home} v {away}", "date": str(date),
               "actual": actual, "notes": notes}
        if len(hit) == 0:
            missing.append(row)
            continue
        x = hit.iloc[0]
        row["fd"] = int(x.yellows)
        row["fouls"] = f"{int(x.fouls)}"
        if not bool(x.cards_confirmed):
            pending.append(row)
        elif row["fd"] == actual:
            agree.append(row)
        elif bool(x.cards_suspicious):
            # football-data disagrees AND its row has the shape of an
            # unfilled second booking. Do not trade a label for that.
            review.append(row)
        elif MACHINE_FILLED not in notes:
            protected.append(row)
        else:
            change.append(row)
    ready_on = None
    if pending:
        latest = max(pd.Timestamp(r["date"]) for r in pending)
        ready_on = (latest
                    + pd.Timedelta(days=loader.CARD_CONFIRMATION_LAG_DAYS)).date()
    return {"frontier": frontier, "agree": agree, "change": change,
            "protected": protected, "pending": pending, "missing": missing,
            "review": review,
            "ready_on": ready_on}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="settle the labels that ARE confirmed even though "
                         "others are still inside the lag")
    ap.add_argument("--file", default=BOOK)
    ap.add_argument("--sheet", default="Predictions")
    ap.add_argument("--quiet", action="store_true",
                    help="one line, for the daily job")
    a = ap.parse_args()

    s = survey(a.file, a.sheet)
    ready = not s["pending"] and not s["missing"]
    n = sum(len(s[k]) for k in ("agree", "change", "protected", "pending",
                                "missing"))

    if a.quiet:
        if ready:
            print(f"labels: READY -- all {n} confirmed, "
                  f"{len(s['change'])} need changing, re-score now")
        else:
            when = f", ready {s['ready_on']}" if s["ready_on"] else ""
            print(f"labels: {len(s['pending']) + len(s['missing'])} of {n} "
                  f"still inside the card lag{when} -- not ready to re-score")
        return 0

    print(f"football-data runs to {s['frontier'].date()}; cards confirmed "
          f"through "
          f"{(s['frontier'] - pd.Timedelta(days=loader.CARD_CONFIRMATION_LAG_DAYS)).date()}\n")
    print(f"  {len(s['agree']):>3} labels already agree with football-data")
    print(f"  {len(s['change']):>3} differ and would be changed")
    print(f"  {len(s['protected']):>3} differ but are hand-entered and are kept")
    print(f"  {len(s['pending']):>3} still inside the card lag")
    print(f"  {len(s['missing']):>3} not in football-data at all")
    print(f"  {len(s['review']):>3} differ, but football-data's row looks "
          "unfilled (red with no yellows) -- kept, check by hand")

    for r in s["change"]:
        print(f"\n  CHANGE  {r['fixture']}: {r['actual']} -> {r['fd']}")
    for r in s["protected"]:
        print(f"\n  KEEP    {r['fixture']}: {r['actual']} "
              f"(football-data says {r['fd']}, but this was entered by hand)")
    for r in s["pending"][:5]:
        print(f"\n  WAIT    {r['fixture']} ({r['date']}): football-data says "
              f"{r['fd']}, not yet confirmed")
    if len(s["pending"]) > 5:
        print(f"\n  ... and {len(s['pending']) - 5} more inside the lag")

    if not ready:
        print(f"\nNOT READY. {len(s['pending']) + len(s['missing'])} fixture(s) "
              "are still inside the lag.")
        if s["ready_on"]:
            print(f"All of them clear it on {s['ready_on']}.")
        print("Re-scoring costs a holdout look, so it is worth spending once, "
              "on settled labels.")
        if a.write and not a.force:
            print("Refusing to write. Pass --force to settle the confirmed "
                  "ones anyway.")
            return 1
    else:
        print("\nREADY -- every labelled fixture is out of the lag.")

    if not a.write:
        print("\ndry run -- pass --write to settle the labels")
        return 0

    if not s["change"]:
        print("\nnothing to change.")
        return 0

    wb = load_workbook(a.file)
    ws = wb[a.sheet]
    h = {ws.cell(1, c).value: c for c in range(1, ws.max_column + 1)}
    for r in s["change"]:
        ws.cell(r["row"], h["ACTUAL yellows"], r["fd"])
        old = ws.cell(r["row"], h["Notes"]).value or ""
        ws.cell(r["row"], h["Notes"],
                old + f"; settled {pd.Timestamp.today().date()} from "
                      f"{r['actual']} to {r['fd']} (football-data, the "
                      "training convention)")
    wb.save(a.file)
    print(f"\n{len(s['change'])} label(s) settled, saved {a.file}")
    print("The recorded score is now stale -- re-run score_forward_test.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
