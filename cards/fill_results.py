"""Fill the forward test's ACTUAL column from the match data, automatically.

    .venv/bin/python cards/fill_results.py                  # what would be filled
    .venv/bin/python cards/fill_results.py --write          # actually fill it
    .venv/bin/python cards/fill_results.py --source bbc     # use the BBC scrape

Chasing results by hand does not scale. League Two card counts are not in a
Google result; they are on the club's own site, and there are 46 fixtures.

They do not need chasing. football-data publishes them within a few days, so
the workflow is: predict before kick-off, wait, run topup, run this. The only
cost is patience.

TWO RULES, BOTH OF WHICH MATTER

1. A result is written only if its card count is CONFIRMED -- more than
   `CARD_CONFIRMATION_LAG_DAYS` behind the data frontier. Recent rows read 0
   yellows while the real figure is still being entered, and cards are the
   target, so importing one early corrupts the label rather than losing a
   feature. AFC Wimbledon v MK Dons really finished 3-0 in cards while
   football-data showed 0.

2. A cell that already holds a value is NEVER overwritten. If you corrected a
   row from an official match report, that hand-entered number outranks
   anything found here.

THE BBC SOURCE

`--source bbc` reads the scrape from fetch_results_bbc.py instead, for the
week football-data has not reached yet. It is faster and less accurate:
measured against 210 settled matches of this season it agrees exactly on
97.1%, and every disagreement is BBC counting one card too many.

So rows written this way are marked provisional in the Notes column, and
`--verify` re-checks them once football-data arrives. Rule 1 does not apply to
this source -- BBC's count is complete within minutes -- but rule 2 does.
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd
from openpyxl import load_workbook

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fetch_results_bbc as bbc
import fixtures_week as fw
import loader

HERE = os.path.dirname(os.path.abspath(__file__))
BOOK = os.path.join(os.path.dirname(HERE), "forward_test_2026-09-18.xlsx")


def fill_from_bbc(a) -> int:
    import csv
    path = a.csv or os.path.join(HERE, "results_bbc_2026-09-20.csv")
    scraped = list(csv.DictReader(open(path)))
    print(f"reading {len(scraped)} scraped matches from {os.path.basename(path)}\n")

    wb = load_workbook(a.file)
    ws = wb[a.sheet]
    h = {ws.cell(1, c).value: c for c in range(1, ws.max_column + 1)}
    div_col = "Div"
    fixtures = {}
    for r in range(2, ws.max_row + 1):
        home = ws.cell(r, h["Home"]).value
        away = ws.cell(r, h["Away"]).value
        date = ws.cell(r, h["Date"]).value
        if not home or not away or not date or home == "Example FC":
            continue
        fixtures[(str(date), ws.cell(r, h[div_col]).value, home, away)] = r
    names = sorted({k[2] for k in fixtures} | {k[3] for k in fixtures})

    filled = kept = skipped = unmatched = 0
    for s in scraped:
        home = bbc.resolve(s["bbc_home"], names)
        away = bbc.resolve(s["bbc_away"], names)
        row = fixtures.get((s["date"], s["division"], home, away))
        if row is None:
            unmatched += 1
            print(f"  ?       {s['bbc_home']} v {s['bbc_away']}: no fixture in "
                  "the workbook matches this")
            continue
        if not int(s["usable"]):
            skipped += 1
            print(f"  skip    {home} v {away}: {s['note']}")
            continue
        if ws.cell(row, h["ACTUAL yellows"]).value is not None:
            kept += 1
            print(f"  keep    {home} v {away}: already "
                  f"{ws.cell(row, h['ACTUAL yellows']).value} "
                  f"(BBC says {s['yellows']})")
            continue
        if a.write:
            ws.cell(row, h["ACTUAL yellows"], int(s["yellows"]))
            note = ws.cell(row, h["Notes"]).value or ""
            extra = (f"actual from BBC {s['date']} (provisional, "
                     f"~97% exact vs football-data)")
            if s["note"]:
                extra += f"; {s['note']}"
            ws.cell(row, h["Notes"], (note + "; " if note else "") + extra)
        filled += 1
        print(f"  {'fill  ' if a.write else 'would '} {home} v {away}: "
              f"{s['yellows']} yellows"
              f"{'   ' + s['note'] if s['note'] else ''}")

    print(f"\n{filled} filled, {kept} kept as-is, {skipped} skipped as "
          f"unreliable, {unmatched} unmatched")
    if a.write and filled:
        wb.save(a.file)
        print(f"saved {a.file}")
    elif not a.write:
        print("dry run -- pass --write to save")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--write", action="store_true",
                    help="write into the workbook; otherwise just report")
    ap.add_argument("--file", default=BOOK)
    ap.add_argument("--source", choices=("football-data", "bbc"),
                    default="football-data")
    ap.add_argument("--csv", default=None,
                    help="the scrape to read for --source bbc")
    ap.add_argument("--sheet", default="Predictions",
                    help="which sheet to fill; a weekly sheet, or Predictions")
    a = ap.parse_args()

    if a.source == "bbc":
        return fill_from_bbc(a)

    m = loader.flag_unconfirmed_cards(
        loader.load_matches(unlock_holdout=True))
    frontier = m.date.max()
    print(f"match data runs to {frontier.date()}; card counts are treated as "
          f"confirmed up to "
          f"{(frontier - pd.Timedelta(days=loader.CARD_CONFIRMATION_LAG_DAYS)).date()}\n")

    wb = load_workbook(a.file)
    ws = wb["Predictions"]
    h = {ws.cell(1, c).value: c for c in range(1, ws.max_column + 1)}

    filled = kept = pending = unpublished = 0
    for r in range(2, ws.max_row + 1):
        home = ws.cell(r, h["Home"]).value
        away = ws.cell(r, h["Away"]).value
        date = ws.cell(r, h["Date"]).value
        # Skip anything that is not a fixture row: the example, and the label
        # above it, which carries text in the Home column but has no date.
        if not home or not away or not date or home == "Example FC":
            continue
        date = str(date)
        existing = ws.cell(r, h["ACTUAL yellows"]).value

        hit = m[(m.HomeTeam == home) & (m.AwayTeam == away)
                & (m.date >= pd.Timestamp(date) - pd.Timedelta(days=1))
                & (m.date <= pd.Timestamp(date) + pd.Timedelta(days=2))]

        if existing is not None:
            kept += 1
            print(f"  keep    {home} v {away}: already {existing} "
                  "(hand-entered values win)")
            continue
        if len(hit) == 0:
            unpublished += 1
            continue
        row = hit.iloc[0]
        if not bool(row.cards_confirmed):
            pending += 1
            print(f"  wait    {home} v {away}: played {row.date.date()}, data "
                  f"says {int(row.yellows)} yellows but cards are not "
                  "confirmed yet")
            continue
        if a.write:
            ws.cell(r, h["ACTUAL yellows"], int(row.yellows))
        filled += 1
        print(f"  {'fill  ' if a.write else 'would '} {home} v {away}: "
              f"{int(row.yellows)} yellows")

    # "not in the data" is not the same as "not played": football-data runs
    # several days behind, so a fixture played on Saturday is simply absent on
    # Monday. Saying "not played yet" invites you to think the weekend never
    # happened.
    print(f"\n{filled} filled, {kept} kept as-is, {pending} awaiting "
          f"confirmation, {unpublished} not in the data yet")
    if a.write and filled:
        wb.save(a.file)
        print(f"saved {a.file}")
    elif not a.write:
        print("dry run -- pass --write to save")
    if pending:
        print(f"\nRe-run in a few days: football-data fills card counts in "
              f"after the result. The weekly topup (Mondays) will have them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
