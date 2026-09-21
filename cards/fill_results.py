"""Fill the forward test's ACTUAL column from the match data, automatically.

    .venv/bin/python cards/fill_results.py            # what would be filled
    .venv/bin/python cards/fill_results.py --write    # actually fill it

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
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd
from openpyxl import load_workbook

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fixtures_week as fw
import loader

BOOK = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "forward_test_2026-09-18.xlsx")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--write", action="store_true",
                    help="write into the workbook; otherwise just report")
    ap.add_argument("--file", default=BOOK)
    a = ap.parse_args()

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
