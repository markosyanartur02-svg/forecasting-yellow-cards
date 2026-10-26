"""Build or refresh this week's forecast sheet, unattended.

    .venv/bin/python cards/auto_week_sheet.py [--dry-run]

The scheduled job runs this after the referee watcher, twice a day. It exists
because building the sheet used to be a manual step, and when nobody ran it
the rounds of 16-20 and 23-26 October went by with no forecasts at all. Those
cannot be made up afterwards: a forecast written after kick-off is not one.

THE WEEK

A fixed Thursday-to-Wednesday window, so every run in a week updates the same
sheet ("Week 29 Oct-4 Nov") instead of starting a new one each day. Thursday
is early enough for Friday-night matches, and a weekend round shares a sheet
with the midweek round after it.

Each run re-predicts only fixtures that have not kicked off, so referees
published after the first run are picked up, and run_week_sheet.py keeps every
started match's row exactly as it was.

WHEN IT DOES NOTHING, IT SAYS WHY

- The workbook is open in Excel. Excel would overwrite the new sheet the next
  time you saved, or this would overwrite what you had typed, so it waits.
- football-data lists no upcoming fixtures for the week yet. It adds them a
  few days ahead, and the next run picks them up.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import subprocess
import sys
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import run_week_sheet as rws

# run_week_sheet.py output worth repeating in the job's log
KEEP = ("have a referee", "already kicked off", "wrote sheet", "predicted now",
        "REFUSING", "ABORTING", "nothing left", "no playable", "dry run")


def week_window(day: dt.date) -> tuple[dt.date, dt.date]:
    """The Thursday-to-Wednesday week containing `day`."""
    start = day - dt.timedelta(days=(day.weekday() - 3) % 7)
    return start, start + dt.timedelta(days=6)


def open_in_excel(path: str) -> bool:
    """Excel keeps a '~$name' lock file next to a workbook it has open."""
    folder, name = os.path.split(path)
    return os.path.exists(os.path.join(folder, "~$" + name))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--file", default=rws.BOOK)
    a = ap.parse_args()

    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    def say(msg: str) -> None:
        print(f"{stamp} week sheet: {msg}", flush=True)

    now = rws.uk_now()
    start, end = week_window(now.date())
    sheet = rws.sheet_name(start, end)

    if open_in_excel(a.file):
        say(f"SKIPPED -- {os.path.basename(a.file)} is open in Excel. Close "
            f"it and the next run updates '{sheet}'.")
        return 0

    try:
        fixtures, _ = rws.collect_fixtures(start, end)
    except Exception as e:
        say(f"could not read football-data's fixtures ({type(e).__name__}) "
            "-- will try again next run")
        return 0
    upcoming = [f for f in fixtures if not rws.kicked_off(f[1], f[5], now)]
    if not upcoming:
        started = len(fixtures) - len(upcoming)
        say(f"no upcoming fixtures listed for '{sheet}' yet"
            + (f" ({started} listed have already kicked off)" if started else ""))
        return 0

    say(f"{len(upcoming)} upcoming fixtures -> '{sheet}'")
    cmd = [sys.executable, os.path.join(HERE, "run_week_sheet.py"),
           "--file", a.file, "--sheet", sheet,
           "--from", start.isoformat(), "--to", end.isoformat()]
    if a.dry_run:
        cmd.append("--dry-run")
    res = subprocess.run(cmd, capture_output=True, text=True)
    out = res.stdout + res.stderr
    for line in out.splitlines():
        if any(k in line for k in KEEP):
            say(line.strip())

    # The referee is half the model's value. Say so when a match is close and
    # still has none, rather than leaving it to a "3 of 12" in the middle.
    m = re.search(r"(\d+) of (\d+) fixtures to predict have a referee", out)
    soon = min(f[1] for f in upcoming) <= (now.date() + dt.timedelta(days=1)).isoformat()
    if m and int(m.group(1)) < int(m.group(2)) and soon:
        say(f"WARNING: {int(m.group(2)) - int(m.group(1))} fixture(s) play "
            "within a day and have no referee yet -- the EFL has not published "
            "appointments, so those use the league average")

    if res.returncode != 0:
        say(f"FAILED (exit {res.returncode}). To see why, run: "
            f".venv/bin/python cards/run_week_sheet.py --from {start} --to {end} "
            f"--sheet '{sheet}'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
