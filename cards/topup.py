"""Keep the current season's files up to date.

    .venv/bin/python cards/topup.py            # fetch what is new
    .venv/bin/python cards/topup.py --dry-run  # say what it would do

Tries football-data.co.uk first, falls back to the Internet Archive's most
recent capture. Intended to run weekly from cron; safe to run any number of
times.

**The guard that matters: a file is never replaced by one with fewer matches.**
The archive holds several captures of each file taken at different points in a
season, and the naive "closest snapshot to date X" query happily returns a
13KB capture from four matches in when a complete 52KB one exists. That is not
hypothetical -- it happened while assembling 2025/26, and a partial file would
silently shrink the training data with no error anywhere.
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import time
from datetime import datetime, timezone

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import loader  # noqa: E402

RAW = loader.RAW_DIR
LOG = os.path.join(HERE, "topup_log.txt")

LIVE = "https://www.football-data.co.uk/mmz4281/{s}/{d}.csv"
CDX = ("http://web.archive.org/cdx/search/cdx?url=football-data.co.uk"
       "/mmz4281/{s}/{d}.csv&output=text&fl=timestamp,statuscode"
       "&filter=statuscode:200")
WAYBACK = ("https://web.archive.org/web/{ts}id_/"
           "https://www.football-data.co.uk/mmz4281/{s}/{d}.csv")

UA = {"User-Agent": "Mozilla/5.0"}


def log(msg: str) -> None:
    line = f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M} {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def rows_in(text: str) -> int:
    """Data rows, ignoring the header and any trailing blank lines."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return max(len(lines) - 1, 0)


def looks_like_results(text: str) -> bool:
    # A UTF-8 BOM decoded as latin-1 becomes "ï»¿", not "\ufeff", so strip
    # both. This check rejected a perfectly good file on its first real run:
    # the fetch succeeded, the content was valid, and the header simply did
    # not start with "Div," because three invisible bytes sat in front of it.
    head = text.lstrip("\ufeff").lstrip("ï»¿").split("\n", 1)[0]
    return head.startswith("Div,") and "HomeTeam" in head


def fetch_live(season: str, div: str) -> str | None:
    try:
        r = requests.get(LIVE.format(s=season, d=div), headers=UA, timeout=60,
                         allow_redirects=True)
    except requests.RequestException:
        return None
    if r.status_code != 200:
        return None
    return r.content.decode("latin-1", errors="replace")


def fetch_archive(season: str, div: str) -> tuple[str | None, str]:
    """The LAST 200-status capture, which is the fullest one.

    Not "the capture nearest some date" -- that is the query that returns a
    part-season file.

    The second element is a human-readable reason on failure. An unattended
    weekly job needs "the archive is down" and "this file has never been
    captured" to look different in its log, because the first resolves itself
    and the second never will.
    """
    try:
        idx = requests.get(CDX.format(s=season, d=div), headers=UA, timeout=60)
    except requests.RequestException as exc:
        return None, f"archive unreachable ({type(exc).__name__})"
    if idx.status_code != 200:
        return None, f"archive returned HTTP {idx.status_code}"
    if "<html" in idx.text[:200].lower():
        return None, "archive is offline (served an HTML error page)"

    stamps = [ln.split()[0] for ln in idx.text.splitlines() if ln.strip()]
    if not stamps:
        return None, "never captured by the archive"
    ts = stamps[-1]
    try:
        r = requests.get(WAYBACK.format(ts=ts, s=season, d=div), headers=UA,
                         timeout=120)
    except requests.RequestException as exc:
        return None, f"capture {ts} unreachable ({type(exc).__name__})"
    if r.status_code != 200:
        return None, f"capture {ts} returned HTTP {r.status_code}"
    return r.content.decode("latin-1", errors="replace"), f"archive@{ts}"


def topup(season: str, dry_run: bool = False) -> dict:
    stats = {"updated": 0, "unchanged": 0, "refused": 0, "missing": 0}
    for div in sorted(loader.DIVISIONS):
        path = os.path.join(RAW, f"{season}_{div}.csv")
        have = 0
        if os.path.exists(path):
            with open(path, encoding="latin-1") as f:
                have = rows_in(f.read())

        text, src = fetch_live(season, div), "live site"
        if not (text and looks_like_results(text)):
            text, src = fetch_archive(season, div)
        if text and not looks_like_results(text):
            log(f"  {season}/{div}: fetched from {src} but the content does "
                f"not look like results — keeping our {have} rows")
            stats["missing"] += 1
            continue
        if not text:
            log(f"  {season}/{div}: unchanged at {have} rows — {src}")
            stats["missing"] += 1
            continue

        got = rows_in(text)
        if got > have:
            if dry_run:
                log(f"  {season}/{div}: WOULD update {have} -> {got} ({src})")
            else:
                os.makedirs(RAW, exist_ok=True)
                with open(path, "w", encoding="latin-1", errors="replace") as f:
                    f.write(text)
                log(f"  {season}/{div}: updated {have} -> {got} rows ({src})")
            stats["updated"] += 1
        elif got == have:
            stats["unchanged"] += 1
        else:
            # the guard
            log(f"  {season}/{div}: REFUSED — source has {got} rows, "
                f"we already have {have}. Keeping ours.")
            stats["refused"] += 1
        time.sleep(2)          # be polite to the archive
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--season", default=None,
                    help="default: the current season by date")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    season = a.season
    if season is None:
        now = datetime.now()
        start = now.year if now.month >= 7 else now.year - 1
        season = f"{start % 100:02d}{(start + 1) % 100:02d}"

    log(f"top-up {season}{' (dry run)' if a.dry_run else ''}")
    st = topup(season, dry_run=a.dry_run)
    log(f"  done: {st['updated']} updated, {st['unchanged']} unchanged, "
        f"{st['refused']} refused, {st['missing']} unavailable")

    if season in loader.HOLDOUT_SEASONS:
        log(f"  NOTE: {season} is the holdout. Fetching it is not a look — "
            "no model was scored here.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
