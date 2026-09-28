"""Download and normalise English football match data.

Source: football-data.co.uk, free, no API key.

The whole job of this file is to turn 104 CSV files written over 26 years into
one table with a stable schema, and to be loud about anything it loses on the
way. That is not boilerplate: the column set genuinely changes across seasons
(odds bookmakers come and go, some stats appear only later), date formats
change between two- and four-digit years, and files carry trailing blank rows.
Every one of those silently drops rows if you let it.

The preceding oil project lost 35% of its data to a filter nobody checked, and
a `drop_duplicates` on the wrong columns once collapsed 26,000 matches into
1,254. Hence the assertions.
"""

from __future__ import annotations

import difflib
import io
import os
import re
import sys

import numpy as np
import pandas as pd
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(os.path.dirname(HERE), "data", "raw")

BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/{div}.csv"

# PROJECT_SPEC section 2. The National League is deliberately absent: it loses
# fouls/corners/shots after 2017/18 and only six of its referees have 50+
# matches, so the referee feature cannot be estimated there.
DIVISIONS = {
    "E0": "Premier League",
    "E1": "Championship",
    "E2": "League One",
    "E3": "League Two",
}

# First season carrying referee, cards, fouls and shots.
FIRST_SEASON_START = 2000
LAST_SEASON_START = 2026

# PROJECT_SPEC section 3.
#
# CHANGED 2026-09-17. Was ("2425", "2526"); is now ("2627",).
#
# This is not the boundary being moved to flatter a result -- it is the
# opposite. The original holdout was LOOKED AT TWICE (see holdout_log.json),
# and a holdout that has been looked at twice is not a holdout any more: the
# model was developed with knowledge of how it scored. Continuing to call it
# one would be the dishonest option.
#
# So 2024/25 and 2025/26 are reclassified as what they have actually become --
# development data -- and 2026/27 becomes the fresh, untouched test set. It is
# genuinely clean: the season is being played now, no model has been fitted
# with it, and no result from it has been seen.
#
# The looks counter starts again at zero for this holdout. The two spent on the
# old one remain in the log permanently; they are part of this project's
# record, not something that gets reset by redrawing a line.
HOLDOUT_SEASONS = ("2627",)
PREVIOUS_HOLDOUT = ("2425", "2526")   # spent: 2 looks, kept for the record

# Expected fixture counts. 20 teams -> 380, 24 teams -> 552. Deviations are
# real (COVID curtailed 2019/20, and division sizes changed historically), so
# these drive a WARNING and a report, never a silent pass.
EXPECTED = {"E0": 380, "E1": 552, "E2": 552, "E3": 552}

CORE = ["Div", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]

# Pre-match betting odds, resolved to canonical names. These are PRE-match
# prices, so using them is point-in-time legitimate -- they were public before
# kick-off. The available bookmaker changes across 26 seasons, so each
# canonical column is filled from the first source present, market averages
# first because a single book's price carries that book's quirks.
ODDS_SOURCES = {
    "odds_h":   ["AvgH", "BbAvH", "PSH", "B365H"],
    "odds_d":   ["AvgD", "BbAvD", "PSD", "B365D"],
    "odds_a":   ["AvgA", "BbAvA", "PSA", "B365A"],
    "odds_o25": ["Avg>2.5", "BbAv>2.5", "P>2.5", "B365>2.5"],
    "odds_u25": ["Avg<2.5", "BbAv<2.5", "P<2.5", "B365<2.5"],
}
STATS = ["Referee", "HY", "AY", "HR", "AR", "HF", "AF",
         "HC", "AC", "HS", "AS", "HST", "AST"]


def canonical_referee(name) -> str | None:
    """Collapse one official's many spellings into a single identity.

    football-data changed naming convention across eras and made typos on the
    way, so the same referee appears as several people. Mike Dean is
    "M Dean" (615), "Mike Dean" (31), "Dean, M. L" (17), "M. L Dean" (12) and
    "M. L. Dean" (1); Alan Wiley has five spellings across 309 matches.

    This matters more than it looks. Referee features carry 43% of the model's
    value, and a referee split across five identities has each rolling rate
    computed from a fragment of his record -- while variants with fewer than
    MIN_REF_MATCHES appearances return nothing at all.

    Everything is reduced to "first-initial surname": Mike Dean, M. L. Dean and
    Dean, M. L all become "m dean".
    """
    if not isinstance(name, str):
        return None
    s = name.strip().replace(".", " ")
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        return None

    if "," in s:                       # "Dean, M L"
        sur, _, inits = s.partition(",")
        sur, inits = sur.strip(), inits.strip()
    else:
        parts = s.split()
        if len(parts) == 1:
            # "APenn", "KAWoolmer" -- initials run into the surname
            mm = re.match(r"^([A-Z]+)([A-Z][a-z].*)$", parts[0])
            if mm:
                inits, sur = mm.group(1), mm.group(2)
            else:
                inits, sur = "", parts[0]
        else:
            sur, inits = parts[-1], " ".join(parts[:-1])

    first = inits[:1].upper() if inits else ""
    sur = re.sub(r"[^A-Za-z'\-]", "", sur)
    if not sur:
        return None
    return f"{first} {sur}".strip().lower()


def merge_referee_typos(names: pd.Series, rare_below: int = 20,
                        similarity: float = 0.86) -> dict:
    """Fold rare misspellings into the common spelling of the same name.

    Canonicalisation cannot fix "C Boweson" for "C Boyeson" or "N Swabrick"
    for "N Swarbrick", because those are typos rather than format differences.
    A rare variant is merged into a common one only when the first initial is
    identical and the surnames are very close, so "C Oliver" and "M Oliver"
    -- genuinely different officials -- are never conflated.
    """
    counts = names.value_counts()
    common = counts[counts >= rare_below]
    rare = counts[counts < rare_below]
    mapping = {}
    by_initial: dict[str, list[str]] = {}
    for nm in common.index:
        by_initial.setdefault(nm.split(" ")[0], []).append(nm)

    for nm in rare.index:
        head = nm.split(" ")[0]
        pool = by_initial.get(head, [])
        if not pool:
            continue
        sur = nm[len(head):].strip()
        best, best_score = None, 0.0
        for cand in pool:
            cs = cand[len(head):].strip()
            score = difflib.SequenceMatcher(None, sur, cs).ratio()
            if score > best_score:
                best, best_score = cand, score
        if best and best_score >= similarity:
            mapping[nm] = best
    return mapping


def season_codes(first: int = FIRST_SEASON_START,
                 last: int = LAST_SEASON_START) -> list[str]:
    """['0001', '0102', ... '2526'] — football-data's season naming."""
    out = []
    for y in range(first, last + 1):
        out.append(f"{y % 100:02d}{(y + 1) % 100:02d}")
    return out


def is_holdout(season: str) -> bool:
    return season in HOLDOUT_SEASONS


# ------------------------------------------------------------------ download


def _raw_path(season: str, div: str) -> str:
    return os.path.join(RAW_DIR, f"{season}_{div}.csv")


def download(season: str, div: str, refresh: bool = False) -> str | None:
    path = _raw_path(season, div)
    if os.path.exists(path) and not refresh:
        return path
    os.makedirs(RAW_DIR, exist_ok=True)
    url = BASE_URL.format(season=season, div=div)
    resp = requests.get(url, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
    if resp.status_code != 200 or len(resp.content) < 200:
        return None
    with open(path, "wb") as f:
        f.write(resp.content)
    return path


def download_all(refresh: bool = False) -> dict:
    got, missing = 0, []
    for s in season_codes():
        for d in DIVISIONS:
            if download(s, d, refresh=refresh):
                got += 1
            else:
                missing.append(f"{s}/{d}")
    return {"downloaded": got, "missing": missing}


# ----------------------------------------------------------------- normalise


def _parse_dates(raw: pd.Series) -> pd.Series:
    """football-data writes dd/mm/yy in early seasons and dd/mm/yyyy later.

    Parsed with an explicit dayfirst and both widths tried, because pandas'
    inference silently reads 01/02/2015 as 2 January in some versions and
    1 February in others. A date that flips meaning by season would scramble
    every rolling feature in this project.
    """
    s = raw.astype(str).str.strip()
    out = pd.to_datetime(s, format="%d/%m/%Y", errors="coerce")
    fallback = pd.to_datetime(s, format="%d/%m/%y", errors="coerce")
    out = out.fillna(fallback)
    return out


def load_one(season: str, div: str) -> pd.DataFrame | None:
    path = _raw_path(season, div)
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, encoding="latin-1", on_bad_lines="skip",
                     low_memory=False)
    # Some files (the 2025/26 ones recovered from the Internet Archive) begin
    # with a UTF-8 byte-order mark. Read as latin-1 it becomes "ï»¿Div"; read
    # as utf-8 it becomes "\ufeffDiv". Either way the first column stops
    # matching its own name, which is the kind of defect that silently drops a
    # column rather than raising.
    df.columns = [str(c).replace("\ufeff", "").replace("ï»¿", "").strip()
                  for c in df.columns]

    # trailing blank rows are common at the end of these files
    if "HomeTeam" not in df.columns:
        return None
    n_raw = len(df)
    df = df[df["HomeTeam"].notna() & (df["HomeTeam"].astype(str).str.len() > 0)]

    missing_stats = [c for c in STATS if c not in df.columns]
    for c in missing_stats:
        df[c] = np.nan

    keep = [c for c in CORE + STATS if c in df.columns]
    out = df[keep].copy()
    for canon, sources in ODDS_SOURCES.items():
        col = next((c for c in sources if c in df.columns), None)
        out[canon] = pd.to_numeric(df[col], errors="coerce") if col else np.nan
        out.attrs.setdefault("odds_source", {})[canon] = col
    out["season"] = season
    out["division"] = div
    out["date"] = _parse_dates(out["Date"])
    out = out.drop(columns=["Date", "Div"], errors="ignore")

    out.attrs["n_raw"] = n_raw
    out.attrs["missing_stats"] = missing_stats
    return out


def load_matches(seasons: list[str] | None = None,
                 divisions: list[str] | None = None,
                 unlock_holdout: bool = False,
                 verbose: bool = False) -> pd.DataFrame:
    """The one table everything downstream uses.

    Holdout seasons are refused unless `unlock_holdout=True`; see
    PROJECT_SPEC section 3.
    """
    seasons = seasons or season_codes()
    divisions = divisions or sorted(DIVISIONS)

    blocked = [s for s in seasons if is_holdout(s)]
    if blocked and not unlock_holdout:
        seasons = [s for s in seasons if not is_holdout(s)]
        if verbose:
            print(f"  holdout seasons withheld: {blocked} "
                  "(pass unlock_holdout=True to read them)")

    frames, report = [], []
    for s in seasons:
        for d in divisions:
            one = load_one(s, d)
            if one is None:
                report.append({"season": s, "division": d, "rows": 0,
                               "note": "file missing"})
                continue
            exp = EXPECTED.get(d)
            note = ""
            if exp and len(one) != exp:
                note = f"expected {exp}, got {len(one)}"
            if one.attrs.get("missing_stats"):
                note += f" | missing cols: {','.join(one.attrs['missing_stats'])}"
            report.append({"season": s, "division": d, "rows": len(one),
                           "note": note})
            frames.append(one)

    if not frames:
        raise RuntimeError("no data loaded — run download_all() first")

    m = pd.concat(frames, ignore_index=True)

    # ---- assertions. Each of these has a specific failure it is watching for.
    before = len(m)
    m = m[m["date"].notna()]
    lost_dates = before - len(m)

    for c in ("HY", "AY", "HR", "AR", "HF", "AF", "HC", "AC",
              "HS", "AS", "HST", "AST", "FTHG", "FTAG",
              "odds_h", "odds_d", "odds_a", "odds_o25", "odds_u25"):
        if c in m.columns:
            m[c] = pd.to_numeric(m[c], errors="coerce")

    # Match identity is (season, division, date, home, away) — NOT the stat
    # values. Deduplicating on values collapsed 26,719 matches into 1,254 in an
    # earlier draft of this analysis, because thousands of matches share the
    # same scoreline and card count.
    key = ["season", "division", "date", "HomeTeam", "AwayTeam"]
    before = len(m)
    m = m.drop_duplicates(subset=key)
    dupes = before - len(m)

    m["yellows"] = m["HY"] + m["AY"]
    m["reds"] = m["HR"].fillna(0) + m["AR"].fillna(0)
    m["fouls"] = m["HF"] + m["AF"]
    m["corners"] = m["HC"] + m["AC"]
    m["shots"] = m["HS"] + m["AS"]
    m["Referee_raw"] = m["Referee"].astype(str).str.strip().replace(
        {"nan": np.nan, "": np.nan})
    canon = m["Referee_raw"].map(canonical_referee)
    fixes = merge_referee_typos(canon.dropna())
    m["Referee"] = canon.replace(fixes)
    m.attrs["referee_variants_before"] = int(m["Referee_raw"].nunique())
    m.attrs["referee_after_canonical"] = int(canon.nunique())
    m.attrs["referee_after_typo_merge"] = int(m["Referee"].nunique())
    m.attrs["typo_merges"] = fixes

    m = m.sort_values(["date", "division", "HomeTeam"],
                      kind="mergesort").reset_index(drop=True)

    m.attrs["report"] = pd.DataFrame(report)
    m.attrs["lost_to_bad_dates"] = lost_dates
    m.attrs["duplicates_dropped"] = dupes
    return m


# football-data publishes a match's result and most of its stats promptly, but
# CARD counts lag by a few days. Until they arrive the row reads 0 yellows --
# not blank, zero -- while shots, corners and fouls are already correct.
#
# Measured on 2026/27: the 0-yellow rate is 2.2% over the season so far, 6.7%
# across the last 30 matches and 20.0% across the last 10, with mean fouls
# unchanged at 23.5. The cards are missing, not absent.
#
# Cards are this project's TARGET, so a spurious zero is a corrupted label, not
# a missing feature. It cannot be detected from the row itself: 0 yellows with
# 20+ fouls is unusual but genuinely happens in 4.9% of settled matches, so any
# foul-based rule throws away real data. Recency is the only reliable signal.
CARD_CONFIRMATION_LAG_DAYS = 10


def flag_unconfirmed_cards(m: pd.DataFrame,
                           lag_days: int = CARD_CONFIRMATION_LAG_DAYS,
                           asof: pd.Timestamp | None = None) -> pd.DataFrame:
    """Mark matches whose card counts may not have been filled in yet.

    Adds `cards_confirmed`. Not dropped, because for a settled match a zero is
    real and for a fresh one you may simply want to wait.

    MEASURED FROM TODAY, NOT FROM THE NEWEST MATCH IN THE DATA

    This used to ask whether a match was `lag_days` behind the data frontier.
    That tracks elapsed time only while fixtures keep arriving. Across the
    September 2026 international break it stopped: the last matches were
    played on the 26th, the next were not until 10 October, so the frontier
    froze and matches played on 20 September would not have counted as
    confirmed until the middle of October -- three weeks after the fact, for
    rows that were demonstrably complete.

    Demonstrably, because two independent sources agreed on all six of them,
    fouls included. What the rule is actually trying to measure is how long
    football-data has had to fill the row in, and that is elapsed time.

    The default is 10 days rather than the old 7 to keep the same strictness
    this rule had in normal weeks: football-data usually runs 3 to 4 days
    behind, so "7 days behind the frontier" was about 10 days behind today.
    This is a faithful translation of the old behaviour, not a loosening.
    """
    out = m.copy()
    now = pd.Timestamp.today().normalize() if asof is None else pd.Timestamp(asof)
    out["cards_confirmed"] = out["date"] <= now - pd.Timedelta(days=lag_days)
    return out


def audit(m: pd.DataFrame) -> None:
    """Print what the loader kept, lost, and could not explain."""
    rep = m.attrs["report"]
    print(f"matches loaded: {len(m):,}")
    print(f"  seasons {m.season.min()}..{m.season.max()}   "
          f"divisions {sorted(m.division.unique())}")
    print(f"  date range {m.date.min().date()} .. {m.date.max().date()}")
    print(f"  rows dropped for unparseable date : {m.attrs['lost_to_bad_dates']}")
    print(f"  duplicate matches dropped         : {m.attrs['duplicates_dropped']}")

    problems = rep[rep.note != ""]
    print(f"\nseason/division files with anomalies: {len(problems)} of {len(rep)}")
    for _, r in problems.head(12).iterrows():
        print(f"   {r.season} {r.division}: {r.rows:>4} rows   {r.note}")
    if len(problems) > 12:
        print(f"   ... and {len(problems) - 12} more")

    print("\ncompleteness of the columns the model needs:")
    for c in ["yellows", "reds", "fouls", "corners", "shots", "Referee",
              "odds_h", "odds_o25"]:
        print(f"   {c:<9} {m[c].notna().mean():6.2%} populated")

    fl = flag_unconfirmed_cards(m)
    unc = (~fl.cards_confirmed).sum()
    if unc:
        z = fl[~fl.cards_confirmed]
        print(f"\nWARNING: {unc} matches are within "
              f"{CARD_CONFIRMATION_LAG_DAYS} days of the data frontier; their "
              f"card counts may not be filled in yet "
              f"({(z.yellows==0).mean():.0%} show 0 yellows vs "
              f"{(fl[fl.cards_confirmed].yellows==0).mean():.0%} for settled "
              "matches).")
    print(f"\nyellows: mean {m.yellows.mean():.3f}  var {m.yellows.var():.3f}  "
          f"var/mean {m.yellows.var()/m.yellows.mean():.3f}")
    print("per division:")
    for d, g in m.groupby("division"):
        print(f"   {d} {DIVISIONS[d]:<16} n={len(g):>6,}  "
              f"yellows {g.yellows.mean():.2f}  fouls {g.fouls.mean():.1f}  "
              f"refs {g.Referee.nunique():>3}")


if __name__ == "__main__":
    if "--download" in sys.argv:
        print(download_all(refresh="--refresh" in sys.argv))
    m = load_matches(verbose=True)
    audit(m)
