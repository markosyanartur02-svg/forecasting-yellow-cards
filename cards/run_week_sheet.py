"""Add a sheet of upcoming fixtures, with the betting arithmetic wired up.

    .venv/bin/python cards/run_week_sheet.py --from 2026-10-03 --to 2026-10-03
    .venv/bin/python cards/run_week_sheet.py --from 2026-10-09 --to 2026-10-12
    .venv/bin/python cards/run_week_sheet.py --dry-run --from ... --to ...

Fixtures come from football-data's fixtures.csv -- the same site and the same
team spellings as the training data, so nothing has to be matched by name.
It is a rolling file covering roughly the next week, so build a sheet in the
days before its fixtures, not a fortnight ahead. It does not mark
postponements; a postponed fixture simply never gets a result and drops out
of every score.

This used to read BBC Sport's fixtures feed. That was retired along with the
rest of the BBC scraping: BBC's terms prohibit it and its API host disallows
crawlers.

The sheet is named for the range unless --sheet says otherwise, and re-running
the same range updates that sheet in place rather than making another.

Adds a new sheet to forward_test_2026-09-18.xlsx rather than a new file, so
one workbook holds the whole season's running record.

THE UNITS TRAP, WHICH THIS SHEET GUARDS AGAINST

The model predicts YELLOW cards. Most bookmakers price TOTAL cards, counting
a red as two. Over 51,554 matches that is worth about 5-6 percentage points at
every line -- larger than the 5% edge threshold, and always in the same
direction. A yellows model priced against a total-cards line shows a phantom
OVER edge on nearly every fixture, which looks exactly like finding an edge.

So "Settles on" is a required column, and the BET? formula refuses to give a
verdict until it says yellow cards only. The market to look for is the one in
the Chelsea screenshot -- "Yellow Cards", not "Cards" or "Booking Points".

REFEREES

From fixtures.csv itself when football-data has filled it in, otherwise from
cards/referee_appointments.json, which check_referees.py writes when the EFL
publishes them. A fixture with neither falls back to the league average and
says so.

The referee is 48% of the model's value, so a sheet built before the
appointments is the weaker version of the model. Re-run this script once
check_referees.py reports them and every row updates.

ALREADY-PLAYED FIXTURES ARE NEVER REWRITTEN

Re-running this is legitimate while a fixture is still ahead: a referee
appointment published after the first run is new PRE-match information and
genuinely improves the forecast. It stops being legitimate the moment the
match kicks off, because the model refits on everything known so far and
would quietly produce a prediction informed by the result.

So a row for a fixture that has kicked off keeps exactly the cells it already
had -- its prediction, and any prices you had written into it -- and a fixture
that has kicked off without a row never gets one. Kick-off is judged by the
date and UK time in fixtures.csv, so a sheet refreshed on a Saturday evening
still adds Sunday's referees but leaves Saturday's rows alone.

Played rows survive even when fixtures.csv no longer lists the match, which it
stops doing once football-data posts the result. Before that was handled, a
refresh after an update deleted the weekend's rows, prices and all.

This guard exists because it was needed. The 26 September appointments were
published the day before the round and the watcher failed to notice, so that
sheet went into the weekend without referees. Re-running afterwards to "fix"
it would have produced numbers that look like forecasts and are not.

SCORING THIS IS A HOLDOUT LOOK

2026/27 is the holdout. Predicting costs nothing, but reading the results back
is a look and should be logged like any other.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np
from openpyxl import load_workbook
from openpyxl.formula.translate import Translator
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import check_referees as refs
import polite
import teamnames
import freeze as freeze_mod
import loader
import predict as predict_mod

BOOK = os.path.join(os.path.dirname(HERE), "forward_test_2026-09-18.xlsx")
STAKE = 100.0
EDGE_THRESHOLD = 0.05
LINES = (1.5, 2.5, 3.5, 4.5, 5.5, 6.5)

FIXTURES_CSV = "https://www.football-data.co.uk/fixtures.csv"
UK = ZoneInfo("Europe/London")      # football-data gives kick-offs in UK time


def uk_now() -> datetime:
    return datetime.now(UK)


def kicked_off(date: str, time: str, now: datetime) -> bool:
    """Has this fixture started? A same-day fixture with no usable time counts
    as started: a missing time must not let a prediction slip in after
    kick-off."""
    today = now.date().isoformat()
    if date != today:
        return date < today
    try:
        hh, mm = (int(x) for x in time.replace(".", ":").split(":")[:2])
    except (ValueError, AttributeError):
        return True
    return (hh, mm) <= (now.hour, now.minute)


def collect_fixtures(start: dt.date, end: dt.date) -> tuple[list, list]:
    """(fixtures in range, []) for E0-E3, from football-data's fixtures.csv.

    Each fixture is (div, date, home, away, referee, kick-off time). The
    second element is kept for the caller's sake; this source does not
    report postponements.
    """
    import csv
    import io
    text = polite.get(FIXTURES_CSV, encoding="latin-1")
    text = text.lstrip("\ufeff").replace("\u00ef\u00bb\u00bf", "")
    out = []
    for r in csv.DictReader(io.StringIO(text)):
        div = (r.get("Div") or "").strip()
        if div not in loader.DIVISIONS:
            continue
        try:
            d = dt.datetime.strptime(r["Date"].strip(), "%d/%m/%Y").date()
        except (KeyError, ValueError):
            continue
        if not (start <= d <= end):
            continue
        ref = (r.get("Referee") or "").strip()
        out.append((div, d.isoformat(), r["HomeTeam"].strip(),
                    r["AwayTeam"].strip(),
                    loader.canonical_referee(ref) if ref else None,
                    (r.get("Time") or "").strip()))
    out.sort(key=lambda x: (x[1], x[0], x[2]))
    return out, []


def sheet_name(start: dt.date, end: dt.date) -> str:
    if start == end:
        return f"Week {start.strftime('%-d %b')}"
    return f"Week {start.strftime('%-d')}-{end.strftime('%-d %b')}"

def played_rows(path: str, sheet: str,
                started: frozenset = frozenset()) -> dict[tuple[str, str, str], dict]:
    """Whole rows for fixtures dated before today, or in `started`, kept
    verbatim: {(date, home, away): {"row": n, "cells": [(header, value, col)]}}.
    The row and column are where the cell was, so a formula can be moved."""
    if not os.path.exists(path):
        return {}
    wb = load_workbook(path, data_only=False)
    if sheet not in wb.sheetnames:
        return {}
    ws = wb[sheet]
    head = {ws.cell(1, c).value: c for c in range(1, ws.max_column + 1)}
    for need in ("Date", "Home", "Away"):
        if need not in head:
            return {}
    today = uk_now().date().isoformat()
    out = {}
    for r in range(2, ws.max_row + 1):
        date = ws.cell(r, head["Date"]).value
        home = ws.cell(r, head["Home"]).value
        away = ws.cell(r, head["Away"]).value
        if not date or not home or not away:
            continue
        key = (str(date), str(home), str(away))
        if str(date) >= today and key not in started:
            continue
        out[key] = {"row": r, "cells": [
            (ws.cell(1, c).value, ws.cell(r, c).value, c)
            for c in range(1, ws.max_column + 1)]}
    return out


def appointed_referees(known: list[str]) -> dict[tuple[str, str], str]:
    """(home, away) in football-data spelling -> referee, from the last EFL
    article seen. The EFL writes full club names, so they are resolved."""
    rec = refs.load_state()
    out = {}
    for f in rec.get("fixtures", []):
        if not f.get("referee"):
            continue
        h = teamnames.resolve(f["home"], known)
        a = teamnames.resolve(f["away"], known)
        if h and a:
            out[(h, a)] = f["referee"]
    return out


FONT = "Arial"
HEAD = PatternFill("solid", fgColor="1F3864")
GROUP = PatternFill("solid", fgColor="2E5C9A")
FILLIN = PatternFill("solid", fgColor="FFF2CC")     # you fill these in
CALC = PatternFill("solid", fgColor="E2EFDA")       # computed for you
WARN = PatternFill("solid", fgColor="FCE4D6")
THIN = Side(style="thin", color="BFBFBF")
BORDER = Border(bottom=THIN)


def build_columns() -> list[tuple[str, int, str]]:
    """(header, width, kind) where kind drives colour and number format."""
    c: list[tuple[str, int, str]] = [
        ("Date", 11, "plain"), ("Div", 5, "plain"), ("League", 14, "plain"),
        ("Home", 20, "plain"), ("Away", 20, "plain"),
        ("Referee", 16, "fill"), ("Ref matches", 11, "plain"),
        ("Ref vs avg", 10, "pct1"),
        ("League avg", 10, "num2"), ("Model expects", 13, "num2"),
        ("vs league avg", 13, "pctsigned"),
        ("Most likely", 11, "plain"), ("80% range", 10, "plain"),
    ]
    c += [(f"P({k})", 7, "pct0") for k in range(9)]
    c += [(f"P(o{ln})", 8, "pct1") for ln in LINES]
    c += [(f"FAIR over {ln}", 12, "num2") for ln in LINES]
    c += [(f"FAIR under {ln}", 13, "num2") for ln in LINES]
    c += [
        ("Bookmaker", 13, "fill"), ("Settles on", 17, "fill"),
        ("Card line", 9, "fill"), ("Over price", 10, "fill"),
        ("Under price", 11, "fill"), ("Stake EUR", 10, "fill"),
        ("Model P(over)", 13, "calcpct"), ("Implied P(over)", 14, "calcpct"),
        ("Edge", 8, "calcpct"), ("BET?", 10, "calc"),
        ("EV EUR", 9, "calcnum2"),
        ("ACTUAL yellows", 14, "fill"),
        ("Won?", 8, "calc"), ("P/L EUR", 9, "calcnum2"),
        ("Running P/L", 12, "calcnum2"),
        ("Notes", 34, "fill"),
    ]
    return c


FMT = {"pct0": "0%", "pct1": "0.0%", "pctsigned": "+0.0%;-0.0%",
       "num2": "0.00", "calcnum2": "0.00", "calcpct": "0.0%"}


def interval(pmf: np.ndarray, mass: float = 0.80) -> str:
    """Shortest run of counts holding `mass` of the probability."""
    n = len(pmf)
    best = None
    for lo in range(n):
        s = 0.0
        for hi in range(lo, n):
            s += pmf[hi]
            if s >= mass:
                if best is None or (hi - lo) < (best[1] - best[0]):
                    best = (lo, hi)
                break
    return f"{best[0]}-{best[1]}" if best else ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--file", default=BOOK)
    ap.add_argument("--sheet", default=None)
    ap.add_argument("--from", dest="start", required=True)
    ap.add_argument("--to", dest="end", required=True)
    a = ap.parse_args()
    start = dt.date.fromisoformat(a.start)
    end = dt.date.fromisoformat(a.end)
    sheet = a.sheet or sheet_name(start, end)

    print("checking the model is still frozen ...")
    if not freeze_mod.do_verify():
        print("\nREFUSING. The specification has changed since the freeze, so "
              "these predictions would not be the model the forward test is "
              "measuring. Re-freeze if the change was cosmetic.")
        return 1
    fp = freeze_mod.load_freeze()["fingerprint"]

    print(f"\nfixtures for {start} .. {end} ...", flush=True)
    FIXTURES, postponed = collect_fixtures(start, end)
    if not FIXTURES:
        print("no playable fixtures in that range.")
        return 1
    print(f"  {len(FIXTURES)} fixtures, "
          f"{sum(1 for f in FIXTURES if f[4])} with a referee in the file")

    print("\nfitting model ...", flush=True)
    p = predict_mod.CardPredictor().fit(verbose=True)

    known = sorted(set(p.matches.HomeTeam) | set(p.matches.AwayTeam)) \
        if hasattr(p, "matches") else None
    if known is None:
        m = loader.load_matches(unlock_holdout=True)
        known = sorted(set(m.HomeTeam) | set(m.AwayTeam))

    appointed = appointed_referees(known)
    if appointed:
        rec = refs.load_state()
        print(f"\nreferee appointments: {len(appointed)} fixtures from "
              f"{rec.get('slug', '?')} (published {rec.get('published', '?')})")
    else:
        print("\nno referee appointments on file yet -- run "
              "cards/check_referees.py")

    rows, unresolved, started = [], [], set()
    n_appointed = 0
    now = uk_now()
    for div, date, home, away, ref, kickoff in FIXTURES:
        h = teamnames.resolve(home, known)
        w = teamnames.resolve(away, known)
        if not h or not w:
            unresolved.append((home, away, h, w))
            continue
        if kicked_off(date, kickoff, now):
            started.add((date, h, w))
            continue
        if ref is None:
            ref = appointed.get((home, away))
        if ref:
            n_appointed += 1
        rows.append({"division": div, "date": date, "home": h, "away": w,
                     "referee": ref})
    if unresolved:
        print(f"\nABORTING: {len(unresolved)} team name(s) unresolved")
        for u in unresolved:
            print("   ", u)
        return 1
    if started:
        print(f"{len(started)} fixture(s) have already kicked off -- no new "
              "prediction is made for them")
    if not rows:
        print("nothing left to predict in that range.")
        return 1
    print(f"{n_appointed} of {len(rows)} fixtures to predict have a referee")

    print(f"\npredicting {len(rows)} fixtures ...\n", flush=True)
    preds = []
    for r in rows:
        res = p.predict(r["home"], r["away"], r["division"], r["date"],
                        referee=r.get("referee"), odds=None)
        pmf = np.array([float(res["pmf"][k]) for k in range(9)])
        cdf = res["cdf"]
        ref_rel = (float(res["row"].ref_yel10.iloc[0])
                   if res["ref_known"] else None)
        preds.append({
            **r,
            "mu": res["mu"], "baseline": res["baseline"], "pmf": pmf,
            "over": [float(1.0 - cdf[int(np.floor(ln))]) for ln in LINES],
            "ref_n": int(res["ref_n_career"]), "ref_rel": ref_rel,
            "mode": int(np.argmax(pmf)), "iv": interval(pmf),
        })
        print(f"   {r['division']} {r['home']:<18} v {r['away']:<18} "
              f"mu {res['mu']:.2f}  (league avg {res['baseline']:.2f}, "
              f"{res['mu'] / res['baseline'] - 1:+.1%})  "
              f"P(o3.5) {1.0 - cdf[3]:.1%}   "
              f"{'ref ' + str(r['referee']) if r['referee'] else 'no referee'}")

    if a.dry_run:
        print("\ndry run -- nothing written")
        return 0

    write(a.file, sheet, preds, fp, frozenset(started))
    return 0


def write(path: str, sheet: str, preds: list[dict], fingerprint: str,
          started: frozenset = frozenset()) -> None:
    frozen = played_rows(path, sheet, started)
    if frozen:
        print(f"\n{len(frozen)} fixture(s) have already kicked off -- their "
              "existing rows are kept exactly as they are:")
        for d, h, a in sorted(frozen):
            print(f"   {d} {h} v {a}")

    # Every row of the new sheet, in date order: a prediction, or a kept row.
    # A kept row stays even when fixtures.csv no longer lists the match.
    def div_of(kept):
        return next((v for name, v, _ in kept["cells"] if name == "Div"), "") or ""
    entries = [((q["date"], q["division"], q["home"]), q, None) for q in preds
               if (q["date"], q["home"], q["away"]) not in frozen]
    entries += [((d, div_of(k), h), None, k) for (d, h, a), k in frozen.items()]
    entries.sort(key=lambda e: e[0])

    wb = load_workbook(path)
    if sheet in wb.sheetnames:
        del wb[sheet]
    ws = wb.create_sheet(sheet)

    cols = build_columns()
    C = {name: i for i, (name, _, _) in enumerate(cols, start=1)}
    L = {name: get_column_letter(i) for name, i in C.items()}

    for name, width, kind in cols:
        i = C[name]
        cell = ws.cell(1, i, name)
        cell.font = Font(FONT, bold=True, color="FFFFFF", size=9)
        cell.fill = GROUP if kind in ("fill",) else HEAD
        cell.alignment = Alignment(horizontal="center", vertical="center",
                                   wrap_text=True)
        ws.column_dimensions[get_column_letter(i)].width = width
    ws.row_dimensions[1].height = 34
    ws.freeze_panes = "F2"

    first_row = 2
    for n, (_, q, keep) in enumerate(entries):
        i = first_row + n
        if keep:
            for name, value, col in keep["cells"]:
                if name not in C:
                    continue
                if isinstance(value, str) and value.startswith("=") \
                        and (col, keep["row"]) != (C[name], i):
                    # The row may have moved; its formulas must point at
                    # their own row, not at whatever now sits where it was.
                    value = Translator(
                        value, origin=f"{get_column_letter(col)}{keep['row']}"
                    ).translate_formula(f"{L[name]}{i}")
                ws.cell(i, C[name], value)
            for name, _, kind in cols:
                cell = ws.cell(i, C[name])
                cell.font = Font(FONT, size=9)
                cell.border = BORDER
                if kind in FMT:
                    cell.number_format = FMT[kind]
                if kind == "fill":
                    cell.fill = FILLIN
                elif kind.startswith("calc"):
                    cell.fill = CALC
            continue
        put = lambda name, v: ws.cell(i, C[name], v)
        put("Date", q["date"])
        put("Div", q["division"])
        put("League", loader.DIVISIONS[q["division"]])
        put("Home", q["home"])
        put("Away", q["away"])
        put("Referee", q["referee"] or "not known")
        put("Ref matches", q["ref_n"])
        if q["ref_rel"] is not None:
            put("Ref vs avg", q["ref_rel"] - 1.0)
        put("League avg", q["baseline"])
        put("Model expects", q["mu"])
        put("vs league avg", q["mu"] / q["baseline"] - 1.0)
        put("Most likely", q["mode"])
        put("80% range", q["iv"])
        for k in range(9):
            put(f"P({k})", float(q["pmf"][k]))
        for ln, pv in zip(LINES, q["over"]):
            put(f"P(o{ln})", pv)
            put(f"FAIR over {ln}", (1.0 / pv) if pv > 0 else None)
            put(f"FAIR under {ln}", (1.0 / (1.0 - pv)) if pv < 1 else None)
        put("Stake EUR", STAKE)

        f_first, f_last = L[f"P(o{LINES[0]})"], L[f"P(o{LINES[-1]})"]
        line, over, under = L["Card line"], L["Over price"], L["Under price"]
        stake, settles = L["Stake EUR"], L["Settles on"]
        mp, ip, edge = L["Model P(over)"], L["Implied P(over)"], L["Edge"]
        bet, actual = L["BET?"], L["ACTUAL yellows"]
        won = L["Won?"]

        formulas = {
            # the model's own probability for whichever line you recorded
            "Model P(over)":
                f'=IFERROR(IF(${line}{i}="","",'
                f'INDEX(${f_first}{i}:${f_last}{i},${line}{i}-0.5)),"")',
            # two-way de-vig; falls back to the raw implied price if you only
            # wrote one side down, which overstates it by the vig -- so record
            # both sides whenever you can
            "Implied P(over)":
                f'=IF(${over}{i}="","",IF(${under}{i}="",1/${over}{i},'
                f'(1/${over}{i})/((1/${over}{i})+(1/${under}{i}))))',
            "Edge":
                f'=IF(OR(${mp}{i}="",${ip}{i}=""),"",${mp}{i}-${ip}{i})',
            # refuses to advise until the market is known to settle on yellows
            "BET?":
                f'=IF(${edge}{i}="","",'
                f'IF(${settles}{i}="","units?",'
                f'IF(ISNUMBER(SEARCH("yellow",${settles}{i})),'
                f'IF(${edge}{i}>={EDGE_THRESHOLD},"OVER",'
                f'IF(${edge}{i}<=-{EDGE_THRESHOLD},"UNDER","skip")),'
                f'"UNITS?")))',
            "EV EUR":
                f'=IF(OR(${bet}{i}="",${bet}{i}="skip",${bet}{i}="units?",'
                f'${bet}{i}="UNITS?"),"",'
                f'IF(${bet}{i}="OVER",${stake}{i}*(${mp}{i}*${over}{i}-1),'
                f'${stake}{i}*((1-${mp}{i})*${under}{i}-1)))',
            "Won?":
                f'=IF(OR(${bet}{i}="",${bet}{i}="skip",${bet}{i}="units?",'
                f'${bet}{i}="UNITS?",${actual}{i}="",${line}{i}=""),"",'
                f'IF(${bet}{i}="OVER",'
                f'IF(${actual}{i}>${line}{i},"WON","LOST"),'
                f'IF(${actual}{i}<${line}{i},"WON","LOST")))',
            "P/L EUR":
                f'=IF(${won}{i}="","",IF(${won}{i}="WON",'
                f'IF(${bet}{i}="OVER",${stake}{i}*(${over}{i}-1),'
                f'${stake}{i}*(${under}{i}-1)),-${stake}{i}))',
            "Running P/L":
                f'=IF(COUNT(${L["P/L EUR"]}${first_row}:${L["P/L EUR"]}{i})=0,'
                f'"",SUM(${L["P/L EUR"]}${first_row}:${L["P/L EUR"]}{i}))',
        }
        for name, f in formulas.items():
            ws.cell(i, C[name], f)

        for name, _, kind in cols:
            cell = ws.cell(i, C[name])
            cell.font = Font(FONT, size=9)
            cell.border = BORDER
            if kind in FMT:
                cell.number_format = FMT[kind]
            if kind == "fill":
                cell.fill = FILLIN
            elif kind.startswith("calc"):
                cell.fill = CALC

    last = first_row + len(entries) - 1
    s = last + 2
    pl, betc, wonc = L["P/L EUR"], L["BET?"], L["Won?"]
    summary = [
        ("SUMMARY", None, None),
        ("bets placed",
         f'=COUNTIF(${betc}${first_row}:${betc}{last},"OVER")'
         f'+COUNTIF(${betc}${first_row}:${betc}{last},"UNDER")', "0"),
        ("bets won", f'=COUNTIF(${wonc}${first_row}:${wonc}{last},"WON")', "0"),
        ("total staked",
         f'=SUMPRODUCT((${wonc}${first_row}:${wonc}{last}<>"")'
         f'*${L["Stake EUR"]}${first_row}:${L["Stake EUR"]}{last})', "0.00"),
        ("profit / loss EUR",
         f'=IF(COUNT(${pl}${first_row}:${pl}{last})=0,"",'
         f'SUM(${pl}${first_row}:${pl}{last}))', "0.00"),
        ("return on stake",
         f'=IF(${L["Away"]}{s + 3}=0,"",${L["Away"]}{s + 4}/${L["Away"]}{s + 3})',
         "0.0%"),
    ]
    for n, (label, formula, fmt) in enumerate(summary):
        i = s + n
        c1 = ws.cell(i, C["Date"], label)
        c1.font = Font(FONT, size=9, bold=(n == 0))
        if formula:
            c2 = ws.cell(i, C["Away"], formula)
            c2.font = Font(FONT, size=9, bold=True)
            c2.number_format = fmt

    note_row = s + len(summary) + 2
    played = sum(1 for p_ in preds
                 if p_["date"] < dt.date.today().isoformat())
    notes = [
        "HOW TO USE THIS SHEET",
        "",
        "1. Fill the yellow columns only. Everything green is computed.",
        "",
        "2. 'Settles on' must say what the market actually pays out on. Write "
        "'yellow cards' only if the bookmaker settles on yellows alone. If it "
        "says total cards or booking points, BET? shows UNITS? and gives no "
        "verdict -- on purpose.",
        "",
        "   The model predicts yellows. Counting a red as two cards adds about "
        "5-6 percentage points at every line, always upward, which is larger "
        "than the 5% edge threshold. Priced against the wrong market the model "
        "shows an OVER edge on nearly every game, and that looks identical to "
        "finding an edge.",
        "",
        "   The market to look for is 'Yellow Cards', as in the Chelsea "
        "screenshot -- not 'Cards' and not 'Booking Points'.",
        "",
        "3. Record BOTH sides of the price. With one side only, the de-vig "
        "cannot run and the implied probability is overstated by the whole "
        "margin, which invents edge that is not there.",
        "",
        "4. Record the price when you see it, before kick-off. A price looked "
        "up afterwards is not one you could have taken.",
        "",
        "5. Stake is EUR 100 a bet. Change the column if you want; the P/L "
        "follows it.",
        "",
        "WHAT THIS SHEET IS AND IS NOT",
        "",
        f"{len(entries)} fixtures. A sample this size tells you nothing about "
        "whether the model beats a bookmaker -- a verdict on profitability "
        "needs thousands of bets. Treat a single week as practice for the "
        "workflow.",
        "",
        "The referee is 48% of the model's value. If the Referee column says "
        "'not known', the EFL has not published appointments yet (they come "
        "one to three days ahead) and these are the weaker version of the "
        "model. Re-run run_week_sheet.py for the same dates once they are out "
        "and every unplayed row updates.",
        "",
        f"Model fingerprint {fingerprint}, frozen for the season.",
        "Scoring these results against the model is a holdout look.",
    ]
    for n, t in enumerate(notes):
        cell = ws.cell(note_row + n, C["Date"], t)
        cell.font = Font(FONT, size=9,
                         bold=t.isupper() and len(t) > 3)
        cell.alignment = Alignment(wrap_text=False, vertical="top")

    wb.save(path)
    print(f"\nwrote sheet '{sheet}' into {os.path.basename(path)}")
    print(f"  {len(entries)} fixtures ({len(entries) - len(frozen)} predicted "
          f"now, {len(frozen)} kept), stake EUR {STAKE:.0f}, "
          f"edge threshold {EDGE_THRESHOLD:.0%}")
    print("  fill the yellow columns; green ones calculate themselves")


if __name__ == "__main__":
    raise SystemExit(main())
