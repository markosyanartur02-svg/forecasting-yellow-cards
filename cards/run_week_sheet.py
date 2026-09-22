"""Add a sheet of upcoming fixtures, with the betting arithmetic wired up.

    .venv/bin/python cards/run_week_sheet.py            # write the sheet
    .venv/bin/python cards/run_week_sheet.py --dry-run  # print, write nothing

Adds a new sheet to forward_test_2026-09-18.xlsx rather than a new file, so
one workbook holds the whole season's running record.

THIS WEEK IS SMALL, AND NOT BECAUSE ANYTHING BROKE

26 September begins a two-weekend international break. There are no Premier
League or Championship fixtures until 10 October, and seven League One and
League Two matches are postponed for clubs losing players. That leaves
fourteen fixtures, all in tiers three and four.

Fourteen matches is a very small sample. It is fine for practising the
workflow and useless as evidence about the model.

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

Taken from cards/referee_appointments.json when it is there, which
check_referees.py writes as soon as the EFL publishes them. A fixture with no
appointment yet falls back to the league average and says so.

The referee is 48% of the model's value, so a sheet built before the
appointments is the weaker version of the model. Re-run this script once
check_referees.py reports them and every row updates.

SCORING THIS IS A HOLDOUT LOOK

2026/27 is the holdout. Predicting costs nothing, but reading the results back
is a look and should be logged like any other.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone

import numpy as np
from openpyxl import load_workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import check_referees as refs
import fetch_results_bbc as bbc
import freeze as freeze_mod
import loader
import predict as predict_mod

BOOK = os.path.join(os.path.dirname(HERE), "forward_test_2026-09-18.xlsx")
SHEET = "Week 26 Sep"
STAKE = 100.0
EDGE_THRESHOLD = 0.05
LINES = (1.5, 2.5, 3.5, 4.5, 5.5, 6.5)

# Fixtures from the BBC fixtures feed, read 2026-09-21, PreEvent only.
# Names as BBC writes them; resolved onto football-data's spellings below.
#
# Postponed that day and therefore absent: Bradford City v Barnsley, MK Dons
# v Bromley, Mansfield Town v Leyton Orient, Notts County v Blackpool,
# Stevenage v Sheffield Wednesday (all E2), Crawley Town v Barnet and Port
# Vale v Northampton Town (E3).
#
# Referee: fill in when the EFL publishes appointments, then re-run.
FIXTURES = [
    ("E2", "2026-09-26", "Cambridge United", "AFC Wimbledon", None),
    ("E2", "2026-09-26", "Plymouth Argyle", "Burton Albion", None),
    ("E2", "2026-09-26", "Stockport County", "Peterborough United", None),
    ("E2", "2026-09-26", "Wycombe Wanderers", "Reading", None),
    ("E3", "2026-09-26", "Bristol Rovers", "Exeter City", None),
    ("E3", "2026-09-26", "Cheltenham Town", "Chesterfield", None),
    ("E3", "2026-09-26", "Fleetwood Town", "Rochdale", None),
    ("E3", "2026-09-26", "Newport County", "Grimsby Town", None),
    ("E3", "2026-09-26", "Oldham Athletic", "Salford City", None),
    ("E3", "2026-09-26", "Rotherham United", "Crewe Alexandra", None),
    ("E3", "2026-09-26", "Shrewsbury Town", "Colchester United", None),
    ("E3", "2026-09-26", "Swindon Town", "Accrington Stanley", None),
    ("E3", "2026-09-26", "Tranmere Rovers", "Walsall", None),
    ("E3", "2026-09-26", "York City", "Gillingham", None),
]

def appointed_referees() -> dict[tuple[str, str], str]:
    """(home, away) lowercased -> referee, from the last EFL article seen."""
    rec = refs.load_state()
    return {(f["home"].lower(), f["away"].lower()): f["referee"]
            for f in rec.get("fixtures", []) if f.get("referee")}


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
    ap.add_argument("--sheet", default=SHEET)
    a = ap.parse_args()

    print("checking the model is still frozen ...")
    if not freeze_mod.do_verify():
        print("\nREFUSING. The specification has changed since the freeze, so "
              "these predictions would not be the model the forward test is "
              "measuring. Re-freeze if the change was cosmetic.")
        return 1
    fp = freeze_mod.load_freeze()["fingerprint"]

    print("\nfitting model ...", flush=True)
    p = predict_mod.CardPredictor().fit(verbose=True)

    known = sorted(set(p.matches.HomeTeam) | set(p.matches.AwayTeam)) \
        if hasattr(p, "matches") else None
    if known is None:
        m = loader.load_matches(unlock_holdout=True)
        known = sorted(set(m.HomeTeam) | set(m.AwayTeam))

    appointed = appointed_referees()
    if appointed:
        rec = refs.load_state()
        print(f"\nreferee appointments: {len(appointed)} fixtures from "
              f"{rec.get('slug', '?')} (published {rec.get('published', '?')})")
    else:
        print("\nno referee appointments on file yet -- run "
              "cards/check_referees.py")

    rows, unresolved = [], []
    n_appointed = 0
    for div, date, home, away, ref in FIXTURES:
        if ref is None:
            ref = appointed.get((home.lower(), away.lower()))
            if ref:
                n_appointed += 1
        h = bbc.resolve(home, known)
        w = bbc.resolve(away, known)
        if not h or not w:
            unresolved.append((home, away, h, w))
            continue
        rows.append({"division": div, "date": date, "home": h, "away": w,
                     "referee": ref})
    print(f"{n_appointed} of {len(FIXTURES)} fixtures have a referee")
    if unresolved:
        print(f"\nABORTING: {len(unresolved)} team name(s) unresolved")
        for u in unresolved:
            print("   ", u)
        return 1

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

    write(a.file, a.sheet, preds, fp)
    return 0


def write(path: str, sheet: str, preds: list[dict], fingerprint: str) -> None:
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
    for n, q in enumerate(preds):
        i = first_row + n
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

    last = first_row + len(preds) - 1
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
        "WHAT THIS WEEK IS AND IS NOT",
        "",
        "Fourteen fixtures, League One and League Two only -- 26 September "
        "starts a two-weekend international break, and seven more EFL matches "
        "are postponed. No Premier League or Championship football until "
        "10 October.",
        "",
        "Fourteen matches is far too few to tell you anything about whether "
        "the model beats a bookmaker. Treat this as practice for the workflow. "
        "A verdict on profitability needs thousands of bets.",
        "",
        "Referees are not in yet. The EFL publishes appointments one to three "
        "days ahead, and the referee is 48% of the model's value -- so these "
        "numbers are the weaker version of the model. Re-run "
        "cards/run_week_sheet.py once the appointments are out.",
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
    print(f"  {len(preds)} fixtures, stake EUR {STAKE:.0f}, "
          f"edge threshold {EDGE_THRESHOLD:.0%}")
    print("  fill the yellow columns; green ones calculate themselves")


if __name__ == "__main__":
    raise SystemExit(main())
