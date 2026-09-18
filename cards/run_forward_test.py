"""Pre-registered forward test: predict a week of fixtures before they happen.

    .venv/bin/python cards/run_forward_test.py

Writes `forward_test_2026-09-18.xlsx` with one row per fixture, the model's
full card distribution, and empty columns for the actual results.

Why this is the best test available. The holdout has been redrawn once already
and the walk-forward, honest as it is, still scores seasons that existed when
the model was designed. These 44 matches have not been played. The predictions
are written to disk now, the results arrive next week, and nothing about the
model can change in between. That is a pre-registered experiment, and it is
worth more than any number of retrospective re-scorings.

Predicting these fixtures is NOT a holdout look -- nothing is scored here.
Scoring them next week is, and it will be logged as such.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone

import numpy as np
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import features
import fixtures_week as fw
import loader
import models
import predict as predict_mod

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "forward_test_2026-09-18.xlsx")

# Prices you have actually seen, keyed by (home, away) as they appear in our
# data. Filled in by hand as you find them; anything not here is left blank for
# you to complete.
#
# Brentford v Chelsea: "Yellow cards - Total" market, read from the bookmaker
# 2026-09-17. Note this market settles on YELLOW cards specifically, which is
# exactly what the model predicts -- most card markets count a red as two, and
# would not line up.
PRICES = {
    ("Brentford", "Chelsea"): {"book": "screenshot", "line": 3.5,
                               "over": 1.50, "under": 2.34,
                               "note": "also offered o4.5 2.14 / u4.5 1.60"},
}

FONT = "Arial"
HEAD = PatternFill("solid", fgColor="1F3864")
FILLIN = PatternFill("solid", fgColor="FFF2CC")     # you fill these in
THIN = Side(style="thin", color="BFBFBF")
BORDER = Border(bottom=THIN)


def frozen_predictions(path: str) -> dict:
    """Predictions already written for fixtures that have since been played.

    A pre-registered test is worthless if re-running the script quietly
    rewrites predictions for matches whose results now exist. Re-running is
    legitimate for a fixture still in the future -- a referee appointment
    published after the first run is new PRE-match information and improves
    the forecast. It is not legitimate once the match has finished.

    So any fixture dated before today keeps exactly the numbers it was first
    given, read back from the existing workbook.
    """
    if not os.path.exists(path):
        return {}
    from openpyxl import load_workbook
    ws = load_workbook(path, data_only=False)["Predictions"]
    head = {ws.cell(1, c).value: c for c in range(1, ws.max_column + 1)}
    today = datetime.now().date().isoformat()
    out = {}
    for r in range(2, ws.max_row + 1):
        home = ws.cell(r, head["Home"]).value
        date = ws.cell(r, head["Date"]).value
        if not home or home == "Example FC" or not date:
            continue
        if str(date) >= today:
            continue                      # still ahead of us; may be redone
        away = ws.cell(r, head["Away"]).value
        out[(home, away)] = {
            "mu": ws.cell(r, head["Model expects"]).value,
            "baseline": ws.cell(r, head["League avg"]).value,
            "over": {ln: ws.cell(r, head[f"P(o{ln})"]).value
                     for ln in (1.5, 2.5, 3.5, 4.5, 5.5, 6.5)},
            "p": [ws.cell(r, head[f"P({k})"]).value for k in range(9)],
            "ref_used": ws.cell(r, head["Referee used"]).value or "",
            "ref_n": ws.cell(r, head["Ref matches"]).value or 0,
            "ref_rel": ws.cell(r, head["Ref x avg"]).value,
        }
    return out


def main() -> int:
    print("fitting model ...", flush=True)
    p = predict_mod.CardPredictor().fit(verbose=True)
    rows, unresolved = fw.resolve_all(p)
    if unresolved:
        print(f"ABORTING: {len(unresolved)} team names unresolved")
        return 1

    frozen = frozen_predictions(OUT)
    if frozen:
        print(f"{len(frozen)} fixture(s) already played -- their original "
              f"predictions are kept unchanged:")
        for h, a in frozen:
            print(f"   {h} v {a}")
    print(f"predicting {len(rows)} fixtures ...", flush=True)
    preds = []
    for r in rows:
        key = (r["home"], r["away"])
        if key in frozen:
            fz = frozen[key]
            preds.append({**r, **fz,
                          "ref_rel": (fz["ref_rel"] if fz["ref_rel"] is not None
                                      else float("nan"))})
            print(f"   {r['division']} {r['home']} v {r['away']}: "
                  f"mu {fz['mu']:.2f}  [FROZEN, already played]", flush=True)
            continue
        res = p.predict(r["home"], r["away"], r["division"], r["date"],
                        referee=r.get("referee"), odds=None)
        cdf = res["cdf"]
        preds.append({
            **r,
            "mu": res["mu"],
            "baseline": res["baseline"],
            "p": [float(res["pmf"][k]) for k in range(9)],
            "over": {ln: float(1.0 - cdf[int(np.floor(ln))])
                     for ln in (1.5, 2.5, 3.5, 4.5, 5.5, 6.5)},
            "ref_used": (r.get("referee") or ""),
            "ref_n": int(res["ref_n_career"]),
            "ref_rel": (float(res["row"].ref_yel10.iloc[0])
                        if res["ref_known"] else float("nan")),
        })
        tag = (f"ref {r['referee']} ({res['ref_n_career']})"
               if r.get("referee") else "no referee")
        print(f"   {r['division']} {r['home']} v {r['away']}: "
              f"mu {res['mu']:.2f}  [{tag}]", flush=True)

    write_workbook(preds, p)
    print(f"\nwrote {OUT}")
    return 0


def write_workbook(preds: list[dict], predictor) -> None:
    """One row per fixture.

    Probabilities are shown twice: as percentages, and as FAIR DECIMAL ODDS
    (1 / probability). The second form is what a bookmaker shows you, so the
    comparison becomes a direct one -- if their price is longer than the
    model's fair price, that side is value. No mental arithmetic required.
    """
    LINES = (1.5, 2.5, 3.5, 4.5, 5.5, 6.5)
    wb = Workbook()
    ws = wb.active
    ws.title = "Predictions"

    headers = (
        [("Date", 11), ("Div", 6), ("League", 15), ("Home", 16), ("Away", 16),
         ("League avg", 10), ("Model expects", 12)]
        + [(f"P(o{ln})", 9) for ln in LINES]
        + [(f"FAIR over {ln}", 11) for ln in LINES]
        + [(f"FAIR under {ln}", 12) for ln in LINES]
        + [(f"P({k})", 7) for k in range(9)]
        + [("Referee used", 14), ("Ref x avg", 9), ("Ref matches", 10)]
        + [("ACTUAL yellows", 13), ("Bookmaker", 12), ("Card line", 9),
           ("Over price", 10), ("Under price", 10)]
        + [("Model P(over)", 12), ("Model FAIR over", 13),
           ("Model FAIR under", 13), ("Implied P(over)", 13), ("Edge", 8),
           ("BET?", 9), ("EV", 8), ("Won?", 8), ("P/L units", 9),
           ("Notes", 20)]
    )
    C = {name: i for i, (name, _) in enumerate(headers, 1)}
    for c, (name, width) in enumerate(headers, 1):
        cell = ws.cell(1, c, name)
        cell.font = Font(name=FONT, bold=True, color="FFFFFF", size=9)
        cell.fill = HEAD
        cell.alignment = Alignment(horizontal="center", wrap_text=True,
                                   vertical="center")
        ws.column_dimensions[get_column_letter(c)].width = width
    ws.row_dimensions[1].height = 34
    ws.freeze_panes = "F2"

    FILL_COLS = {C["ACTUAL yellows"], C["Bookmaker"], C["Card line"],
                 C["Over price"], C["Under price"], C["Notes"]}
    PCT_COLS = ({C[f"P(o{ln})"] for ln in LINES}
                | {C[f"P({k})"] for k in range(9)})
    ODDS_COLS = ({C[f"FAIR over {ln}"] for ln in LINES}
                 | {C[f"FAIR under {ln}"] for ln in LINES})

    for i, r in enumerate(preds, start=2):
        vals = ([r["date"], r["division"], loader.DIVISIONS[r["division"]],
                 r["home"], r["away"], r["baseline"], r["mu"]]
                + [r["over"][ln] for ln in LINES]
                + [fair(r["over"][ln]) for ln in LINES]
                + [fair(1.0 - r["over"][ln]) for ln in LINES]
                + list(r["p"])
                + [r["ref_used"] or "not known",
                   (None if not r["ref_used"] or not np.isfinite(r["ref_rel"])
                    else r["ref_rel"]),
                   (r["ref_n"] if r["ref_used"] else None)])
        for c, v in enumerate(vals, 1):
            cell = ws.cell(i, c, v)
            cell.font = Font(name=FONT, size=9)
            cell.border = BORDER
            if c in (C["League avg"], C["Model expects"], C["Ref x avg"]):
                cell.number_format = "0.00"
            elif c in PCT_COLS:
                cell.number_format = "0.0%"
            elif c in ODDS_COLS:
                cell.number_format = "0.00"
            if c >= C["League avg"]:
                cell.alignment = Alignment(horizontal="center")
        for c in FILL_COLS:
            cell = ws.cell(i, c)
            cell.fill = FILLIN
            cell.font = Font(name=FONT, size=9)
            cell.border = BORDER
        known = PRICES.get((r["home"], r["away"]))
        if known:
            ws.cell(i, C["Bookmaker"], known["book"])
            ws.cell(i, C["Card line"], known["line"])
            ws.cell(i, C["Over price"], known["over"])
            ws.cell(i, C["Under price"], known["under"])
            ws.cell(i, C["Notes"], known.get("note", ""))
            for c in FILL_COLS:
                ws.cell(i, c).font = Font(name=FONT, size=9, bold=True)

        L_ = {k: get_column_letter(C[k]) for k in
              ("ACTUAL yellows", "Card line", "Over price", "Under price",
               "Model P(over)", "Model FAIR over", "Model FAIR under",
               "Implied P(over)", "Edge", "BET?", "EV", "Won?", "P/L units")}
        first, lastp = get_column_letter(C["P(o1.5)"]), get_column_letter(C["P(o6.5)"])
        fm = {
            "Model P(over)":
                f'=IFERROR(IF(${L_["Card line"]}{i}="","",'
                f'INDEX(${first}{i}:${lastp}{i},${L_["Card line"]}{i}-0.5)),'
                f'"line must be 1.5-6.5")',
            "Model FAIR over":
                f'=IF(ISNUMBER(${L_["Model P(over)"]}{i}),'
                f'1/${L_["Model P(over)"]}{i},"")',
            "Model FAIR under":
                f'=IF(ISNUMBER(${L_["Model P(over)"]}{i}),'
                f'1/(1-${L_["Model P(over)"]}{i}),"")',
            "Implied P(over)":
                f'=IF(${L_["Over price"]}{i}="","",'
                f'IF(${L_["Under price"]}{i}="",1/${L_["Over price"]}{i},'
                f'(1/${L_["Over price"]}{i})/((1/${L_["Over price"]}{i})'
                f'+(1/${L_["Under price"]}{i}))))',
            "Edge":
                f'=IF(OR(NOT(ISNUMBER(${L_["Model P(over)"]}{i})),'
                f'${L_["Implied P(over)"]}{i}=""),"",'
                f'${L_["Model P(over)"]}{i}-${L_["Implied P(over)"]}{i})',
            "BET?":
                f'=IF(${L_["Edge"]}{i}="","",IF(${L_["Edge"]}{i}>=0.05,"OVER",'
                f'IF(${L_["Edge"]}{i}<=-0.05,"UNDER","skip")))',
            "EV":
                f'=IF(${L_["BET?"]}{i}="OVER",'
                f'${L_["Model P(over)"]}{i}*${L_["Over price"]}{i}-1,'
                f'IF(${L_["BET?"]}{i}="UNDER",'
                f'IF(${L_["Under price"]}{i}="","need under price",'
                f'(1-${L_["Model P(over)"]}{i})*${L_["Under price"]}{i}-1),""))',
            "Won?":
                f'=IF(OR(${L_["BET?"]}{i}="",${L_["BET?"]}{i}="skip",'
                f'${L_["ACTUAL yellows"]}{i}="",${L_["Card line"]}{i}=""),"",'
                f'IF(${L_["BET?"]}{i}="OVER",'
                f'IF(${L_["ACTUAL yellows"]}{i}>${L_["Card line"]}{i},"WON","LOST"),'
                f'IF(${L_["ACTUAL yellows"]}{i}<${L_["Card line"]}{i},"WON","LOST")))',
            "P/L units":
                f'=IF(${L_["Won?"]}{i}="","",IF(${L_["Won?"]}{i}="WON",'
                f'IF(${L_["BET?"]}{i}="OVER",${L_["Over price"]}{i}-1,'
                f'${L_["Under price"]}{i}-1),-1))',
        }
        for name, formula in fm.items():
            cell = ws.cell(i, C[name], formula)
            cell.font = Font(name=FONT, size=9)
            cell.border = BORDER
            cell.alignment = Alignment(horizontal="center")
            if name in ("Model P(over)", "Implied P(over)", "Edge", "EV"):
                cell.number_format = "0.0%"
            elif name in ("Model FAIR over", "Model FAIR under"):
                cell.number_format = "0.00"
            elif name == "P/L units":
                cell.number_format = "+0.00;-0.00;0.00"

    last = len(preds) + 1
    write_example_row(ws, C, last + 2)
    write_scoring(wb, C, last)
    write_readme(wb, predictor)
    wb.save(OUT)


def write_example_row(ws, C, row: int) -> None:
    """One filled-in row showing the format, clearly marked as not real.

    Placed below the data with a gap so it cannot be mistaken for a fixture,
    and excluded from every formula range on the Scoring tab.
    """
    grey = PatternFill("solid", fgColor="EDEDED")
    ws.cell(row - 1, 4, "EXAMPLE OF A COMPLETED ROW (not a real fixture, "
                        "not counted anywhere)").font = Font(
        name=FONT, size=9, bold=True, italic=True)
    demo = {
        "Date": "2026-09-19", "Div": "E0", "League": "Premier League",
        "Home": "Example FC", "Away": "Sample United",
        "League avg": 3.87, "Model expects": 4.40,
        "P(o3.5)": 0.63, "FAIR over 3.5": 1.59, "FAIR under 3.5": 2.70,
        "Referee used": "a referee", "Ref x avg": 1.20, "Ref matches": 200,
        "ACTUAL yellows": 5, "Bookmaker": "Bet365", "Card line": 3.5,
        "Over price": 1.80, "Under price": 2.05,
        "Model P(over)": 0.63, "Model FAIR over": 1.59,
        "Model FAIR under": 2.70, "Implied P(over)": 0.532,
        "Edge": 0.098, "BET?": "OVER", "EV": 0.134, "Won?": "WON",
        "P/L units": 0.80,
        "Notes": "5 yellows > 3.5, so the OVER won at 1.80: +0.80 units",
    }
    for name, val in demo.items():
        cell = ws.cell(row, C[name], val)
        cell.font = Font(name=FONT, size=9, italic=True)
        cell.fill = grey
        cell.alignment = Alignment(horizontal="center")
        if name in ("P(o3.5)", "Model P(over)", "Implied P(over)", "Edge", "EV"):
            cell.number_format = "0.0%"
        elif name in ("FAIR over 3.5", "FAIR under 3.5", "Model FAIR over",
                      "Model FAIR under", "League avg", "Model expects",
                      "Ref x avg", "Over price", "Under price", "Card line"):
            cell.number_format = "0.00"
        elif name == "P/L units":
            cell.number_format = "+0.00;-0.00;0.00"
    ws.cell(row, C["Notes"]).alignment = Alignment(horizontal="left")


def fair(p: float) -> float | None:
    """Probability -> fair decimal odds. 0.50 -> 2.00, 0.25 -> 4.00."""
    if p is None or not np.isfinite(p) or p <= 0:
        return None
    return float(1.0 / p)


def write_scoring(wb, C, last: int) -> None:
    sc = wb.create_sheet("Scoring")
    sc.column_dimensions["A"].width = 32
    sc.column_dimensions["B"].width = 13
    sc.column_dimensions["C"].width = 62
    g = get_column_letter
    A, BET, WON, PL = (g(C["ACTUAL yellows"]), g(C["BET?"]),
                       g(C["Won?"]), g(C["P/L units"]))
    W = f"Predictions!${A}$2:${A}${last}"
    mask, nf = f"--ISNUMBER({W})", f"COUNT({W})"

    def mm(col):
        cl = g(C[col])
        return f"SUMPRODUCT({mask},Predictions!${cl}$2:${cl}${last})/{nf}"

    def guard(e):
        return f'=IF({nf}=0,"",{e})'

    nbets = (f'COUNTIF(Predictions!${BET}$2:${BET}${last},"OVER")'
             f'+COUNTIF(Predictions!${BET}$2:${BET}${last},"UNDER")')
    nsettled = (f'COUNTIF(Predictions!${WON}$2:${WON}${last},"WON")'
                f'+COUNTIF(Predictions!${WON}$2:${WON}${last},"LOST")')
    nwon = f'COUNTIF(Predictions!${WON}$2:${WON}${last},"WON")'

    rows = [
        ("Metric", "Value", "What it means"),
        ("Matches predicted", f'=COUNTA(Predictions!$D$2:$D${last})', ""),
        ("Results entered", f"={nf}", "Fill the yellow ACTUAL column in."),
        ("Mean predicted (filled rows)", guard(mm("Model expects")),
         "Averaged over only the matches you have entered."),
        ("Mean actual", guard(f"AVERAGE({W})"), ""),
        ("Bias (actual - predicted)",
         guard(f'AVERAGE({W})-{mm("Model expects")}'),
         "Positive = the model under-predicts cards."),
        ("Mean absolute error",
         guard(f'SUMPRODUCT({mask},ABS({W}-Predictions!$G$2:$G${last}))/{nf}'),
         "Expect ~1.5. Match cards are genuinely noisy."),
        ("", "", ""),
        ("CALIBRATION -- THE NUMBER THAT MATTERS", "", ""),
        ("Predicted over 3.5", guard(mm("P(o3.5)")), ""),
        ("Actual share over 3.5", guard(f'COUNTIF({W},">3.5")/{nf}'), ""),
        ("Calibration gap",
         guard(f'{mm("P(o3.5)")}-COUNTIF({W},">3.5")/{nf}'),
         "Near zero = well calibrated. This is the whole test."),
        ("Predicted over 2.5", guard(mm("P(o2.5)")), ""),
        ("Actual share over 2.5", guard(f'COUNTIF({W},">2.5")/{nf}'), ""),
        ("Predicted over 4.5", guard(mm("P(o4.5)")), ""),
        ("Actual share over 4.5", guard(f'COUNTIF({W},">4.5")/{nf}'), ""),
        ("", "", ""),
        ("BETTING -- ONLY IF YOU ENTERED PRICES", "", ""),
        ("Bets the model would place", f"={nbets}",
         "Edge of 5 points or more. Most fixtures should be skips."),
        ("  of which OVER",
         f'=COUNTIF(Predictions!${BET}$2:${BET}${last},"OVER")', ""),
        ("  of which UNDER",
         f'=COUNTIF(Predictions!${BET}$2:${BET}${last},"UNDER")',
         "A model below the price is an edge too."),
        ("Bets settled", f"={nsettled}", ""),
        ("Bets won", f"={nwon}", ""),
        ("Strike rate", f'=IF({nsettled}=0,"",{nwon}/({nsettled}))',
         "Means nothing on its own -- a 40% strike rate at long prices wins."),
        ("Profit / loss (units)",
         f'=IF(COUNT(Predictions!${PL}$2:${PL}${last})=0,"",'
         f'SUM(Predictions!${PL}$2:${PL}${last}))',
         "One unit staked per bet. This is the number that matters."),
        ("ROI",
         f'=IF({nsettled}=0,"",SUM(Predictions!${PL}$2:${PL}${last})/({nsettled}))',
         "Over one weekend this is NOISE. A few hundred bets before it means "
         "anything."),
    ]
    for i, (a, b, c) in enumerate(rows, 1):
        sc.cell(i, 1, a).font = Font(name=FONT, size=10,
                                     bold=(i == 1 or a.isupper() and a))
        sc.cell(i, 2, b).font = Font(name=FONT, size=10, bold=(i == 1))
        sc.cell(i, 3, c).font = Font(name=FONT, size=10, italic=(i != 1))
        if i == 1:
            for col in (1, 2, 3):
                sc.cell(i, col).fill = HEAD
                sc.cell(i, col).font = Font(name=FONT, bold=True,
                                            color="FFFFFF", size=10)
        if a.startswith(("Predicted", "Actual share", "Calibration")) \
                or a in ("Strike rate", "ROI"):
            sc.cell(i, 2).number_format = "0.0%"
        elif a.startswith(("Mean", "Bias")):
            sc.cell(i, 2).number_format = "0.00"
        elif a.startswith("Profit"):
            sc.cell(i, 2).number_format = "+0.00;-0.00;0.00"


def write_readme(wb, predictor) -> None:
    rm = wb.create_sheet("Read me")
    rm.column_dimensions["A"].width = 100
    text = [
        ("HOW TO USE THIS FILE", True),
        ("", False),
        ("Yellow columns are yours to fill in. Everything else calculates.",
         False),
        ("  ACTUAL yellows -- both teams combined, YELLOW cards only.", False),
        ("  Bookmaker / Card line / Over price / Under price -- what you were "
         "offered.", False),
        ("Give BOTH prices where you can: with only one, the margin cannot be "
         "removed and your edge looks smaller than it is.", False),
        ("", False),
        ("READING THE ODDS COLUMNS", True),
        ("Every probability is also shown as FAIR DECIMAL ODDS, which is "
         "1 / probability -- the same form a bookmaker quotes.", False),
        ("A 50% chance is 2.00. A 67% chance is 1.50. A 25% chance is 4.00.",
         False),
        ("The comparison is then direct: if the bookmaker's price is LONGER "
         "than the model's fair price, that side is value.", False),
        ("Example: model fair under 3.5 = 2.11, bookmaker offers 2.34. "
         "2.34 > 2.11, so the under is value.", False),
        ("", False),
        ("WHAT WAS PREDICTED, AND WITH WHAT", True),
        (f"Negative binomial GLM, 38 features, trained on "
         f"{len(predictor.history_):,} matches to the end of 2025/26.", False),
        ("Premier League fixtures carry their real referee appointment. The "
         "EFL had not published its appointments when this was generated, so "
         "those 36 fixtures ran with the referee unknown.", False),
        ("That matters: referee features carry 47% of this model's value. The "
         "PL rows test the model at full strength; the EFL rows test it "
         "without its most important input.", False),
        ("No match odds were used as model inputs -- only partial prices were "
         "available and the margin cannot be removed from one side alone.",
         False),
        ("", False),
        ("WHAT WOULD COUNT AS SUCCESS", True),
        ("Calibration gap within about 5 percentage points.", False),
        ("Mean absolute error near 1.5 cards is EXPECTED, not a failure: the "
         "match-level standard deviation of yellows is about 1.9.", False),
        ("Profit or loss over one weekend is noise. It is recorded so that a "
         "season of weekends eventually means something.", False),
        ("", False),
        ("PROVENANCE", True),
        (f"Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC, before "
         "any of these matches kicked off.", False),
        ("Fixtures: ESPN, cross-checked against Sky Sports and the EFL news "
         "feed -- ESPN alone was missing two Friday/Thursday evening games.",
         False),
        ("Referees: premierleague.com Matchweek 5 officials.", False),
        ("These fixtures are the 2026/27 holdout. Generating predictions is "
         "not a holdout look. Scoring them is, and it will be logged.", False),
    ]
    for i, (t, bold) in enumerate(text, 1):
        c = rm.cell(i, 1, t)
        c.font = Font(name=FONT, size=11, bold=bold)
        c.alignment = Alignment(wrap_text=True, vertical="top")


if __name__ == "__main__":
    raise SystemExit(main())
