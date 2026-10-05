"""Track whether the model is drifting low, and decide whose fault it is.

    .venv/bin/python cards/bias_watch.py
    .venv/bin/python cards/bias_watch.py --quiet     # one line, for the daily job

The forward test under-predicted by 0.41 cards a match, and the 3 October
round by 0.85. Same direction twice is worth watching: one weekend is noise,
a run is a finding.

THE QUESTION THAT MATTERS IS NOT "IS THE MODEL LOW"

It is "is the model lower than its own baseline". The baseline is a negative
binomial fitted to each league-season, and most of the seasons it learns from
are in the past. If 2026/27 is simply a higher-card season than history, then
the baseline under-predicts too, and so does the model, and nothing about the
model is wrong -- the league moved.

Those two cases need opposite responses and look identical if you only measure
the model. So this reports both, and the difference between them:

  both low by a similar amount  -> the SEASON is running hot. Expected, and
                                   the model's edge over the baseline is
                                   unaffected because both shift together.
  model lower than the baseline -> the MODEL is the problem. Its features are
                                   pushing predictions down when they should
                                   not be.

WHY WATCHING THIS DOES NOT SPEND A HOLDOUT LOOK

A look is expensive because it is a chance to adjust the model to the test
set. The model is frozen for the season and its fingerprint is checked before
every score, so nothing observed here can be fitted to. The budget exists to
stop a particular kind of cheating, and this is not that kind -- it is the
monitoring the freeze was put in place to make safe.

What it must not become is a reason to unfreeze. If the bias turns out to be
real, the honest response is to record it, finish the season, and build the
next model knowing it -- not to adjust mid-experiment and report the result as
one number.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from openpyxl import load_workbook

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

BOOK = os.path.join(os.path.dirname(HERE), "forward_test_2026-09-18.xlsx")


def gather(path: str) -> pd.DataFrame:
    """Every prediction whose outcome was explicitly recorded in the workbook.

    Deliberately does NOT reach into football-data for a missing label. The
    first version did, and swallowed Rochdale v Swindon's 0 -- a count already
    known to be impossible, since a player was sent off there for two
    bookings. A monitor that helps itself to unverified numbers reports drift
    that is really a data error.

    So a blank ACTUAL means "not established yet" and is skipped. Filling it
    is fill_results' job, and that goes through the verification.
    """
    wb = load_workbook(path, data_only=True)
    rows = []
    for name in wb.sheetnames:
        ws = wb[name]
        h = {ws.cell(1, c).value: c for c in range(1, ws.max_column + 1)}
        if not all(k in h for k in ("Home", "Away", "Date", "Model expects",
                                    "League avg")):
            continue
        has_actual = "ACTUAL yellows" in h
        for r in range(2, ws.max_row + 1):
            home = ws.cell(r, h["Home"]).value
            away = ws.cell(r, h["Away"]).value
            date = ws.cell(r, h["Date"]).value
            mu = ws.cell(r, h["Model expects"]).value
            base = ws.cell(r, h["League avg"]).value
            if not home or home == "Example FC" or not away or not date:
                continue
            if not isinstance(mu, (int, float)) or not isinstance(base, (int, float)):
                continue
            actual = ws.cell(r, h["ACTUAL yellows"]).value if has_actual else None
            if not isinstance(actual, (int, float)):
                continue
            rows.append({"sheet": name, "date": str(date), "fixture":
                         f"{home} v {away}", "mu": float(mu),
                         "base": float(base), "actual": float(actual)})
    return pd.DataFrame(rows)


def stats(d: pd.DataFrame) -> dict:
    em = (d.actual - d.mu).to_numpy()
    eb = (d.actual - d.base).to_numpy()
    n = len(d)
    out = {"n": n, "model_bias": em.mean(), "base_bias": eb.mean()}
    out["model_se"] = em.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
    out["base_se"] = eb.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
    out["model_sigma"] = (out["model_bias"] / out["model_se"]
                          if out["model_se"] else np.nan)
    # paired: is the model lower than the baseline on the SAME matches?
    diff = eb - em                      # >0 means the model sits above the baseline
    out["vs_base"] = diff.mean()
    out["vs_base_sigma"] = (diff.mean() / (diff.std(ddof=1) / np.sqrt(n))
                            if n > 1 and diff.std(ddof=1) > 0 else np.nan)
    return out


def verdict(s: dict) -> tuple[str, str]:
    if abs(s["model_sigma"]) < 2:
        return ("WITHIN NOISE",
                "the shortfall is not yet distinguishable from variance")
    if abs(s["vs_base_sigma"]) < 2:
        return ("SEASON RUNNING HOT",
                "the baseline is low by a similar amount, so the league has "
                "moved rather than the model breaking; the model's edge over "
                "the baseline is unaffected")
    return ("MODEL DRIFTING LOW",
            "the model is low by more than its own baseline is -- this is the "
            "model, not the season. Record it, finish the season, and build "
            "the next one knowing it")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--file", default=BOOK)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    d = gather(a.file)
    if len(d) < 2:
        print("not enough labelled predictions yet")
        return 0
    s = stats(d)
    v, why = verdict(s)

    if a.quiet:
        print(f"bias: model {s['model_bias']:+.2f} "
              f"({s['model_sigma']:+.1f} sigma), baseline "
              f"{s['base_bias']:+.2f}, n={s['n']} -- {v}")
        return 0

    print(f"{s['n']} predictions with an outcome\n")
    print(f"  {'':22}{'bias':>8}{'sigma':>8}")
    print(f"  {'model':22}{s['model_bias']:>+8.2f}{s['model_sigma']:>+8.1f}")
    print(f"  {'league-season baseline':22}{s['base_bias']:>+8.2f}"
          f"{s['base_bias'] / s['base_se']:>+8.1f}")
    print(f"  {'model minus baseline':22}{s['vs_base']:>+8.2f}"
          f"{s['vs_base_sigma']:>+8.1f}")
    print(f"\n  VERDICT: {v}\n           {why}")

    print("\n  BY WEEK")
    for sheet, g in d.groupby("sheet", sort=False):
        print(f"     {sheet:<16} n={len(g):>3}  predicted {g.mu.mean():.2f}  "
              f"actual {g.actual.mean():.2f}  model {(g.actual - g.mu).mean():+.2f}"
              f"  baseline {(g.actual - g.base).mean():+.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
