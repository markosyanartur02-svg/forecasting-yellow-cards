"""Spend a holdout look. PROJECT_SPEC section 3.

This is deliberately awkward to run by accident: it requires an explicit flag,
it appends to `cards/holdout_log.json` BEFORE printing anything, and it prints
the running total of looks taken. Every look is an in-sample fit with extra
steps, and the log is what stops that being deniable later.

    .venv/bin/python cards/run_holdout.py --i-am-spending-a-holdout-look
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import features
import loader
import models

HOLDOUT_LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "holdout_log.json")


def log_look(entry: dict) -> int:
    log = []
    if os.path.exists(HOLDOUT_LOG):
        with open(HOLDOUT_LOG) as f:
            log = json.load(f)
    entry["at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    log.append(entry)
    with open(HOLDOUT_LOG, "w") as f:
        json.dump(log, f, indent=2)
    return len(log)


def looks_taken(current_only: bool = True) -> int:
    """Looks spent against the CURRENT holdout definition.

    Looks against a retired holdout stay in the log but do not count against
    the new one -- they were spent on different seasons. The total is still
    reported, because "we have looked at test data 3 times in this project" is
    a fact a reader deserves even when the definitions changed.
    """
    if not os.path.exists(HOLDOUT_LOG):
        return 0
    with open(HOLDOUT_LOG) as f:
        log = json.load(f)
    if not current_only:
        return len(log)
    cur = list(loader.HOLDOUT_SEASONS)
    return sum(1 for e in log if e.get("holdout_definition") == cur)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--i-am-spending-a-holdout-look", action="store_true",
                    dest="spend")
    a = ap.parse_args()
    if not a.spend:
        print(f"Refusing. This spends a holdout look "
              f"({looks_taken()} taken so far).\n"
              "Pass --i-am-spending-a-holdout-look to proceed.")
        return 1

    print(f"looks against the current holdout {loader.HOLDOUT_SEASONS}: "
          f"{looks_taken()}")
    print(f"looks against any holdout, all time: {looks_taken(current_only=False)}")

    # Features are built over ALL matches so that a holdout fixture's rolling
    # form is computed from the matches that actually preceded it. That is not
    # a leak: the builder only ever looks backwards. The model, however, is
    # fitted on development rows alone.
    allm = loader.load_matches(
        seasons=[s for s in loader.season_codes() if s >= "0506"],
        unlock_holdout=True)
    feat = features.build(allm)
    cols = features.feature_columns(feat)

    dev = feat[~feat.season.isin(loader.HOLDOUT_SEASONS)]
    hold = feat[feat.season.isin(loader.HOLDOUT_SEASONS)]
    print(f"train {len(dev):,} matches ({dev.season.min()}..{dev.season.max()})")
    print(f"holdout {len(hold):,} matches "
          f"({', '.join(sorted(hold.season.unique()))})")
    for s in sorted(hold.season.unique()):
        g = hold[hold.season == s]
        print(f"   {s}: {len(g):,} matches, divisions "
              f"{sorted(g.division.unique())}")

    fitted = {}
    for label, fac in (("league-season NB", models.LeagueSeasonNB),
                       ("NB-GLM", lambda: models.NBGLM(cols))):
        fitted[label] = fac().fit(dev)

    scores = {}
    for label, mdl in fitted.items():
        mu = mdl.predict_mu(hold)
        scores[label] = models.evaluate(
            hold.yellows.to_numpy(float), mu, mdl.alpha_, hold.division)

    # ---- record the look BEFORE reporting, so a crash still costs it
    n = log_look({
        "model": "NBGLM",
        "features": len(cols),
        "trained_on": [str(dev.season.min()), str(dev.season.max())],
        "holdout_seasons": sorted(hold.season.unique()),
        "n_holdout": int(len(hold)),
        "results": {k: {"nll": v["nll"], "alpha": v["alpha"],
                        "cal_gap": v["mean_abs_calibration_gap"]}
                    for k, v in scores.items()},
        "holdout_definition": list(loader.HOLDOUT_SEASONS),
        "note": "against the 2026/27 holdout declared on 2026-09-17; the two "
                "earlier looks in this log were against the retired "
                "2024/25+2025/26 holdout and do not count toward this one",
    })

    print(f"\n{'=' * 66}\nHOLDOUT RESULT   (look #{n})\n{'=' * 66}")
    for label, sc in scores.items():
        print(f"  {label:<18} NLL {sc['nll']:.5f}   alpha {sc['alpha']:.4f}   "
              f"mean |cal gap| {sc['mean_abs_calibration_gap']:.4f}")

    base = scores["league-season NB"]["nll"]
    glm = scores["NB-GLM"]["nll"]
    print(f"\n  GLM improvement over baseline: {base - glm:+.5f} NLL")
    print(f"  development walk-forward said:  +0.02545 NLL (16.8 sigma)")

    print("\n  per division (NB-GLM):")
    for d, v in sorted(scores["NB-GLM"]["per_division"].items()):
        b = scores["league-season NB"]["per_division"][d]
        print(f"    {d} {loader.DIVISIONS[d]:<16} n={v['n']:>5}  "
              f"NLL {v['nll']:.4f}  vs baseline {b['nll']:.4f}  "
              f"({b['nll'] - v['nll']:+.4f})")

    print("\n  per season (NB-GLM):")
    for s in sorted(hold.season.unique()):
        g = hold[hold.season == s]
        mu = fitted["NB-GLM"].predict_mu(g)
        sc = models.evaluate(g.yellows.to_numpy(float), mu,
                             fitted["NB-GLM"].alpha_)
        print(f"    {s}: n={sc['n']:>5}  NLL {sc['nll']:.5f}  "
              f"cal gap {sc['mean_abs_calibration_gap']:.4f}")

    print("\n  calibration on the holdout — the number that decides whether\n"
          "  this distribution can be compared to a price:")
    for line, v in scores["NB-GLM"]["calibration"].items():
        flag = "" if abs(v["gap"]) < 0.02 else "   <-- off by >2pp"
        print(f"    over {line}: predicted {v['pred']:.3f}  "
              f"actual {v['actual']:.3f}  gap {v['gap']:+.3f}{flag}")

    mu = fitted["NB-GLM"].predict_mu(hold)
    rel = models.reliability(hold.yellows.to_numpy(float), mu,
                             fitted["NB-GLM"].alpha_, line=3.5)
    print("\n  reliability at the 3.5 line (does P=0.6 happen 60% of the time?):")
    for _, r in rel.iterrows():
        print(f"    predicted {r.predicted:.3f}  actual {r.actual:.3f}  "
              f"gap {r.gap:+.3f}   n={int(r.n)}")

    print(f"\n{'=' * 66}")
    print(f"holdout looks now spent: {n}. Recorded in {os.path.basename(HOLDOUT_LOG)}.")
    print("Every further look is another in-sample fit. Report this count.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
