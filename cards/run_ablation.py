"""What is each family of features actually worth?

Drop one group, refit, and measure how much worse the model gets. A group that
costs nothing when removed was not contributing, however plausible it looked
and however large its coefficients appeared.

Same data and same walk-forward folds for every arm, so the only thing that
changes is the feature set. The deltas are paired by fold, which is what makes
the noise floor meaningful: fold-to-fold variation is large compared to the
effects being measured, and an unpaired comparison would drown in it.

Every arm is written to the model registry. Running six ablations is six
chances for one to look good by accident, and the count is part of the evidence
about any of them.

    .venv/bin/python cards/run_ablation.py
"""

from __future__ import annotations

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
from run_gbm import REGISTRY, register


def groups(cols: list[str]) -> dict[str, list[str]]:
    """Feature families. Every column lands in exactly one, so the ablation
    cannot silently leave a group half-removed."""
    g = {
        "market": [c for c in cols if c.startswith("mkt_")],
        "referee": [c for c in cols if c.startswith("ref_")],
        "team_form": [c for c in cols if c.startswith(("h_yel", "a_yel",
                                                       "h_fouls", "a_fouls",
                                                       "h_oppyel", "a_oppyel"))],
        "evidence_counts": [c for c in cols
                            if c.startswith(("h_n", "a_n"))],
        "context": [c for c in cols if c in ("h_rest_days", "a_rest_days",
                                             "match_no_in_season",
                                             "season_progress")],
    }
    assigned = {c for v in g.values() for c in v}
    leftover = [c for c in cols if c not in assigned and c != "baseline_yellows"]
    if leftover:
        g["other"] = leftover
    return g


def paired(full: pd.DataFrame, reduced: pd.DataFrame) -> dict:
    """How much worse is the reduced model? Positive = the group mattered."""
    j = full.merge(reduced, on="test_season", suffixes=("_full", "_red"))
    d = j.nll_red - j.nll_full
    sd = float(d.std(ddof=1))
    mean = float(d.mean())
    se = sd / np.sqrt(len(d)) if len(d) else np.nan
    sigma = mean / se if se and se > 0 else np.nan
    verdict = ("MATTERS" if sigma > 2 else
               "HURTS_TO_KEEP" if sigma < -2 else "NO_EVIDENCE")
    return {"cost_of_removing": mean, "sd": sd, "sigma": float(sigma),
            "verdict": verdict, "folds": int(len(d)),
            "worse_in_n_folds": int((d > 0).sum())}


def main() -> int:
    m = loader.load_matches(
        verbose=True)   # all development seasons, 2000/01 onward
    feat = features.build(m)
    cols = features.feature_columns(feat)
    fams = groups(cols)

    print(f"matches {len(feat):,}   features {len(cols)}")
    print("feature families:")
    for k, v in fams.items():
        print(f"   {k:<16} {len(v):>3} features")

    print("\nfitting full model...")
    full = models.walk_forward(feat, lambda: models.NBGLM(cols))
    print(f"   full model: {len(full)} folds, mean NLL {full.nll.mean():.5f}")

    base = models.walk_forward(feat, models.LeagueSeasonNB)
    print(f"   no features at all: mean NLL {base.nll.mean():.5f}"
          f"   (total value of all features: "
          f"{base.nll.mean() - full.nll.mean():+.5f})")

    print(f"\n{'group removed':<18}{'mean NLL':>10}{'cost of removing':>19}"
          f"{'sigma':>8}{'worse in':>10}  verdict")
    rows = []
    for name, drop in fams.items():
        if not drop:
            continue
        keep = [c for c in cols if c not in drop]
        wf = models.walk_forward(feat, lambda k=keep: models.NBGLM(k))
        res = paired(full, wf)
        rows.append({"group": name, "n_features": len(drop),
                     "mean_nll": float(wf.nll.mean()), **res})
        print(f"{name:<18}{wf.nll.mean():>10.5f}{res['cost_of_removing']:>+19.5f}"
              f"{res['sigma']:>8.1f}{res['worse_in_n_folds']:>6}/{res['folds']}"
              f"   {res['verdict']}")
        register({"model": "NBGLM_ablation", "group_removed": name,
                  "n_features_removed": len(drop),
                  "mean_nll": float(wf.nll.mean()), "vs_full": res})

    rows.sort(key=lambda r: -r["cost_of_removing"])
    print("\nranked by how much removing them costs:")
    for r in rows:
        share = (r["cost_of_removing"]
                 / max(base.nll.mean() - full.nll.mean(), 1e-9))
        print(f"   {r['group']:<16} {r['cost_of_removing']:+.5f} NLL"
              f"   = {share:6.1%} of everything the features add")

    n = len(json.load(open(REGISTRY))) if os.path.exists(REGISTRY) else 0
    print(f"\nconfigurations now in the registry: {n}")
    print("Six arms were tried. At 2 sigma, one in twenty null results clears "
          "the bar by chance; read the sigmas with that in mind.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
