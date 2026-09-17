"""Does gradient boosting beat the NB-GLM by more than the noise floor?

PROJECT_SPEC section 4: the GBM is kept only if it clears the GLM by more than
fold-to-fold variation. Run:

    .venv/bin/python cards/run_gbm.py
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

REGISTRY = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "model_registry.json")


def register(entry: dict) -> int:
    """Every configuration evaluated goes on the record.

    At any single comparison a model has a real chance of winning by luck; the
    number of configurations tried is part of the evidence about the one that
    won, and is worthless if nobody wrote them down.
    """
    reg = []
    if os.path.exists(REGISTRY):
        with open(REGISTRY) as f:
            reg = json.load(f)
    entry["at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    reg.append(entry)
    with open(REGISTRY, "w") as f:
        json.dump(reg, f, indent=2)
    return len(reg)


def compare(a: pd.DataFrame, b: pd.DataFrame, name_a: str, name_b: str) -> dict:
    j = a.merge(b, on="test_season", suffixes=("_a", "_b"))
    d = j.nll_a - j.nll_b                       # positive => b is better
    sd = float(d.std(ddof=1))
    mean = float(d.mean())
    se = sd / np.sqrt(len(d)) if len(d) else np.nan
    sigma = mean / se if se and se > 0 else np.nan
    verdict = ("IMPROVED" if sigma > 2 else
               "HURT" if sigma < -2 else "NO_EVIDENCE")
    print(f"\n{name_b} vs {name_a}, per fold:")
    for _, r in j.iterrows():
        print(f"   {r.test_season}: {r.nll_a:.5f} -> {r.nll_b:.5f} "
              f"({r.nll_a - r.nll_b:+.5f})")
    print(f"\n   mean improvement {mean:+.5f} NLL   sd across folds {sd:.5f}"
          f"   -> {sigma:.1f} sigma")
    print(f"   VERDICT: {verdict}"
          + ("" if verdict != "NO_EVIDENCE" else
             "  — smaller than fold-to-fold variation; NOT evidence of a gain"))
    return {"mean_delta": mean, "sd": sd, "sigma": float(sigma),
            "verdict": verdict, "folds": len(d)}


def main() -> int:
    m = loader.load_matches(
        verbose=True)   # all development seasons, 2000/01 onward
    feat = features.build(m)
    cols = features.feature_columns(feat)
    print(f"matches {len(feat):,}   features {len(cols)}\n")

    runs = {}
    for label, fac in (
        ("league-season NB", models.LeagueSeasonNB),
        ("NB-GLM", lambda: models.NBGLM(cols)),
        ("NB-GBM", lambda: models.NBGBM(cols)),
    ):
        wf = models.walk_forward(feat, fac)
        runs[label] = wf
        print(f"{label:<18} folds {len(wf)}  mean NLL {wf.nll.mean():.5f}"
              f"  sd {wf.nll.std():.5f}  mean |cal gap| {wf.cal_gap.mean():.4f}")

    res = compare(runs["NB-GLM"], runs["NB-GBM"], "NB-GLM", "NB-GBM")
    n = register({"model": "NBGBM", "params": models.NBGBM.DEFAULTS,
                  "mean_nll": float(runs["NB-GBM"].nll.mean()),
                  "vs_glm": res})
    print(f"\n   configurations in the registry: {n}")

    # detail on the last development season
    seasons = sorted(feat.season.unique())
    tr, te = feat[feat.season < seasons[-1]], feat[feat.season == seasons[-1]]
    print(f"\nDETAIL on {seasons[-1]}  (train {len(tr):,} / test {len(te):,})")
    gbm = models.NBGBM(cols).fit(tr)
    mu = gbm.predict_mu(te)
    sc = models.evaluate(te.yellows.to_numpy(float), mu, gbm.alpha_, te.division)
    print(f"   NB-GBM NLL {sc['nll']:.5f}  alpha {sc['alpha']:.4f}  "
          f"trees {gbm.best_iteration_}")
    print("   per-division " + " ".join(
        f"{d}:{v['nll']:.4f}" for d, v in sorted(sc["per_division"].items())))
    print("\n   calibration:")
    for line, v in sc["calibration"].items():
        print(f"     over {line}: predicted {v['pred']:.3f}  "
              f"actual {v['actual']:.3f}  gap {v['gap']:+.3f}")

    imp = gbm.importances().head(12)
    total = imp.gain.sum() or 1.0
    print("\n   most informative features (share of total gain):")
    for _, r in imp.iterrows():
        print(f"     {r.feature:<24} {r.gain / gbm.importances().gain.sum():6.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
