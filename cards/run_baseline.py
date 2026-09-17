"""Fit the three baselines and report the scoreboard from PROJECT_SPEC s4.

Run:  .venv/bin/python cards/run_baseline.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import features
import loader
import models


def main() -> int:
    m = loader.load_matches(
        verbose=True)   # all development seasons, 2000/01 onward
    feat = features.build(m)
    cols = features.feature_columns(feat)
    print(f"matches {len(feat):,}   seasons {feat.season.min()}..{feat.season.max()}"
          f"   features {len(cols)}\n")

    factories = [
        ("1. constant NB      ", models.ConstantNB),
        ("2. league-season NB ", models.LeagueSeasonNB),
        ("3. NB-GLM + features", lambda: models.NBGLM(cols)),
    ]

    print("WALK-FORWARD (train on all prior seasons, test on the next)")
    print(f"{'model':<22}{'folds':>6}{'mean NLL':>11}{'sd across folds':>17}"
          f"{'mean |cal gap|':>16}")
    results = {}
    for name, fac in factories:
        wf = models.walk_forward(feat, fac)
        results[name.strip()] = wf
        print(f"{name:<22}{len(wf):>6}{wf.nll.mean():>11.5f}"
              f"{wf.nll.std():>17.5f}{wf.cal_gap.mean():>16.4f}")

    # improvement vs the noise floor -- the AML/oil rule, ported
    base = results["2. league-season NB"]
    glm = results["3. NB-GLM + features"]
    joined = base.merge(glm, on="test_season", suffixes=("_base", "_glm"))
    delta = joined.nll_base - joined.nll_glm          # positive = GLM better
    sd = float(delta.std(ddof=1))
    mean = float(delta.mean())
    sigma = mean / (sd / np.sqrt(len(delta))) if sd > 0 else np.nan
    print(f"\nGLM vs league-season baseline, per fold:")
    for _, r in joined.iterrows():
        print(f"   {r.test_season}: {r.nll_base:.5f} -> {r.nll_glm:.5f} "
              f"({r.nll_base - r.nll_glm:+.5f})")
    print(f"\n   mean improvement {mean:+.5f} NLL   sd across folds {sd:.5f}"
          f"   -> {sigma:.1f} sigma")
    verdict = ("IMPROVED" if sigma > 2 else
               "HURT" if sigma < -2 else "NO_EVIDENCE")
    print(f"   VERDICT: {verdict}"
          + ("" if verdict != "NO_EVIDENCE" else
             "  — the change is smaller than fold-to-fold variation; "
             "it is NOT evidence of an improvement"))

    # a single holdout-style fit on the last dev season, for detail
    seasons = sorted(feat.season.unique())
    tr, te = feat[feat.season < seasons[-1]], feat[feat.season == seasons[-1]]
    print(f"\nDETAIL on the last development season ({seasons[-1]}), "
          f"train {len(tr):,} / test {len(te):,}")
    for name, fac in factories:
        mdl = fac().fit(tr)
        mu = mdl.predict_mu(te)
        sc = models.evaluate(te.yellows.to_numpy(float), mu, mdl.alpha_,
                             te.division)
        pd_ = sc.get("per_division", {})
        print(f"  {name.strip():<22} NLL {sc['nll']:.5f}  alpha {sc['alpha']:.4f}"
              f"  per-division " +
              " ".join(f"{d}:{v['nll']:.4f}" for d, v in sorted(pd_.items())))

    mdl = models.NBGLM(cols).fit(tr)
    mu = mdl.predict_mu(te)
    sc = models.evaluate(te.yellows.to_numpy(float), mu, mdl.alpha_)
    print("\nCALIBRATION of the GLM on that season (predicted vs actual):")
    for line, v in sc["calibration"].items():
        print(f"   over {line}: predicted {v['pred']:.3f}  actual {v['actual']:.3f}"
              f"  gap {v['gap']:+.3f}")

    print("\nLARGEST COEFFICIENTS (log-scale, relative to league average):")
    for _, r in mdl.coefficients().head(12).iterrows():
        print(f"   {r.feature:<26} {r.coef:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
