"""Prove the point-in-time guards actually fire.

PROJECT_SPEC section 6: a guard nobody has watched reject something is not a
guard. Four of the checks below hand a *deliberately broken* feature builder to
a test and require the test to fail it. If the test passes both the correct and
the broken implementation, it is proving nothing, which is exactly what
happened on the first attempt in the preceding oil project.

There are two distinct leaks here and they need two distinct tests:

  * **future leakage** — a later match changes an earlier match's features.
    Caught by truncation: build knowing N matches, build again knowing 2N,
    require the first N to be identical.
  * **self leakage** — a match's own result feeds its own features. Truncation
    CANNOT catch this, because the row is present in both builds. Caught by
    perturbation: change one match's card count and require its own feature
    row to be unchanged while later rows move.

Run:  .venv/bin/python cards/test_features.py
"""

from __future__ import annotations

import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import features
import loader

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, passed: bool, detail: str = "") -> None:
    RESULTS.append((name, passed, detail))
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}"
          + (f"\n         {detail}" if detail else ""))


def sample_matches(n: int = 4000) -> pd.DataFrame:
    m = loader.load_matches(
        seasons=[s for s in loader.season_codes() if s >= "1516"])
    return m.sort_values(["date", "division", "HomeTeam"],
                         kind="mergesort").head(n).reset_index(drop=True)


# ---------------------------------------------------- broken implementations


def build_future_leaky(matches: pd.DataFrame) -> pd.DataFrame:
    """Baseline computed over the WHOLE season instead of the season so far.

    A one-line change of the kind that gets written without thinking: instead
    of "average cards in this division-season up to now", it uses the finished
    season average — which on matchday one already knows how the season ends.
    """
    m = matches.sort_values(["date", "division", "HomeTeam"],
                            kind="mergesort").reset_index(drop=True)
    full = m.groupby(["division", "season"]).yellows.transform("mean")
    base = features.build(m)
    base["baseline_yellows"] = full.values
    return base


def build_self_leaky(matches: pd.DataFrame) -> pd.DataFrame:
    """State updated BEFORE the row is emitted, so a match sees its own result."""
    m = matches.sort_values(["date", "division", "HomeTeam"],
                            kind="mergesort").reset_index(drop=True)
    hist: dict = defaultdict(list)
    rows = []
    for r in m.itertuples(index=False):
        hist[r.HomeTeam].append(r.HY)          # <-- the bug: update first
        hist[r.AwayTeam].append(r.AY)
        rows.append({
            "date": r.date, "home": r.HomeTeam, "away": r.AwayTeam,
            "yellows": r.yellows,
            "h_yel5": float(np.mean(hist[r.HomeTeam][-5:])),
            "a_yel5": float(np.mean(hist[r.AwayTeam][-5:])),
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------- tests


def test_truncation():
    """Features built knowing less must match features built knowing more."""
    m = sample_matches(4000)
    n = 2000
    early = features.build(m.head(n))
    late = features.build(m).head(n)
    cols = [c for c in features.feature_columns(early) if c in late.columns]

    a = early[cols].to_numpy(dtype=float)
    b = late[cols].to_numpy(dtype=float)
    same = np.array_equal(np.nan_to_num(a, nan=-999), np.nan_to_num(b, nan=-999))
    diffs = int((~np.isclose(np.nan_to_num(a, nan=-999),
                             np.nan_to_num(b, nan=-999))).sum())
    check(
        "features are bit-identical when later matches are added",
        same,
        f"compared {n:,} rows x {len(cols)} features; differing cells: {diffs}",
    )


def test_truncation_catches_a_future_leak():
    """Hand the truncation test a leak and require it to notice."""
    m = sample_matches(4000)
    n = 2000
    early = build_future_leaky(m.head(n))
    late = build_future_leaky(m).head(n)
    a = early["baseline_yellows"].to_numpy(float)
    b = late["baseline_yellows"].to_numpy(float)
    differs = not np.allclose(a, b, equal_nan=True)
    check(
        "the truncation test rejects a full-season (future-peeking) baseline",
        differs,
        f"{int((~np.isclose(a, b, equal_nan=True)).sum()):,} of {n:,} baselines "
        "changed once the rest of the season existed — that is the leak",
    )


def test_no_self_leakage():
    """A match's own cards must not reach its own features.

    Truncation cannot catch this: the row is in both builds. So we change one
    match's card count and require its OWN feature row to be untouched while
    later rows move.
    """
    m = sample_matches(3000)
    i = 1500
    f0 = features.build(m)
    m2 = m.copy()
    m2.loc[i, "HY"] = 9.0                    # a wild result for one match
    m2.loc[i, "yellows"] = m2.loc[i, "HY"] + m2.loc[i, "AY"]
    f1 = features.build(m2)

    cols = features.feature_columns(f0)
    own_same = np.allclose(f0.loc[i, cols].to_numpy(float),
                           f1.loc[i, cols].to_numpy(float), equal_nan=True)
    later = [j for j in range(i + 1, len(f0))
             if f0.loc[j, "home"] in (f0.loc[i, "home"], f0.loc[i, "away"])
             or f0.loc[j, "away"] in (f0.loc[i, "home"], f0.loc[i, "away"])]
    moved = 0
    for j in later[:40]:
        if not np.allclose(f0.loc[j, cols].to_numpy(float),
                           f1.loc[j, cols].to_numpy(float), equal_nan=True):
            moved += 1
    check(
        "a match's own result does not reach its own features, but does reach later ones",
        own_same and moved > 0,
        f"changed match {i}'s cards: own row unchanged={own_same}; "
        f"{moved} of the next {min(40, len(later))} matches involving those "
        "teams did change, which is correct",
    )


def test_self_leak_detector_catches_it():
    """Prove the perturbation test can fail a builder that leaks."""
    m = sample_matches(3000)
    i = 1500
    f0 = build_self_leaky(m)
    m2 = m.copy()
    m2.loc[i, "HY"] = 9.0
    f1 = build_self_leaky(m2)
    own_changed = not np.allclose(
        f0.loc[i, ["h_yel5", "a_yel5"]].to_numpy(float),
        f1.loc[i, ["h_yel5", "a_yel5"]].to_numpy(float), equal_nan=True)
    check(
        "the perturbation test rejects a builder that updates state before emitting",
        own_changed,
        f"leaky builder: match {i}'s own h_yel5 moved from "
        f"{f0.loc[i,'h_yel5']:.3f} to {f1.loc[i,'h_yel5']:.3f} when its own "
        "result changed — it was reading its own answer",
    )


def test_no_identity_features():
    """PROJECT_SPEC section 5: behaviour, never identity."""
    m = sample_matches(2000)
    f = features.build(m)
    cols = features.feature_columns(f)
    banned = {"home", "away", "referee", "division", "season"}
    leaked = [c for c in cols if c in banned]
    non_numeric = [c for c in cols
                   if not pd.api.types.is_numeric_dtype(f[c])]
    check(
        "no team, referee, division or season identity reaches the model",
        not leaked and not non_numeric,
        f"{len(cols)} features, all numeric; identity columns present: "
        f"{leaked or 'none'}; non-numeric: {non_numeric or 'none'}",
    )


def test_referee_feature_degrades_instead_of_guessing():
    m = sample_matches(4000)
    f = features.build(m)
    thin = f[f.ref_n_career < features.MIN_REF_MATCHES]
    check(
        "a referee with little history yields NaN, not a fabricated rate",
        bool(thin["ref_yel30"].isna().all()) if len(thin) else True,
        f"{len(thin):,} rows had a referee with < {features.MIN_REF_MATCHES} "
        "prior matches; all of their referee rates are NaN, and ref_n_career "
        "records how thin the evidence was",
    )


def test_holdout_is_locked():
    raised_rows = loader.load_matches(
        seasons=["2324", "2425"], verbose=False)
    leaked = raised_rows[raised_rows.season.isin(loader.HOLDOUT_SEASONS)]
    check(
        "holdout seasons are withheld unless explicitly unlocked",
        len(leaked) == 0,
        f"asked for 2324 + 2425; got {len(raised_rows):,} rows, "
        f"{len(leaked)} of them from holdout seasons",
    )
    unlocked = loader.load_matches(seasons=["2425"], unlock_holdout=True)
    check(
        "holdout data is reachable when deliberately unlocked",
        len(unlocked) > 0,
        f"{len(unlocked):,} holdout rows loaded with unlock_holdout=True",
    )


def main() -> int:
    for fn in (
        test_truncation,
        test_truncation_catches_a_future_leak,
        test_no_self_leakage,
        test_self_leak_detector_catches_it,
        test_no_identity_features,
        test_referee_feature_degrades_instead_of_guessing,
        test_holdout_is_locked,
    ):
        print(fn.__name__)
        fn()
        print()
    failed = [n for n, ok, _ in RESULTS if not ok]
    print("-" * 70)
    print(f"{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed")
    for n in failed:
        print(f"  FAILED: {n}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
