"""Freeze the model specification for the season, and check it stayed frozen.

    .venv/bin/python cards/freeze.py --freeze "reason"   # record the spec
    .venv/bin/python cards/freeze.py --verify            # has it changed?

WHY

The forward test only means something if it is ONE experiment. Scoring it every
week is fine -- that is monitoring a single accumulating result. Changing the
model halfway through is not: every week before the change measured a different
model, and adding the two together measures nothing.

So the model is frozen for the season. The point of this file is that "frozen"
is checkable rather than remembered.

WHAT IS FROZEN -- THE SPECIFICATION, NOT THE COEFFICIENTS

The model still refits as new matches arrive, and it should: fitting on
everything known before a fixture is correct point-in-time behaviour and is how
the thing would really be used. What must not change is the RECIPE --
which features, which model family, which hyperparameters, which de-vig method,
which training window rule.

So the fingerprint covers the specification and is deliberately invariant to
new training data. A coefficient that shifts because another weekend has been
played is expected. A feature list that gains a column is not.

WHEN IT TRIPS

`--verify` failing does not always mean something bad. Editing a docstring
changes the hash without changing a single prediction. The honest response is
to re-freeze with `--freeze` and say why in the reason, which leaves a dated
record of what changed and when. What you must not do is score the accumulated
forward test across a genuine specification change and report it as one number.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import re
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import features
import loader
import models

HERE = os.path.dirname(os.path.abspath(__file__))
FREEZE_FILE = os.path.join(HERE, "model_freeze.json")


def _strip(src: str) -> str:
    """Drop comments and blank lines so formatting noise does not trip the hash."""
    out = []
    for line in src.splitlines():
        s = line.split("#")[0].rstrip()
        if s.strip():
            out.append(" ".join(s.split()))
    return "\n".join(out)


def spec() -> dict:
    """Everything that decides what the model predicts, given the same data."""
    import pandas as pd
    m = loader.load_matches(seasons=["2324"])          # small, just for columns
    cols = sorted(features.feature_columns(features.build(m)))
    logic = "\n".join(_strip(inspect.getsource(fn)) for fn in (
        features.build, features._add_market_features,
        features.shin_probabilities, features.feature_columns,
        models.NBGLM.fit, models.NBGLM.predict_mu, models.NBGLM._design,
        models.LeagueSeasonNB.fit, models.LeagueSeasonNB.predict_mu,
        models.nb_logpmf, models.fit_alpha,
    ))
    return {
        "n_features": len(cols),
        "features": cols,
        "model": "NBGLM",
        "ridge": models.NBGLM().ridge,
        "devig_default": inspect.signature(features.build)
                                .parameters["devig"].default,
        "team_windows": list(features.TEAM_WINDOWS),
        "ref_windows": list(features.REF_WINDOWS),
        "min_team_matches": features.MIN_TEAM_MATCHES,
        "min_ref_matches": features.MIN_REF_MATCHES,
        "min_baseline_matches": features.MIN_BASELINE_MATCHES,
        "holdout": list(loader.HOLDOUT_SEASONS),
        "divisions": sorted(loader.DIVISIONS),
        "logic_hash": hashlib.sha256(logic.encode()).hexdigest()[:16],
    }


def fingerprint(sp: dict) -> str:
    return hashlib.sha256(
        json.dumps(sp, sort_keys=True).encode()).hexdigest()[:16]


def load_freeze() -> dict | None:
    if not os.path.exists(FREEZE_FILE):
        return None
    with open(FREEZE_FILE) as f:
        return json.load(f)


def do_freeze(reason: str) -> int:
    sp = spec()
    fp = fingerprint(sp)
    prev = load_freeze()
    record = {
        "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "fingerprint": fp,
        "reason": reason,
        "spec": sp,
        "history": (prev.get("history", []) + [{
            "fingerprint": prev["fingerprint"], "frozen_at": prev["frozen_at"],
            "reason": prev["reason"]}]) if prev else [],
    }
    with open(FREEZE_FILE, "w") as f:
        json.dump(record, f, indent=2)
    print(f"frozen: {fp}")
    print(f"  {sp['n_features']} features, {sp['model']}, ridge {sp['ridge']}, "
          f"de-vig {sp['devig_default']}")
    print(f"  holdout {sp['holdout']}")
    print(f"  reason: {reason}")
    if prev:
        print(f"  previous fingerprint {prev['fingerprint']} moved to history")
    return 0


def do_verify(quiet: bool = False) -> bool:
    rec = load_freeze()
    if rec is None:
        if not quiet:
            print("No freeze recorded. Run --freeze first.")
        return False
    now = spec()
    fp_now, fp_then = fingerprint(now), rec["fingerprint"]
    if fp_now == fp_then:
        if not quiet:
            print(f"UNCHANGED  {fp_now}   frozen {rec['frozen_at'][:10]}")
            print(f"  {now['n_features']} features, {now['model']}")
        return True

    print(f"CHANGED    frozen {fp_then}   now {fp_now}")
    old = rec["spec"]
    for k in sorted(set(old) | set(now)):
        a, b = old.get(k), now.get(k)
        if a == b:
            continue
        if k == "features":
            added = sorted(set(b or []) - set(a or []))
            gone = sorted(set(a or []) - set(b or []))
            if added:
                print(f"  features added:   {added}")
            if gone:
                print(f"  features removed: {gone}")
        else:
            print(f"  {k}: {a!r} -> {b!r}")
    if old.get("features") == now.get("features") and \
            old.get("logic_hash") != now.get("logic_hash"):
        print("  NOTE: the feature list is identical and only the logic hash "
              "moved. A docstring edit does that without changing any "
              "prediction. If that is all it was, re-freeze and say so.")
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--freeze", metavar="REASON")
    g.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    if a.freeze:
        return do_freeze(a.freeze)
    return 0 if do_verify() else 1


if __name__ == "__main__":
    raise SystemExit(main())
