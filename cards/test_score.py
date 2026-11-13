"""The forward-test score must compare the model and the baseline on the same event.

    .venv/bin/python cards/test_score.py

The sheets store P(0)..P(8) for exactly that many yellows. A match with 9 or
more must be scored as "9 or more" on both sides. Until 13 November 2026 the
model was given P(exactly 8) against the baseline's P(exactly 11) for Barnet v
York, which credited it with 1.9 it had not earned.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import models
import score_forward_test as sft

PASS = FAIL = 0


def check(name: str, got, want, tol: float = 1e-9) -> None:
    global PASS, FAIL
    ok = abs(float(got) - float(want)) <= tol if isinstance(want, float) else got == want
    if ok:
        PASS += 1
        print(f"  [PASS] {name}: {got}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}: got {got!r}, expected {want!r}")


def main() -> int:
    # Barnet v York's committed forecast, near enough: P(0..8) leave 0.9% beyond 8
    pmf = [0.04, 0.13, 0.20, 0.21, 0.18, 0.12, 0.068, 0.033, 0.010]
    mu, alpha = 3.90, 0.034
    print("\nup to 8 yellows: the exact count on both sides")
    m, b = sft.loglik_pair([pmf], np.array([3.0]), np.array([mu]), alpha)
    check("model reads its own P(3)", m[0], float(np.log(0.21)))
    check("baseline reads its own P(3)", b[0], float(models.nb_logpmf([3.0], [mu], alpha)[0]))

    print("\n9 or more: both sides scored on '9 or more'")
    m, b = sft.loglik_pair([pmf], np.array([11.0]), np.array([mu]), alpha)
    tail_model = 1.0 - sum(pmf)
    tail_base = 1.0 - float(np.exp(models.nb_logpmf(np.arange(9.0), np.full(9, mu), alpha)).sum())
    check("model gets what its P(0..8) leave over", m[0], float(np.log(tail_model)))
    check("baseline gets its tail beyond 8", b[0], float(np.log(tail_base)))
    old_gain = float(np.log(pmf[8]) - models.nb_logpmf([11.0], [mu], alpha)[0])
    check("the old comparison flattered the model", old_gain > (m[0] - b[0]) + 1.0, True)

    print("\n" + "-" * 62)
    print(f"{PASS}/{PASS + FAIL} checks passed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
