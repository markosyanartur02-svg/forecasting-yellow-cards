"""Settle the labels and re-score the forward test, the day it becomes possible.

    .venv/bin/python cards/rescore_when_ready.py

Called by the daily job. Most days it prints one line saying how long is left.
On the day every label is out of the confirmation lag it settles them against
football-data and re-scores, then records that it has done so and never does
it again.

WHY THIS RUNS UNATTENDED, WHEN A HOLDOUT LOOK NORMALLY SHOULD NOT

Because the decision was already made deliberately, and what is left is only
waiting. The thing that makes a look expensive is choosing to take it -- each
one is a chance to learn something about the test set and adjust. Nothing is
being chosen here: the labels come from football-data, the scorer is
pre-registered, the model is frozen and verified before scoring, and this runs
exactly once.

What it will not do is spend the look early. `reconcile_labels` refuses to
write while any fixture is still inside the lag, and this refuses to score
unless that write succeeded. A look spent on labels that are still moving
would have to be spent again.

THE FIRST SCORE IS NOT DISCARDED

It stays in the holdout log. This is a correction to one experiment, not a
second experiment -- the same fixtures, the same predictions, better labels --
and the log shows both so the difference is visible rather than quietly
replaced.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import net
import reconcile_labels as rl

MARKER = os.path.join(HERE, "rescore_done.json")


def main() -> int:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    if os.path.exists(MARKER):
        with open(MARKER) as f:
            done = json.load(f)
        print(f"{stamp} labels: settled and re-scored on "
              f"{done['at'][:10]} -- nothing further to do")
        return 0

    if not net.wait_for_network(log=lambda m: print(f"{stamp} labels: {m}")):
        print(f"{stamp} labels: skipped, no network")
        return 0

    try:
        s = rl.survey(rl.BOOK)
    except Exception as e:
        print(f"{stamp} labels: could not survey ({type(e).__name__}: {e})")
        return 0

    n = sum(len(s[k]) for k in ("agree", "change", "protected", "pending",
                                "missing"))
    waiting = len(s["pending"]) + len(s["missing"])
    if waiting:
        when = f", ready {s['ready_on']}" if s["ready_on"] else ""
        print(f"{stamp} labels: {waiting} of {n} still inside the card "
              f"lag{when} -- not re-scoring yet")
        return 0

    print(f"{stamp} labels: READY -- all {n} confirmed. Settling and "
          "re-scoring now.")

    argv = sys.argv[:]
    try:
        sys.argv = ["reconcile_labels.py", "--write"]
        if rl.main() != 0:
            print(f"{stamp} labels: reconcile refused -- not scoring")
            return 0
        import score_forward_test as sft
        sys.argv = ["score_forward_test.py", "--i-am-spending-a-holdout-look"]
        rc = sft.main()
    except Exception as e:
        print(f"{stamp} labels: re-score failed ({type(e).__name__}: {e})")
        return 0
    finally:
        sys.argv = argv

    if rc != 0:
        print(f"{stamp} labels: scorer returned {rc} -- marker not written, "
              "so this will be retried tomorrow")
        return 0

    with open(MARKER, "w") as f:
        json.dump({"at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                   "n_labels": n, "n_changed": len(s["change"]),
                   "n_protected": len(s["protected"])}, f, indent=2)
    print(f"{stamp} labels: done -- {len(s['change'])} label(s) settled, "
          "forward test re-scored")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
