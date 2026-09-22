"""Check the EFL appointment parser against referees that are known correct.

    .venv/bin/python cards/test_check_referees.py

This parser feeds referee names straight into the model, and the referee is
48% of the model's value. A silent mis-parse would not crash anything -- it
would just attach the wrong official to a fixture and quietly move the
prediction, which is the kind of bug that never announces itself.

So it is checked against ground truth rather than inspected: the 18-20
September article is parsed and compared with the 35 EFL referees that were
read off that same article by hand last week and typed into fixtures_week.py.

It needs the network. If the EFL is unreachable the checks are reported as
skipped, never as passed.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import check_referees as cr
import fetch_results_bbc as bbc
import fixtures_week as fw

ARTICLE = ("https://www.efl.com/news/2026/september/17/"
           "referee-appointments--18-20-september-2026/")

PASS = FAIL = SKIP = 0


def check(name, passed, detail=""):
    global PASS, FAIL
    if passed:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}")
    if detail:
        print(f"         {detail}")


def main() -> int:
    global SKIP

    print("test_model_name")
    check("full name becomes initial and surname",
          cr.model_name("Oliver Langford") == "o langford")
    check("a hyphenated surname keeps its last token",
          cr.model_name("Matt Gunn-Smith") == "m gunn-smith")
    check("stray markup and spacing are stripped",
          cr.model_name("  Paul   Tierney ") == "p tierney")

    print("\ntest_cup_articles_are_not_league_rounds")
    arts = [{"slug": "referee-appointments--trophy"},
            {"slug": "referee-appointments--carabao-cup-round-three"},
            {"slug": "referee-appointments--18-20-september-2026"}]
    flags = [any(c in a["slug"] for c in cr.CUPS) for a in arts]
    check("Trophy and Carabao are excluded, the league round is not",
          flags == [True, True, False],
          f"is_cup flags: {flags}")

    print("\ntest_parses_a_real_article_correctly")
    try:
        raw = bbc.get(ARTICLE)
    except Exception as e:
        SKIP += 1
        print(f"  [SKIP] EFL unreachable: {e}")
        raw = None

    if raw is not None:
        fx = cr.parse_article(raw)
        parsed = {(f["home"].lower(), f["away"].lower()): f["referee"]
                  for f in fx}
        truth = {(h.lower(), a.lower()): r
                 for _, _, h, a, r in fw.FIXTURES if r}
        both = set(parsed) & set(truth)
        wrong = [k for k in both if parsed[k] != truth[k]]

        check("all 35 EFL fixtures in the article are found",
              len(fx) == 35, f"parsed {len(fx)}")
        check("every one of them matches a hand-entered fixture",
              len(both) == 35, f"overlap {len(both)}")
        check("every referee name agrees with the hand-entered one",
              not wrong, f"mismatches: {wrong}")
        check("divisions are read from the competition headings",
              {f["division"] for f in fx} == {"E1", "E2", "E3"},
              f"divisions seen: {sorted({f['division'] for f in fx})}")
        check("the Premier League is absent, as the EFL does not appoint it",
              len(set(truth) - set(parsed)) == 10,
              f"{len(set(truth) - set(parsed))} hand-entered fixtures not in "
              "the article")

    print("\n" + "-" * 62)
    print(f"{PASS}/{PASS + FAIL} checks passed"
          + (f", {SKIP} skipped" if SKIP else ""))
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
