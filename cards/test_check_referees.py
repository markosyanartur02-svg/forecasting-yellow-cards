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
import net

ARTICLE = ("https://www.efl.com/news/2026/september/17/"
           "referee-appointments--18-20-september-2026/")
# The format that cost a week: date in an <h3>, competition in a
# <p><strong>, fixtures as <p>...<br>NAME<br>. Parsed fine except that every
# fixture came out with no division, so the filter that keeps league fixtures
# discarded all fourteen and the watcher reported "still waiting" on the day
# the appointments were published.
ARTICLE_STRONG = ("https://www.efl.com/news/2026/september/25/"
                  "referee-appointments--26-september/")
# A cup round: parses, but must yield no league fixtures.
ARTICLE_CUP = ("https://www.efl.com/news/2026/september/21/"
               "referee-appointments--trophy/")

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

    print("\ntest_survives_no_network")
    # This is the case that actually happened. launchd runs a missed job the
    # instant the Mac wakes, before wi-fi is up; on 23 and 24 September DNS
    # failed and the checker died with a traceback, which in a log is
    # indistinguishable from never having run.
    real_probes = net.PROBES
    try:
        net.PROBES = ("no-such-host.invalid",)
        check("an unresolvable host is reported offline",
              net.online("no-such-host.invalid", timeout=2) is False)
        check("wait_for_network gives up and returns False",
              net.wait_for_network(timeout_s=2, interval_s=1,
                                   log=lambda m: None) is False)
        # main() waits ten minutes for the network in production, which is
        # right there and useless here, so shorten just that call.
        argv, real_wait = sys.argv[:], net.wait_for_network
        sys.argv = ["check_referees.py"]
        cr.net.wait_for_network = lambda log=print, **kw: real_wait(
            timeout_s=2, interval_s=1, log=log)
        try:
            rc = cr.main()
        except Exception as e:
            rc = f"raised {type(e).__name__}"
        finally:
            sys.argv = argv
            cr.net.wait_for_network = real_wait
        check("the checker exits cleanly instead of raising",
              rc == 0, f"returned {rc!r}")
    finally:
        net.PROBES = real_probes

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

    print("\ntest_pending_comes_from_the_workbook")
    # This coupling has broken twice. It used to read another module's
    # FIXTURES constant; when that module moved to date ranges the constant
    # vanished, and the watcher compared 31 published appointments against an
    # empty list and reported "still waiting". The workbook is what actually
    # knows which fixtures are pending.
    pend = cr.pending_fixtures()
    check("pending fixtures are read without importing run_week_sheet",
          isinstance(pend, set), f"{type(pend).__name__}")
    check("they are (home, away) pairs of plain strings",
          all(isinstance(x, tuple) and len(x) == 2
              and all(isinstance(n, str) for n in x) for x in pend),
          f"{len(pend)} pending")
    import run_week_sheet as rws
    check("run_week_sheet no longer exposes a FIXTURES constant, and that is "
          "no longer fatal",
          not hasattr(rws, "FIXTURES"))

    print("\ntest_reads_the_other_two_layouts")
    try:
        fx = cr.parse_article(bbc.get(ARTICLE_STRONG))
        div = [f for f in fx if f["division"]]
        check("competition in a <p><strong> is still recognised",
              len(div) == 14, f"{len(div)} of {len(fx)} fixtures got a division")
        check("divisions are League One and League Two",
              sorted({f["division"] for f in div}) == ["E2", "E3"],
              f"{sorted({f['division'] for f in div})}")
        by = {(f["home"], f["away"]): f["referee"] for f in div}
        check("a known fixture has the right referee",
              by.get(("Cambridge United", "AFC Wimbledon")) == "b toner",
              f"{by.get(('Cambridge United', 'AFC Wimbledon'))!r}")
        check("no fixture swallowed the page header as a team name",
              all(len(f["home"]) <= 40 and "window." not in f["home"]
                  for f in fx),
              f"longest home name: {max(len(f['home']) for f in fx)}")

        cup = cr.parse_article(bbc.get(ARTICLE_CUP))
        check("a cup round parses but yields no league fixtures",
              len(cup) > 0 and not [f for f in cup if f["division"]],
              f"{len(cup)} fixtures, "
              f"{len([f for f in cup if f['division']])} with a division")
    except Exception as e:
        SKIP += 1
        print(f"  [SKIP] EFL unreachable: {e}")

    print("\n" + "-" * 62)
    print(f"{PASS}/{PASS + FAIL} checks passed"
          + (f", {SKIP} skipped" if SKIP else ""))
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
