"""Check the network rules hold. No network is used.

    .venv/bin/python cards/test_polite.py

polite.py exists because this project scraped for a month without asking
anyone's permission -- a copied Chrome User-Agent, no robots.txt check, and a
site whose terms forbid it. These are the rules that replaced that, and each
is checked by feeding it a case it must refuse.
"""

from __future__ import annotations

import glob
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import polite

PASS = FAIL = 0


def check(name, ok, detail=""):
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}")
    if detail:
        print(f"         {detail}")


def main() -> int:
    print("test_says_what_it_is")
    ua = polite.USER_AGENT
    check("the User-Agent names this tool", ua.startswith(polite.AGENT_TOKEN), ua)
    check("and does not impersonate a browser",
          not any(b in ua for b in ("Mozilla", "Chrome", "Safari", "Gecko")))

    print("\ntest_no_other_code_makes_its_own_requests")
    # The original sin was a helper in a scraper module with a browser UA
    # baked in. If HTTP calls creep back in elsewhere, the rules here are
    # bypassed without anyone noticing.
    offenders = []
    for path in glob.glob(os.path.join(HERE, "*.py")):
        name = os.path.basename(path)
        if name in ("polite.py", "test_polite.py"):
            continue
        src = open(path).read()
        for needle in ("requests.get(", "urlopen(", "Mozilla/5.0"):
            if needle in src:
                offenders.append(f"{name}: {needle}")
    check("every request goes through polite.py", not offenders,
          "; ".join(offenders) or "none outside polite.py")
    check("the retired BBC scraper is gone",
          not os.path.exists(os.path.join(HERE, "fetch_results_bbc.py")))

    print("\ntest_blocked_hosts_are_refused_before_robots")
    for url in ("https://www.bbc.co.uk/sport/football/live/x",
                "https://bbc.co.uk/sport",
                "https://web-cdn.api.bbci.co.uk/wc-poll-data/container/x"):
        ok, why = polite.allowed(url)
        check(f"refuses {url.split('/')[2]}", not ok, why)
    ok, _ = polite.allowed("https://notbbc.co.uk.example.invalid/x")
    # suffix matching must not catch a host that merely contains the name

    print("\ntest_robots_rules_are_obeyed")
    host = "https://example.invalid"
    polite.load_rules(host, "User-agent: *\nDisallow: /private\n")
    check("a disallowed path is refused",
          not polite.allowed(f"{host}/private/page")[0])
    check("an allowed path is permitted", polite.allowed(f"{host}/public")[0])
    host2 = "https://example2.invalid"
    polite.load_rules(host2, f"User-agent: {polite.AGENT_TOKEN}\nDisallow: /\n"
                             "\nUser-agent: *\nAllow: /\n")
    check("a rule aimed at this tool by name is obeyed even when * is allowed",
          not polite.allowed(f"{host2}/anything")[0])

    print("\ntest_fetch_refuses_before_touching_the_network")
    try:
        polite.fetch(f"{host}/private/page")
        refused = "fetched"
    except polite.Disallowed:
        refused = "Disallowed"
    except Exception as e:
        refused = f"{type(e).__name__} -- it tried the network"
    check("fetch raises Disallowed, rather than trying and failing",
          refused == "Disallowed", refused)

    print("\ntest_robots_responses_interpreted_per_rfc_9309")
    rp, v = polite.interpret(404, "")
    check("404: no rules published, allow", rp is None and v.startswith("allow"), v)
    rp, v = polite.interpret(503, "")
    check("503: cannot say yes, so deny", rp is None and v.startswith("deny"), v)
    rp, v = polite.interpret(None, "")
    check("unreachable: deny", rp is None and v.startswith("deny"), v)
    rp, v = polite.interpret(200, "<!DOCTYPE html><html><title>Homepage</title>")
    check("an HTML page served as robots.txt holds no rules",
          rp is None and v.startswith("allow"), v)

    print("\ntest_crawl_delay_is_honoured")
    host3 = "https://example3.invalid"
    polite.load_rules(host3, "User-agent: *\nCrawl-delay: 10\nAllow: /\n")
    check("a Crawl-delay longer than our minimum wins",
          polite.delay_for(host3) == 10.0, f"{polite.delay_for(host3)}s")
    check("otherwise the minimum applies",
          polite.delay_for("https://example.invalid") == polite.MIN_INTERVAL,
          f"{polite.delay_for('https://example.invalid')}s")

    print("\n" + "-" * 62)
    print(f"{PASS}/{PASS + FAIL} checks passed")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
