"""Watch the EFL for referee appointments, and say when they land.

    .venv/bin/python cards/check_referees.py            # check, quietly
    .venv/bin/python cards/check_referees.py --verbose  # print what was parsed

Run from the daily launchd job. Most days it prints one line and exits. On the
day the appointments appear it says so, saves them, and names the fixtures it
can fill in.

The referee is 48% of the model's value, so a week's predictions made before
the appointments are the weaker version of the model. Knowing the hour they
land is worth a daily HTTP request.

TWO FORMATS, AND WHY BOTH ARE PARSED

League articles put the fixture in an <h4> and the officials in a <p>:

    <h4>Bristol City v Watford (20:00)</h4>
    <p>Referee: Oliver Langford Assistants: ... Fourth Official: ...</p>

Cup articles (EFL Trophy, Carabao) use a flat paragraph with <br> and the
referee in capitals. Both are read, and the competition heading above each
block decides the division, so a Trophy article cannot be mistaken for a
League Two round.

The content is inside the page's Nuxt payload rather than its rendered body,
so this reads the raw HTML rather than stripped text.

VALIDATED AGAINST KNOWN GROUND TRUTH

The 18-20 September article is parsed and checked against the 45 referees that
were entered by hand into fixtures_week.py last week. All 45 agree, which is
the only reason this is trusted to fill them in automatically.
"""

from __future__ import annotations

import argparse
import html as htmlmod
import json
import os
import re
import sys
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import fetch_results_bbc as bbc
import net

CATEGORY = "https://www.efl.com/category/match-officials/"
STATE = os.path.join(HERE, "referee_appointments.json")

COMPETITIONS = {
    "championship": "E1",
    "league one": "E2",
    "league two": "E3",
}
# Articles that are not a league round. Kept separate rather than ignored, so
# "appointments are out" never fires on a cup article.
CUPS = ("trophy", "carabao", "play-off", "playoff", "laws-of-the-game", "laws")

LINK = re.compile(r'"(/news/(\d{4})/([a-z]+)/(\d{2})/([a-z0-9\-]*referee[a-z0-9\-]*)/)"')
H2 = re.compile(r"<h2[^>]*>(.*?)</h2>", re.S)
H3 = re.compile(r"<h3[^>]*>(.*?)</h3>", re.S)
FIXTURE_H4 = re.compile(
    r"<h4[^>]*>\s*(.+?)\s+v\s+(.+?)\s*\((\d{1,2}[:.]\d{2})\)\s*</h4>"
    r".{0,12}?<p[^>]*>\s*Referee:\s*(.+?)\s+Assistants?:", re.S | re.I)
FIXTURE_BR = re.compile(
    r"<p[^>]*>\s*(.+?)\s+v\s+(.+?)\s*\((\d{1,2}[:.]\d{2})\)\s*<br\s*/?>"
    r"\s*([A-Z][A-Z '&.\-]{3,40}?)\s*<br", re.S)


def clean(s: str) -> str:
    return " ".join(htmlmod.unescape(re.sub(r"<[^>]+>", "", s)).split())


def model_name(full: str) -> str:
    """'Oliver Langford' -> 'o langford', the form the model's data uses."""
    parts = clean(full).split()
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0].lower()
    return f"{parts[0][0].lower()} {parts[-1].lower()}"


def list_articles(raw: str) -> list[dict]:
    out, seen = [], set()
    for path, year, month, day, slug in LINK.findall(raw):
        if path in seen:
            continue
        seen.add(path)
        out.append({
            "url": "https://www.efl.com" + path,
            "slug": slug,
            "is_cup": any(c in slug for c in CUPS),
            "published": f"{year}-{month}-{day}",
        })
    return out


def parse_article(raw: str) -> list[dict]:
    """Fixtures with date, competition and referee, in document order."""
    marks = []
    for m in H2.finditer(raw):
        marks.append((m.start(), "date", clean(m.group(1))))
    for m in H3.finditer(raw):
        marks.append((m.start(), "comp", clean(m.group(1))))
    for pat, kind in ((FIXTURE_H4, "h4"), (FIXTURE_BR, "br")):
        for m in pat.finditer(raw):
            marks.append((m.start(), "fixture",
                          (clean(m.group(1)), clean(m.group(2)),
                           clean(m.group(3)), clean(m.group(4)))))
    marks.sort(key=lambda x: x[0])

    date = comp = None
    out = []
    for _, kind, val in marks:
        if kind == "date":
            date = val
        elif kind == "comp":
            comp = val
        elif kind == "fixture":
            home, away, time, ref = val
            div = None
            if comp:
                low = comp.lower()
                for needle, code in COMPETITIONS.items():
                    if needle in low:
                        div = code
                        break
            out.append({"date_text": date, "competition": comp,
                        "division": div, "home": home, "away": away,
                        "time": time, "referee_full": ref,
                        "referee": model_name(ref)})
    return out


def load_state() -> dict:
    if os.path.exists(STATE):
        with open(STATE) as f:
            return json.load(f)
    return {}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--url", help="parse this article instead of looking one up")
    a = ap.parse_args()

    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    if a.url:
        fx = parse_article(bbc.get(a.url))
        print(f"{stamp} parsed {len(fx)} fixtures from {a.url}")
        for f in fx:
            print(f"   {f['division'] or '--'} {f['date_text']}  "
                  f"{f['home']} v {f['away']} ({f['time']})  "
                  f"{f['referee_full']}  ->  {f['referee']}")
        return 0

    if not net.wait_for_network(
            log=lambda m: print(f"{stamp} referees: {m}")):
        print(f"{stamp} referees: skipped, no network")
        return 0

    try:
        arts = list_articles(bbc.get(CATEGORY))
    except Exception as e:
        # A scheduled job that dies with a traceback looks identical in the
        # log to one that never ran. Say what happened and exit cleanly.
        print(f"{stamp} referees: could not reach the EFL ({type(e).__name__}) "
              "-- will try again tomorrow")
        return 0
    league = [x for x in arts if not x["is_cup"]]
    if not league:
        print(f"{stamp} referees: no league appointment article found")
        return 0
    newest = league[0]

    state = load_state()
    if state.get("url") == newest["url"]:
        print(f"{stamp} referees: nothing new "
              f"(latest is still '{newest['slug']}')")
        return 0

    fx = parse_article(bbc.get(newest["url"]))
    covered = [f for f in fx if f["division"]]

    # "New article" is not the same as "this week's appointments". On a first
    # run the newest article is often last week's, and announcing that would
    # be a false alarm every time the state file is missing. What matters is
    # whether it covers the round we are waiting on.
    pending, hit = set(), set()
    try:
        import run_week_sheet as rws
        pending = {(h.lower(), aw.lower()) for _, _, h, aw, _ in rws.FIXTURES}
        got = {(f["home"].lower(), f["away"].lower()) for f in covered}
        hit = pending & got
    except Exception as e:
        print(f"{stamp} referees: could not read the pending sheet ({e})")

    record = {
        "url": newest["url"], "slug": newest["slug"],
        "published": newest["published"],
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_fixtures": len(covered),
        "fixtures": covered,
    }
    with open(STATE, "w") as f:
        json.dump(record, f, indent=2)

    if hit:
        print(f"{stamp} REFEREE APPOINTMENTS ARE OUT: {newest['slug']}")
        print(f"{stamp}   {newest['url']}")
        print(f"{stamp}   {len(covered)} league fixtures parsed, covering "
              f"{len(hit)} of the {len(pending)} in the pending sheet")
        for h, aw in sorted(pending - hit):
            print(f"{stamp}      still without one: {h} v {aw}")
        print(f"{stamp}   re-run: .venv/bin/python cards/run_week_sheet.py")
    else:
        print(f"{stamp} referees: new article '{newest['slug']}' "
              f"({len(covered)} fixtures), but none of them are the "
              f"{len(pending)} in the pending sheet -- still waiting")

    if a.verbose:
        for f in covered:
            print(f"   {f['division']} {f['home']} v {f['away']}: "
                  f"{f['referee_full']} -> {f['referee']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
