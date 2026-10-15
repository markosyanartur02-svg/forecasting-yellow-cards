"""Every network request this project makes goes through here.

    import polite
    text = polite.get("https://www.efl.com/category/match-officials/")

Three rules, enforced rather than remembered:

1. SAY WHAT YOU ARE. The User-Agent names this tool. The first versions sent a
   copy of Chrome's, which made the scraper invisible to the very rules meant
   to govern it -- a site's robots.txt addresses crawlers by User-Agent, and a
   crawler pretending to be a browser has opted out of being addressed.

2. ASK FIRST. Before fetching a URL, read the host's robots.txt and refuse if
   it disallows the path. Nothing in this project checked until October 2026,
   and when it finally did, BBC's internal API turned out to disallow all
   crawlers -- after several hundred requests to it. `Disallowed` is raised,
   not logged and skipped, so a refused fetch cannot pass for an empty page.

3. DON'T HURRY. At least `MIN_INTERVAL` seconds between requests to the same
   host, longer if robots.txt sets a Crawl-delay. During development this
   project fetched one EFL article dozens of times in a few minutes while
   debugging a parser; the fix for that is to save a copy (see testdata/) and
   parse the copy, and this limit is the backstop.

robots.txt IS NECESSARY, NOT SUFFICIENT

It is machine-readable and says only which paths a crawler may fetch. A site's
terms of use can forbid what robots.txt permits: BBC's match pages pass the
robots check for a generic crawler, while BBC states in the same file -- in
comments a parser ignores -- "No scraping, crawling, or systematic extraction
of content" and no agentic-AI use. So hosts whose terms rule this project out
are listed in BLOCKED with the reason, and refused before robots.txt is even
read.

How robots.txt is interpreted follows RFC 9309:
  - 2xx: obey the rules.
  - 4xx: the site has published no rules; everything is allowed.
  - 5xx or unreachable: assume everything is disallowed. A server that cannot
    answer "may I?" has not said yes.
  - A 200 response that is plainly an HTML page (efl.com serves its homepage
    at /robots.txt) contains no rules, so it is treated like a 404.
"""

from __future__ import annotations

import time
import urllib.robotparser
from urllib.parse import urlsplit

import requests

AGENT_TOKEN = "football-cards"
USER_AGENT = (f"{AGENT_TOKEN}/1.0 (personal, non-commercial research project; "
              "one request per page, at most daily)")
MIN_INTERVAL = 3.0          # seconds between requests to the same host
TIMEOUT = 60


# Host suffix -> why. Checked before robots.txt.
BLOCKED = {
    "bbc.co.uk": "BBC terms prohibit scraping, systematic extraction and "
                 "agentic-AI use of BBC content",
    "bbci.co.uk": "BBC's API host; robots.txt disallows all crawlers, and the "
                  "BBC terms above apply",
}


class Disallowed(Exception):
    """robots.txt for this host does not permit fetching this URL."""


_session = requests.Session()
_session.headers["User-Agent"] = USER_AGENT
_robots: dict[str, tuple[urllib.robotparser.RobotFileParser | None, str]] = {}
_last_hit: dict[str, float] = {}


def _rules_for(scheme_host: str) -> tuple[urllib.robotparser.RobotFileParser | None, str]:
    """(parser, verdict). parser None means 'allow all' or 'deny all' per verdict."""
    if scheme_host in _robots:
        return _robots[scheme_host]
    url = f"{scheme_host}/robots.txt"
    try:
        r = _session.get(url, timeout=TIMEOUT, allow_redirects=True)
    except requests.RequestException:
        result = interpret(None, "")
    else:
        result = interpret(r.status_code, r.text if r.status_code == 200 else "")
    _robots[scheme_host] = result
    return result


def interpret(status: int | None, body: str):
    """What a robots.txt response means, per RFC 9309. No network here."""
    if status is None:
        return (None, "deny: robots.txt unreachable")
    if 400 <= status < 500:
        return (None, f"allow: no robots.txt (HTTP {status})")
    if status >= 500:
        return (None, f"deny: robots.txt returned HTTP {status}")
    head = body[:2000].lower()
    if "<html" in head or "<!doctype" in head:
        return (None, "allow: robots.txt is an HTML page, no rules")
    rp = urllib.robotparser.RobotFileParser()
    rp.parse(body.splitlines())
    return (rp, "rules")


def load_rules(scheme_host: str, robots_txt: str) -> None:
    """Install rules from text instead of fetching them. For tests."""
    rp = urllib.robotparser.RobotFileParser()
    rp.parse(robots_txt.splitlines())
    _robots[scheme_host] = (rp, "rules")


def allowed(url: str) -> tuple[bool, str]:
    parts = urlsplit(url)
    host = parts.hostname or ""
    for suffix, why in BLOCKED.items():
        if host == suffix or host.endswith("." + suffix):
            return False, f"blocked: {why}"
    scheme_host = f"{parts.scheme}://{parts.netloc}"
    rp, verdict = _rules_for(scheme_host)
    if rp is None:
        return verdict.startswith("allow"), verdict
    ok = rp.can_fetch(AGENT_TOKEN, url)
    return ok, "allowed by robots.txt" if ok else "disallowed by robots.txt"


def delay_for(host: str) -> float:
    """Seconds to leave between requests to this host."""
    rp, _ = _robots.get(host, (None, ""))
    delay = MIN_INTERVAL
    if rp is not None:
        cd = rp.crawl_delay(AGENT_TOKEN)
        if cd:
            delay = max(delay, float(cd))
    return delay


def _wait_turn(host: str) -> None:
    delay = delay_for(host)
    last = _last_hit.get(host)
    if last is not None:
        gap = time.monotonic() - last
        if gap < delay:
            time.sleep(delay - gap)
    _last_hit[host] = time.monotonic()


def fetch(url: str, tries: int = 3) -> requests.Response:
    """The response, after the robots check and the per-host wait.

    Raises Disallowed if robots.txt says no, and RuntimeError if every attempt
    fails. A non-200 response is returned, not raised -- callers decide what a
    404 means.
    """
    ok, why = allowed(url)
    if not ok:
        raise Disallowed(f"{url}: {why}")
    parts = urlsplit(url)
    host = f"{parts.scheme}://{parts.netloc}"
    last = None
    for i in range(tries):
        _wait_turn(host)
        try:
            return _session.get(url, timeout=TIMEOUT, allow_redirects=True)
        except requests.RequestException as e:
            last = e
            time.sleep(2.0 * (i + 1))
    raise RuntimeError(f"failed after {tries} tries: {url}\n  {last}")


def get(url: str, tries: int = 3, encoding: str | None = None) -> str:
    """Page text for a 200 response; RuntimeError otherwise."""
    r = fetch(url, tries=tries)
    if r.status_code != 200:
        raise RuntimeError(f"{url}: HTTP {r.status_code}")
    if encoding:
        return r.content.decode(encoding, errors="replace")
    return r.text
