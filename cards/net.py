"""Wait for the network before a scheduled job touches it.

launchd runs a missed job the moment the Mac wakes, which is the whole reason
it replaced cron. But at that moment the network is usually not up yet: DNS
fails with "nodename nor servname provided", every fetch fails, and the job
records a clean-looking "unavailable" for the day.

That happened on 23 and 24 September. The top-up reported 4 unavailable and
the referee check died with a traceback, on a machine whose network was
perfectly fine a minute later.

So anything scheduled waits for a name to resolve before it starts. Ten
minutes is long enough for wi-fi to associate after a wake and short enough
that a genuinely offline machine gives up rather than hanging until the next
run.
"""

from __future__ import annotations

import socket
import time

PROBES = ("www.football-data.co.uk", "www.efl.com")


def online(host: str, timeout: float = 5.0) -> bool:
    try:
        socket.setdefaulttimeout(timeout)
        socket.getaddrinfo(host, 443, proto=socket.IPPROTO_TCP)
        return True
    except OSError:
        return False


def _spell(seconds: float) -> str:
    if seconds >= 120:
        return f"{int(seconds // 60)} minutes"
    return f"{int(seconds)}s"


def wait_for_network(timeout_s: float = 600.0, interval_s: float = 20.0,
                     log=print) -> bool:
    """Block until a probe host resolves. True if the network came up.

    Messages carry no leading indent; each caller prefixes them to match its
    own log format.
    """
    if any(online(h) for h in PROBES):
        return True
    log(f"network is not up yet -- waiting up to {_spell(timeout_s)} "
        "(this is normal just after the Mac wakes)")
    start = time.monotonic()
    deadline = start + timeout_s
    while time.monotonic() < deadline:
        time.sleep(interval_s)
        if any(online(h) for h in PROBES):
            log(f"network came up after about "
                f"{_spell(time.monotonic() - start)}")
            return True
    log(f"still offline after {_spell(timeout_s)} -- giving up until the next "
        "scheduled run")
    return False
