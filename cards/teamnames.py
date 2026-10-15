"""Map a club name as some other source writes it onto football-data's spelling.

    teamnames.resolve("Nottingham Forest", known_names)  ->  "Nott'm Forest"

football-data writes short names ("Man City", "Sheffield Weds"); the EFL's
appointment articles write full ones ("Manchester City", "Sheffield
Wednesday"). Matching them wrongly would attach one fixture's referee to
another fixture's prediction, which is worse than failing, so anything that
is not an unambiguous match returns None.

Moved here from fetch_results_bbc.py when the BBC scraper was retired, since
the EFL articles still need it.
"""

from __future__ import annotations

# BBC writes full club names, football-data writes short ones. Most pairs fall
# out of dropping the common suffixes; the rest are listed, because a wrong
# guess here attaches one match's result to another match's prediction, which
# is worse than a crash.
_SUFFIXES = (" football club", " fc", " afc", " united", " city", " town",
             " athletic", " rovers", " wanderers", " albion", " county",
             " alexandra", " argyle", " stanley", " hotspur", " orient",
             " north end", " and hove albion", " rangers", " forest",
             " villa", " dons", " palace")
MANUAL = {
    "nottingham forest": "nott'm forest",
    "queens park rangers": "qpr",
    "sheffield wednesday": "sheffield weds",
    "manchester united": "man united",
    "manchester city": "man city",
    "west bromwich albion": "west brom",
    "brighton and hove albion": "brighton",
    "wolverhampton wanderers": "wolves",
    "peterborough united": "peterboro",
    "bristol rovers": "bristol rvs",
    "tottenham hotspur": "tottenham",
    "mk dons": "milton keynes dons",
}


def _base(s: str) -> str:
    s = (s or "").lower().replace("&", "and").replace(".", "").replace("'", "")
    return " ".join(s.split())


def _strip(s: str) -> str:
    for suf in sorted(_SUFFIXES, key=len, reverse=True):
        if s.endswith(suf) and len(s) > len(suf) + 2:
            return s[:-len(suf)].strip()
    return s


def resolve(bbc_name: str, choices) -> str | None:
    """BBC's club name -> the football-data name, or None if not unambiguous."""
    b = _base(bbc_name)
    if b in MANUAL:
        want = _base(MANUAL[b])          # MANUAL is written readably, so it is
        for c in choices:                # normalised here rather than by hand
            if _base(c) == want:
                return c
    for c in choices:
        if _base(c) == b:
            return c
    bs = _strip(b)
    for rule in (lambda c: _strip(_base(c)) == bs,
                 lambda c: _base(c).startswith(bs) or bs.startswith(_base(c))):
        hits = [c for c in choices if rule(c)]
        if len(hits) == 1:
            return hits[0]
    return None


def correction(dismissals: list[dict], enabled: bool = False) -> tuple[int, int, int]:
    """(yellows to add back, two-caution dismissals, straight reds).

    `enabled` is False by default -- see the note at the top of this file.
    The dismissal counts are still reported, because "this match had a
    sending-off" is useful even when the adjustment is not trusted.
    """
    two = sum(1 for d in dismissals
              if (d.get("type") or "").strip().lower() == TWO_YELLOWS)
    straight = len(dismissals) - two
    return (2 * two if enabled else 0), two, straight
