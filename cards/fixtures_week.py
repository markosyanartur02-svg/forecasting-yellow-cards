"""The fixture list for the forward test, 18-20 September 2026.

Recorded here as data rather than fetched at run time, on purpose: this is a
pre-registered test. The predictions must be fixed before the results exist,
and a fixture list that could change underneath them would make that claim
unverifiable later.

Source: ESPN schedule pages for eng.1 through eng.4, read 2026-09-17. There is
no midweek round that week -- ESPN shows no Championship fixtures for 22-24
September.

Names are as ESPN writes them; `resolve_all` maps them onto the abbreviated
forms football-data.co.uk uses ("Tottenham Hotspur" -> "Tottenham").
"""

from __future__ import annotations

# (division, date, home, away, referee or None)
#
# Premier League referees from premierleague.com "Match officials for
# Matchweek 5", published 14 Sep 2026, read in a browser -- the page is
# JavaScript-rendered and plain fetching returns the officials without the
# fixtures they belong to.
#
# EFL referees are NOT here: as of 2026-09-17 the EFL's latest appointments
# article covers 12-17 September. The 19-24 September round had not been
# published. Those 34 fixtures run with the referee unknown.
FIXTURES = [
    # ---- Premier League
    ("E0", "2026-09-18", "Brentford", "Chelsea", "a madley"),
    ("E0", "2026-09-19", "Tottenham Hotspur", "Aston Villa", "s barrott"),
    ("E0", "2026-09-19", "Brighton & Hove Albion", "Arsenal", "d england"),
    ("E0", "2026-09-19", "Everton", "Ipswich Town", "s attwell"),
    ("E0", "2026-09-19", "Newcastle United", "Hull City", "t harrington"),
    ("E0", "2026-09-19", "Nottingham Forest", "Coventry City", "j gillett"),
    ("E0", "2026-09-20", "AFC Bournemouth", "Liverpool", "m oliver"),
    ("E0", "2026-09-20", "Leeds United", "Crystal Palace", "a herczeg"),
    ("E0", "2026-09-20", "Manchester City", "Sunderland", "r jones"),
    ("E0", "2026-09-20", "Fulham", "Manchester United", "p bankes"),

    # ---- Championship
    #
    # Bristol City v Watford was absent from ESPN's 19 September page because
    # it is a Friday-night fixture. A 24-team division plays 12 matches a
    # round, and ESPN listed 11 -- the count is what caught it.
    ("E1", "2026-09-18", "Bristol City", "Watford", None),
    ("E1", "2026-09-19", "Cardiff City", "Charlton Athletic", None),
    ("E1", "2026-09-19", "Millwall", "West Ham United", None),
    ("E1", "2026-09-19", "Stoke City", "Sheffield United", None),
    ("E1", "2026-09-19", "Birmingham City", "Middlesbrough", None),
    ("E1", "2026-09-19", "Burnley", "Derby County", None),
    ("E1", "2026-09-19", "Lincoln City", "Swansea City", None),
    ("E1", "2026-09-19", "Portsmouth", "Blackburn Rovers", None),
    ("E1", "2026-09-19", "Queens Park Rangers", "Preston North End", None),
    ("E1", "2026-09-19", "Wrexham", "Southampton", None),
    ("E1", "2026-09-20", "Wolverhampton Wanderers", "West Bromwich Albion", None),
    ("E1", "2026-09-20", "Norwich City", "Bolton Wanderers", None),

    # ---- League One
    #
    # AFC Wimbledon v MK Dons kicks off tonight, Thursday 17 September, and was
    # likewise missing from ESPN's weekend page. Found because the EFL news
    # feed carried a pre-match stat pack for it.
    ("E2", "2026-09-17", "AFC Wimbledon", "MK Dons", None),
    ("E2", "2026-09-19", "Barnsley", "Leicester City", None),
    ("E2", "2026-09-19", "Burton Albion", "Mansfield Town", None),
    ("E2", "2026-09-19", "Blackpool", "Plymouth Argyle", None),
    ("E2", "2026-09-19", "Bromley", "Huddersfield Town", None),
    ("E2", "2026-09-19", "Leyton Orient", "Stevenage", None),
    ("E2", "2026-09-19", "Luton Town", "Bradford City", None),
    ("E2", "2026-09-19", "Oxford United", "Cambridge United", None),
    ("E2", "2026-09-19", "Peterborough United", "Doncaster Rovers", None),
    ("E2", "2026-09-19", "Reading", "Notts County", None),
    ("E2", "2026-09-19", "Sheffield Wednesday", "Stockport County", None),
    ("E2", "2026-09-19", "Wigan Athletic", "Wycombe Wanderers", None),

    # ---- League Two
    ("E3", "2026-09-19", "Rochdale", "Oldham Athletic", None),
    ("E3", "2026-09-19", "Walsall", "Port Vale", None),
    ("E3", "2026-09-19", "Accrington Stanley", "Newport County", None),
    ("E3", "2026-09-19", "Barnet", "Fleetwood Town", None),
    ("E3", "2026-09-19", "Chesterfield", "York City", None),
    ("E3", "2026-09-19", "Colchester United", "Cheltenham Town", None),
    ("E3", "2026-09-19", "Crewe Alexandra", "Shrewsbury Town", None),
    ("E3", "2026-09-19", "Exeter City", "Tranmere Rovers", None),
    ("E3", "2026-09-19", "Gillingham", "Bristol Rovers", None),
    ("E3", "2026-09-19", "Grimsby Town", "Crawley Town", None),
    ("E3", "2026-09-19", "Northampton Town", "Rotherham United", None),
    ("E3", "2026-09-19", "Salford City", "Swindon Town", None),
]

# Resolution is by LONGEST PREFIX, not fuzzy similarity, plus the explicit
# table below.
#
# Fuzzy matching was tried first and is not safe here. difflib ranked
# "Newport County" above "Derby" for "Derby County", "Northampton" above
# "Wolves" for "Wolverhampton Wanderers", and "Newport County" above
# "Stockport" for "Stockport County". Accepting its top suggestion would have
# predicted the wrong fixture while looking entirely successful.
#
# Prefix matching handles 80 of the 88 names unambiguously. These eight cannot
# be reached that way and are written out by hand.
MANUAL = {
    "AFC Bournemouth": "Bournemouth",
    "Bristol Rovers": "Bristol Rvs",
    "Manchester City": "Man City",
    "Manchester United": "Man United",
    "Nottingham Forest": "Nott'm Forest",
    "Queens Park Rangers": "QPR",
    "Sheffield Wednesday": "Sheffield Weds",
    "Wolverhampton Wanderers": "Wolves",
    "MK Dons": "Milton Keynes Dons",
}


def resolve_all(predictor, verbose: bool = True):
    """Map ESPN names onto the names in our data.

    Resolution runs against EVERY team in the dataset, not just those seen in
    the fixture's own division. Teams are promoted and relegated constantly, so
    a club playing in League One this season may appear in our history only
    under League Two -- filtering by division would fail to find it.
    """
    pool = predictor.teams()
    out, unresolved = [], []
    for div, date, home, away, ref in FIXTURES:
        row = {"division": div, "date": date, "referee": ref,
               "espn_home": home, "espn_away": away}
        for side, name in (("home", home), ("away", away)):
            if name in MANUAL:
                row[side] = MANUAL[name]
                continue
            cands = [t for t in pool if name.lower().startswith(t.lower())]
            if cands:
                row[side] = max(cands, key=len)   # longest wins: "Bristol City" over "Bristol"
            else:
                row[side] = None
                unresolved.append((name, []))
        out.append(row)
    if verbose and unresolved:
        print(f"could not resolve {len(unresolved)} name(s):")
        for nm, close in unresolved:
            print(f"   {nm!r} -> closest: {', '.join(close) or '(none)'}")
    return out, unresolved
