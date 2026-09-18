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
# EFL referees added 2026-09-18 from efl.com "Referee appointments: 18-20
# September 2026", published 17 Sep. On 17 September the EFL's latest article
# still covered 12-17 September, so the first run of this forward test had the
# referee for the ten Premier League fixtures only.
#
# All 35 resolve against our data and every one clears the 5-match floor,
# ranging from Emily Heaslip (6) to Dean Whitestone (580).
#
# AFC Wimbledon v MK Dons is not in the EFL article because it was played on
# Thursday 17 September, before the window it covers. Its prediction was made
# before kick-off and stands as-is.
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
    ("E1", "2026-09-18", "Bristol City", "Watford", "o langford"),
    ("E1", "2026-09-19", "Cardiff City", "Charlton Athletic", "e bell"),
    ("E1", "2026-09-19", "Millwall", "West Ham United", "p tierney"),
    ("E1", "2026-09-19", "Stoke City", "Sheffield United", "t robinson"),
    ("E1", "2026-09-19", "Birmingham City", "Middlesbrough", "d webb"),
    ("E1", "2026-09-19", "Burnley", "Derby County", "s martin"),
    ("E1", "2026-09-19", "Lincoln City", "Swansea City", "a backhouse"),
    ("E1", "2026-09-19", "Portsmouth", "Blackburn Rovers", "j busby"),
    ("E1", "2026-09-19", "Queens Park Rangers", "Preston North End", "t parsons"),
    ("E1", "2026-09-19", "Wrexham", "Southampton", "g ward"),
    ("E1", "2026-09-20", "Wolverhampton Wanderers", "West Bromwich Albion", "j smith"),
    ("E1", "2026-09-20", "Norwich City", "Bolton Wanderers", "t nield"),

    # ---- League One
    #
    # AFC Wimbledon v MK Dons kicks off tonight, Thursday 17 September, and was
    # likewise missing from ESPN's weekend page. Found because the EFL news
    # feed carried a pre-match stat pack for it.
    ("E2", "2026-09-17", "AFC Wimbledon", "MK Dons", None),
    ("E2", "2026-09-19", "Barnsley", "Leicester City", "s stockbridge"),
    ("E2", "2026-09-19", "Burton Albion", "Mansfield Town", "o mackey"),
    ("E2", "2026-09-19", "Blackpool", "Plymouth Argyle", "d drysdale"),
    ("E2", "2026-09-19", "Bromley", "Huddersfield Town", "c brook"),
    ("E2", "2026-09-19", "Leyton Orient", "Stevenage", "c hicks"),
    ("E2", "2026-09-19", "Luton Town", "Bradford City", "r martin"),
    ("E2", "2026-09-19", "Oxford United", "Cambridge United", "t kirk"),
    ("E2", "2026-09-19", "Peterborough United", "Doncaster Rovers", "w davis"),
    ("E2", "2026-09-19", "Reading", "Notts County", "s mather"),
    ("E2", "2026-09-19", "Sheffield Wednesday", "Stockport County", "b toner"),
    ("E2", "2026-09-19", "Wigan Athletic", "Wycombe Wanderers", "d whitestone"),

    # ---- League Two
    ("E3", "2026-09-19", "Rochdale", "Oldham Athletic", "s parkinson"),
    ("E3", "2026-09-19", "Walsall", "Port Vale", "j oldham"),
    ("E3", "2026-09-19", "Accrington Stanley", "Newport County", "j robinson"),
    ("E3", "2026-09-19", "Barnet", "Fleetwood Town", "s jackson"),
    ("E3", "2026-09-19", "Chesterfield", "York City", "b speedie"),
    ("E3", "2026-09-19", "Colchester United", "Cheltenham Town", "a young"),
    ("E3", "2026-09-19", "Crewe Alexandra", "Shrewsbury Town", "a chilowicz"),
    ("E3", "2026-09-19", "Exeter City", "Tranmere Rovers", "i searle"),
    ("E3", "2026-09-19", "Gillingham", "Bristol Rovers", "s gill"),
    ("E3", "2026-09-19", "Grimsby Town", "Crawley Town", "e heaslip"),
    ("E3", "2026-09-19", "Northampton Town", "Rotherham United", "w finnie"),
    ("E3", "2026-09-19", "Salford City", "Swindon Town", "n hair"),
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
