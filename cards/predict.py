"""Predict the yellow-card distribution for a fixture that has not happened.

    .venv/bin/python cards/predict.py --home Bolton --away Wigan \
        --division E2 --date 2026-09-20 --referee "Salisbury"

The trick that keeps this honest: a fixture is appended to the match history as
a row with no result, and the ordinary feature builder is run over the whole
thing. Because that builder updates its state only after a row is emitted, and
only when the row has a result, the fixture gets exactly the features it would
have had on the day — computed by the same code path that produced the training
features. There is no separate "inference" implementation to drift out of sync
with the training one, which is where this class of system usually goes wrong.

Ask about a past date and you get the same thing: what the model would have
said at the time, using only what was known then.
"""

from __future__ import annotations

import argparse
import difflib
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import features
import loader
import models

DIVISION_NAMES = loader.DIVISIONS


class CardPredictor:
    """Fits once, then answers fixtures.

    `history_seasons` defaults to every season on disk, holdout included.
    That is deliberate and is NOT a holdout look: no model is scored here, and
    nothing about the holdout's outcomes reaches a reported metric. Predicting
    next Saturday's match genuinely requires last Saturday's results, and
    withholding them would produce a worse prediction for no methodological
    gain. Holdout looks are counted when a model is *evaluated*, in
    `models.walk_forward` and the holdout runner — not here.
    """

    def __init__(self, train_max_season: str | None = None):
        self.train_max_season = train_max_season
        self.history_: pd.DataFrame | None = None
        self.model_: models.NBGLM | None = None

    def fit(self, verbose: bool = False):
        hist = loader.load_matches(unlock_holdout=True)
        self.history_ = hist

        feat = features.build(hist)
        cols = features.feature_columns(feat)
        train = feat
        if self.train_max_season:
            train = feat[feat.season <= self.train_max_season]
        self.cols_ = cols
        self.model_ = models.NBGLM(cols).fit(train)
        if verbose:
            print(f"trained on {len(train):,} matches "
                  f"({train.season.min()}..{train.season.max()}), "
                  f"history through {hist.season.max()}, alpha "
                  f"{self.model_.alpha_:.4f}")
        return self

    # ---------------------------------------------------------- name lookup

    def teams(self, division: str | None = None) -> list[str]:
        h = self.history_
        if division:
            h = h[h.division == division]
        return sorted(set(h.HomeTeam) | set(h.AwayTeam))

    def referees(self) -> list[str]:
        return sorted(x for x in self.history_.Referee.dropna().unique())

    def resolve(self, name: str, pool: list[str]) -> tuple[str | None, list[str]]:
        """Exact match, else case-insensitive, else close suggestions.

        Team names in this data are abbreviated in ways nobody guesses first
        time ("Man United", "Nott'm Forest", "Sheffield Weds"), so a miss
        returns candidates rather than an error.
        """
        if name in pool:
            return name, []
        low = {p.lower(): p for p in pool}
        if name.lower() in low:
            return low[name.lower()], []
        close = difflib.get_close_matches(name, pool, n=6, cutoff=0.5)
        if not close:
            close = [p for p in pool if name.lower() in p.lower()][:6]
        return None, close

    # ------------------------------------------------------------- predict

    def predict(self, home: str, away: str, division: str, date: str,
                referee: str | None = None,
                odds: dict | None = None) -> dict:
        if self.model_ is None:
            raise RuntimeError("call fit() first")
        hist = self.history_
        when = pd.Timestamp(date)

        season = _season_code_for(when)
        # Pre-match prices, if supplied. These drive the mkt_* features, which
        # are the only inputs carrying what the market knows about squad
        # quality, injuries and motivation. Without them the model still runs,
        # but it runs blind to all of that -- so the output says so.
        o = odds or {}
        fixture = pd.DataFrame([{
            "odds_h": o.get("home", np.nan), "odds_d": o.get("draw", np.nan),
            "odds_a": o.get("away", np.nan),
            "odds_o25": o.get("over25", np.nan),
            "odds_u25": o.get("under25", np.nan),
            "season": season, "division": division, "date": when,
            "HomeTeam": home, "AwayTeam": away, "Referee": referee,
            "HY": np.nan, "AY": np.nan, "HR": np.nan, "AR": np.nan,
            "HF": np.nan, "AF": np.nan, "HC": np.nan, "AC": np.nan,
            "HS": np.nan, "AS": np.nan, "HST": np.nan, "AST": np.nan,
            "FTHG": np.nan, "FTAG": np.nan, "FTR": None,
            "yellows": np.nan, "reds": np.nan, "fouls": np.nan,
            "corners": np.nan, "shots": np.nan,
        }])
        combined = pd.concat([hist, fixture], ignore_index=True, sort=False)

        feat = features.build(combined)
        row = feat[(feat.date == when) & (feat.home == home)
                   & (feat.away == away)]
        if len(row) == 0:
            raise RuntimeError("fixture row vanished during feature build")
        row = row.tail(1)

        mu = float(self.model_.predict_mu(row)[0])
        alpha = self.model_.alpha_
        pmf = models.nb_pmf_matrix(np.array([mu]), alpha)[0]
        cdf = np.cumsum(pmf)

        return {
            "home": home, "away": away, "division": division,
            "date": str(when.date()), "referee": referee,
            "season": season,
            "mu": mu, "alpha": alpha,
            "pmf": pmf, "cdf": cdf,
            "row": row,
            "mkt_known": bool(row.mkt_known.iloc[0]),
            "mkt_closeness": float(row.mkt_closeness.iloc[0]),
            "mkt_p_draw": float(row.mkt_p_draw.iloc[0]),
            "mkt_home_edge": float(row.mkt_home_edge.iloc[0]),
            "mkt_p_over25": float(row.mkt_p_over25.iloc[0]),
            "baseline": float(row.baseline_yellows.iloc[0]),
            "baseline_source": str(row.baseline_source.iloc[0]),
            "ref_n_career": int(row.ref_n_career.iloc[0]),
            "ref_known": bool(row.ref_known.iloc[0]),
        }

    # -------------------------------------------------------- explanation

    def contributions(self, row: pd.DataFrame, top: int = 6) -> pd.DataFrame:
        """Which features pushed this fixture away from its league baseline."""
        X, names = self.model_._design(row)
        contrib = X[0] * self.model_.params_
        df = pd.DataFrame({"feature": names, "contribution": contrib})
        df = df[df.feature != "intercept"]
        df = df[df.contribution.abs() > 1e-9]
        return df.reindex(df.contribution.abs().sort_values(
            ascending=False).index).head(top)


def _season_code_for(d: pd.Timestamp) -> str:
    """Football seasons run Aug-May; a July date belongs to the season ending."""
    start = d.year if d.month >= 7 else d.year - 1
    return f"{start % 100:02d}{(start + 1) % 100:02d}"


# ---------------------------------------------------------------- rendering


def render(res: dict, predictor: CardPredictor, prices: dict | None = None
           ) -> str:
    pmf, cdf = res["pmf"], res["cdf"]
    lines = []
    lines.append(f"{res['home']} v {res['away']} — "
                 f"{DIVISION_NAMES.get(res['division'], res['division'])} — "
                 f"{res['date']}")
    if res["ref_known"] and res["ref_n_career"] >= features.MIN_REF_MATCHES:
        f10 = res["row"].ref_yel10.iloc[0]
        lines.append(f"Referee: {res['referee']} "
                     f"({res['ref_n_career']} matches on record"
                     + (f", {f10:.2f}x league average)" if np.isfinite(f10)
                        else ")"))
    elif res["referee"]:
        lines.append(f"Referee: {res['referee']} — too few matches on record "
                     f"({res['ref_n_career']}); using league average")
    else:
        lines.append("Referee: not supplied — using league average")

    if res.get("mkt_known"):
        ph = (1.0 + res["mkt_home_edge"] - res["mkt_p_draw"]) / 2.0
        pa = ph - res["mkt_home_edge"]
        lines.append(f"Market: home {ph:.0%} / draw {res['mkt_p_draw']:.0%} / "
                     f"away {pa:.0%}  — closeness {res['mkt_closeness']:.2f}"
                     + (f", P(over 2.5 goals) {res['mkt_p_over25']:.0%}"
                        if np.isfinite(res["mkt_p_over25"]) else ""))
    else:
        lines.append("Market: no odds supplied — running without the market "
                     "features, which carry squad quality, injuries and "
                     "motivation. Pass --odds-home/--odds-draw/--odds-away "
                     "for a better estimate.")
    lines.append(f"League baseline: {res['baseline']:.2f} yellows "
                 f"(source: {res['baseline_source']})")
    lines.append(f"Model expectation: {res['mu']:.2f} yellows"
                 f"   [dispersion alpha {res['alpha']:.4f}]")
    lines.append("")
    lines.append("  yellows   P(exactly)        line    P(over)")
    for k in range(0, 9):
        over = 1.0 - cdf[k] if k < len(cdf) else 0.0
        lines.append(f"     {k:<2}       {pmf[k]:6.2%}          "
                     f"{k + 0.5:<4}     {over:6.2%}")
    tail = 1.0 - cdf[8]
    lines.append(f"     9+       {tail:6.2%}")

    con = predictor.contributions(res["row"])
    if len(con):
        lines.append("")
        lines.append("  what moved it from the league baseline:")
        for _, r in con.iterrows():
            arrow = "raises" if r.contribution > 0 else "lowers"
            lines.append(f"    {r.feature:<24} {arrow} by "
                         f"{abs(np.expm1(r.contribution)):.1%}")

    if prices:
        lines.append("")
        lines.append("  against your prices:")
        for line, price in sorted(prices.items()):
            k = int(np.floor(line))
            p_over = 1.0 - cdf[k]
            for side, p in (("over", p_over), ("under", 1.0 - p_over)):
                ev = p * price - 1.0
                if side == "over":
                    tag = "VALUE" if ev > 0 else "pass"
                    lines.append(f"    {side} {line} @ {price:.2f}  "
                                 f"implied {1/price:.1%}  model {p:.1%}  "
                                 f"EV {ev:+.1%}  {tag}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--home")
    ap.add_argument("--away")
    ap.add_argument("--division", required=True, choices=sorted(loader.DIVISIONS))
    ap.add_argument("--date", help="YYYY-MM-DD")
    ap.add_argument("--referee", default=None)
    ap.add_argument("--odds-home", type=float, default=None,
                    help="decimal match odds, home win")
    ap.add_argument("--odds-draw", type=float, default=None)
    ap.add_argument("--odds-away", type=float, default=None)
    ap.add_argument("--odds-over25", type=float, default=None,
                    help="decimal odds, over 2.5 GOALS (not cards)")
    ap.add_argument("--odds-under25", type=float, default=None)
    ap.add_argument("--price-over", type=float, default=None,
                    help="decimal odds for the over, paired with --line")
    ap.add_argument("--line", type=float, default=3.5)
    ap.add_argument("--list-teams", action="store_true")
    a = ap.parse_args()

    if a.list_teams:
        p = CardPredictor()
        p.history_ = loader.load_matches(unlock_holdout=True)
        print("\n".join(p.teams(a.division)))
        return 0
    if not (a.home and a.away and a.date):
        ap.error("--home, --away and --date are required unless --list-teams")

    p = CardPredictor().fit(verbose=True)

    pool = p.teams(a.division)
    problems = []
    for label, val in (("home", a.home), ("away", a.away)):
        got, close = p.resolve(val, pool)
        if got is None:
            problems.append(f"{label} team {val!r} not found in "
                            f"{a.division}. Did you mean: {', '.join(close) or '(no close match)'}")
        else:
            setattr(a, label, got)
    ref = a.referee
    if ref:
        got, close = p.resolve(ref, p.referees())
        if got is None:
            print(f"note: referee {ref!r} not on record"
                  + (f"; close matches: {', '.join(close)}" if close else "")
                  + " — continuing with league average")
            ref = None
        else:
            ref = got
    if problems:
        print("\n".join(problems))
        return 1

    odds = {}
    if a.odds_home and a.odds_draw and a.odds_away:
        odds.update(home=a.odds_home, draw=a.odds_draw, away=a.odds_away)
    elif any((a.odds_home, a.odds_draw, a.odds_away)):
        print("note: match odds need all three of home/draw/away to remove the "
              "bookmaker's margin — ignoring the partial set")
    if a.odds_over25 and a.odds_under25:
        odds.update(over25=a.odds_over25, under25=a.odds_under25)
    elif a.odds_over25 or a.odds_under25:
        print("note: goals over/under needs both sides to de-vig — ignoring")

    res = p.predict(a.home, a.away, a.division, a.date, ref, odds or None)
    prices = {a.line: a.price_over} if a.price_over else None
    print()
    print(render(res, p, prices))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
