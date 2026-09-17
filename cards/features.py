"""Point-in-time feature construction.

Every feature for a match on date D is computed from matches played strictly
before D. That is enforced structurally rather than by care: the builder walks
matches in chronological order and updates its state *after* emitting each
row's features, so a future match physically cannot reach a past feature.

The alternative — groupby/rolling/shift over the whole table — is faster and
much easier to get subtly wrong. At 25k matches the explicit walk costs a
second and is obviously correct, which is the better trade for the one thing
in this project that must not be wrong.

PROJECT_SPEC section 5: the model consumes behaviour, never identity. Nothing
here emits a team name, a referee name, or a division label. Every rate is
divided by its own league-season baseline, so a feature means "relative to a
typical match in this context".
"""

from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import pandas as pd

# Rolling windows, in matches.
TEAM_WINDOWS = (5, 10, 20)
REF_WINDOWS = (10, 30)

# Below this many observations a rolling rate is too noisy to use; the feature
# falls back to the league baseline and a companion *_n column records how much
# evidence it rested on, so the model can learn to distrust thin estimates.
MIN_TEAM_MATCHES = 3
MIN_REF_MATCHES = 5

# Empirical-Bayes shrinkage, in matches. A referee's observed rate is pulled
# toward the league average with weight equivalent to this many league-average
# matches, so a referee with 4 matches on record contributes little and one
# with 300 contributes almost all of their own rate.
#
# This is the principled fix for the problem the `*_n` evidence-count features
# were supposed to solve and measurably did not (0.1% of model value, 1.7
# sigma). Telling a linear model "this estimate rests on 4 matches" and hoping
# it learns to discount it asks the model to represent an interaction it has no
# way to express. Doing the discounting in the feature costs nothing.
#
# REMOVED 2026-09-17, after measurement.
#
# Ten expanded referee features were built here -- career rate, empirical-Bayes
# shrunk rate, division-specific rate, home/away bias, strictness trend,
# consistency, cards-per-foul, red rate -- on the reasoning that the ablation
# showed referee features carrying 43% of the model's value.
#
# They did not help: +0.00069 NLL at +1.4 sigma over 16 folds (NO_EVIDENCE),
# and calibration got WORSE, 0.0129 -> 0.0142. `ref_yel10` and `ref_yel30`
# already extract essentially all of the referee signal; ten correlated ways of
# restating one quantity add estimation noise, not information. The shrinkage
# in particular solves a problem this data rarely has, since most matches are
# officiated by experienced referees whose shrunk and raw rates barely differ.
#
# The lesson worth keeping: "this feature family is 43% of model value" means
# "it matters and we already measure it well", NOT "there is more signal here
# to extract". Those read identically in an ablation.
#
# Kept in the registry as NBGLM_ref_expanded. Do not re-add without a reason
# that is not this one.

# League-season baselines need a burn-in before they mean anything. Until then
# the previous season's division average is used.
MIN_BASELINE_MATCHES = 30


class _Roll:
    """Fixed-length history of (value...) tuples with cheap means."""

    def __init__(self, maxlen: int):
        self.dq: deque = deque(maxlen=maxlen)

    def add(self, *vals) -> None:
        self.dq.append(vals)

    def mean(self, i: int) -> float:
        if not self.dq:
            return np.nan
        return float(np.mean([v[i] for v in self.dq]))

    def __len__(self) -> int:
        return len(self.dq)


def build(matches: pd.DataFrame, verbose: bool = False,
          devig: str = "shin") -> pd.DataFrame:
    """Return one feature row per match, in chronological order.

    `matches` must carry: date, division, season, HomeTeam, AwayTeam, Referee,
    HY, AY, HF, AF, HC, AC, HS, AS, yellows, fouls.
    """
    m = matches.sort_values(
        ["date", "division", "HomeTeam"], kind="mergesort"
    ).reset_index(drop=True)

    # ---- state, all of it strictly "what has happened so far"
    team_all: dict[tuple, dict[int, _Roll]] = defaultdict(
        lambda: {w: _Roll(w) for w in TEAM_WINDOWS})
    team_home: dict[tuple, _Roll] = defaultdict(lambda: _Roll(10))
    team_away: dict[tuple, _Roll] = defaultdict(lambda: _Roll(10))
    ref_roll: dict[str, dict[int, _Roll]] = defaultdict(
        lambda: {w: _Roll(w) for w in REF_WINDOWS})
    ref_total: dict[str, int] = defaultdict(int)
    last_played: dict[tuple, pd.Timestamp] = {}

    # league-season running totals, and the previous season's finished average
    ls_sum: dict[tuple, float] = defaultdict(float)
    ls_n: dict[tuple, int] = defaultdict(int)
    ls_fouls: dict[tuple, float] = defaultdict(float)
    prev_season_avg: dict[str, float] = {}
    prev_season_fouls: dict[str, float] = {}
    season_order = sorted(m.season.unique())

    rows = []
    for r in m.itertuples(index=False):
        div, season = r.division, r.season
        key_h, key_a = (r.HomeTeam,), (r.AwayTeam,)
        ls_key = (div, season)

        # ---------------- baseline: this division-season, so far only
        n_seen = ls_n[ls_key]
        if n_seen >= MIN_BASELINE_MATCHES:
            baseline = ls_sum[ls_key] / n_seen
            baseline_fouls = ls_fouls[ls_key] / n_seen
            baseline_src = "season"
        else:
            # Walk BACK through earlier seasons until one has a recorded
            # average for this division. Looking only one season back failed
            # whenever that season's file was missing for that division -- a
            # 2026/27 Premier League fixture fell through to no baseline at
            # all, because the 2025/26 Premier League file does not exist.
            idx = season_order.index(season)
            baseline, prev = np.nan, None
            for back in range(idx - 1, -1, -1):
                cand = prev_season_avg.get(f"{div}|{season_order[back]}", np.nan)
                if np.isfinite(cand):
                    baseline, prev = cand, season_order[back]
                    break
            # The fouls baseline needs the same fallback as the cards one.
            # Without it every *_fouls* feature was NaN for the first 30
            # matches of any season -- including, always, the fixtures you
            # actually want to predict in August.
            baseline_fouls = (prev_season_fouls.get(f"{div}|{prev}", np.nan)
                              if prev else np.nan)
            baseline_src = "prev_season"
            if not np.isfinite(baseline):
                baseline = np.nan
                baseline_src = "none"

        def rel(x, base):
            if not np.isfinite(x) or not np.isfinite(base) or base <= 0:
                return np.nan
            return x / base

        feat = {
            "date": r.date, "season": season, "division": div,
            "home": r.HomeTeam, "away": r.AwayTeam,
            "referee": r.Referee if isinstance(r.Referee, str) else None,
            "yellows": r.yellows,                      # the target
            "baseline_yellows": baseline,
            "baseline_source": baseline_src,
        }

        # ---------------- team form, relative to the league baseline
        for side, key in (("h", key_h), ("a", key_a)):
            rolls = team_all[key]
            for w in TEAM_WINDOWS:
                rr = rolls[w]
                n = len(rr)
                # `n` still gates the feature below, but is NOT emitted as a
                # feature itself. Two independent ablations, on 16 and 21
                # folds, measured the six h_n*/a_n* columns at 0.1% of model
                # value (1.1 and 1.7 sigma, NO_EVIDENCE both times).
                #
                # The idea was that the model could learn to distrust a rolling
                # average resting on few matches. It cannot: a linear model has
                # no way to express "discount that other coefficient when this
                # one is small". The principled version -- empirical-Bayes
                # shrinkage -- was tried too and also failed, because this data
                # has too few thin estimates for either to matter. Removed
                # 2026-09-17; 44 features -> 38.
                if n >= MIN_TEAM_MATCHES:
                    feat[f"{side}_yel{w}"] = rel(rr.mean(0), baseline)
                    feat[f"{side}_fouls{w}"] = rel(rr.mean(1), baseline_fouls)
                    feat[f"{side}_oppyel{w}"] = rel(rr.mean(2), baseline)
                else:
                    feat[f"{side}_yel{w}"] = np.nan
                    feat[f"{side}_fouls{w}"] = np.nan
                    feat[f"{side}_oppyel{w}"] = np.nan

        hr, ar = team_home[key_h], team_away[key_a]
        feat["h_yel_home10"] = (rel(hr.mean(0), baseline)
                                if len(hr) >= MIN_TEAM_MATCHES else np.nan)
        feat["a_yel_away10"] = (rel(ar.mean(0), baseline)
                                if len(ar) >= MIN_TEAM_MATCHES else np.nan)

        # ---------------- referee, optional by design
        ref = r.Referee if isinstance(r.Referee, str) else None
        feat["ref_known"] = int(ref is not None)
        feat["ref_n_career"] = ref_total[ref] if ref else 0
        for w in REF_WINDOWS:
            if ref and len(ref_roll[ref][w]) >= MIN_REF_MATCHES:
                rr = ref_roll[ref][w]
                feat[f"ref_yel{w}"] = rel(rr.mean(0), baseline)
                fpm = rr.mean(1)
                feat[f"ref_yel_per_foul{w}"] = (
                    rr.mean(0) / fpm if np.isfinite(fpm) and fpm > 0 else np.nan)
            else:
                feat[f"ref_yel{w}"] = np.nan
                feat[f"ref_yel_per_foul{w}"] = np.nan

        # ---------------- context
        for side, key in (("h", key_h), ("a", key_a)):
            lp = last_played.get(key)
            feat[f"{side}_rest_days"] = (
                (r.date - lp).days if lp is not None else np.nan)
        feat["match_no_in_season"] = n_seen
        feat["season_progress"] = (
            n_seen / 552.0 if div != "E0" else n_seen / 380.0)

        rows.append(feat)

        # ================= state update happens AFTER the row is emitted.
        # This ordering is the whole point-in-time guarantee.
        y_h, y_a = r.HY, r.AY
        f_h, f_a = r.HF, r.AF
        if np.isfinite(y_h) and np.isfinite(y_a):
            for w in TEAM_WINDOWS:
                team_all[key_h][w].add(y_h, f_h, y_a)
                team_all[key_a][w].add(y_a, f_a, y_h)
            team_home[key_h].add(y_h)
            team_away[key_a].add(y_a)
            if ref:
                for w in REF_WINDOWS:
                    ref_roll[ref][w].add(r.yellows, r.fouls)
                ref_total[ref] += 1
            ls_sum[ls_key] += r.yellows
            ls_fouls[ls_key] += r.fouls if np.isfinite(r.fouls) else 0.0
            ls_n[ls_key] += 1
        last_played[key_h] = r.date
        last_played[key_a] = r.date

        if ls_n[ls_key]:
            prev_season_avg[f"{div}|{season}"] = ls_sum[ls_key] / ls_n[ls_key]
            prev_season_fouls[f"{div}|{season}"] = ls_fouls[ls_key] / ls_n[ls_key]

    out = pd.DataFrame(rows)
    out = _add_market_features(out, m, devig=devig)
    if verbose:
        print(f"built {len(out):,} feature rows, {out.shape[1]} columns")
        thin = out.filter(like="_yel").isna().mean().mean()
        print(f"  mean NaN rate across rolling features: {thin:.1%} "
              "(early matches have no history yet — expected)")
    return out


def shin_probabilities(odds: list[np.ndarray], iters: int = 60
                       ) -> tuple[list[np.ndarray], np.ndarray]:
    """Strip the bookmaker's margin using Shin's (1993) method.

    Proportional de-vigging -- dividing every raw 1/odds by their sum -- spreads
    the margin evenly across outcomes. Bookmakers do not price that way: they
    load more margin onto longshots, because a minority of bettors are better
    informed than the price and the book protects itself where it is most
    exposed. Shin models that explicitly with a single parameter `z`, the
    notional share of insider money, and solves for the `z` that makes the
    de-vigged probabilities sum to one:

        p_i = [ sqrt(z^2 + 4(1-z) * pi_i^2 / B) - z ] / (2(1-z))

    with pi_i = 1/odds_i and B = sum(pi). Solved by vectorised bisection over
    z in [0, 1); the sum is monotonically decreasing in z, so bisection is
    exact enough in 60 steps and runs over all 47,000 matches at once.
    """
    pis = [1.0 / o for o in odds]
    B = sum(pis)
    lo = np.zeros_like(B)
    hi = np.full_like(B, 0.999)

    def total(z):
        acc = np.zeros_like(B)
        for pi in pis:
            num = np.sqrt(z * z + 4.0 * (1.0 - z) * pi * pi / B) - z
            acc = acc + num / (2.0 * (1.0 - z))
        return acc

    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        with np.errstate(invalid="ignore", divide="ignore"):
            t = total(mid)
        too_high = t > 1.0          # sum falls as z rises
        lo = np.where(too_high, mid, lo)
        hi = np.where(too_high, hi, mid)
    z = 0.5 * (lo + hi)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = [(np.sqrt(z * z + 4.0 * (1.0 - z) * pi * pi / B) - z)
               / (2.0 * (1.0 - z)) for pi in pis]
    return out, z


def _add_market_features(out: pd.DataFrame, m: pd.DataFrame,
                         devig: str = "shin") -> pd.DataFrame:
    """Features derived from the PRE-MATCH betting odds.

    These are legitimate point-in-time inputs: the prices were public before
    kick-off. They are also the only features here that summarise information
    the model has no other way of seeing -- squad quality, injuries, motivation,
    everything the market has priced -- compressed into three numbers.

    The specific reason to expect them to matter for cards: **closely matched
    games are fought harder than mismatches**. A one-sided game is often placid;
    a tight one produces the tackles that produce the bookings. Nothing else in
    the feature set measures how evenly matched the two sides are.

    Overround ("vig") is removed before anything is computed. Raw 1/odds sum to
    1.075 on average in this data -- a 7.5% margin -- and that margin is not
    constant: 10.8% in 2005/06, 5.9% in 2023/24. Left in, the feature would
    encode how greedy the bookmaker was that decade alongside anything about
    the match.

    Default method is Shin's, which is the more principled of the two and
    measured very slightly better (+0.000021 NLL, 2.5 sigma, better in 14 of 21
    folds). Worth being honest about that number: it clears the significance
    bar only because both methods are deterministic functions of the same odds,
    so the comparison carries almost no noise. The effect is about one
    two-hundredth of one percent of what the features contribute in total.
    It is kept because it costs nothing and is correct, not because it matters
    here -- these features measure how BALANCED a match is, and Shin mostly
    shifts the level.
    """
    o_h = m["odds_h"].to_numpy(float)
    o_d = m["odds_d"].to_numpy(float)
    o_a = m["odds_a"].to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        if devig == "shin":
            (p_h, p_d, p_a), _z = shin_probabilities([o_h, o_d, o_a])
        else:
            ih, id_, ia = 1.0 / o_h, 1.0 / o_d, 1.0 / o_a
            tot = ih + id_ + ia
            p_h, p_d, p_a = ih / tot, id_ / tot, ia / tot

    o_o, o_u = m["odds_o25"].to_numpy(float), m["odds_u25"].to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        io, iu = 1.0 / o_o, 1.0 / o_u
        p_over = io / (io + iu)

    known = np.isfinite(p_h) & np.isfinite(p_a)
    with np.errstate(invalid="ignore"):
        ent = -(p_h * np.log(p_h) + p_d * np.log(p_d) + p_a * np.log(p_a))

    out = out.copy()
    out["mkt_known"] = known.astype(int)
    out["mkt_closeness"] = np.where(known, 1.0 - np.abs(p_h - p_a), np.nan)
    out["mkt_p_draw"] = np.where(known, p_d, np.nan)
    out["mkt_fav_prob"] = np.where(known, np.maximum(p_h, p_a), np.nan)
    out["mkt_home_edge"] = np.where(known, p_h - p_a, np.nan)
    out["mkt_entropy"] = np.where(known, ent, np.nan)
    out["mkt_p_over25"] = np.where(np.isfinite(p_over), p_over, np.nan)
    return out


FEATURE_COLUMNS = None  # populated by feature_columns()


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Model inputs only: no identities, no target, no bookkeeping."""
    drop = {"date", "season", "division", "home", "away", "referee",
            "yellows", "baseline_source"}
    return [c for c in df.columns if c not in drop]
