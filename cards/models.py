"""Negative binomial models, and the scoreboard from PROJECT_SPEC section 4.

Three models, each of which must beat the one before it or not be kept:

  1. `ConstantNB`      one distribution for every match in every division
  2. `LeagueSeasonNB`  the league-season baseline only, no features at all
  3. `NBGLM`           the baseline as an offset, plus the relative features

Nothing here decides whether a difference is real. `evaluate` returns numbers;
`walk_forward` returns them per fold; and the spread across folds is the noise
floor an improvement has to clear. That separation is deliberate — the moment a
model gets to grade itself, the grade stops meaning anything.

Scoring is mean negative log-likelihood per match. It is a proper scoring rule:
it rewards being right *and* being honestly uncertain, and cannot be gamed by
shading predictions toward the mean. Calibration is reported alongside and is
non-negotiable — a better NLL with worse calibration is not an improvement,
because a miscalibrated probability cannot be compared to a betting price.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import gammaln

MAX_CARDS = 15          # P(y) is computed out to here; the tail beyond is tiny
LINES = (1.5, 2.5, 3.5, 4.5, 5.5, 6.5)


# --------------------------------------------------------- the distribution


def nb_logpmf(y: np.ndarray, mu: np.ndarray, alpha: float) -> np.ndarray:
    """NB2 log-likelihood: variance = mu + alpha * mu^2.

    alpha -> 0 recovers Poisson. Measured on this data the overdispersion is
    mild but real (var/mean 1.13), so alpha lands small and positive; forcing
    it to zero would understate exactly the tails the betting lines sit on.
    """
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    if alpha <= 1e-10:
        return y * np.log(mu) - mu - gammaln(y + 1.0)
    r = 1.0 / alpha
    return (gammaln(y + r) - gammaln(r) - gammaln(y + 1.0)
            + r * np.log(r / (r + mu)) + y * np.log(mu / (r + mu)))


def nb_pmf_matrix(mu: np.ndarray, alpha: float,
                  max_k: int = MAX_CARDS) -> np.ndarray:
    """(n_matches, max_k+1) array of P(exactly k cards). This is the product."""
    ks = np.arange(max_k + 1)
    return np.exp(nb_logpmf(ks[None, :], np.asarray(mu)[:, None], alpha))


def fit_alpha(y: np.ndarray, mu: np.ndarray) -> float:
    """MLE for the dispersion, given fitted means."""
    def nll(log_alpha):
        return -nb_logpmf(y, mu, float(np.exp(log_alpha))).sum()
    res = minimize_scalar(nll, bounds=(-9.0, 2.0), method="bounded")
    return float(np.exp(res.x))


# ----------------------------------------------------------------- scoring


def evaluate(y: np.ndarray, mu: np.ndarray, alpha: float,
             divisions=None) -> dict:
    y = np.asarray(y, float)
    mu = np.asarray(mu, float)
    # `divisions` arrives as a Series carrying its original index; convert to a
    # plain array before masking, or the boolean mask silently becomes a label
    # lookup and raises (or worse, reorders).
    div = None if divisions is None else np.asarray(divisions)
    ok = np.isfinite(y) & np.isfinite(mu) & (mu > 0)
    y, mu = y[ok], mu[ok]
    if div is not None:
        div = div[ok]

    nll = float(-nb_logpmf(y, mu, alpha).mean())
    pmf = nb_pmf_matrix(mu, alpha)
    cdf = np.cumsum(pmf, axis=1)

    out = {"n": int(len(y)), "nll": nll, "alpha": alpha,
           "mean_pred": float(mu.mean()), "mean_actual": float(y.mean())}

    # calibration at each betting line: predicted P(over) vs what happened
    cal = {}
    for line in LINES:
        k = int(np.floor(line))
        p_over = 1.0 - cdf[:, k]
        actual = (y > line).mean()
        cal[line] = {"pred": float(p_over.mean()), "actual": float(actual),
                     "gap": float(p_over.mean() - actual)}
    out["calibration"] = cal
    out["mean_abs_calibration_gap"] = float(
        np.mean([abs(v["gap"]) for v in cal.values()]))

    if div is not None:
        per = {}
        for d in sorted(pd.unique(div)):
            sel = div == d
            if sel.sum() == 0:
                continue
            per[d] = {"n": int(sel.sum()),
                      "nll": float(-nb_logpmf(y[sel], mu[sel], alpha).mean())}
        out["per_division"] = per
    return out


def reliability(y: np.ndarray, mu: np.ndarray, alpha: float,
                line: float = 3.5, bins: int = 10) -> pd.DataFrame:
    """Does P(over)=0.6 actually happen 60% of the time? Binned answer."""
    pmf = nb_pmf_matrix(np.asarray(mu, float), alpha)
    p = 1.0 - np.cumsum(pmf, axis=1)[:, int(np.floor(line))]
    y = np.asarray(y, float)
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    edges[0] -= 1e-9
    idx = np.digitize(p, edges[1:-1])
    rows = []
    for b in range(bins):
        sel = idx == b
        if sel.sum() < 20:
            continue
        rows.append({"bin": b, "n": int(sel.sum()),
                     "predicted": float(p[sel].mean()),
                     "actual": float((y[sel] > line).mean())})
    df = pd.DataFrame(rows)
    if len(df):
        df["gap"] = df.predicted - df.actual
    return df


# ------------------------------------------------------------------ models


class ConstantNB:
    """One distribution for every match. The floor under everything."""

    name = "constant"

    def fit(self, df: pd.DataFrame):
        y = df["yellows"].to_numpy(float)
        self.mu_ = float(np.nanmean(y))
        self.alpha_ = fit_alpha(y[np.isfinite(y)],
                                np.full(np.isfinite(y).sum(), self.mu_))
        return self

    def predict_mu(self, df: pd.DataFrame) -> np.ndarray:
        return np.full(len(df), self.mu_)


class LeagueSeasonNB:
    """The point-in-time league-season baseline, and nothing else.

    This is the number every feature must justify itself against. It already
    knows that League Two cards less than the Championship and that 2019 cards
    less than 2024 — so any credit the feature model claims for those is not
    credit at all.
    """

    name = "league_season"

    def fit(self, df: pd.DataFrame):
        mu = self._mu(df)
        y = df["yellows"].to_numpy(float)
        ok = np.isfinite(y) & np.isfinite(mu) & (mu > 0)
        self.alpha_ = fit_alpha(y[ok], mu[ok])
        self.fallback_ = float(np.nanmean(y))
        return self

    @staticmethod
    def _mu(df: pd.DataFrame) -> np.ndarray:
        return df["baseline_yellows"].to_numpy(float)

    def predict_mu(self, df: pd.DataFrame) -> np.ndarray:
        mu = self._mu(df).copy()
        bad = ~np.isfinite(mu) | (mu <= 0)
        mu[bad] = getattr(self, "fallback_", np.nanmean(mu))
        return mu


class NBGLM:
    """Baseline as offset, relative features as a log-linear correction.

        log(mu) = log(baseline) + b0 + b.x

    The offset carries the level; the coefficients only have to explain how
    this fixture differs from a typical one in its own context. Features are
    centred at zero (ratio - 1), so a coefficient reads as "effect of being 1x
    the league average above league average", and a missing feature imputes to
    zero — i.e. "no information, assume typical" — with a companion indicator
    so the model can learn that absence itself is informative.
    """

    name = "nb_glm"

    def __init__(self, feature_cols: list[str] | None = None,
                 ridge: float = 1.0):
        self.feature_cols = feature_cols
        self.ridge = ridge
        self.scale_ = None

    # ---- design matrix
    RATIO_PREFIXES = ("h_yel", "a_yel", "h_fouls", "a_fouls", "h_oppyel",
                      "a_oppyel", "ref_yel")

    def _design(self, df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        cols = self.feature_cols
        X, names = [], []
        for c in cols:
            v = df[c].to_numpy(float)
            if c.startswith(self.RATIO_PREFIXES) and "per_foul" not in c:
                v = v - 1.0                       # centre ratios at 0
            elif c in ("h_rest_days", "a_rest_days"):
                v = np.clip(v, 0, 30) / 7.0 - 1.0
            elif c.startswith(("h_n", "a_n", "ref_n")):
                v = np.log1p(np.clip(v, 0, None)) - 2.0
            elif c == "baseline_yellows":
                continue                          # it is the offset, not a feature
            miss = ~np.isfinite(v)
            v = np.where(miss, 0.0, v)
            # The indicator is emitted for EVERY feature, always. Emitting it
            # only when this particular frame happens to contain a NaN makes
            # the design matrix a function of the data it is given, so train
            # and test silently end up with different column counts.
            X.append(v); names.append(c)
            X.append(miss.astype(float)); names.append(c + "__missing")
        X = np.column_stack(X) if X else np.zeros((len(df), 0))
        # Standardise using statistics learned at FIT time. Without this the
        # ridge penalty is applied to raw scales -- ref_n_career enters at
        # ~3.9 and h_yel5 at ~0.2, so the same lambda shrinks them by wildly
        # different amounts and the coefficients stop being comparable. It
        # also made the per-fixture explanation nonsense: bookkeeping features
        # with large magnitudes dominated the contribution ranking while the
        # strongest real signal, the referee rate, never appeared.
        if getattr(self, "scale_", None) is None:
            mu = X.mean(axis=0)
            sd = X.std(axis=0)
            sd[sd < 1e-9] = 1.0
            self.scale_ = (mu, sd)
        mu, sd = self.scale_
        if X.shape[1] == len(mu):
            X = (X - mu) / sd
        X = np.column_stack([np.ones(len(df)), X])
        return X, ["intercept"] + names

    def fit(self, df: pd.DataFrame):
        import statsmodels.api as sm
        y = df["yellows"].to_numpy(float)
        # The offset's fallback is fitted HERE and reused at predict time.
        # Refitting it inside predict_mu would let the test set's own mean
        # supply the fallback for its own rows -- small, but a leak.
        self.scale_ = None          # re-learned from this training set
        self.offset_model_ = LeagueSeasonNB().fit(df)
        off = np.log(np.clip(self.offset_model_.predict_mu(df), 1e-6, None))
        X, names = self._design(df)
        ok = np.isfinite(y)
        # Alpha and coefficients are estimated by alternating: fit the GLM at
        # the current alpha, re-estimate alpha from the fitted means, repeat.
        # Two passes is plenty at this dispersion and avoids statsmodels'
        # joint optimiser, which is fragile with many near-collinear features.
        alpha = 0.05
        for _ in range(3):
            fam = sm.families.NegativeBinomial(alpha=alpha)
            mod = sm.GLM(y[ok], X[ok], family=fam, offset=off[ok])
            res = mod.fit_regularized(alpha=self.ridge, L1_wt=0.0) \
                if self.ridge else mod.fit()
            mu = np.exp(off[ok] + X[ok] @ np.asarray(res.params))
            alpha = fit_alpha(y[ok], mu)
        self.params_ = np.asarray(res.params)
        self.names_ = names
        self.alpha_ = alpha
        return self

    def predict_mu(self, df: pd.DataFrame) -> np.ndarray:
        off = np.log(np.clip(self.offset_model_.predict_mu(df), 1e-6, None))
        X, names = self._design(df)
        if X.shape[1] != len(self.params_):
            raise ValueError(
                f"design matrix has {X.shape[1]} columns but the model was fitted "
                f"with {len(self.params_)}; the two must match exactly")
        return np.exp(off + X @ self.params_)

    def coefficients(self) -> pd.DataFrame:
        return (pd.DataFrame({"feature": self.names_, "coef": self.params_})
                .assign(abs=lambda d: d.coef.abs())
                .sort_values("abs", ascending=False)
                .drop(columns="abs"))


class NBGBM:
    """Gradient boosting for the mean, negative binomial for the spread.

    Same contract as NBGLM and the same offset, so the comparison is fair:
    the league-season baseline enters as `init_score`, and the trees only ever
    learn how a fixture departs from a typical match in its own context. Then
    the dispersion is fitted by MLE on the resulting means, exactly as for the
    GLM, so both models are scored by the same likelihood.

    Early stopping uses the LAST SEASON OF THE TRAINING DATA as its validation
    set -- never the fold being tested. Using the test fold to decide when to
    stop is the quiet way a boosted model beats a linear one on paper and
    nowhere else.

    Hyperparameters are fixed defaults, deliberately untuned. Tuning them
    against the walk-forward folds would be selection on the thing being
    measured; if it is done later it belongs in the registry with every
    configuration tried, per PROJECT_SPEC section 6.
    """

    name = "nb_gbm"

    DEFAULTS = dict(
        objective="poisson",
        learning_rate=0.03,
        num_leaves=15,
        min_data_in_leaf=200,
        feature_fraction=0.7,
        bagging_fraction=0.8,
        bagging_freq=1,
        lambda_l2=5.0,
        verbose=-1,
        num_threads=4,
    )
    MAX_ROUNDS = 2000
    EARLY_STOP = 50

    def __init__(self, feature_cols: list[str], **params):
        self.feature_cols = [c for c in feature_cols if c != "baseline_yellows"]
        self.params = {**self.DEFAULTS, **params}
        self.booster_ = None

    def _X(self, df: pd.DataFrame) -> np.ndarray:
        # Trees handle NaN natively -- no imputation, no missing indicators.
        # That is a real advantage here: "this referee has no history" is a
        # branch the model can learn rather than a value it must be told.
        return df[self.feature_cols].to_numpy(float)

    def _offset(self, df: pd.DataFrame) -> np.ndarray:
        return np.log(np.clip(self.offset_model_.predict_mu(df), 1e-6, None))

    def fit(self, df: pd.DataFrame):
        import lightgbm as lgb

        self.offset_model_ = LeagueSeasonNB().fit(df)
        y = df["yellows"].to_numpy(float)
        ok = np.isfinite(y)
        d = df[ok]
        y = y[ok]
        off = self._offset(d)
        X = self._X(d)

        seasons = sorted(d.season.unique())
        if len(seasons) >= 2:
            val = (d.season == seasons[-1]).to_numpy()
        else:
            val = np.zeros(len(d), dtype=bool)
            val[int(0.8 * len(d)):] = True
        tr = ~val

        dtrain = lgb.Dataset(X[tr], label=y[tr], init_score=off[tr])
        dvalid = lgb.Dataset(X[val], label=y[val], init_score=off[val],
                             reference=dtrain)
        self.booster_ = lgb.train(
            self.params, dtrain, num_boost_round=self.MAX_ROUNDS,
            valid_sets=[dvalid],
            callbacks=[lgb.early_stopping(self.EARLY_STOP, verbose=False)],
        )
        self.best_iteration_ = self.booster_.best_iteration
        mu = np.exp(off + self.booster_.predict(X, raw_score=True))
        self.alpha_ = fit_alpha(y, mu)
        return self

    def predict_mu(self, df: pd.DataFrame) -> np.ndarray:
        off = self._offset(df)
        raw = self.booster_.predict(self._X(df), raw_score=True)
        return np.exp(off + raw)

    def importances(self) -> pd.DataFrame:
        return (pd.DataFrame({
            "feature": self.feature_cols,
            "gain": self.booster_.feature_importance("gain")})
            .sort_values("gain", ascending=False))


# ---------------------------------------------------------- walk-forward CV


def walk_forward(feat: pd.DataFrame, model_factory, min_train_seasons: int = 3
                 ) -> pd.DataFrame:
    """Train on every season before S, test on S. Never the other way round.

    This is the only validation shape that matches how the model would be used,
    and the spread of its fold scores is the noise floor that any claimed
    improvement has to clear.
    """
    seasons = sorted(feat.season.unique())
    rows = []
    for i in range(min_train_seasons, len(seasons)):
        tr = feat[feat.season.isin(seasons[:i])]
        te = feat[feat.season == seasons[i]]
        if len(te) == 0 or len(tr) < 500:
            continue
        m = model_factory().fit(tr)
        mu = m.predict_mu(te)
        sc = evaluate(te.yellows.to_numpy(float), mu, m.alpha_, te.division)
        rows.append({"test_season": seasons[i], "n_train": len(tr),
                     "n_test": sc["n"], "nll": sc["nll"], "alpha": sc["alpha"],
                     "cal_gap": sc["mean_abs_calibration_gap"]})
    return pd.DataFrame(rows)
