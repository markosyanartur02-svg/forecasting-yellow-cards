# Yellow-card model — the specification

Written before any modelling code, and frozen. Nothing in this file may be
changed to accommodate a result that has already been measured.

Carried over from the crude-oil harness that preceded this project: the
scoreboard is decided first, because a scoreboard written after seeing the
scores is not a scoreboard.

---

## 1. What this predicts

Given a fixture, the **probability of every possible number of yellow cards**
in the match — not a point estimate.

```
input   home team, away team, division, date, referee (optional)
output  P(0), P(1), P(2), ...  and hence  P(over 3.5) = 0.454
```

A point estimate cannot be compared to a betting line. A distribution can, and
comparing it to a price is the only thing this model is ultimately for.

### Why yellows only

Measured over 26,719 matches in E0–E3:

| | mean | variance | var/mean |
|---|---|---|---|
| yellow cards | 3.283 | 3.712 | **1.131** |
| red cards | 0.149 | — | — |

Reds appear in only 13.4% of matches and two or more in 1.35%. A target that is
mostly zeros teaches a model very little and adds variance to everything else.
Reds are excluded from the target and may later be modelled as a separate small
component if market alignment requires it — most card markets count a red as two
cards, so a yellows-only model is deliberately slightly off-market.

### Why negative binomial

Variance exceeds the mean (ratio 1.13), so the count is **overdispersed**.
Poisson assumes variance equals mean and would understate the tails — which is
exactly where the betting lines sit. Negative binomial is the baseline family.

---

## 2. Data

`football-data.co.uk`, free, no key.

| division | code | matches/season |
|---|---|---|
| Premier League | E0 | 380 |
| Championship | E1 | 552 |
| League One | E2 | 552 |
| League Two | E3 | 552 |

Seasons **2000/01 onward** — the first season carrying referee, cards, fouls and
shots. Roughly 53,000 matches.

**The National League (E5/EC) is excluded.** It loses fouls, corners and shots
from 2017/18 onward, and only 6 of its referees have 50+ matches on record, so
the referee feature — which the whole design leans on — cannot be estimated
there. It is the one English division that fails the quality bar.

Columns used: `Referee`, `HY`, `AY`, `HR`, `AR`, `HF`, `AF`, `HC`, `AC`, `HS`,
`AS`, `HST`, `AST`, plus teams, date and result.

---

## 3. The split, frozen

| | seasons | matches |
|---|---|---|
| development | 2000/01 .. 2025/26 | 51,556 |
| **holdout — locked** | 2026/27 | accumulating (159 as of 2026-09-17) |

### The holdout was redrawn once, on 2026-09-17

Originally 2024/25 + 2025/26. **It was looked at twice** — both looks are in
`cards/holdout_log.json` — and a holdout looked at twice is no longer a
holdout: the model that followed was developed with knowledge of how it
scored. Continuing to call it one would be the dishonest option, not the
careful one.

So those two seasons are reclassified as what they had in practice become,
development data, and **2026/27 becomes the fresh test set**. It is genuinely
clean: the season is being played now, nothing has been fitted on it, and no
result from it has been seen.

The counter restarts at zero for the new holdout. The two spent looks stay in
the log permanently and are reported alongside — "test data was examined three
times in this project" is a fact a reader deserves, and redrawing a boundary
does not erase it.

**This is the last time the boundary moves.** The next redraw would need the
same justification, and a project that redraws whenever the holdout runs out
has simply chosen a slower way of fitting to its test set.

The holdout is not a second scoreboard, it is a finite resource. Every
evaluation against it is appended to `cards/holdout_log.json` with a timestamp
and the full model definition, and the count is reported in any write-up.
Loading holdout data requires an explicit `unlock_holdout=True`.

Within development, model selection uses **walk-forward** validation: train on
seasons up to N, evaluate on N+1, roll forward. That mirrors how the model
would actually be used and never lets a later season inform an earlier one.

---

## 4. What makes the model good

Decided here, before building, because "better" is otherwise decided by
whichever number happened to move.

**Primary metric: mean negative log-likelihood** of the observed card count
under the predicted distribution, per match. It is a proper scoring rule — it
rewards being right *and* being honestly uncertain, and cannot be gamed by
shading predictions toward the mean.

**Secondary, and non-negotiable: calibration.** When the model says
`P(over 3.5) = 0.60`, it must happen about 60% of the time. Reported as a
reliability curve and as mean absolute calibration error across deciles. A
model with a better NLL and worse calibration is not an improvement — a
miscalibrated probability cannot be compared to a price.

**Per-division breakdown is required.** The model must work in each of the four
divisions, not on average. A model that is strong in the Premier League and
useless in League Two has not done what was asked.

### Baselines, in order

Each must be beaten by the next, or the next is not kept.

1. **League-season mean** — a single negative binomial fitted to the league and
   season, no features at all. The number every feature must justify itself
   against.
2. **Negative binomial GLM** with the league-season baseline as an offset and
   the relative features of section 5.
3. **Gradient boosting** (LightGBM/CatBoost) on the same features. Kept only if
   it beats the GLM by more than the noise floor.

### The noise floor

No improvement is believed until it is compared to run-to-run variation. NLL is
re-measured across seeds and across resampled walk-forward folds, and an
improvement smaller than that spread is reported as `NO_EVIDENCE`, not as a
gain. This is arithmetic in code, never a judgement call.

---

## 5. Features

**The model consumes behaviour, never identity.** No team IDs, no referee
names, no division as a bare category. Every feature is a rolling statistic
expressed relative to its own league-season baseline, so the model learns *"this
fixture is 15% more card-prone than a typical match in its context"*.

Entity embeddings are deliberately not used. Over 26 seasons of promotion,
relegation and squad turnover, "Arsenal 2003" and "Arsenal 2026" are not the
same entity, and an identity vector would encode a team that no longer exists.

Families:

* **team aggression** — rolling fouls committed and drawn, cards received, over
  the last 5/10/20 matches, home and away split, ÷ league-season average
* **opponent-adjusted** — fouling heavily against strong opposition is not the
  same as against weak; adjusted for opponent quality
* **referee** — cards per foul ÷ league average, plus the match count that
  estimate rests on. **Optional**: only England and Scotland publish referees,
  and a referee with little history falls back to league average, reported
  explicitly rather than silently
* **match context** — home/away, days rest, league position gap, season stage
* **pace proxies** — shots, corners and the rhythm they imply

### Point-in-time, enforced

Every rolling feature for a match on date *D* uses only matches played strictly
before *D*. Verified by a **truncation test**: build the features knowing 1,000
matches, build them again knowing 2,000, and require the first 1,000 values to
be bit-identical. A value that changes when later matches arrive has seen the
future.

This test caught a real bug in the preceding oil project, where an unstable
sort silently reordered tied timestamps between runs.

---

## 6. Guards

Ported from the oil harness, where every one of them was wrong on first writing
and only fixed by deliberately feeding it cases it should catch.

| guard | what it prevents |
|---|---|
| truncation test | rolling features that peek forward |
| row-count assertions in the loader | silent row loss across 26 seasons of shifting schemas |
| locked holdout + counted looks | the holdout quietly becoming a training set |
| noise floor | believing a gain smaller than run-to-run variation |
| model registry | forgetting how many variants were tried before one worked |

**A guard nobody has watched reject something is not a guard.** Each one gets a
test that feeds it a case it must catch, including deliberately broken
implementations that must fail.

---

## 7. Known limits, stated up front

Not in the data and therefore not in the model: injuries and suspensions,
weather, managerial pressure, cup-tie intensity, and derby rivalry (not
derivable — would need a manual lookup).

Most of the variance in a match's card count is irreducible. Bookmakers face the
same noise. The question this project answers is not "can cards be predicted"
but **"is this distribution better calibrated than the market's"**, and the
honest prior is that beating a well-priced market is hard and beating it after
5–8% vig is harder.

A model that turns out no better than the league average, reported honestly, is
a real result and the expected one.
