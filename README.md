# Forecasting Yellow Cards

**A negative binomial model of yellow cards in English league football, tested on matches that had not been played yet.**

For every match in the Premier League, Championship, League One and League Two, the model gives the
probability of each number of yellow cards (0, 1, 2, …), not just an average. That is what a card
market prices, so a distribution can be compared with a bookmaker's line and a single number cannot.

Around the model sits a small pipeline that runs twice a day: it collects results and referee
appointments, checks them, writes the week's forecasts before kick-off, and scores them once the
results are in.

## Results

| Test | Matches | Result |
|---|---|---|
| Held-out seasons 2024/25 and 2025/26, scored once | 3,140 | Better log-likelihood than the league-season baseline by about 0.02 per match; calibration gap 0.8% against the baseline's 2.4% |
| Forward test, 2026/27: every forecast written before kick-off | 114 | +0.024 per match over the baseline, +1.5 sigma, better in 75 of 114 matches. Verdict **consistent**, on a scale fixed before any result was seen |

The 2026/27 season is running about 0.4 cards a match above history. The baseline is low by the same
amount, so this is the season, not the model drifting; it is monitored, not "fixed" mid-season.

## What drives it

Removing each group of features in turn, across 21 cross-validation folds:

| Feature group removed | Share of the total cost |
|---|---|
| Referee | 48% (worse in 21 of 21 folds) |
| Team form | 38% |
| Match context | 8% |
| Betting market | 6% |
| Evidence counts | 0% (no evidence it helps) |

A gradient-boosted model was also tried and rejected: it was worse than the GLM by 1.5 sigma in
development.

## How it works

1. **Data.** [football-data.co.uk](https://www.football-data.co.uk) results for England's top four
   divisions from 2000/01 onward (about 53,000 matches), plus the EFL's published referee
   appointments.
2. **Target.** Yellow cards only. Their variance exceeds the mean (ratio 1.13), so the count is
   modelled as **negative binomial**; Poisson would understate the tails, which is where betting
   lines sit.
3. **Features.** Point-in-time only: rolling team form over 5, 10 and 20 matches, fouls, opponents'
   cards, rest days, the referee's record, and the match context. Tests check both kinds of leak:
   a later match changing an earlier match's features, and a match's own result feeding its own
   features. Each test is also shown a deliberately leaky version and must fail it.
4. **Rules written first.** [PROJECT_SPEC.md](PROJECT_SPEC.md) fixed the target, the split, the
   baselines and the scoring before any model was fitted. Every look at held-out data is logged in
   [cards/holdout_log.json](cards/holdout_log.json), and the model is frozen for the season under a
   fingerprint that is checked before every forecast and every score.

## The pipeline

Runs at 09:30 and 18:30 (macOS launchd, [cards/com.footballcards.topup.plist](cards/com.footballcards.topup.plist)):

| Step | What it does |
|---|---|
| [topup.py](cards/topup.py) | Fetches the season's results. Never replaces a file with one holding fewer matches, and warns when the source is days behind |
| [check_referees.py](cards/check_referees.py) | Reads the EFL's referee appointments. The parser reads the article as lines rather than by HTML tag, which survived five page layouts, and is tested against saved copies |
| [auto_week_sheet.py](cards/auto_week_sheet.py) | Builds or refreshes the week's forecast sheet. A match that has kicked off is never re-forecast, and its row (with any prices typed in) is kept |
| [score_forward_test.py](cards/score_forward_test.py), [bias_watch.py](cards/bias_watch.py) | Score the forecasts against what happened, and separate model drift from a high-card season |

Every request goes through one fetcher, [polite.py](cards/polite.py): an honest user-agent,
robots.txt read as RFC 9309 says, and at least 3 seconds between requests to a site.

**Data checks.** Impossible card counts are caught by arithmetic; card counts from the last few days
are treated as unconfirmed; a result typed in by hand is never overwritten by an unverified one; and
104 test checks cover the parsers, the features, the sheet's betting arithmetic and the refresh rules.

## Run it

Python 3.9 or later.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python cards/loader.py --download
.venv/bin/python cards/app.py
```

`loader.py --download` fetches every season (about 110 files, a few seconds apart) into `data/raw/`.
`app.py` serves a local page at http://127.0.0.1:5001 for forecasting any fixture.

A week's forecast sheet: `.venv/bin/python cards/run_week_sheet.py --from 2026-10-29 --to 2026-11-04`.
Tests: `.venv/bin/python cards/test_week_sheet.py`, and the other `cards/test_*.py` files. The
referee-parser tests read saved EFL pages that are not included here, so they report SKIP.

## Limits

- **Not yet tested against bookmaker prices.** The forward test measures whether the model is
  good, not whether it is profitable; that needs prices recorded before kick-off, over many more
  matches.
- **114 forward matches is a small sample.** The pre-registered estimate was about 1,000 matches for a
  real verdict.
- **Yellows only.** Many card markets count a red as two cards, which moves every line by 5 to 6
  points; the forecast sheet refuses to suggest a bet unless the market settles on yellows alone.

## License

[MIT](LICENSE)
