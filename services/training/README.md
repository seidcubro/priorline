# Training

Everything between the feature table and the board. All of it runs in one
image, `docker compose run --rm training python <file>`.

## The pipeline

What `scripts/scheduled_update.sh` runs, in order. If one of these breaks, the
site goes stale.

| file | job |
|---|---|
| `train.py` | fit the point model for one market |
| `train_quantiles.py` | fit the five quantile models in the same feature space |
| `eval.py` | score the point model on a held-out split |
| `build_projections.py` | project every eligible player, priced or not |
| `build_prop_edges.py` | price those projections against the book, pick a side, tier them |
| `grade_edges.py` | settle played picks against the box score |
| `replay_current_rules.py` | replay the season under today's tiering, walk forward, for the record page |
| `eval_clv.py` | score picks against the closing line |
| `eval_strategy.py` | regenerate the published record the site quotes |
| `audit_freshness.py` | the gate: exits non-zero when anything is stale |
| `audit_plausibility.py` | flag projections a person would not believe |

## Calibration

Fitted from graded results and read at serving time. Each fitter writes a JSON
artifact and has an apply module beside it, so a missing artifact means the
correction is skipped, not that the build fails.

| fitter | applier | corrects |
|---|---|---|
| `fit_probability_calibrator.py` | read in `build_prop_edges.py` | P(over) |
| `fit_display_probability.py` | read in `build_prop_edges.py` | the probability the site shows, per side |
| `fit_interval_calibrator.py` | `interval_calibration.py`, `interval_scale.py` | the published range |
| `fit_median_anchor.py` | `median_anchor.py` | the median against the point projection |
| `fit_spread_calibrator.py` | `spread_calibration.py` | how far predictions spread |
| `fit_season_bias.py` | `season_bias.py` | drift between seasons |
| `fit_vacated_volume.py` | `vacated_volume.py` | a backup quarterback inheriting the starter's workload |
| `fit_gap_tiers.py` | `gap_tier.py` | which markets earn a ranking, and which way it runs |
| `eval_stale_role.py` | read in `build_prop_edges.py` | a player whose role just changed |

## Shared modules

`gap_tier.py` holds the tiering rules. `stats_ci.py` is the clustered bootstrap
every interval on the site comes from. `odds_markets.py` maps the provider's
market keys to this project's codes and is mounted read-only into the API.
`market_blend.py` and `thin_sample.py` are small helpers the board imports.

## Model selection

`bakeoff.py` picks the model family per market on expanding-window folds and
refuses a change that is inside the noise. `tune.py` searches hyperparameters
scored on the same folds. `explain_model.py` checks that what a model leans on
agrees with football.

## Backtests, backfills and recovery

Run by hand, usually once, usually after something changed underneath.

| file | job |
|---|---|
| `backtest_season.py`, `backtest_quantile_ev.py` | rebuild a season's picks on refit models |
| `backtest_gap_system.py` | the walk-forward backtest the tiering was measured on |
| `backfill_track_record.py` | write a rebuilt season into the graded record |
| `backfill_historical_odds.py` | buy and store past prices |
| `backfill_projection_history.py` | fill in what was projected for games already played |
| `recover_live_picks.py` | rebuild a published board that was lost before grading |
| `retier_live_record.py` | restore tiers a since-fixed bug mislabelled |

## The rest

`simulate.py` is the Monte Carlo game simulation. `eda.py` is the exploratory
data analysis; its figures and write-up are in `docs/eda/`.

    docker compose run --rm training python eda.py
