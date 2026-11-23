"""Which way of projecting a quarterback is right about his props?

The record says passing is where this board is best. Published and graded:

    2025   pass_td  427 picks 57.6%    pass_yds  394 picks 53.0%
    2026   pass_td  219 picks 56.2%    pass_yds  124 picks 54.8%

and no other market is above 54.4% in either season. The walk-forward test on
22 November 2026 said the opposite about passing yards: 50.1% on 674 props,
ranking at a coin. Both cannot describe the same thing, and they do not. The
test projected every priced quarterback with one model family, hgb, taken
from a table in the backtest that nobody had checked against what is live.
The record is the picks the board chose to publish, from whatever family is
live. So either the live family is better than hgb, or the board's choice of
which props to publish is doing the work, or both.

This separates them. For each passing market, walking forward (each season
projected by a model that saw only the seasons before it), every family the
trainer can build, plus two that are not models:

    window     the quarterback's weighted average over his window, untouched
    the live family, named from active_models, marked with a star

For each: how far its number was from what happened (and the line's own miss
beside it), how often its side of the line was right on every priced prop,
how well its gap from the line sorts the overs from the unders, and how often
it was right on the fifth of props where it disagreed with the line most,
which is how the board picks its best. Both seasons, then each.

Passing touchdowns are a count: the side is the likelier one under a Poisson
with the projection as its rate, as the board does it.

What to look for: a family that is right more often than the rest in both
seasons, on all props and on its top fifth. If it is not the live one, the
fix is to train passing as that family. If no family separates, the record's
edge is in the selection, and the next test is which props the board chose.

Prices are the best any book showed on each side. Read-only.

    sh lab/run.sh research_passing.py
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from psycopg2.extras import RealDictCursor
from scipy.stats import poisson

import backtest_gap_system as bt
import research_reproject_blanks as RB
import train as tr
from research_who_sorts import sorts

MARKETS = [m.strip() for m in os.getenv(
    "MARKETS", "pass_yds,pass_td,pass_att,pass_completions").split(",") if m.strip()]
ODDS = {"pass_td": "player_pass_tds"}
FAMILIES = ("ridge_v1", "enet_v2", "rf_v1", "xtrees_v1", "hgb_v1", "gb_v1", "vote_v1")
COUNTS = {"pass_td"}
MIN_PAST = 600


def live_family(code):
    with tr.connect() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("""SELECT a.model_name FROM active_models a
                       JOIN prop_markets p ON p.id = a.market_id
                       WHERE p.code = %s""", (code,))
        r = cur.fetchone()
    return (r or {}).get("model_name")


def lines(code):
    if code in bt.ODDS_KEY:
        return bt.lines(code)
    bt.ODDS_KEY[code] = ODDS[code]
    try:
        return bt.lines(code)
    finally:
        bt.ODDS_KEY.pop(code, None)


def project(d, family, seasons):
    """One column of walk-forward projections for one family."""
    out = pd.Series(np.nan, index=d.index)
    for s in seasons:
        past, now = d[d["season"] < s], d.index[d["season"] == s]
        if len(past) < MIN_PAST or not len(now):
            continue
        X_tr, cols = tr._build_feature_dataframe(past.copy())
        X_te, _ = tr._build_feature_dataframe(d.loc[now].copy())
        for c in cols:
            if c not in X_te.columns:
                X_te[c] = 0.0
        try:
            m = tr.build_model(family)
            m.fit(X_tr[cols], past["label_actual"].astype(float))
            out.loc[now] = np.maximum(0.0, m.predict(X_te[cols]))
        except Exception as e:
            print(f"    {family} {s}: failed ({type(e).__name__}: {e})")
    return out


def score(j, col, code):
    """The same priced props, sided by one projection."""
    j = j[j[col].notna()].copy()
    if code in COUNTS:
        p_over = poisson.sf(np.floor(j["line"].to_numpy(dtype=float)),
                            j[col].clip(lower=0.01).to_numpy(dtype=float))
        over = p_over > 0.5
        j["lean"] = p_over
    else:
        over = (j[col] > j["line"]).to_numpy()
        j["lean"] = (j[col] - j["line"]) / j["line"]
    price = np.where(over, j["over"], j["under"]).astype(float)
    j["needed"] = np.where(price > 0, 100.0 / (price + 100.0), -price / (-price + 100.0))
    j["hit"] = np.where(over, j["went_over"], ~j["went_over"]).astype(float)
    j["far"] = (j["lean"] - (0.5 if code in COUNTS else 0.0)).abs()
    j["top"] = False
    for _, g in j.groupby("slate"):
        n = max(1, int(round(len(g) * 0.2)))
        j.loc[g["far"].nlargest(n).index, "top"] = True
    return j


def row(name, j, col, code):
    s = score(j, col, code)
    if len(s) < 40:
        return f"    {name:<12}{len(s):>5}  too few"
    mae = (s[col] - s["actual"]).abs().mean()
    return (f"    {name:<12}{len(s):>5}{mae:>9.2f}{s['hit'].mean():>9.1%}"
            f"{s['needed'].mean():>9.1%}{sorts(s['lean'], s['went_over']):>9.3f}"
            f"   {RB.bet(s[s['top']], 20)}")


def table(j, cols, code, title):
    if len(j) < 40:
        return
    line_miss = (j["line"] - j["actual"]).abs().mean()
    print(f"\n  {title}: {len(j):,} priced props, over came in "
          f"{j['went_over'].mean():.1%}, the line missed by {line_miss:.2f} on average")
    print(f"    {'':<12}{'props':>5}{'missed':>9}{'right':>9}{'needed':>9}"
          f"{'sorts':>9}   its top fifth: picks, right, needed, difference")
    for name, col in cols:
        print(row(name, j, col, code))


def main():
    seasons = RB.SEASONS
    for code in MARKETS:
        d = bt.features(code)
        o = lines(code)
        if d is None or o is None or o.empty:
            print(f"\n{code}: nothing to read")
            continue
        live = live_family(code)
        print(f"\n{code.upper()}: {len(d):,} labelled rows at lookback "
              f"{d.attrs.get('lookback')}. Live family: {live}.")
        fams = list(FAMILIES)
        if live and not any(live.startswith(f.split('_')[0]) for f in fams):
            fams.append(live)
        cols = [("window", "weighted_mean")]
        for f in fams:
            d[f] = project(d, f, seasons)
            star = "*" if live and live.startswith(f.split("_")[0]) else ""
            cols.append((f.split("_")[0] + star, f))
        d = d.rename(columns={"name": "player_name"})
        d["d"] = pd.to_datetime(d["as_of_game_date"])
        j = d[d["season"].isin(seasons)].merge(o, on=["player_name", "d"], how="inner")
        j = j[j["over"].notna() & j["under"].notna() & ((j["line"] * 2) % 2 == 1)].copy()
        j["actual"] = j["label_actual"].astype(float)
        j["went_over"] = j["actual"] > j["line"]
        j["weighted_mean"] = pd.to_numeric(j["weighted_mean"], errors="coerce")
        j["cluster"] = j["d"].dt.strftime("%Y-%m-%d")
        j["slate"] = (j["d"] - pd.Timedelta(days=3)).dt.strftime("%G-%V")
        table(j, cols, code, "both seasons")
        for s in seasons:
            table(j[j["season"] == s], cols, code, str(s))
    print("\n'missed' is the average distance from what happened. 'sorts' is how "
          "often the projection's lean rated a prop that went over above one "
          "that did not; 0.50 is a coin. The star is the family that is live.")


if __name__ == "__main__":
    main()
