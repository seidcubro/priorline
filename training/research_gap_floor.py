"""Is the elite tier being filled by half-point lines, and does it cost us?

Week 10 was the worst week since September: 48.0% on 573 graded picks where
the prices needed 54.7%. The elite tier went 20 of 41. Eleven of those 41 were
the same bet, over 0.5 catches on a backup tight end or a depth receiver at a
price of -155 to -260, and six of the eleven finished on zero (Van Jefferson,
Noah Fant, Efton Chism, Tory Horton, Foster Moreau, Matthew Hibner). With the
58 already in the record that is 34 of 69 this season, 49%, at prices that
need about 64%.

Why they are elite at all: tiers rank on the gap between my number and the
line as a share of the line, with the line floored at 0.5. A projection of
1.4 catches against a line of 0.5 is a gap of 180 percent. A projection of 70
yards against a line of 60 is 17 percent. So the half-point line outranks
almost anything a starter can produce, on exactly the player whose role is
least certain, and whose blank games the model has never been shown
(diagnose_zero_games.py: one priced tight end game in eight).

The picks stay on the board. The question is only whether they belong at the
top of it. This replays the season the way replay_current_rules.py does, walk
forward, with the floor under the line raised, so a half-point line is
measured as if it were a 1.0, 1.5 or 2.5 line and has to earn its rank in
catches, not in percent. Nothing else changes: same picks, same prices, same
direction rule, same share of each market stamped elite.

For each floor: how many elite picks, how often they were right, what their
prices needed, the difference with its interval, and how many of them sat on
a line under 1.

A higher floor is worth shipping if elite's difference rises, its interval
does not fall, and the result holds when the season is cut in two. One number
moving on 300 picks is not that, and the halves are printed so it cannot be
read as if it were.

Read-only. It stores nothing: REPLAY_SAVE is not consulted here.

    docker compose run --rm training python research_gap_floor.py
"""

import numpy as np
import pandas as pd
import stats_ci as S
from sqlalchemy import create_engine

import build_prop_edges as bp
import gap_tier
import replay_current_rules as R

FLOORS = (0.5, 1.0, 1.5, 2.5)


def walk(d, floor):
    d = d.copy()
    d["gap"] = (d["p50"] - d["line"]).abs() / d["line"].abs().clip(lower=floor)
    live = d[(d["source"] == "live") & (d["week"] > 0)].copy()
    back = d[d["source"] == "backtest"]
    live["tier_wf"] = ""
    for w in sorted(live["week"].unique()):
        cfg = R.decide(live[live["week"] < w], back)
        idx = live.index[live["week"] == w]
        labels = R.tier_week(live.loc[idx], cfg).tolist()
        if gap_tier.PRICE_TIERS:
            implied = [None if pd.isna(v) else float(v)
                       for v in live.loc[idx, "needed"]]
            labels = gap_tier.recut_by_price(labels, implied, [True] * len(idx))
        live.loc[idx, "tier_wf"] = labels
    return live


def margin(s):
    s = s[s["needed"].notna()]
    if len(s) < 30:
        return f"{len(s):>6}  too few"
    diff = (s["hit"] - s["needed"]).to_numpy(dtype=float)
    r, lo, hi = S.clustered_bootstrap(
        lambda i: float(np.mean(diff[i])), s["cluster"].to_numpy(), n_boot=2000)
    return (f"{len(s):>6}{s['hit'].mean():>8.1%}{s['needed'].mean():>8.1%}"
            f"{r:>+8.1%}  [{lo:+.1%}, {hi:+.1%}]")


def main():
    eng = create_engine(bp.DATABASE_URL, future=True)
    d = R.load(eng)
    weeks = sorted(d.loc[(d["source"] == "live") & (d["week"] > 0), "week"].unique())
    half = weeks[len(weeks) // 2]
    print(f"{int(((d['source'] == 'live') & (d['week'] > 0)).sum()):,} live "
          f"graded props, weeks {weeks[0]} to {weeks[-1]}. The season is cut "
          f"before week {half} for the two halves.\n")
    print(f"{'floor':<8}{'tier':<8}{'picks':>6}{'hit':>8}{'needed':>8}"
          f"{'diff':>8}  interval            on a line under 1   early half"
          f"                          late half")
    for floor in FLOORS:
        live = walk(d, floor)
        for tier in ("elite", "strong"):
            s = live[live["tier_wf"] == tier]
            low = s[s["line"] < 1]
            lowtxt = (f"{len(low):>3} at {low['hit'].mean():.0%}" if len(low)
                      else "  0")
            print(f"{floor:<8}{tier:<8}{margin(s)}   {lowtxt:<18}"
                  f"{margin(s[s['week'] < half])}   {margin(s[s['week'] >= half])}")
        el = live[live["tier_wf"] == "elite"]
        by = el.groupby("market_code")["hit"].agg(["size", "mean"])
        print("        elite by market: " + ", ".join(
            f"{m} {int(r['size'])} at {r['mean']:.0%}" for m, r in by.iterrows()))
        print()
    print("floor 0.5 is what the board does today. diff is hit minus needed. "
          "The halves use the same tiers, split by week.")


if __name__ == "__main__":
    main()
