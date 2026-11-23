"""Why do the overs the model takes lose, in every market?

Four markets projected again on 22 November 2026, and the same thing in each:
the unders the median takes are right 52% to 56% of the time, and the overs it
takes are right 45% to 50%. Correcting the ladder's level against results
(research_calibrated_side.py) moved some overs to unders and was right more
often on catches in both seasons, but the overs it kept still lost. So the
model does not know which overs are good. This looks at what its overs are.

An over is taken when the median is above the line, which means the model
thinks more of the player than the book does. There are only a few ways that
happens, and each is cut here, market by market and all together:

    line against the window   the line as a share of the player's average over
                              his window. A line set well under what he has
                              been doing is the book knowing something the
                              window does not (a role cut, a returning
                              starter, a bad matchup). A line at or above it
                              is the book agreeing he is good.
    trend                     whether his window was rising or falling.
    how far over              the median's gap above the line, in thirds.
    position

Each row: the overs the model took, how often the over came in, and what the
price needed. Under them, the same cut for every prop whatever the side, so a
row can be read against how often the over comes in for that kind of prop
anyway.

Read-only. Reads the projections research_reproject_blanks.py kept.

    MARKETS=recs,rec_yds,rush_yds,rush_att sh lab/run.sh research_overs.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import research_reproject_blanks as RB
from research_side_by_price import implied


def cell(s, least=30):
    if len(s) < least:
        return f"{len(s):>5}  too few"
    return (f"{len(s):>5}  over came in {s['went_over'].mean():>6.1%}, "
            f"needed {s['need_o'].mean():>6.1%}")


def prepare(j):
    j = j.copy()
    j["need_o"] = implied(j["over"])
    avg = j["mean"].astype(float)
    ratio = np.where(avg > 0, j["line"].astype(float) / avg, np.nan)
    j["line vs window"] = pd.cut(
        ratio, [-np.inf, 0.6, 0.8, 0.95, 1.1, np.inf],
        labels=["line under 60% of his average", "60 to 80%", "80 to 95%",
                "about his average", "line above his average"])
    t = j["trend"].astype(float)
    j["trend"] = np.where(t > 0, "window rising", np.where(t < 0, "window falling", "flat"))
    return j


def cuts(j, title):
    print(f"\n{title}")
    overs = j[j["q50"] > j["line"]].copy()
    print(f"    every over the model took   {cell(overs)}")
    print(f"    every prop                  {cell(j)}")
    if len(overs) >= 90:
        g = overs["gap"].astype(float).abs()
        overs["how far over"] = pd.qcut(g.rank(method="first"), 3,
                                        labels=["nearest the line", "middle", "furthest over"])
    for col in ("line vs window", "trend", "how far over", "position"):
        if col not in overs:
            continue
        print(f"\n    by {col}")
        for name, s in overs.groupby(col, observed=True):
            ref = f"   | all props: {cell(j[j[col] == name])}" if col in j else ""
            print(f"      {str(name):<32}{cell(s)}{ref}")


def main():
    every = []
    for code in RB.MARKETS:
        arms = RB.cached_arms(code)
        if arms is None:
            print(f"{code}: no kept projections. Run research_reproject_blanks.py first.")
            continue
        j = prepare(arms["with"])
        every.append(j)
        cuts(j, f"{code.upper()}: {len(j):,} priced props")
        for s in RB.SEASONS:
            cuts(j[j["season"] == s], f"{code.upper()} {s}")
    if len(every) > 1:
        cuts(pd.concat(every, ignore_index=True), "ALL OF THESE MARKETS TOGETHER")


if __name__ == "__main__":
    main()
