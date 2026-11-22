"""On a low count line, should the side come from the chance against the price?

research_reproject_blanks.py projected two seasons of catches again with the
blank games in. The projection under a half-point line is now close to honest:
it says 1.13 where the players averaged 1.05, and the ladder says the over
comes in 58% of the time where it came in 54.5%. And the board still takes
the over on 84% of them, 56.5% right at prices that need 60.7%.

That is not a projection problem any more. It is the side rule. The board
takes the over whenever the median is above the line. On a line of 0.5 the
median is above the line whenever the man is more likely than not to catch
one pass, and the over is priced for a man who catches one six times in ten
or better. A 58% chance at a price that needs 61% is a losing bet the model
itself can see, and the rule takes it anyway. Where the model did say under,
the under won 55.6% at plus money that needed 46.4%, on 45 props.

So this asks the model's own question on the same projections and prices.
For every prop, the chance the ladder gives each side, minus what that
side's price needs:

    median      the side the median falls on. What the board does today.
    better      the side whose chance is further over its price.
    clear only  the same, kept only where that side's chance is over its
                price at all. The rest are props the model has no opinion
                about once the price is paid.

By size of line and by season: picks, how often right, what the prices
needed, the difference. And the overs and unders each rule takes on a
half-point line, since that is where the loss is.

'better' is worth building into the board if it is ahead of 'median' on the
low lines in both seasons and not behind it on the rest. Nothing leaves the
board in it: the same props, some on the other side.

The prices are the best any book showed on each side, as in the file this
reads from, so the differences against the price are flattered in every row
by the same amount. Read the rows against each other.

It reads the projections research_reproject_blanks.py kept, and fits them
first if they are not there (about twenty minutes).

Read-only.

    sh lab/run.sh research_side_by_price.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import research_reproject_blanks as RB

RULES = ("median", "better", "clear only")
BANDS = ("0.5", "1.5", "2.5", "3.5 to 5.5", "6.5 and up")


def implied(p):
    p = p.astype(float)
    return np.where(p > 0, 100.0 / (p + 100.0), -p / (-p + 100.0))


def sided(j, rule):
    """The same props with the side chosen by one rule."""
    j = j.copy()
    need_o, need_u = implied(j["over"]), implied(j["under"])
    edge_o = j["p_over"] - need_o
    edge_u = (1.0 - j["p_over"]) - need_u
    if rule == "median":
        over = (j["q50"] > j["line"]).to_numpy()
    else:
        over = (edge_o >= edge_u).to_numpy()
    j["side"] = np.where(over, "over", "under")
    j["needed"] = np.where(over, need_o, need_u)
    j["edge"] = np.where(over, edge_o, edge_u)
    j["hit"] = np.where(over, j["went_over"], ~j["went_over"]).astype(float)
    if rule == "clear only":
        j = j[j["edge"] > 0]
    return j


def on_side(j, side):
    return j[j["side"] == side]


def table(j, title):
    print(f"\n{title}")
    j = j.assign(band=j["line"].map(RB.band))
    print(f"    {'line':<12}" + "".join(f"{r:<42}" for r in RULES))
    for name in BANDS + ("every line",):
        s = j if name == "every line" else j[j["band"] == name]
        if len(s) < 25:
            continue
        print(f"    {name:<12}" + "".join(
            f"{RB.bet(sided(s, r)):<42}" for r in RULES))
    half = j[j["band"] == "0.5"]
    if len(half) >= 25:
        print("    on a line of 0.5, by the side taken")
        for side in ("over", "under"):
            print(f"    {side + 's':<12}" + "".join(
                f"{RB.bet(on_side(sided(half, r), side), 15):<42}"
                for r in RULES))
    low = j[j["band"].isin(["0.5", "1.5"])]
    if len(low) >= 25:
        print("    lines of 0.5 and 1.5 together, by the side taken")
        for side in ("over", "under"):
            print(f"    {side + 's':<12}" + "".join(
                f"{RB.bet(on_side(sided(low, r), side), 15):<42}"
                for r in RULES))


def main():
    for code in RB.MARKETS:
        arms = RB.cached_arms(code)
        if arms is None:
            print(f"{code}: no kept projections, fitting them first.")
            RB.main()
            arms = RB.cached_arms(code)
        if arms is None:
            print(f"{code}: nothing to read")
            continue
        j = arms["with"]
        print(f"\n{code.upper()}: {len(j):,} priced props, projected with the "
              f"blank games in. The ladder said over "
              f"{j['p_over'].mean():.1%} of the time and over came in "
              f"{j['went_over'].mean():.1%}.")
        cal = j.assign(band=j["line"].map(RB.band)).groupby("band").agg(
            n=("hit", "size"), said=("p_over", "mean"), came=("went_over", "mean"))
        for name in BANDS:
            if name in cal.index:
                r = cal.loc[name]
                print(f"    line {name:<11}{int(r['n']):>5} props   ladder said "
                      f"over {r['said']:.1%}, came in {r['came']:.1%}")
        table(j, "BOTH SEASONS")
        for s in RB.SEASONS:
            table(j[j["season"] == s], str(s))
    print("\nEach cell: picks, how often right, what the prices needed, the "
          "difference, its interval. 'median' is what the board does today.")


if __name__ == "__main__":
    main()
