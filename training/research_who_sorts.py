"""Does the model sort catch props at all, and does it sort them better than the price?

research_side_by_price.py took the side from the ladder's chance against the
price. It was right less often on every size of line (54.9% to 52.5% over all
of them) and its margin over the price moved by half a point, inside the noise.
So choosing the side differently is not the fix.

The reason would be that the chance itself carries little. On a half-point
line the over came in 54.5% of the time; the overs the median took came in
56.5% and the overs the price rule took came in 53.4%. Neither is far from
taking every over blind. This measures that directly.

For every size of line, two numbers between 0.5 and 1. Take one prop that went
over and one that did not: how often did the ruler rate the first one higher?
0.50 is a coin, and a ruler has to beat the book's to be worth a bet.

    ladder   the model's chance of the over
    book     the over's price with the hold taken out
    both     the two averaged

Then the props cut into fifths by the ladder's chance: what it said, how often
the over came in, and what the over's price needed. A model that sorts shows
the over coming in more often in each fifth. A model that can be bet shows the
over coming in more often than the price needs in the top fifth, or less
often than the under needs in the bottom one.

The prices are the best any book showed, so 'needed' is low in every row.

It reads the projections research_reproject_blanks.py kept. Read-only.

    sh lab/run.sh research_who_sorts.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import research_reproject_blanks as RB
from research_side_by_price import BANDS, implied


def sorts(score, went):
    """Chance a prop that went over was rated above one that did not."""
    went = np.asarray(went, dtype=bool)
    n1, n0 = went.sum(), (~went).sum()
    if n1 < 10 or n0 < 10:
        return float("nan")
    r = pd.Series(np.asarray(score, dtype=float)).rank().to_numpy()
    return (r[went].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def rulers(j):
    need_o, need_u = implied(j["over"]), implied(j["under"])
    book = need_o / (need_o + need_u)
    return {"ladder": j["p_over"].to_numpy(dtype=float), "book": book,
            "both": (j["p_over"].to_numpy(dtype=float) + book) / 2.0}, need_o, need_u


def fifths(j, name):
    r, need_o, need_u = rulers(j)
    j = j.assign(need_o=need_o, need_u=need_u, book=r["book"])
    try:
        j["fifth"] = pd.qcut(j["p_over"].rank(method="first"), 5, labels=False)
    except ValueError:
        return
    print(f"\n    line {name}, by fifth of the ladder's chance")
    print(f"    {'fifth':<8}{'props':>6}{'ladder said':>13}{'book said':>11}"
          f"{'over came in':>14}{'over needs':>12}{'under needs':>13}")
    for k, g in j.groupby("fifth"):
        print(f"    {int(k) + 1:<8}{len(g):>6}{g['p_over'].mean():>13.1%}"
              f"{g['book'].mean():>11.1%}{g['went_over'].mean():>14.1%}"
              f"{g['need_o'].mean():>12.1%}{g['need_u'].mean():>13.1%}")


def section(j, title):
    print(f"\n{title}")
    j = j.assign(band=j["line"].map(RB.band))
    print(f"    {'line':<12}{'props':>6}{'ladder':>9}{'book':>9}{'both':>9}")
    for name in BANDS + ("every line",):
        s = j if name == "every line" else j[j["band"] == name]
        if len(s) < 50:
            continue
        r, _, _ = rulers(s)
        print(f"    {name:<12}{len(s):>6}" + "".join(
            f"{sorts(r[k], s['went_over']):>9.3f}" for k in ("ladder", "book", "both")))


def main():
    for code in RB.MARKETS:
        arms = RB.cached_arms(code)
        if arms is None:
            print(f"{code}: no kept projections. Run research_reproject_blanks.py first.")
            continue
        j = arms["with"]
        print(f"{code.upper()}: {len(j):,} priced props, projected with the blank games in.")
        print("How often each ruler rated a prop that went over above one that did not.")
        section(j, "BOTH SEASONS")
        for s in RB.SEASONS:
            section(j[j["season"] == s], str(s))
        print("\nBOTH SEASONS, cut by the ladder's chance")
        b = j.assign(band=j["line"].map(RB.band))
        for name in BANDS:
            s = b[b["band"] == name]
            if len(s) >= 100:
                fifths(s, name)


if __name__ == "__main__":
    main()
