"""Take the side from the ladder's chance once that chance has been checked against results.

research_who_sorts.py: the ladder sorts catch props about as well as the
book's price (0.544 against 0.551) and its chances are wrong by size of line.
On lines of 3.5 to 5.5 its top fifth said the over was 52% and the over came
in 43%. On lines of 2.5 to 5.5 the over came in under half the time in every
fifth, and the board still takes 528 overs there and is right on 46% of them.

So the ladder's order is worth something and its level is not. This corrects
the level walking forward. Each week, using only the props already played,
it learns for each size of line how often the over really came in at each
ladder chance (an isotonic fit, so the order is kept), and takes the over
only where that corrected chance is above a half.

    median       the side the median falls on. What the board does today.
    corrected    the side the corrected chance favours.
    line only    the side that has come in more often on that size of line so
                 far, the model ignored. The control: 'corrected' has to beat
                 this to show the model adds anything to the habit of the line.

The same props under every rule: every week that has at least 400 earlier
props to learn from. Each cell: picks, how often right, what the prices
needed, the difference, its interval. Prices are the best any book showed.

'corrected' is worth building into the board if it is right more often than
'median' in both seasons. It flips sides; nothing leaves the board.

Read-only. Reads the projections research_reproject_blanks.py kept.

    sh lab/run.sh research_calibrated_side.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

import research_reproject_blanks as RB
from research_side_by_price import BANDS, implied

RULES = ("median", "corrected", "line only")
WARMUP = 400
BAND_MIN = 150


def walk(j):
    """Corrected chance and the line's own rate, from earlier weeks only."""
    j = j.assign(band=j["line"].map(RB.band)).sort_values("slate").copy()
    j["p_fix"] = np.nan
    j["p_line"] = np.nan
    for slate in sorted(j["slate"].unique()):
        past = j[j["slate"] < slate]
        if len(past) < WARMUP:
            continue
        now = j["slate"] == slate
        for name in j.loc[now, "band"].unique():
            rows = now & (j["band"] == name)
            own = past[past["band"] == name]
            src = own if len(own) >= BAND_MIN else past
            iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            iso.fit(src["p_over"].to_numpy(dtype=float),
                    src["went_over"].to_numpy(dtype=float))
            j.loc[rows, "p_fix"] = iso.predict(j.loc[rows, "p_over"].to_numpy(dtype=float))
            j.loc[rows, "p_line"] = float(src["went_over"].mean())
    return j[j["p_fix"].notna()].copy()


def sided(j, rule):
    j = j.copy()
    if rule == "median":
        over = (j["q50"] > j["line"]).to_numpy()
    elif rule == "corrected":
        over = (j["p_fix"] > 0.5).to_numpy()
    else:
        over = (j["p_line"] > 0.5).to_numpy()
    j["side"] = np.where(over, "over", "under")
    j["needed"] = np.where(over, implied(j["over"]), implied(j["under"]))
    j["hit"] = np.where(over, j["went_over"], ~j["went_over"]).astype(float)
    return j


def table(j, title):
    print(f"\n{title}")
    print(f"    {'line':<12}" + "".join(f"{r:<42}" for r in RULES))
    for name in BANDS + ("every line",):
        s = j if name == "every line" else j[j["band"] == name]
        if len(s) < 25:
            continue
        print(f"    {name:<12}" + "".join(
            f"{RB.bet(sided(s, r)):<42}" for r in RULES))
    for side in ("over", "under"):
        print(f"    {side + 's':<12}" + "".join(
            f"{RB.bet(sided(j, r)[lambda d: d['side'] == side], 15):<42}"
            for r in RULES))
    m, c = sided(j, "median"), sided(j, "corrected")
    moved = (m["side"] != c["side"]).to_numpy()
    if moved.sum() >= 15:
        print(f"    the {moved.sum()} props 'corrected' puts on the other side: "
              f"median was right {m['hit'][moved].mean():.1%}, "
              f"corrected {c['hit'][moved].mean():.1%}")
    top = j[j["top"]] if "top" in j else j.iloc[0:0]
    if len(top) >= 25:
        print(f"    {'top fifth':<12}" + "".join(
            f"{RB.bet(sided(top, r)):<42}" for r in RULES))


def main():
    for code in RB.MARKETS:
        arms = RB.cached_arms(code)
        if arms is None:
            print(f"{code}: no kept projections. Run research_reproject_blanks.py first.")
            continue
        j = walk(arms["with"])
        print(f"{code.upper()}: {len(j):,} priced props with at least {WARMUP} "
              f"earlier ones to learn from.")
        table(j, "BOTH SEASONS")
        for s in RB.SEASONS:
            table(j[j["season"] == s], str(s))
    print("\n'top fifth' is the top fifth of the gap each week, the board's elite, "
          "with the side each rule gives it.")


if __name__ == "__main__":
    main()
