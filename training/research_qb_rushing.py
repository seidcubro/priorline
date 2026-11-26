"""Why quarterback rushing picks lose, and what the middle game of a quarterback looks like.

On the published record, the under on a quarterback's rushing prop came in
53.6% (2025) and 65.2% (2026) for attempts, and 53.5% and 56.6% for yards. Our
side on those props was right 48.1%, 55.4%, 49.1% and 52.0%: worse than taking
every under, in both seasons and both markets.

The projection is not the problem. Our mean for a quarterback's rushing yards
was 16.8 and they averaged 16.6. The problem is that the mean is not where a
quarterback's game usually lands. A scramble for 30 or a designed-run game
drags the average up, and the usual game sits well under it: the median was
12 yards against a mean of 16.6. The book hangs its line near the mean. Our
range put its median near the mean too, so our side came out a coin.

This measures the ratio of the median outcome to our projection in one season
and applies it in the other, so each season is scored by a number it did not
produce:

    side      over if ratio x our projection is above the line, else under

and prints how that side did against the published one and against taking
every under. Reads the record pulled by pull_record.py; runs anywhere.

    python lab/training/research_qb_rushing.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CACHE = Path(__file__).resolve().parents[2] / "marketing" / ".cache"


def main():
    d = pd.read_pickle(CACHE / "record.pkl")
    for c in ("line", "actual", "projection"):
        d[c] = d[c].astype(float)
    d = d[d["season"].isin([2025, 2026]) & (d["actual"] != d["line"])
          & (d["position"] == "QB") & d["market_code"].isin(["rush_att", "rush_yds"])]
    pooled = {}
    for code in ("rush_att", "rush_yds"):
        x = d[d["market_code"] == code]
        print(f"\n{code.upper()}: {len(x):,} quarterback props")
        ratio = {}
        for s in (2025, 2026):
            y = x[x["season"] == s]
            # Each game's outcome over what we projected for it; the middle one.
            ratio[s] = float(np.median(y["actual"] / y["projection"].clip(lower=0.5)))
            print(f"  {s}: {len(y)} props. We projected {y['projection'].mean():.1f}, the line "
                  f"was {y['line'].mean():.1f}, they averaged {y['actual'].mean():.1f}, the "
                  f"middle game was {y['actual'].median():.1f}. Middle outcome over our "
                  f"projection: {ratio[s]:.2f}")
        for fit, test in ((2025, 2026), (2026, 2025)):
            y = x[x["season"] == test]
            over = y["projection"] * ratio[fit] > y["line"]
            right = np.where(over, y["over"], 1 - y["over"])
            print(f"  ratio from {fit} ({ratio[fit]:.2f}) on {test}: right {right.mean():.1%} on "
                  f"{len(y)}; published side {y['hit'].mean():.1%}; every under "
                  f"{(1 - y['over']).mean():.1%}; it calls under on {(~over).mean():.0%}")
        pooled[code] = float(np.median(x["actual"] / x["projection"].clip(lower=0.5)))
    print("\nBoth seasons together, the number to use: "
          + ", ".join(f"{k} {v:.2f}" for k, v in pooled.items()))


if __name__ == "__main__":
    main()
