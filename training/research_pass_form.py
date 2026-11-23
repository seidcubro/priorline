"""Take the passing yards side from where the line sits against the quarterback's recent games.

Of 66 numbers the feature build holds on a passing yards prop, one separated
the overs from the unders in all three seasons: the line as a share of the
quarterback's average over his window. 0.565 in 2024, 0.519 in 2025, 0.567 in
2026, and 2024 was bought after the idea was written down, so it was a season
the idea had never seen. Everything that measures his recent form leans the
other way with it (his weighted average, his trend, his best game), which is
the same fact from the other side.

What it says: when the book hangs a number above what he has been doing, he
goes over it more often than not, and when it hangs one below, he goes under.
The book has already moved off his recent games for a reason (who he is
playing, who is back, what the game should look like), and it does not move
far enough. Our model is built from those recent games, so it leans back
toward them and lands on the wrong side.

This prices it. No model is fitted: the side is a comparison of two numbers.

    model     the side today's projections took (read from the kept ones)
    form      over if the line is above his window average, under if below
    form, clear only
              the same, kept only where the line is at least 2% away from the
              average. The rest are props where the two agree and the rule
              has nothing to say.

By season, on every priced prop: picks, how often right, what the prices
needed, the difference, its interval. Then by how far the line sits from his
average and by side, to see where it is strong and where it gives out. The
same for attempts and completions, which have one season of prices.

Prices are the best any book showed on each side.

Read-only.

    SEASONS=2024,2025,2026 sh lab/run.sh research_pass_form.py
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

import backtest_gap_system as bt
import research_passing as P
import research_reproject_blanks as RB
from research_side_by_price import implied

MARKETS = [m.strip() for m in os.getenv(
    "MARKETS", "pass_yds,pass_att,pass_completions").split(",") if m.strip()]
BANDS = ((0.0, 0.85, "line under 85% of it"), (0.85, 0.93, "85 to 93%"),
         (0.93, 0.98, "93 to 98%"), (0.98, 1.02, "within 2%"),
         (1.02, 1.08, "102 to 108%"), (1.08, 1.20, "108 to 120%"),
         (1.20, 99.0, "over 120%"))


def sided(j, over):
    j = j.copy()
    over = np.asarray(over, dtype=bool)
    j["needed"] = np.where(over, implied(j["over"]), implied(j["under"]))
    j["hit"] = np.where(over, j["went_over"], ~j["went_over"]).astype(float)
    return j


def props(code):
    d, o = bt.features(code), P.lines(code)
    if d is None or o is None or o.empty:
        return None
    d = d.rename(columns={"name": "player_name"})
    d["d"] = pd.to_datetime(d["as_of_game_date"])
    j = d.merge(o, on=["player_name", "d"], how="inner")
    j = j[j["over"].notna() & j["under"].notna() & ((j["line"] * 2) % 2 == 1)].copy()
    j["actual"] = j["label_actual"].astype(float)
    j["went_over"] = j["actual"] > j["line"]
    avg = pd.to_numeric(j["mean"], errors="coerce")
    j["ratio"] = np.where(avg > 0, j["line"] / avg, np.nan)
    j = j[j["ratio"].notna()].copy()
    j["cluster"] = j["d"].dt.strftime("%Y-%m-%d")
    arms = RB.cached_arms(code)
    if arms is not None:
        m = arms["with"][["player_name", "d", "q50"]].drop_duplicates(["player_name", "d"])
        j = j.merge(m, on=["player_name", "d"], how="left")
    else:
        j["q50"] = np.nan
    return j


def block(j, title):
    if len(j) < 60:
        return
    print(f"\n  {title}: {len(j):,} priced props, over came in "
          f"{j['went_over'].mean():.1%}")
    m = j[j["q50"].notna()]
    print(f"    {'model':<28}{RB.bet(sided(m, m['q50'] > m['line']))}")
    print(f"    {'form':<28}{RB.bet(sided(j, j['ratio'] > 1))}")
    clear = j[(j["ratio"] - 1).abs() >= 0.02]
    print(f"    {'form, clear only':<28}{RB.bet(sided(clear, clear['ratio'] > 1))}")
    both = m[(m["q50"] > m["line"]) == (m["ratio"] > 1)]
    apart = m[(m["q50"] > m["line"]) != (m["ratio"] > 1)]
    print(f"    {'model and form agree':<28}{RB.bet(sided(both, both['ratio'] > 1))}")
    if len(apart) >= 25:
        a = sided(apart, apart["ratio"] > 1)
        print(f"    they disagree on {len(apart)}: form was right {a['hit'].mean():.1%}, "
              f"the model {1 - a['hit'].mean():.1%}")
    print("    by where the line sits against his average, on the form side")
    for lo, hi, name in BANDS:
        s = j[(j["ratio"] >= lo) & (j["ratio"] < hi)]
        print(f"      {name:<24}{RB.bet(sided(s, s['ratio'] > 1), 20)}")


def main():
    for code in MARKETS:
        j = props(code)
        if j is None or len(j) < 60:
            print(f"\n{code}: nothing to read")
            continue
        print(f"\n{code.upper()}")
        block(j, "every season with prices")
        for s in sorted(j["season"].unique()):
            block(j[j["season"] == s], str(int(s)))
    print("\nEach cell: picks, how often right, what the prices needed, the "
          "difference, its interval. 'model' covers the seasons the kept "
          "projections cover, so its count can be smaller.")


if __name__ == "__main__":
    main()
