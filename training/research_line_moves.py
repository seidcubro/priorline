"""Does the way a line moved during the week say which side of it to take?

research_what_the_line_misses.py looked at 81 numbers the feature build writes
for a receiving yards prop and 77 for rush yards. Not one separated the overs
from the unders in both seasons. The line has priced everything in the
feature store.

There is one thing in the database the feature store does not hold: the
line's own history. Every odds sync is kept, so for a prop seen more than
once before kickoff there is a first line and a last one. A line that climbed
three yards did so because money or news pushed it, and the question is
whether the last line has finished moving or tends to stop short (take the
side it was moving toward) or to overshoot (take the other one).

For every market, every player and game seen at two or more different times
before kickoff: the middle line across books at the first look and at the
last, and how often the over came in at the last line.

    moved up     the last line is higher than the first
    did not move
    moved down

by how far, as a share of the first line, and by season. And how many hours
apart the two looks were, because a first look taken an hour before the last
one is not an opening line.

Read-only. Pushes are left out.

    sh lab/run.sh research_line_moves.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from psycopg2.extras import RealDictCursor

import backtest_gap_system as bt
import train as tr

CUTS = ((-9.9, -0.06, "down 6% or more"), (-0.06, -0.02, "down 2 to 6%"),
        (-0.02, 0.0, "down under 2%"), (0.0, 0.0, "did not move"),
        (0.0, 0.02, "up under 2%"), (0.02, 0.06, "up 2 to 6%"),
        (0.06, 9.9, "up 6% or more"))


def history(code):
    with tr.connect() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("""
            SELECT player_name, provider_event_id, observed_at,
                   (commence_time AT TIME ZONE 'America/New_York')::date AS d,
                   percentile_cont(0.5) WITHIN GROUP (ORDER BY line) AS line
            FROM odds_snapshots
            WHERE market_key = %s AND line IS NOT NULL
              AND observed_at <= commence_time
              AND lower(trim(outcome_name)) = 'over'
            GROUP BY 1, 2, 3, 4
        """, (bt.ODDS_KEY[code],))
        o = pd.DataFrame(cur.fetchall())
    if o.empty:
        return o
    o["line"] = o["line"].astype(float)
    o = o.sort_values("observed_at")
    g = o.groupby(["player_name", "d"])
    out = g.agg(first=("line", "first"), last=("line", "last"),
                looks=("observed_at", "nunique"),
                t0=("observed_at", "first"), t1=("observed_at", "last")).reset_index()
    out = out[out["looks"] >= 2].copy()
    out["hours"] = (pd.to_datetime(out["t1"], utc=True)
                    - pd.to_datetime(out["t0"], utc=True)).dt.total_seconds() / 3600.0
    out["d"] = pd.to_datetime(out["d"])
    return out


def cell(s):
    if len(s) < 30:
        return f"{len(s):>6}  too few"
    return f"{len(s):>6}  over came in {s['went_over'].mean():>6.1%}"


def table(j, title):
    if len(j) < 60:
        return
    print(f"\n  {title}: {len(j):,} props, over came in "
          f"{j['went_over'].mean():.1%} at the last line, looks "
          f"{j['hours'].median():.0f} hours apart in the middle case")
    for lo, hi, name in CUTS:
        if lo == hi:
            s = j[j["move"] == 0]
        elif hi <= 0:
            s = j[(j["move"] >= lo) & (j["move"] < hi)]
        else:
            s = j[(j["move"] > lo) & (j["move"] <= hi)]
        print(f"    {name:<18}{cell(s)}")
    day = j[j["hours"] >= 24]
    print(f"    looks a day or more apart: moved up {cell(day[day['move'] > 0])}"
          f"   moved down {cell(day[day['move'] < 0])}")


def main():
    every = []
    for code in bt.ODDS_KEY:
        h = history(code)
        f = bt.features(code)
        if h.empty or f is None:
            print(f"\n{code}: nothing to read")
            continue
        f = f.rename(columns={"name": "player_name"})
        f["d"] = pd.to_datetime(f["as_of_game_date"])
        f = f[["player_name", "d", "label_actual", "season"]].drop_duplicates(
            ["player_name", "d"])
        j = h.merge(f, on=["player_name", "d"], how="inner")
        j = j[(j["first"] > 0) & (j["label_actual"].astype(float) != j["last"])].copy()
        j["move"] = (j["last"] - j["first"]) / j["first"]
        j["went_over"] = j["label_actual"].astype(float) > j["last"]
        j["market"] = code
        every.append(j)
        print(f"\n{code.upper()}")
        table(j, "every season")
        for s in sorted(j["season"].unique()):
            table(j[j["season"] == s], str(s))
    if len(every) > 1:
        a = pd.concat(every, ignore_index=True)
        print("\nEVERY MARKET TOGETHER")
        table(a, "every season")
        for s in sorted(a["season"].unique()):
            table(a[a["season"] == s], str(s))


if __name__ == "__main__":
    main()
