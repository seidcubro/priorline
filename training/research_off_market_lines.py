"""When one book's line is off from the others, does the side toward the others win?

Every test on 22 November 2026 says the same thing about the yardage markets.
The model ranks a yardage prop no better than a coin. Carrying the side over
from the catch model, which is right 55% of the time on catches, is right
51.3% of the time on receiving yards. The model already has target share, air
yards, depth of target and yards after the catch. The yardage line is simply
a good forecast, and nothing in the box score is beating it.

So this stops asking the model and asks the books. They do not always agree.
When one book hangs 45.5 and the rest are at 48.5, either that book knows
something or it is slow, and the bet toward the others (over 45.5) is taken
against a number the rest of the market thinks is wrong. No projection is
involved. It is the same idea as shopping for the best price, which is worth
more on this board than any model change has been, applied to the line.

For every market, every player and game, at the last look before kickoff
where at least three books were up: each book's line against the middle of the
other books' lines. Where a book is off, the side toward the others, at that
book's own price for that side:

    by how far off     as a share of the others' line
    by direction       the off book is low (take the over) or high (the under)
    by season

Each cell: bets, how often right, what the price needed, the difference, its
interval. Pushes are left out. One bet per player, game and book.

If a row is right well over what its price needs in every season, the board
should show it: which book is off, by how much, on which side. It would be the
first thing on the site that does not depend on our projection being good.

The prices here are the real price at that book at that moment, not the best
any book showed, so these margins are not flattered the way the others are.

Read-only.

    sh lab/run.sh research_off_market_lines.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from psycopg2.extras import RealDictCursor

import backtest_gap_system as bt
import research_reproject_blanks as RB
import train as tr

MIN_BOOKS = 3
BANDS = ((0.0, 0.03, "under 3%"), (0.03, 0.06, "3 to 6%"),
         (0.06, 0.10, "6 to 10%"), (0.10, 9.9, "10% and over"))


def board(code):
    """Every book's line and prices at the last look before kickoff."""
    with tr.connect() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("""
            WITH last AS (
                SELECT player_name, provider_event_id, max(observed_at) AS seen
                FROM odds_snapshots
                WHERE market_key = %s AND observed_at <= commence_time
                GROUP BY 1, 2)
            SELECT o.player_name, o.bookmaker_key,
                   lower(trim(o.outcome_name)) AS side, o.line, o.price_american,
                   (o.commence_time AT TIME ZONE 'America/New_York')::date AS d
            FROM odds_snapshots o
            JOIN last l ON l.player_name = o.player_name
                       AND l.provider_event_id = o.provider_event_id
                       AND l.seen = o.observed_at
            WHERE o.market_key = %s AND o.line IS NOT NULL
              AND o.price_american IS NOT NULL
        """, (bt.ODDS_KEY[code], bt.ODDS_KEY[code]))
        o = pd.DataFrame(cur.fetchall())
    if o.empty:
        return o
    o = o[o["side"].isin(["over", "under"])]
    o["line"] = o["line"].astype(float)
    p = o.pivot_table(index=["player_name", "d", "bookmaker_key", "line"],
                      columns="side", values="price_american",
                      aggfunc="max").reset_index()
    for c in ("over", "under"):
        if c not in p:
            p[c] = np.nan
    p = p[p["over"].notna() & p["under"].notna()]
    # One line a book: the one priced closest to even, which is its main line.
    p["tilt"] = (p["over"] - p["under"]).abs()
    p = p.sort_values("tilt").drop_duplicates(["player_name", "d", "bookmaker_key"])
    p["d"] = pd.to_datetime(p["d"])
    return p


def implied(price):
    price = np.asarray(price, dtype=float)
    return np.where(price > 0, 100.0 / (price + 100.0), -price / (-price + 100.0))


def bets(code):
    p = board(code)
    f = bt.features(code)
    if p.empty or f is None:
        return None, 0
    f = f.rename(columns={"name": "player_name"})
    f["d"] = pd.to_datetime(f["as_of_game_date"])
    f = f[["player_name", "d", "label_actual", "season"]].drop_duplicates(
        ["player_name", "d"])
    p = p.merge(f, on=["player_name", "d"], how="inner")
    rows = []
    games = 0
    for (_, _), g in p.groupby(["player_name", "d"]):
        if len(g) < MIN_BOOKS:
            continue
        games += 1
        lines = g["line"].to_numpy()
        for i, r in enumerate(g.itertuples()):
            others = float(np.median(np.delete(lines, i)))
            if others <= 0 or r.line == others:
                continue
            actual = float(r.label_actual)
            if actual == r.line:
                continue
            over = r.line < others
            rows.append({
                "season": int(r.season), "cluster": r.d.strftime("%Y-%m-%d"),
                "off": abs(r.line - others) / others,
                "side": "over" if over else "under",
                "book": r.bookmaker_key,
                "needed": float(implied(r.over if over else r.under)),
                "hit": float((actual > r.line) if over else (actual < r.line)),
            })
    return pd.DataFrame(rows), games


def report(j, title):
    if len(j) < 25:
        return
    print(f"\n  {title}")
    print(f"    every off line          {RB.bet(j)}")
    for lo, hi, name in BANDS:
        s = j[(j["off"] >= lo) & (j["off"] < hi)]
        print(f"    off by {name:<16} {RB.bet(s)}")
    for side, label in (("over", "book is low, take the over"),
                        ("under", "book is high, take the under")):
        print(f"    {label:<30}{RB.bet(j[j['side'] == side])}")


def main():
    every = []
    for code in bt.ODDS_KEY:
        j, games = bets(code)
        if j is None or j.empty:
            print(f"\n{code}: nothing to read")
            continue
        j["market"] = code
        every.append(j)
        print(f"\n{code.upper()}: {games:,} player-games with {MIN_BOOKS} or "
              f"more books up at the last look, {len(j):,} lines off from the "
              f"rest. Books: {', '.join(sorted(j['book'].unique()))}")
        report(j, "every season")
        for s in sorted(j["season"].unique()):
            report(j[j["season"] == s], str(s))
    if len(every) > 1:
        a = pd.concat(every, ignore_index=True)
        print("\nEVERY MARKET TOGETHER")
        report(a, "every season")
        for s in sorted(a["season"].unique()):
            report(a[a["season"] == s], str(s))
        big = a[a["off"] >= 0.06]
        if len(big) >= 25:
            print("\n  off by 6% or more, by book that was off")
            for b, s in big.groupby("book"):
                print(f"    {b:<24}{RB.bet(s)}")


if __name__ == "__main__":
    main()
