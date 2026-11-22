"""What do half-point props actually do, on both sides, at the prices offered?

The elite tier loses its price on one kind of pick: a line of 0.5. 91 of them
this season, 52.7% right where the prices needed 64.8%. Stamping them
differently does not fix that. Ranked by price they land in strong and take
the loss with them (strong goes from -2.7 to -5.7), and publishing them small
would take them off the page, which is hiding a problem, not fixing one.

The fix has to be in the pick itself, and there are only two things it can
be. A half-point prop is a yes or no question, will he catch one, will he
throw one:

    the side is wrong   the book's "yes" is overpriced on players who draw
                        blanks, and the bet with the edge is the "no" at plus
                        money. Then the fix is a probability that knows about
                        blank games, and the board takes the under.
    there is no side    the book has both sides about right and the vig is
                        the whole result. Then the fix is to stop the model
                        claiming an edge it does not have, and these picks
                        take whatever rank their measured record earns.

The record cannot say which, because it holds only the side I took and only
its price. This reads the prices themselves. For every prop on a line of 0.5
or 1.5 that the books posted this season, catches and passing touchdowns: the
last price on each side before kickoff, the best across books, and what the
player then did. A player who took a snap and recorded nothing counts as
zero. A player who did not play is void and left out.

It needs no model. Each prop is sorted by how often the player has drawn a
blank in his own last ten games with a snap (from before the game, pulled
toward his position's rate), and for each band it prints both bets:

    how often the over won, what its price needed, and what a unit on every
    one returned; and the same for the under.

If the under is ahead of its price where blanks are common, on a real count
and in both halves of the season, that is an edge and the next step is to
build the board's probability for these props around it. If neither side is
ahead anywhere, there is no edge here to find and the model has to be made
to say so.

Read-only.

    docker compose run --rm training python research_yes_no_props.py
"""

import os

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

import build_prop_edges as bp
import eval_blank_game as B

NOW = int(os.getenv("SEASON", "2026"))
MARKETS = {"player_receptions": ("catches", "receptions"),
           "player_pass_tds": ("passing TDs", "passing_tds")}
NORM = "lower(replace(replace({c}, '.', ''), '-', ' '))"
BANDS = [(0, 0.03, "under 3%"), (0.03, 0.08, "3 to 8%"), (0.08, 0.15, "8 to 15%"),
         (0.15, 0.30, "15 to 30%"), (0.30, 1.01, "30% and up")]


def implied(p):
    p = pd.to_numeric(p, errors="coerce")
    return np.where(p < 0, -p / (-p + 100.0), 100.0 / (p + 100.0))


def payout(p):
    p = pd.to_numeric(p, errors="coerce")
    return np.where(p < 0, 100.0 / -p, p / 100.0)


def load(eng):
    props = pd.read_sql(text(f"""
        SELECT DISTINCT ON (s.provider_event_id, s.bookmaker_key, s.market_key,
                            s.player_name, s.outcome_name)
               s.provider_event_id AS event_id,
               (s.commence_time AT TIME ZONE 'America/New_York')::date AS d,
               s.market_key, {NORM.format(c='s.player_name')} AS nm,
               s.player_name, lower(s.outcome_name) AS side, s.line,
               s.price_american AS price
        FROM odds_snapshots s
        WHERE s.market_key = ANY(:keys)
          AND s.line IN (0.5, 1.5)
          AND s.commence_time >= :since
          AND s.commence_time < now() - interval '3 days'
          AND s.observed_at <= s.commence_time
          AND s.price_american IS NOT NULL
          AND lower(s.outcome_name) IN ('over', 'under')
        ORDER BY s.provider_event_id, s.bookmaker_key, s.market_key,
                 s.player_name, s.outcome_name, s.observed_at DESC
    """), eng, params={"keys": list(MARKETS), "since": f"{NOW}-08-01"})
    ids = pd.read_sql(text(f"""
        SELECT external_id AS player_id, {NORM.format(c='name')} AS nm
        FROM players WHERE external_id IS NOT NULL
    """), eng)
    stats = pd.read_sql(text("""
        SELECT player_id, season, week, game_date,
               COALESCE(receptions, 0)::float8 AS receptions,
               COALESCE(passing_tds, 0)::float8 AS passing_tds
        FROM player_game_stats_app WHERE season = :s
    """), eng, params={"s": NOW})
    snaps = pd.read_sql(text("""
        SELECT sc.player_id, sc.season, sc.week, MAX(sc.position) AS position,
               BOOL_OR(s.player_id IS NOT NULL) AS has_row
        FROM snap_counts sc
        LEFT JOIN (
            SELECT DISTINCT player_id, season, week FROM player_game_stats_app
        ) s ON s.player_id = sc.player_id AND s.season = sc.season
           AND s.week = sc.week
        WHERE sc.offense_snaps > 0 AND sc.season >= :s - 2
          AND sc.player_id IS NOT NULL
        GROUP BY 1, 2, 3
    """), eng, params={"s": NOW})
    return props, ids, stats, snaps


def build(props, ids, stats, snaps):
    """One row a prop: the best price on each side, and what he did."""
    key = ["event_id", "d", "market_key", "nm", "line"]
    best = (props.groupby(key + ["side"])["price"].max().unstack("side")
            .reset_index().dropna(subset=["over", "under"]))
    best["d"] = pd.to_datetime(best["d"])
    stats["game_date"] = pd.to_datetime(stats["game_date"])
    cal = stats[["week", "game_date"]].drop_duplicates("game_date")
    week_of = {}
    for day in best["d"].drop_duplicates():
        near = cal[(cal["game_date"] - day).abs() <= pd.Timedelta(days=1)]
        week_of[day] = int(near["week"].iloc[0]) if len(near) else None
    best["week"] = best["d"].map(week_of)
    best = best.dropna(subset=["week"])
    best["week"] = best["week"].astype(int)
    best["season"] = NOW

    # A name can belong to more than one player. Keep the one who took a snap
    # that week, which is the one a prop was written about.
    cand = best.merge(ids, on="nm").merge(
        snaps[["player_id", "season", "week", "position"]],
        on=["player_id", "season", "week"], how="inner")
    cand = cand.drop_duplicates(key)
    played = len(cand)
    rates, base = B.blank_rates(snaps, set(cand["player_id"]))
    cand = cand.merge(rates, on=["player_id", "season", "week"], how="left")
    cand = cand.merge(stats[["player_id", "week", "receptions", "passing_tds"]]
                      .drop_duplicates(["player_id", "week"]),
                      on=["player_id", "week"], how="left")
    for col in ("receptions", "passing_tds"):
        cand[col] = cand[col].fillna(0.0)
    cand["actual"] = np.where(cand["market_key"] == "player_receptions",
                              cand["receptions"], cand["passing_tds"])
    cand["went_over"] = cand["actual"] > cand["line"]
    return cand.dropna(subset=["p_blank"]), len(best), played, base


def show(s):
    n = len(s)
    if n < 15:
        return f"{n:>5}  too few"
    o = s["went_over"].astype(float)
    over_u = float(np.where(s["went_over"], payout(s["over"]), -1.0).mean())
    under_u = float(np.where(~s["went_over"], payout(s["under"]), -1.0).mean())
    half = 1.96 * np.sqrt(o.mean() * (1 - o.mean()) / n)
    return (f"{n:>5}   over won {o.mean():>5.1%} (+/-{half:.0%}), needed "
            f"{implied(s['over']).mean():>5.1%}, returned {over_u:>+6.1%}"
            f"   |   under won {1 - o.mean():>5.1%}, needed "
            f"{implied(s['under']).mean():>5.1%}, returned {under_u:>+6.1%}")


def main():
    eng = create_engine(bp.DATABASE_URL, future=True)
    props, ids, stats, snaps = load(eng)
    d, posted, played, base = build(props, ids, stats, snaps)
    weeks = sorted(d["week"].unique())
    cut = weeks[len(weeks) // 2]
    print(f"{posted} props posted on a line of 0.5 or 1.5 with a price on both "
          f"sides; {played} where the player took a snap; {len(d)} with a "
          f"snap history to measure. Weeks {weeks[0]} to {weeks[-1]}, cut "
          f"before week {cut}.")
    print("Blank rate, priced players, earlier seasons: "
          + ", ".join(f"{p} {v:.1%}" for p, v in base.items()))
    print("'returned' is the result of one unit on every prop in the row at "
          "the best price on that side.\n")
    for mk, (name, _) in MARKETS.items():
        for line in (0.5, 1.5):
            s = d[(d["market_key"] == mk) & (d["line"] == line)]
            if len(s) < 15:
                continue
            print(f"{name.upper()}, LINE {line}:  {show(s)}")
            print(f"    early  {show(s[s['week'] < cut])}")
            print(f"    late   {show(s[s['week'] >= cut])}")
            if mk == "player_receptions":
                print("    by his chance of a blank game")
                for lo, hi, label in BANDS:
                    b = s[(s["p_blank"] >= lo) & (s["p_blank"] < hi)]
                    print(f"      {label:<11}{show(b)}")
                print("    by position")
                for pos, b in s.groupby("position"):
                    print(f"      {pos:<11}{show(b)}")
                print("    by how heavily the over is priced")
                need = pd.Series(implied(s["over"]), index=s.index)
                for lo, hi, label in ((0, 0.6, "to -150"), (0.6, 0.667, "-150 to -200"),
                                      (0.667, 0.75, "-200 to -300"),
                                      (0.75, 1.01, "-300 and shorter")):
                    b = s[(need >= lo) & (need < hi)]
                    print(f"      {label:<16}{show(b)}")
            print()
    print("A side is worth building on only if it is ahead of its price on a "
          "few hundred props and in both halves. Under 15 props a row is left "
          "blank.")


if __name__ == "__main__":
    main()
