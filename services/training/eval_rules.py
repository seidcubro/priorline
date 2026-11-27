"""Grade the rules the board runs on, one by one, on what they have published.

Every rule that changes which side a pick is on, or whether it is ranked, was
chosen on a backtest. A backtest is the past priced at the best number any
book showed. This is the present: for each rule, the picks it actually touched
since the day it went live, graded, beside the picks it left alone.

    against_number   recs, rec_yds, rush_yds, rush_att since 22 November 2026.
                     The side comes from the corrected chance and sits on the
                     other side of the line from our number. Backtest: these
                     were right 53.6% of the time against 46.4% on the
                     median's side, on 1,539 props.
    off_form         pass_yds since 22 November 2026. The line is on the other
                     side of the quarterback's recent games from our pick, so
                     the pick is published as small and not ranked. Backtest:
                     these were a coin, and the ones the line agreed with were
                     right 56.1% on 494.
    under_even       a pick the corrected chance puts at even or worse.

A rule earns its place if its marked picks do what the backtest said: an
against_number pick should be right more often than not, and an off_form pick
should be right less often than an unmarked passing yards pick. With a few
dozen picks a week neither can be read for a month, so each row carries its
count and anything under MIN_PICKS says so and gives no rate.

Read-only. Prints; writes nothing.

Env: RULES_SINCE (default 2026-11-22), MIN_PICKS (default 40).
"""

import os

import pandas as pd
from sqlalchemy import create_engine, text

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://{u}:{p}@{h}:{port}/{db}".format(
        u=os.getenv("POSTGRES_USER", "app"),
        p=os.getenv("POSTGRES_PASSWORD", "app"),
        h=os.getenv("POSTGRES_HOST", "postgres"),
        port=os.getenv("POSTGRES_PORT", "5432"),
        db=os.getenv("POSTGRES_DB", "app"),
    ),
)
SINCE = os.getenv("RULES_SINCE", "2026-11-22")
MIN_PICKS = int(os.getenv("MIN_PICKS", "40"))

SQL = """
SELECT r.market_code,
       COALESCE(h.notes, 'unmarked') AS mark,
       COUNT(*) AS picks,
       AVG(CASE WHEN r.hit THEN 1.0 ELSE 0 END) AS right_rate,
       AVG(CASE WHEN r.price_american > 0 THEN 100.0 / (r.price_american + 100.0)
                WHEN r.price_american < 0 THEN -r.price_american / (-r.price_american + 100.0)
           END) AS needed
FROM prop_edge_results r
JOIN prop_edges_history h ON h.id = r.edge_id
WHERE r.hit IS NOT NULL
  AND r.source = 'live'
  AND r.game_date >= :since
GROUP BY 1, 2
ORDER BY 1, 2
"""


def main():
    engine = create_engine(DATABASE_URL)
    with engine.connect() as conn:
        d = pd.read_sql(text(SQL), conn, params={"since": SINCE})
    print(f"Rules graded on live picks since {SINCE}")
    if d.empty:
        print("  nothing graded yet")
        qb_rush_unders(engine)
        return
    print(f"  {'market':<18}{'mark':<16}{'picks':>6}{'right':>9}{'needed':>9}")
    for r in d.itertuples(index=False):
        if r.picks < MIN_PICKS:
            tail = f"{'':>9}{'':>9}  under {MIN_PICKS}, too few to read"
        else:
            need = f"{r.needed:.1%}" if pd.notna(r.needed) else "n/a"
            tail = f"{r.right_rate:>9.1%}{need:>9}"
        print(f"  {r.market_code:<18}{r.mark:<16}{int(r.picks):>6}{tail}")
    allm = d.groupby("mark").apply(
        lambda g: pd.Series({
            "picks": g["picks"].sum(),
            "right": (g["right_rate"] * g["picks"]).sum() / g["picks"].sum()}),
        include_groups=False)
    print("  every market together")
    for mark, r in allm.iterrows():
        rate = f"{r['right']:.1%}" if r["picks"] >= MIN_PICKS else "too few"
        print(f"  {'':<18}{mark:<16}{int(r['picks']):>6}{rate:>9}")
    qb_rush_unders(engine)


QB_SQL = """
SELECT COUNT(*) AS picks,
       AVG(CASE WHEN r.hit THEN 1.0 ELSE 0 END) AS right_rate,
       AVG(CASE WHEN r.price_american > 0 THEN 100.0 / (r.price_american + 100.0)
                WHEN r.price_american < 0 THEN -r.price_american / (-r.price_american + 100.0)
           END) AS needed
FROM prop_edge_results r
JOIN prop_edges_history h ON h.id = r.edge_id
WHERE r.hit IS NOT NULL
  AND r.source = 'live'
  AND r.game_date >= :since
  AND h.recommended_side = 'under'
  AND ((h.market_code = 'rush_att' AND h.line > 5.5)
       OR (h.market_code = 'rush_yds' AND h.line > 25.5))
  AND h.player_name IN (SELECT name FROM players WHERE position = 'QB')
"""


def qb_rush_unders(engine):
    """The high-line quarterback rushing under, which the board lifts to elite.

    It went live on 27 November 2026 on a backtest of 66.5% on 161 picks. This
    is what it has done since, on its own line so it cannot hide inside a tier.
    """
    since = os.getenv("QB_RUSH_SINCE", "2026-11-27")
    try:
        with engine.connect() as conn:
            r = pd.read_sql(text(QB_SQL), conn, params={"since": since}).iloc[0]
    except Exception as exc:
        print(f"  high-line quarterback rushing unders: could not be read ({exc})")
        return
    n = int(r["picks"] or 0)
    if n < MIN_PICKS:
        print(f"  high-line quarterback rushing unders since {since}: {n} graded, "
              f"under {MIN_PICKS}, too few to read (backtest: 66.5% on 161)")
    else:
        print(f"  high-line quarterback rushing unders since {since}: {n} graded, "
              f"right {r['right_rate']:.1%}, needed {r['needed']:.1%} "
              f"(backtest: 66.5% on 161)")


if __name__ == "__main__":
    main()
