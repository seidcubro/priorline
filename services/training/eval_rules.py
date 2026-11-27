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
    sure_pass_tds(engine)
    shadow_carries(engine)
    short_line_unders(engine)


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


TD_SQL = """
SELECT COUNT(*) AS picks,
       AVG(CASE WHEN r.hit THEN 1.0 ELSE 0 END) AS right_rate,
       AVG(CASE WHEN r.price_american > 0 THEN 100.0 / (r.price_american + 100.0)
                WHEN r.price_american < 0 THEN -r.price_american / (-r.price_american + 100.0)
           END) AS needed
FROM prop_edge_results r
WHERE r.hit IS NOT NULL AND r.source = 'live' AND r.game_date >= :since
  AND r.market_code = 'pass_td' AND r.win_prob >= 0.60
"""


SHADOW_SQL = """
SELECT s.side, s.board_side, s.gap, s.line, r.actual
FROM shadow_numbers s
JOIN LATERAL (
    SELECT actual FROM prop_edge_results r
    WHERE r.player_id = s.player_id AND r.market_code = s.market_code
      AND r.game_date BETWEEN s.game_date - 1 AND s.game_date + 1
      AND r.actual IS NOT NULL
    ORDER BY r.game_date LIMIT 1) r ON TRUE
WHERE s.model = 'football_v1' AND s.market_code = 'rush_att'
"""


def shadow_carries(engine):
    """The football model's carries number, never published, graded against
    the line beside the board's own side on the same props."""
    try:
        with engine.connect() as conn:
            s = pd.read_sql(text(SHADOW_SQL), conn)
    except Exception:
        print("  carries in shadow (football model): nothing written yet")
        return
    s = s[s["actual"].astype(float) != s["line"].astype(float)]
    if len(s) < MIN_PICKS:
        print(f"  carries in shadow (football model): {len(s)} graded, under {MIN_PICKS}, "
              "too few to read (backtest: 52.2% on 946, 55.4% on the 40% furthest from "
              "the line)")
        return
    went_over = s["actual"].astype(float) > s["line"].astype(float)
    mine = (s["side"] == "over") == went_over
    board = (s["board_side"] == "over") == went_over
    far = s["gap"].abs() >= s["gap"].abs().quantile(0.6)
    print(f"  carries in shadow (football model): {len(s)} graded, its side right "
          f"{mine.mean():.1%}, the board's side on the same props {board.mean():.1%}; "
          f"its 40% furthest from the line {mine[far].mean():.1%} on {int(far.sum())} "
          "(backtest: 52.2%, and 55.4% furthest from the line)")


LINE_SQL = SHADOW_SQL.replace("football_v1", "line_out_v1")


def short_line_unders(engine):
    """Carries unders on teams without a usual starting lineman, written by
    line_health.py whether or not the rule is acting on the board."""
    try:
        with engine.connect() as conn:
            s = pd.read_sql(text(LINE_SQL), conn)
    except Exception:
        print("  carries unders behind a short line: nothing written yet")
        return
    s = s[s["actual"].astype(float) != s["line"].astype(float)]
    if len(s) < MIN_PICKS:
        print(f"  carries unders behind a short line: {len(s)} graded, under {MIN_PICKS}, "
              "too few to read (backtest: 59.1% on 303, needed 52.6%)")
        return
    right = s["actual"].astype(float) < s["line"].astype(float)
    board = (s["board_side"] == "under") == right
    print(f"  carries unders behind a short line: {len(s)} graded, right {right.mean():.1%}, "
          f"the board's own side on the same props {board.mean():.1%} "
          "(backtest: 59.1% on 303, needed 52.6%)")


def sure_pass_tds(engine):
    """Passing touchdown picks at a 60% chance or better, lifted to elite on
    27 November 2026 on 65% right across 231 picks in two seasons."""
    since = os.getenv("PASS_TD_SINCE", "2026-11-27")
    try:
        with engine.connect() as conn:
            r = pd.read_sql(text(TD_SQL), conn, params={"since": since}).iloc[0]
    except Exception as exc:
        print(f"  passing touchdown picks at 60% or better: could not be read ({exc})")
        return
    n = int(r["picks"] or 0)
    if n < MIN_PICKS:
        print(f"  passing touchdown picks at 60% or better since {since}: {n} graded, "
              f"under {MIN_PICKS}, too few to read (backtest: 65% on 231)")
    else:
        print(f"  passing touchdown picks at 60% or better since {since}: {n} graded, "
              f"right {r['right_rate']:.1%}, needed {r['needed']:.1%} (backtest: 65% on 231)")


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
