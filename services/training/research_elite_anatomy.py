"""Where inside the elite tier is the price being lost?

After week 10 the elite tier, tiered by the rule the board uses today and
walked forward a week at a time, is 57.9% right on 321 picks where the prices
needed 58.9%. It led its price through week 5 (64.8% against 59.2% on 88) and
has trailed it since (55.4% against 58.8% on 233).

research_gap_floor.py tested the obvious suspect, half-point lines crowding
the top of the ranking, and it is not the answer: measuring those lines
differently moved elite by under three points, inside the noise, and most of
that came from passing touchdowns dropping out of the ranking altogether, not
from the catches the test was about.

So before another rule is tried, this takes the tier apart. The same 321
picks, cut one way at a time: by side, by market, by market and side, by how
heavily the pick was priced, by the size of the line, and by position. Each
cut shows how often it was right, what its prices needed, the difference with
an interval, and the same for the two halves of the season.

It is a map, not a rule. A cut is worth acting on only if it is large, its
interval sits clear of zero, and it says the same thing in both halves. On
321 picks most cuts will fail that, and knowing which ones do is the point:
it says where to look with the full record next, where there are ten times
as many picks.

Strong is printed the same way underneath, because it hits as often as elite
and sits further under its price.

Read-only.

    docker compose run --rm training python research_elite_anatomy.py
"""

import numpy as np
import pandas as pd
import stats_ci as S
from sqlalchemy import create_engine, text

import build_prop_edges as bp
import replay_current_rules as R
import research_gap_floor as G


def cell(s):
    s = s[s["needed"].notna()]
    n = len(s)
    if n < 12:
        return f"{n:>4}  -"
    diff = (s["hit"] - s["needed"]).to_numpy(dtype=float)
    if n >= 30:
        r, lo, hi = S.clustered_bootstrap(
            lambda i: float(np.mean(diff[i])), s["cluster"].to_numpy(), n_boot=1500)
        # Too few game days to resample gives no interval, and then none
        # is printed.
        tail = (f"{r:>+7.1%} [{lo:+.0%},{hi:+.0%}]"
                if np.isfinite(lo) and np.isfinite(hi) else f"{diff.mean():>+7.1%}")
    else:
        tail = f"{diff.mean():>+7.1%}"
    return f"{n:>4}{s['hit'].mean():>7.1%}{s['needed'].mean():>7.1%}{tail}"


def cut(s, half, key, title, order=None):
    print(f"\n  {title}")
    print(f"    {'':<26}{'all season':<38}{'early':<38}{'late'}")
    groups = s.groupby(key(s))
    names = order or sorted(groups.groups, key=str)
    for name in names:
        if name not in groups.groups:
            continue
        g = groups.get_group(name)
        if len(g) < 12:
            continue
        print(f"    {str(name):<26}{cell(g):<38}{cell(g[g['week'] < half]):<38}"
              f"{cell(g[g['week'] >= half])}")


def price_band(s):
    n = s["needed"]
    return pd.cut(n, [0, 0.50, 0.545, 0.60, 0.667, 1.0],
                  labels=["plus money", "near even", "-120 to -150",
                          "-150 to -200", "-200 and shorter"]).astype(str)


def line_band(s):
    def one(r):
        m, ln = r["market_code"], r["line"]
        if m in ("recs", "pass_td", "rush_att", "pass_att", "pass_completions"):
            return ("0.5" if ln < 1 else "1.5" if ln < 2 else "2.5" if ln < 3
                    else "3.5 to 4.5" if ln < 5 else "5.5 and up")
        return ("under 20" if ln < 20 else "20 to 50" if ln < 50
                else "50 to 80" if ln < 80 else "80 and up")
    return s.apply(one, axis=1)


def main():
    eng = create_engine(bp.DATABASE_URL, future=True)
    d = R.load(eng)
    # The side and the position are not in the replay's own load.
    extra = pd.read_sql(text("""
        SELECT r.game_date, r.player_name, r.market_code, r.line,
               r.recommended_side AS side, MAX(p.position) AS position
        FROM prop_edge_results r
        LEFT JOIN players p ON p.external_id = r.player_id
        WHERE r.hit IS NOT NULL
        GROUP BY 1, 2, 3, 4, 5
    """), eng).drop_duplicates(["game_date", "player_name", "market_code", "line"])
    d = d.merge(extra, on=["game_date", "player_name", "market_code", "line"],
                how="left")
    live = G.walk(d, 0.5)
    weeks = sorted(live["week"].unique())
    half = weeks[len(weeks) // 2]
    print(f"{len(live):,} live graded props, weeks {weeks[0]} to {weeks[-1]}, "
          f"tiered walk forward by today's rule. Early is before week {half}.")
    print("Each cell: picks, hit, needed, hit minus needed with its interval "
          "(no interval under 30 picks, nothing under 12).")
    for tier in ("elite", "strong"):
        s = live[live["tier_wf"] == tier].copy()
        print(f"\n{tier.upper()}: {cell(s)}   early {cell(s[s['week'] < half])}"
              f"   late {cell(s[s['week'] >= half])}")
        cut(s, half, lambda x: x["side"], "by side")
        cut(s, half, lambda x: x["market_code"], "by market")
        cut(s, half, lambda x: x["market_code"] + " " + x["side"].fillna("?"),
            "by market and side")
        cut(s, half, price_band, "by price",
            ["plus money", "near even", "-120 to -150", "-150 to -200",
             "-200 and shorter"])
        cut(s, half, line_band, "by size of line")
        cut(s, half, lambda x: x["position"].fillna("?"), "by position")
        cut(s, half, lambda x: x["position"].fillna("?") + " "
            + x["side"].fillna("?"), "by position and side")
        cut(s, half, lambda x: x["week"], "by week")


if __name__ == "__main__":
    main()
