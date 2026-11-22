"""What should rank a count prop: the gap as a share of the line, or the chance?

The elite tier loses its price on one kind of pick, a line of 0.5 (91 picks,
52.7% right, needed 64.8%). Two ways of treating those picks differently were
tested and refused: stamped by price they carry the loss into strong, and held
back they are only hidden.

The blank games are in the game log now, so the projections under these picks
are lower and more honest. The board on 22 November shows what is left of the
problem. Ben Sinnott projects 0.7 catches against a line of 0.5 at -160 and
Jordan Mason 0.9 at -225, and both are stamped elite. The cause is the
ranking, not the projection: the gap is divided by the line, a half is a
small number to divide by, and 0.4 of a catch comes out as an 80 percent gap,
ahead of Justin Jefferson projected 2.2 catches under his line.

The gap as a share of the line is the wrong ruler for a count. Whether a man
catches at least one pass is a question about a chance, and a projection of
0.9 answers it: about 59 times in 100, on a bet priced to need 69. This tests
rulers that say so, on catches, passing touchdowns, carries, attempts and
completions. Yardage markets keep today's ruler in every row.

    share        the gap as a share of the line. What the board does today.
    chance       the chance the pick wins if the count is Poisson around the
                 projection. No price in it.
    over price   that chance minus what the price needs. A pick the price has
                 already paid for cannot rank high.
    model        the board's own stored win probability minus what the price
                 needs.

Nothing is fitted and nothing leaves the board. A half-point pick with a real
chance over its price still ranks at the top; one without it ranks where its
chance puts it.

Two readings, because the season alone is about 320 elite picks and that is
not enough to change a rule on.

    1. This season, walk forward, the whole board the way the replay scores
       it: every tier, both halves, and the elite tier by size of line.
    2. The top fifth of each count market each week under each ruler, on this
       season and on every earlier season in the record separately. Same
       picks, same prices, no direction fitting. This is the larger sample
       and the one that has to agree.

A ruler is worth shipping if its top fifth is ahead of its price by more than
today's in most seasons, not one, and elite this season rises in both halves
without strong paying for it.

What this cannot see: the record holds the projections as they were published,
before the blank games were written. The half-point overs in it are projected
higher than the board would project them now, so a ruler that reads the
projection is being tested on the inflated number. If it works here it works
on worse inputs than it will get.

Read-only.

    docker compose run --rm training python research_count_rank.py
"""

import numpy as np
import pandas as pd
import stats_ci as S
from scipy.stats import poisson
from sqlalchemy import create_engine, text

import build_prop_edges as bp
import gap_tier
import replay_current_rules as R

COUNT = ("recs", "pass_td", "rush_att", "pass_att", "pass_completions")
RULERS = ("share", "chance", "over price", "model")
TIERS = ("elite", "strong", "medium", "small")
KEY = ["game_date", "player_name", "market_code", "line"]


def load(eng):
    d = R.load(eng)
    extra = pd.read_sql(text("""
        SELECT game_date, player_name, market_code, line,
               projection AS mean, recommended_side AS side,
               win_prob AS model_p
        FROM prop_edge_results
        WHERE hit IS NOT NULL
        ORDER BY game_date, player_name, market_code, line, edge_id
    """), eng).drop_duplicates(KEY)
    d = d.merge(extra, on=KEY, how="left")
    d["side"] = d["side"].astype(str).str.lower()
    centre = pd.to_numeric(d["mean"], errors="coerce").fillna(d["p50"])
    centre = centre.clip(lower=0.01)
    # Most a player can record and still be under the line.
    top = np.ceil(d["line"].to_numpy(dtype=float)) - 1
    under = poisson.cdf(top, centre.to_numpy(dtype=float))
    d["p_win"] = np.where(d["side"] == "over", 1.0 - under, under)
    d["share"] = d["gap"]
    d["is_count"] = d["market_code"].isin(COUNT)
    gd = pd.to_datetime(d["game_date"])
    d["yr"] = np.where(gd.dt.month >= 8, gd.dt.year, gd.dt.year - 1)
    # Thursday to Monday falls in one week once the date is moved back three days.
    d["slate"] = (gd - pd.Timedelta(days=3)).dt.strftime("%G-%V")
    return d


def score(d, ruler):
    """The ranking number under one ruler. Yardage markets never change."""
    if ruler == "share":
        s = d["share"]
    elif ruler == "chance":
        s = d["p_win"]
    elif ruler == "over price":
        s = d["p_win"] - d["needed"]
    else:
        s = pd.to_numeric(d["model_p"], errors="coerce") - d["needed"]
    return pd.Series(np.where(d["is_count"], s, d["share"]), index=d.index)


def walk(d, ruler):
    d = d.copy()
    d["gap"] = score(d, ruler)
    live = d[(d["source"] == "live") & (d["week"] > 0)].copy()
    back = d[d["source"] == "backtest"]
    live["tier_wf"] = ""
    for w in sorted(live["week"].unique()):
        cfg = R.decide(live[live["week"] < w], back)
        idx = live.index[live["week"] == w]
        labels = R.tier_week(live.loc[idx], cfg).tolist()
        if gap_tier.PRICE_TIERS:
            implied = [None if pd.isna(v) else float(v)
                       for v in live.loc[idx, "needed"]]
            labels = gap_tier.recut_by_price(labels, implied, [True] * len(idx))
        live.loc[idx, "tier_wf"] = labels
    return live


def cell(s, least=30):
    s = s[s["needed"].notna()]
    if len(s) < least:
        return f"{len(s):>5}  too few"
    diff = (s["hit"] - s["needed"]).to_numpy(dtype=float)
    r, lo, hi = S.clustered_bootstrap(
        lambda i: float(np.mean(diff[i])), s["cluster"].to_numpy(), n_boot=1500)
    tail = (f" [{lo:+.1%},{hi:+.1%}]"
            if np.isfinite(lo) and np.isfinite(hi) else "")
    return (f"{len(s):>5}{s['hit'].mean():>7.1%}{s['needed'].mean():>7.1%}"
            f"{diff.mean():>+7.1%}{tail}")


def line_band(ln):
    return ("0.5" if ln < 1 else "1.5" if ln < 2 else "2.5" if ln < 3
            else "3.5 to 5.5" if ln < 6 else "6.5 and up")


def season_board(d, cut):
    print("1. THIS SEASON, WALK FORWARD, THE WHOLE BOARD")
    print("   each cell: picks, hit, needed, hit minus needed, interval\n")
    for ruler in RULERS:
        live = walk(d, ruler)
        print(f"  {ruler.upper()}")
        print(f"    {'tier':<8}{'all season':<42}{'early':<42}late")
        for t in TIERS:
            s = live[live["tier_wf"] == t]
            print(f"    {t:<8}{cell(s):<42}{cell(s[s['week'] < cut]):<42}"
                  f"{cell(s[s['week'] >= cut])}")
        el = live[live["tier_wf"] == "elite"]
        print("    elite by size of line, count markets:")
        c = el[el["is_count"]]
        for name, g in c.groupby(c["line"].map(line_band)):
            print(f"      {name:<12}{cell(g, 12)}")
        for side, g in el.groupby("side"):
            print(f"      {side:<12}{cell(g, 12)}")
        by = el.groupby("market_code")["hit"].agg(["size", "mean"])
        print("    elite by market: " + ", ".join(
            f"{m} {int(r['size'])} at {r['mean']:.0%}" for m, r in by.iterrows()))
        wk = el.groupby("week").apply(
            lambda g: (g["hit"] - g["needed"]).mean() if len(g) >= 10 else np.nan)
        print("    elite, weeks ahead of its price: "
              f"{int((wk > 0).sum())} of {int(wk.notna().sum())}\n")


def top_fifth(d):
    print("2. THE TOP FIFTH OF EACH COUNT MARKET EACH WEEK, EVERY SEASON")
    print("   no direction fitting; a market needs "
          f"{gap_tier.MIN_RANKABLE} picks in a week to be ranked\n")
    c = d[d["is_count"] & d["needed"].notna()].copy()
    picked = {}
    for ruler in RULERS:
        c["s"] = score(c, ruler)
        sel = pd.Series(False, index=c.index)
        for _, g in c.groupby(["source", "yr", "slate", "market_code"]):
            g = g[g["s"].notna()]
            if len(g) < gap_tier.MIN_RANKABLE:
                continue
            n = int(round(len(g) * gap_tier.ELITE_SHARE))
            sel.loc[g.sort_values("s", ascending=False, kind="stable").index[:n]] = True
        picked[ruler] = sel
    groups = sorted(c.groupby(["yr", "source"]).groups)
    print(f"    {'season':<16}" + "".join(f"{r:<40}" for r in RULERS))
    for yr, src in groups:
        g = c[(c["yr"] == yr) & (c["source"] == src)]
        print(f"    {f'{yr} {src}':<16}" + "".join(
            f"{cell(g[picked[r].loc[g.index]]):<40}" for r in RULERS))
    print(f"    {'every season':<16}" + "".join(
        f"{cell(c[picked[r]]):<40}" for r in RULERS))
    print("\n    the same top fifth, by size of line, every season together")
    for name in ("0.5", "1.5", "2.5", "3.5 to 5.5", "6.5 and up"):
        g = c[c["line"].map(line_band) == name]
        print(f"    {name:<16}" + "".join(
            f"{cell(g[picked[r].loc[g.index]]):<40}" for r in RULERS))
    print("\n    the rest of the count picks (not in the top fifth)")
    print(f"    {'every season':<16}" + "".join(
        f"{cell(c[~picked[r]]):<40}" for r in RULERS))
    print()


def main():
    eng = create_engine(bp.DATABASE_URL, future=True)
    d = load(eng)
    weeks = sorted(d.loc[(d["source"] == "live") & (d["week"] > 0), "week"].unique())
    cut = weeks[len(weeks) // 2]
    n_count = int(d["is_count"].sum())
    print(f"{len(d):,} graded props, {n_count:,} of them in count markets. "
          f"This season weeks {weeks[0]} to {weeks[-1]}, cut before week {cut}.")
    print(f"stored win probability missing on "
          f"{int(d.loc[d['is_count'], 'model_p'].isna().sum()):,} count picks, "
          f"mean projection missing on "
          f"{int(d.loc[d['is_count'], 'mean'].isna().sum()):,} (the median "
          f"stands in).\n")
    top_fifth(d)
    season_board(d, cut)
    print("'share' is what the board does today. A ruler has to be ahead in "
          "reading 2 across seasons before reading 1 counts for anything.")


if __name__ == "__main__":
    main()
