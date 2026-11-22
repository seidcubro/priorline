"""Should a half-point line be able to earn the elite stamp on its gap?

research_elite_anatomy.py took the elite tier apart, weeks 1 to 11, tiered
walk forward by today's rule. One cut carries the whole deficit:

    elite on a line of 0.5      91 picks   52.7% right   needed 64.8%   -12.0 [-19, -3]
    every other elite pick     230 picks   about 60%     needed 56.6%   about +3.5

and it is the same story in both halves of the season (-7.0 early, -14.3
late). The other cuts that look bad are mostly these same picks seen from
another side: priced -200 or shorter (-13.5 on 57), and passing touchdowns
(-13.8 on 41, nearly all "over 0.5" at about -250).

Why they are elite: the tier ranks on the gap as a share of the line, and a
half is a very small number to divide by. A projection of 1.4 against 0.5 is
a gap of 180 percent on a bet that is really a yes or no question, will he
record one, asked about the players most likely to record none. The model has
never been shown their blank games (diagnose_zero_games.py), so it answers
yes too often, and the book charges 65 cents on the dollar for it.

research_gap_floor.py tried dividing by a bigger number and that was too
blunt: it marked down the 1.5 lines along with the 0.5s, and the 1.5 and 2.5
lines are the good part of the tier (+3.6 and +11.0).

This tests the direct version. Three ways of tiering the same season, the
same picks at the same prices, walked forward a week at a time:

    today        as the board does it now
    by price     a half-point pick cannot be elite on its gap. Elite is
                 filled from the other lines. The half-point picks then
                 compete for strong and medium on price, like everything
                 else under elite, which is the only measure that means
                 anything on a yes or no bet.
    held back    a half-point pick is published small and takes no part in
                 either ranking.

Nothing leaves the board in any of them. The question is which stamp a
half-point pick has earned. For each: every tier's picks, how often they
were right, what their prices needed, the difference with its interval, and
both halves of the season.

"by price" is worth shipping if elite's difference rises clear of today's in
both halves and strong does not pay for it. If strong falls by what elite
gains, the picks have only been moved, and "held back" is the comparison
that says what they cost wherever they sit.

Read-only.

    docker compose run --rm training python research_half_point.py
"""

import numpy as np
import pandas as pd
import stats_ci as S
from sqlalchemy import create_engine

import build_prop_edges as bp
import gap_tier
import replay_current_rules as R

TIERS = ("elite", "strong", "medium", "small")
MODES = ("today", "by price", "held back")


def walk(d, mode):
    live = d[(d["source"] == "live") & (d["week"] > 0)].copy()
    back = d[d["source"] == "backtest"]
    live["tier_wf"] = ""
    for w in sorted(live["week"].unique()):
        cfg = R.decide(live[live["week"] < w], back)
        idx = live.index[live["week"] == w]
        wk = live.loc[idx]
        half = (wk["line"] < 1).to_numpy()
        if mode == "today":
            labels = R.tier_week(wk, cfg).tolist()
        else:
            # Elite is cut over the other lines only, by the same rule and
            # the same share of each market.
            labels = pd.Series("small", index=idx, dtype=object)
            rest = wk[~half]
            if len(rest):
                labels.loc[rest.index] = R.tier_week(rest, cfg).values
            labels = labels.tolist()
        if gap_tier.PRICE_TIERS:
            implied = [None if pd.isna(v) else float(v) for v in wk["needed"]]
            eligible = ([True] * len(idx) if mode != "held back"
                        else (~half).tolist())
            labels = gap_tier.recut_by_price(labels, implied, eligible)
        live.loc[idx, "tier_wf"] = labels
    return live


def cell(s):
    s = s[s["needed"].notna()]
    if len(s) < 30:
        return f"{len(s):>5}  too few"
    diff = (s["hit"] - s["needed"]).to_numpy(dtype=float)
    r, lo, hi = S.clustered_bootstrap(
        lambda i: float(np.mean(diff[i])), s["cluster"].to_numpy(), n_boot=2000)
    return (f"{len(s):>5}{s['hit'].mean():>7.1%}{s['needed'].mean():>7.1%}"
            f"{r:>+7.1%} [{lo:+.1%},{hi:+.1%}]")


def main():
    eng = create_engine(bp.DATABASE_URL, future=True)
    d = R.load(eng)
    weeks = sorted(d.loc[(d["source"] == "live") & (d["week"] > 0), "week"].unique())
    cut = weeks[len(weeks) // 2]
    print(f"weeks {weeks[0]} to {weeks[-1]}, cut before week {cut} for the "
          f"halves. Each cell: picks, hit, needed, hit minus needed, interval.\n")
    for mode in MODES:
        live = walk(d, mode)
        print(mode.upper())
        print(f"  {'tier':<8}{'all season':<44}{'early':<44}late")
        for t in TIERS:
            s = live[live["tier_wf"] == t]
            print(f"  {t:<8}{cell(s):<44}{cell(s[s['week'] < cut]):<44}"
                  f"{cell(s[s['week'] >= cut])}")
        top = live[live["tier_wf"].isin(["elite", "strong"])]
        low = top[top["line"] < 1]
        print(f"  elite and strong together: {cell(top)}")
        print(f"  half-point picks stamped elite or strong: {len(low)}"
              + (f" at {low['hit'].mean():.1%}, needing "
                 f"{low['needed'].mean():.1%}" if len(low) else ""))
        el = live[live["tier_wf"] == "elite"]
        by = el.groupby("market_code")["hit"].agg(["size", "mean"])
        print("  elite by market: " + ", ".join(
            f"{m} {int(r['size'])} at {r['mean']:.0%}" for m, r in by.iterrows()))
        wk = el.groupby("week").apply(
            lambda g: (g["hit"] - g["needed"]).mean() if len(g) >= 10 else np.nan)
        print("  elite, weeks ahead of its price: "
              f"{int((wk > 0).sum())} of {int(wk.notna().sum())}\n")
    print("'today' is what the board does. A change is real only if it shows "
          "in both halves and in the weeks count, not in one total.")


if __name__ == "__main__":
    main()
