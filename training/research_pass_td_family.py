"""Is another family really better than the live one on passing touchdowns, or is it noise?

research_passing.py: passing touchdowns are the one passing market where the
projection sorts props, in the test and in the record alike. Every family
rated a prop that went over above one that did not 0.57 to 0.60 of the time,
in both seasons. And the live family, pois, was not the best of them:

                 2025 right   2026 right   both
    pois (live)     54.3%        56.3%     55.0%
    enet            56.2%        59.7%     57.5%
    vote            56.2%        59.7%     57.5%

Two and a half points in both seasons, on 685 props. That is the size of
difference this project has been fooled by before, so it is not shipped on
that table. Two families agree on most props; the whole difference is in the
props where they take different sides, and that is a much smaller number.

So, for each challenger against the live family:

    the props where they disagree, and which was right, each season
    the chance a gap that size is luck (a sign test on those props)

and for each family on its own, by the line (0.5, 1.5, 2.5) and by side,
since a touchdown line of 1.5 at -150 and one at +120 are different bets.

A challenger replaces the live family only if it wins the disagreements in
both seasons and the sign test puts luck under one in twenty.

Read-only.

    sh lab/run.sh research_pass_td_family.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import binomtest

import backtest_gap_system as bt
import research_passing as P
import research_reproject_blanks as RB

CODE = "pass_td"
CHALLENGERS = ("enet_v2", "vote_v1", "ridge_v1", "hgb_v1")


def sign(wins, n):
    return binomtest(int(wins), int(n), 0.5).pvalue if n else float("nan")


def main():
    d = bt.features(CODE)
    o = P.lines(CODE)
    live = P.live_family(CODE) or "pois_v4"
    fams = [live] + [f for f in CHALLENGERS if not live.startswith(f.split("_")[0])]
    for f in fams:
        d[f] = P.project(d, f, RB.SEASONS)
    d["blend"] = d[[live, "enet_v2"]].mean(axis=1) if "enet_v2" in d else np.nan
    d = d.rename(columns={"name": "player_name"})
    d["d"] = pd.to_datetime(d["as_of_game_date"])
    j = d[d["season"].isin(RB.SEASONS)].merge(o, on=["player_name", "d"], how="inner")
    j = j[j["over"].notna() & j["under"].notna() & ((j["line"] * 2) % 2 == 1)].copy()
    j["actual"] = j["label_actual"].astype(float)
    j["went_over"] = j["actual"] > j["line"]
    j["cluster"] = j["d"].dt.strftime("%Y-%m-%d")
    j["slate"] = (j["d"] - pd.Timedelta(days=3)).dt.strftime("%G-%V")
    print(f"{CODE.upper()}: {len(j):,} priced props. Live family: {live}.")

    scored = {f: P.score(j, f, CODE) for f in fams + ["blend"]}
    base = scored[live]
    print("\nAgainst the live family, on the props where the two take different sides")
    for f in fams[1:] + ["blend"]:
        s = scored[f]
        k = base[["hit"]].join(s[["hit", "season"]], lsuffix="_live", how="inner")
        diff = k[k["hit_live"] != k["hit"]]
        line = f"  {f.split('_')[0]:<8}"
        for label, part in [("both", diff)] + [(str(x), diff[diff["season"] == x])
                                               for x in RB.SEASONS]:
            w = int(part["hit"].sum())
            line += (f"  {label}: differ on {len(part):>3}, it was right on {w:>3}"
                     f" ({(w / len(part) if len(part) else float('nan')):.0%})")
        line += f"   luck: {sign(diff['hit'].sum(), len(diff)):.2f}"
        print(line)

    for f in fams + ["blend"]:
        s = scored[f]
        print(f"\n  {f.split('_')[0]}{' (live)' if f == live else ''}: "
              f"every prop {RB.bet(s)}")
        for ln in sorted(s["line"].unique()):
            for side, name in ((True, "over"), (False, "under")):
                over = (s["hit"] == s["went_over"].astype(float))
                part = s[(s["line"] == ln) & (over == side)]
                if len(part) >= 20:
                    by = "  ".join(
                        f"{x}: {int((part['season'] == x).sum())} at "
                        f"{part.loc[part['season'] == x, 'hit'].mean():.0%}"
                        for x in RB.SEASONS)
                    print(f"    line {ln:<4} {name:<6}{RB.bet(part, 20)}   {by}")
    print("\n'luck' is the chance of a split that lopsided if the two families "
          "were equally good. Under 0.05 in the 'both' column, with the "
          "challenger ahead in each season, is the bar.")


if __name__ == "__main__":
    main()
