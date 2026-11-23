"""Can the yardage side be taken from the volume market, where the model works?

Projected again over 2025 and 2026, the model is right on 55% of catch props
and 53% to 54% of rush attempt props, and on 50% of receiving yards and 51% of
rush yards. It ranks a yardage prop no better than a coin (0.51 to 0.53) and
the yardage line is a better forecast than ours by a wide margin: fitted
against results, the weight on our number is 0.10 for receiving yards.

Yards are volume times yards per touch. The volume half is the half the model
gets right, and the per-touch half is mostly noise from one week to the next.
So the question is whether the yardage model is throwing away what the volume
model knows. For every yardage prop that has the same player's volume prop on
the same day (catches for receiving yards, rush attempts for rush yards):

    own      the side the yardage model's median takes. Today.
    volume   the side the volume model takes on the volume prop, carried
             over. The same as projecting yards as our catches times the
             book's own yards per catch (yardage line over catch line),
             because that product is above the yardage line exactly when our
             catches are above the catch line.
    both     the props where the two agree, and how they do.

and, where they disagree, which of the two was right.

If 'volume' is right more often than 'own' in both seasons, the fix for the
yardage markets is to build their projection on the volume model, and the
next step is doing that in the feature build and the trainer.

Prices are the best any book showed. Read-only. Reads the projections
research_reproject_blanks.py kept.

    MARKETS=recs,rec_yds,rush_yds,rush_att sh lab/run.sh research_volume_to_yards.py
"""

from __future__ import annotations

import numpy as np

import research_reproject_blanks as RB
from research_side_by_price import implied

PAIRS = (("rec_yds", "recs"), ("rush_yds", "rush_att"))


def sided(j, over):
    j = j.copy()
    over = np.asarray(over, dtype=bool)
    j["needed"] = np.where(over, implied(j["over"]), implied(j["under"]))
    j["hit"] = np.where(over, j["went_over"], ~j["went_over"]).astype(float)
    return j


def block(j, title):
    if len(j) < 50:
        return
    own = (j["q50"] > j["line"]).to_numpy()
    vol = (j["v_q50"] > j["v_line"]).to_numpy()
    same = own == vol
    print(f"\n  {title}: {len(j):,} yardage props with the volume prop beside them")
    print(f"    own                     {RB.bet(sided(j, own))}")
    print(f"    volume                  {RB.bet(sided(j, vol))}")
    print(f"    both agree              {RB.bet(sided(j[same], own[same]))}")
    for name, pick in (("both say under", ~own & ~vol), ("both say over", own & vol)):
        print(f"      {name:<22}{RB.bet(sided(j[pick], own[pick]))}")
    d = j[~same]
    if len(d) >= 25:
        o = sided(d, own[~same])["hit"].mean()
        print(f"    they disagree on {len(d):,}: own was right {o:.1%}, "
              f"volume {1 - o:.1%}")
        for name, pick in (("own over, volume under", own & ~vol),
                           ("own under, volume over", ~own & vol)):
            s = sided(j[pick], own[pick])
            if len(s) >= 25:
                print(f"      {name:<24}{len(s):>5}  own right "
                      f"{s['hit'].mean():.1%}")
    top = j[j["v_top"].astype(bool)]
    print(f"    volume, where the volume pick was in its top fifth"
          f"   {RB.bet(sided(top, (top['v_q50'] > top['v_line']).to_numpy()))}")
    # The volume model's own record on these same player-days, for scale.
    v = j.copy()
    v_over = (v["v_q50"] > v["v_line"]).to_numpy()
    v["hit"] = np.where(v_over, v["v_went_over"], ~v["v_went_over"]).astype(float)
    print(f"    for scale, the volume model on its own prop, same player-days: "
          f"right {v['hit'].mean():.1%}")


def main():
    for yards, volume in PAIRS:
        a, b = RB.cached_arms(yards), RB.cached_arms(volume)
        if a is None or b is None:
            print(f"{yards}: needs kept projections for {yards} and {volume}. "
                  f"Run research_reproject_blanks.py with both in MARKETS.")
            continue
        v = b["with"][["player_name", "d", "line", "q50", "went_over", "top"]].rename(
            columns={"line": "v_line", "q50": "v_q50",
                     "went_over": "v_went_over", "top": "v_top"})
        v = v.drop_duplicates(["player_name", "d"])
        j = a["with"].merge(v, on=["player_name", "d"], how="inner")
        print(f"\n{yards.upper()} from {volume.upper()}: {len(a['with']):,} "
              f"yardage props, {len(j):,} with the volume prop beside them")
        block(j, "both seasons")
        for s in RB.SEASONS:
            block(j[j["season"] == s], str(s))
    print("\nEach cell: picks, how often right, what the prices needed, the "
          "difference, its interval.")


if __name__ == "__main__":
    main()
