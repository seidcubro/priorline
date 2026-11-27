"""The receiving result, tried on every other market, three different ways.

research_stack.py found that one model across all markets picks no better than
the board, but that its most confident catches and receiving-yards picks come
in 62 to 68% of the time in both seasons. This asks whether rushing and passing
have a confident slice of their own once the model is built for them:

    own model     trained only on that group's props (rushing, passing,
                  receiving), so a quarterback's dropbacks are not being
                  weighed against a tight end's targets
    own features  the same, with only the features that belong to the group,
                  because a small market drowns in 133 columns
    agreement     the picks where three things that share nothing agree: the
                  side we published, the side the price favours, and the side
                  the model takes

Each is read on its most confident fifth and tenth, in 2026 a week at a time
and then the other way round (learned on 2026, scored on 2025). A slice counts
only if it is ahead of its price in both.

    python lab/training/research_stack_markets.py
"""

from __future__ import annotations

import sys
import warnings

import numpy as np
import pandas as pd

import research_stack as S

warnings.filterwarnings("ignore")
GROUPS = {
    "receiving": ("recs", "rec_yds"),
    "rushing": ("rush_att", "rush_yds"),
    "passing volume": ("pass_att", "pass_completions", "pass_yds"),
    "passing TDs": ("pass_td",),
}
KEEP = {
    "receiving": ("tgt", "rec", "air", "yac", "rz_tgt", "x_rec", "d_rec", "d_tgt", "d_x",
                  "d_att", "d_pass", "d_cmp", "pass_rate", "t_"),
    "rushing": ("car", "rush", "rz_car", "x_rush", "scr", "d_car", "d_rush", "d_xr",
                "pass_rate", "t_", "dropbacks"),
    "passing volume": ("att", "cmp", "pass", "dropbacks", "sacks", "ints", "rz_att", "x_pass",
                       "d_att", "d_cmp", "d_pass", "d_sacks", "d_x", "pass_rate", "t_", "scr"),
    "passing TDs": ("att", "cmp", "pass", "dropbacks", "sacks", "rz_att", "x_pass", "d_att",
                    "d_pass", "d_sacks", "pass_rate", "t_"),
}
ALWAYS = ("line", "projection", "proj_gap", "line_vs_8", "line_vs_4", "trend", "m_4", "m_8",
          "m_last", "m_sd8", "p_over_ours", "need_over", "need_under", "we_over", "fav",
          "total_line", "home", "indoor", "wind", "temp", "games", "rest")


def pruned(cols, group):
    keep = KEEP[group]
    return [c for c in cols if c in ALWAYS or c.startswith(("is_", "pos_"))
            or any(c.startswith(k) for k in keep)]


def both_ways(x, cols):
    walk = S.walk(x, cols)
    back = S.swap(x, cols)
    return walk, back


def slices(t, name):
    """Right, needed, and picks for the most confident tenth and fifth, and for
    the picks where published, favourite and model all take the same side."""
    p = t["p_stack"].to_numpy()
    conf = np.maximum(p, 1 - p)
    over = p > 0.5
    right = np.where(over, t["over"], 1 - t["over"])
    need = np.where(over, t["need_over"], t["need_under"])
    agree = (over == (t["we_over"].to_numpy() == 1)) & (
        over == (t["need_over"].to_numpy() > t["need_under"].to_numpy()))
    out = []
    for label, mask in (("top 10%", conf >= np.quantile(conf, 0.9)),
                        ("top 20%", conf >= np.quantile(conf, 0.8)),
                        ("all three agree", agree),
                        ("agree, top 20%", agree & (conf >= np.quantile(conf[agree], 0.8)
                                                    if agree.sum() > 20 else agree))):
        n = int(mask.sum())
        if n < 25:
            out.append(f"{label}: {n} few")
            continue
        r, q = right[mask].mean(), need[mask].mean()
        out.append(f"{label}: {n:>4} {r:.1%} ({r - q:+.1%})")
    return f"    {name:<26}" + "   ".join(out)


def main():
    x = S.bench()
    cols = S.features(x)
    for group, markets in GROUPS.items():
        g = x[x["market_code"].isin(markets)]
        n25, n26 = int((g["season"] == 2025).sum()), int((g["season"] == 2026).sum())
        print(f"\n{group.upper()}: {n25:,} props in 2025, {n26:,} in 2026")
        if min(n25, n26) < 200:
            print("    one of the seasons is too thin to learn from and check on")
        for label, c in (("own model, all features", cols),
                         ("own model, own features", pruned(cols, group))):
            try:
                walk, back = both_ways(g, c)
            except Exception as err:
                print(f"    {label}: could not be fitted ({err})")
                continue
            print(f"  {label} ({len(c)} features)")
            print(slices(walk, "2026, week by week"))
            print(slices(back, "2025, the other way"))
            for m in markets:
                if len(markets) > 1:
                    w, b = walk[walk["market_code"] == m], back[back["market_code"] == m]
                    if len(w) >= 120 and len(b) >= 120:
                        print(slices(w, f"  {m} 2026"))
                        print(slices(b, f"  {m} 2025"))
    print("\nEach cell: picks, how often right, and in brackets that minus what the "
          "prices needed.")


if __name__ == "__main__":
    sys.exit(main())
