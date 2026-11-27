"""The web: let every market on a game inform every other one.

Until now each prop was judged alone: this player, this line, his own history.
But the props on a game are not independent, and the book prices them at
different desks at different times. A quarterback's attempts line, his
receivers' catches lines and his backs' carries lines are all statements about
the same sixty minutes, and they do not have to agree.

So each prop gets, on top of everything research_stack.py gives it:

    his other lines        the same player's lines and our numbers in his other
                           markets, and the rates they imply: the book's yards
                           a carry (yards line over attempts line) against his
                           own, its yards a throw, its completion rate
    his team's lines       the catches and receiving-yards lines of every priced
                           teammate added up, set against the quarterback's
                           completions and passing-yards lines; the backs'
                           carries lines added up; this player's share of them
    his team's lean        whether our numbers are above or below the line
                           across his whole team, not only on him
    the other side         the opposing quarterback's lines, which say what the
                           book thinks the game will look like

Then a model for each group of markets is fitted with and without the web, and
read on its most confident tenth and fifth, in 2026 a week at a time and the
other way round. The web earns its place only where it lifts a slice above its
price in both.

    python lab/training/research_web.py
"""

from __future__ import annotations

import sys
import warnings

import numpy as np
import pandas as pd

import research_stack as S
import research_stack_markets as M

warnings.filterwarnings("ignore")
MK = list(S.STAT)


def web(x):
    x = x.copy()
    x["gkey"] = x["game_id"].astype(str)
    own_l = x.pivot_table(index=["gsis", "gkey"], columns="market_code", values="line",
                          aggfunc="first")
    own_p = x.pivot_table(index=["gsis", "gkey"], columns="market_code", values="projection",
                          aggfunc="first")
    own_l.columns = [f"l_{c}" for c in own_l.columns]
    own_p.columns = [f"pr_{c}" for c in own_p.columns]
    own = own_l.join(own_p).reset_index()
    for m in MK:
        if f"l_{m}" in own and f"pr_{m}" in own:
            own[f"g_{m}"] = (own[f"pr_{m}"] - own[f"l_{m}"]) / own[f"l_{m}"].clip(lower=0.5)
    x = x.merge(own, on=["gsis", "gkey"], how="left")

    def has(*cols):
        return all(c in x.columns for c in cols)

    if has("l_rush_yds", "l_rush_att"):
        x["book_ypc"] = x["l_rush_yds"] / x["l_rush_att"].clip(lower=0.5)
        x["ypc_vs_his"] = x["book_ypc"] / (x["rush_yds_8"] / x["car_8"].clip(lower=0.5)).clip(lower=0.5)
        x["car_line_vs_his"] = x["l_rush_att"] / x["car_8"].clip(lower=0.5)
    if has("l_rec_yds", "l_recs"):
        x["book_ypr"] = x["l_rec_yds"] / x["l_recs"].clip(lower=0.5)
        x["ypr_vs_his"] = x["book_ypr"] / (x["rec_yds_8"] / x["rec_8"].clip(lower=0.5)).clip(lower=0.5)
    if has("l_pass_yds", "l_pass_att"):
        x["book_ypa"] = x["l_pass_yds"] / x["l_pass_att"].clip(lower=1)
        x["ypa_vs_his"] = x["book_ypa"] / (x["pass_yds_8"] / x["att_8"].clip(lower=1)).clip(lower=1)
    if has("l_pass_completions", "l_pass_att"):
        x["book_cmp"] = x["l_pass_completions"] / x["l_pass_att"].clip(lower=1)
        x["cmp_vs_his"] = x["book_cmp"] / (x["cmp_8"] / x["att_8"].clip(lower=1)).clip(lower=0.2)
    if has("l_pass_td", "l_pass_att"):
        x["book_td_rate"] = x["l_pass_td"] / x["l_pass_att"].clip(lower=1)

    one = x.drop_duplicates(["gsis", "gkey", "market_code"])
    tm = one.groupby(["gkey", "posteam", "market_code"]).agg(
        n=("line", "size"), tot=("line", "sum"), lean=("proj_gap", "mean")).unstack("market_code")
    tm.columns = [f"tm_{a}_{b}" for a, b in tm.columns]
    tm = tm.reset_index()
    x = x.merge(tm, on=["gkey", "posteam"], how="left")
    opp = tm.rename(columns={"posteam": "defteam", **{c: "opp_" + c[3:] for c in tm.columns
                                                     if c.startswith("tm_")}})
    keep = ["gkey", "defteam"] + [c for c in opp.columns if c.startswith("opp_tot_pass")
                                  or c.startswith("opp_tot_rush") or c.startswith("opp_lean")]
    x = x.merge(opp[keep], on=["gkey", "defteam"], how="left")
    for a, b, name in (("tm_tot_recs", "tm_tot_pass_completions", "web_recs_over_cmp"),
                       ("tm_tot_rec_yds", "tm_tot_pass_yds", "web_recyds_over_passyds"),
                       ("tm_tot_rush_att", "tm_tot_pass_att", "web_runs_over_throws")):
        if has(a, b):
            x[name] = x[a] / x[b].clip(lower=1)
    for m in MK:
        t = f"tm_tot_{m}"
        if t in x:
            x[f"share_{m}"] = np.where(x["market_code"] == m, x["line"] / x[t].clip(lower=0.5), np.nan)
    return x


def main():
    x = S.bench()
    base = S.features(x)
    w = web(x)
    extra = [c for c in w.columns if c not in x.columns and c != "gkey"
             and pd.api.types.is_numeric_dtype(w[c])]
    print(f"{len(w):,} props, {len(base)} features before, {len(extra)} from the web")
    for group, markets in M.GROUPS.items():
        g = w[w["market_code"].isin(markets)]
        print(f"\n{group.upper()}: {(g['season'] == 2025).sum():,} props in 2025, "
              f"{(g['season'] == 2026).sum():,} in 2026")
        for label, cols in (("without the web", base), ("with the web", base + extra),
                            ("the web and the basics only",
                             [c for c in base if c in M.ALWAYS or c.startswith(("is_", "pos_"))]
                             + extra)):
            try:
                walk, back = M.both_ways(g, cols)
            except Exception as err:
                print(f"  {label}: could not be fitted ({err})")
                continue
            print(f"  {label} ({len(cols)} features)")
            print(M.slices(walk, "2026, week by week"))
            print(M.slices(back, "2025, the other way"))
    print("\nEach cell: picks, how often right, and in brackets that minus what the "
          "prices needed.")


if __name__ == "__main__":
    sys.exit(main())
