"""Pass attempts and completions, modelled as what they are: a team's plan meeting a game.

Every other model here asks what a player has done lately. That is the wrong
question for attempts. How often a quarterback throws is decided by his team
and by the score: bad teams throw because they are behind, teams with a lead
run the clock, a team starting its backup hides him, and some coaches just
throw more than others whatever the score.

So this starts from the team, one row a team a game, with what was known
before kickoff:

    style        the team's pass rate when the score is within a touchdown in
                 the first three quarters, which is the coach's choice and not
                 the scoreboard's; its plays a game; its sacks and scrambles
    the script   the spread and the total, and what they imply the team will
                 score; home or away; roof and wind
    the defence  how often opponents throw on this defence, how many plays it
                 allows, what completion rate it allows
    the man      the quarterback's own attempts and completion rate, how many
                 starts he has made, and whether he started last week

It is fitted on every game before the one predicted, 2022 on, and scored three
ways on 2025 and 2026:

    1. against simply using his eight-game average, on every starter
    2. against the book's line, on the props that were priced (2026 only: the
       book did not price these markets before)
    3. as a side: over if the model is above the line, and how that did, in
       total and where the model and the line are furthest apart

    python lab/training/research_pass_volume.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
CACHE = Path(__file__).resolve().parents[2] / "marketing" / ".cache"
SEASONS = (2022, 2023, 2024, 2025, 2026)


def team_games():
    cols = ["game_id", "season", "week", "season_type", "game_date", "posteam", "defteam",
            "home_team", "spread_line", "total_line", "roof", "wind", "pass_attempt",
            "rush_attempt", "sack", "qb_scramble", "qb_kneel", "complete_pass",
            "score_differential", "qtr", "passer_player_id", "two_point_attempt",
            "yards_gained"]
    d = pd.concat([pd.read_csv(CACHE / f"pbp_{s}.csv.gz", usecols=cols, low_memory=False)
                   for s in SEASONS], ignore_index=True)
    d = d[(d["season_type"] == "REG") & (d["two_point_attempt"] != 1) & d["posteam"].notna()]
    d["game_date"] = pd.to_datetime(d["game_date"])
    d["throw"] = ((d["pass_attempt"] == 1) & (d["sack"] != 1)).astype(int)
    d["run"] = ((d["rush_attempt"] == 1) & (d["qb_kneel"] != 1)).astype(int)
    d["db"] = (d["pass_attempt"] == 1).astype(int) + d["qb_scramble"].fillna(0).astype(int)
    d["play"] = ((d["throw"] + d["run"] + d["sack"].fillna(0)) > 0).astype(int)
    close = (d["score_differential"].abs() <= 7) & (d["qtr"] <= 3)
    d["n_db"] = np.where(close, d["db"], 0)
    d["n_play"] = np.where(close, d["play"], 0)
    key = ["season", "week", "game_id", "game_date", "posteam", "defteam", "home_team",
           "spread_line", "total_line", "roof", "wind"]
    t = d.groupby(key, dropna=False).agg(
        att=("throw", "sum"), cmp=("complete_pass", "sum"), runs=("run", "sum"),
        plays=("play", "sum"), sacks=("sack", "sum"), scr=("qb_scramble", "sum"),
        n_db=("n_db", "sum"), n_play=("n_play", "sum"),
        pass_yds=("yards_gained", lambda v: v[d.loc[v.index, "throw"] == 1].sum())).reset_index()
    qb = d[d["throw"] == 1].groupby(["game_id", "posteam", "passer_player_id"]).agg(
        q_att=("throw", "sum"), q_cmp=("complete_pass", "sum")).reset_index()
    qb = qb.sort_values("q_att").drop_duplicates(["game_id", "posteam"], keep="last")
    t = t.merge(qb, on=["game_id", "posteam"], how="left")
    t["pass_rate"] = t["att"] / t["plays"].clip(lower=1)
    t["neutral"] = t["n_db"] / t["n_play"].clip(lower=1)
    t["cmp_rate"] = t["cmp"] / t["att"].clip(lower=1)
    t["home"] = (t["posteam"] == t["home_team"]).astype(int)
    t["fav"] = np.where(t["home"] == 1, t["spread_line"], -t["spread_line"])
    t["implied"] = t["total_line"] / 2 + t["fav"] / 2
    t["indoor"] = t["roof"].isin(["dome", "closed"]).astype(int)
    t["wind"] = np.where(t["indoor"] == 1, 0, t["wind"].fillna(6))
    return t.sort_values(["game_date", "game_id"]).reset_index(drop=True)


def roll(g, col, n, minp=2):
    return g[col].transform(lambda v: v.shift(1).rolling(n, min_periods=minp).mean())


def features(t):
    t = t.copy()
    off = t.groupby("posteam", sort=False)
    for c in ("att", "cmp", "plays", "runs", "pass_rate", "neutral", "cmp_rate", "sacks",
              "scr", "pass_yds"):
        t[f"o_{c}_4"] = roll(off, c, 4)
        t[f"o_{c}_8"] = roll(off, c, 8)
    t["o_att_last"] = off["att"].shift(1)
    de = t.groupby("defteam", sort=False)
    for c in ("att", "cmp", "plays", "runs", "pass_rate", "cmp_rate", "sacks"):
        t[f"d_{c}_6"] = roll(de, c, 6)
    q = t.groupby("passer_player_id", sort=False)
    t["q_att_4"] = roll(q, "q_att", 4, 1)
    t["q_att_8"] = roll(q, "q_att", 8, 1)
    t["q_cmp_8"] = roll(q, "q_cmp", 8, 1)
    t["q_rate_8"] = t["q_cmp_8"] / t["q_att_8"].clip(lower=1)
    t["q_starts"] = q.cumcount()
    t["q_same"] = (off["passer_player_id"].shift(1) == t["passer_player_id"]).astype(int)
    t["q_new"] = (t["q_starts"] < 4).astype(int)
    t["week_n"] = t["week"]
    return t


COLS = (["fav", "total_line", "implied", "home", "indoor", "wind", "q_att_4", "q_att_8",
         "q_cmp_8", "q_rate_8", "q_starts", "q_same", "q_new", "o_att_last", "week_n"]
        + [f"o_{c}_{n}" for c in ("att", "cmp", "plays", "runs", "pass_rate", "neutral",
                                  "cmp_rate", "sacks", "scr", "pass_yds") for n in (4, 8)]
        + [f"d_{c}_6" for c in ("att", "cmp", "plays", "runs", "pass_rate", "cmp_rate", "sacks")])


def walk(t, target):
    """2025 and 2026 a week at a time, each fitted on every game before it."""
    out = []
    t = t[t[target].notna() & (t["q_att"] >= 10)]
    for (s, w), te in t[t["season"] >= 2025].groupby(["season", "week"]):
        tr = t[(t["season"] < s) | ((t["season"] == s) & (t["week"] < w))]
        a = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=250,
                                          min_samples_leaf=40, l2_regularization=1.0,
                                          random_state=0).fit(tr[COLS], tr[target])
        b = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                          Ridge(alpha=30.0)).fit(tr[COLS], tr[target])
        te = te.copy()
        te["pred"] = (a.predict(te[COLS]) + b.predict(te[COLS])) / 2
        out.append(te)
    return pd.concat(out)


def mae(a, b):
    return float(np.mean(np.abs(np.asarray(a, float) - np.asarray(b, float))))


def main():
    t = features(team_games())
    rec = pd.read_pickle(CACHE / "record.pkl")
    for c in ("line", "actual", "projection"):
        rec[c] = rec[c].astype(float)
    rec["game_date"] = pd.to_datetime(rec["game_date"])
    for market, target, base in (("pass_att", "q_att", "q_att_8"),
                                 ("pass_completions", "q_cmp", "q_cmp_8")):
        p = walk(t, target)
        print(f"\n{market.upper()}: {len(p):,} starts in 2025 and 2026, each predicted "
              f"from the games before it")
        for s in (2025, 2026):
            x = p[(p["season"] == s) & p[base].notna()]
            print(f"  {s}: his 8-game average was off by {mae(x[target], x[base]):.2f}; "
                  f"the team model by {mae(x[target], x['pred']):.2f} "
                  f"({len(x)} starts)")
        r = rec[(rec["market_code"] == market) & (rec["actual"] != rec["line"])]
        j = r.merge(p, left_on=["gsis", "game_date"],
                    right_on=["passer_player_id", "game_date"], how="inner")
        if len(j) < 60:
            print(f"  {len(j)} priced props matched, too few to read against the line")
            continue
        j["over_side"] = j["pred"] > j["line"]
        j["right"] = np.where(j["over_side"], j["over"], 1 - j["over"])
        j["gap"] = (j["pred"] - j["line"]) / j["line"]
        print(f"  {len(j)} priced props (2026). Off by: the line {mae(j['actual'], j['line']):.2f}, "
              f"our published number {mae(j['actual'], j['projection']):.2f}, "
              f"the team model {mae(j['actual'], j['pred']):.2f}")
        print(f"    means: actual {j['actual'].mean():.1f}, line {j['line'].mean():.1f}, "
              f"published {j['projection'].mean():.1f}, team model {j['pred'].mean():.1f}")
        print(f"    side from the team model: right {j['right'].mean():.1%} "
              f"(published side {j['hit'].mean():.1%}); calls over on {j['over_side'].mean():.0%}")
        for lo, hi in ((0.0, 0.03), (0.03, 0.06), (0.06, 0.10), (0.10, 9)):
            s = j[(j["gap"].abs() >= lo) & (j["gap"].abs() < hi)]
            if len(s) >= 15:
                print(f"    model {lo:.0%} to {hi:.0%} from the line: {len(s):>4} props, "
                      f"right {s['right'].mean():.1%}")
        half = j["week"] <= j["week"].median()
        print(f"    first half of the season {j[half]['right'].mean():.1%} on {half.sum()}, "
              f"second half {j[~half]['right'].mean():.1%} on {(~half).sum()}")
        j.to_pickle(CACHE / f"volume_{market}.pkl")


if __name__ == "__main__":
    sys.exit(main())
