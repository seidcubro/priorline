"""Attempts, completions and carries from the football underneath them.

research_pass_volume.py modelled a team's plan meeting a game, with counting
numbers and one bad rule: whether the quarterback started last week. This
replaces the rule and adds the numbers a coach would actually look at.

    who is under centre   The starter is the man the depth chart names, and a
                          returning starter is not a backup. So each start
                          carries: the share of his team's last 17 games he
                          started (the de facto starter has most of them), how
                          many team games since he last started, his career
                          starts, and his team's numbers in HIS starts only,
                          not in whoever played last week.
    efficiency            EPA a play, EPA a dropback and a carry, success rate,
                          completion rate over expected, depth of target,
                          pressure rate (sacks and hits a dropback), points a
                          drive and plays a drive.
    tendency              pass rate over expected, pass rate in a close game,
                          shotgun and no-huddle rates.
    the defence           every one of those as allowed by the defence across
                          from him over its last six.
    the opponent taken    each offence's number set against what the defences
    out                   it faced usually allow, and the same for defences.
                          The idea behind DVOA: 30 throws against a defence
                          everyone throws on says less than 30 against one
                          nobody does.

Carries are built from the team down: how many times this backfield will run,
then this back's share of it, with the share lifted by the work left behind by
backs who were playing and are not in this game.

Each is fitted on every game before the one predicted and scored on 2025 and
2026: against the old feature set, against the book's line, and as a side.

    python lab/training/research_football.py
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
KEY = ["season", "week", "game_id", "game_date", "posteam", "defteam"]
OFF = ["att", "cmp", "runs", "rb_runs", "plays", "dropbacks", "sack_rate", "scr", "neutral",
       "pass_oe", "epa", "pass_epa", "rush_epa", "succ", "cpoe", "adot", "press", "ppd",
       "drives", "ppdrive", "shotgun", "nohuddle", "cmp_rate"]
DEF = ["att", "cmp", "runs", "rb_runs", "plays", "pass_oe", "epa", "pass_epa", "rush_epa",
       "succ", "cpoe", "adot", "press", "ppd", "cmp_rate", "sack_rate"]
ADJ = ["att", "plays", "runs", "pass_oe", "epa", "succ", "cmp_rate", "press", "cpoe", "ppd"]
WITH = ["att", "cmp", "plays", "runs", "pass_oe", "neutral", "epa", "cmp_rate", "cpoe", "adot",
        "press", "sack_rate", "scr"]


def load():
    cols = ["game_id", "season", "week", "season_type", "game_date", "posteam", "defteam",
            "home_team", "spread_line", "total_line", "roof", "wind", "pass_attempt",
            "rush_attempt", "sack", "qb_scramble", "qb_kneel", "qb_spike", "complete_pass",
            "score_differential", "qtr", "passer_player_id", "rusher_player_id",
            "two_point_attempt", "yards_gained", "epa", "success", "cpoe", "air_yards",
            "qb_hit", "qb_dropback", "pass_oe", "shotgun", "no_huddle", "fixed_drive",
            "fixed_drive_result"]
    d = pd.concat([pd.read_csv(CACHE / f"pbp_{s}.csv.gz", usecols=cols, low_memory=False)
                   for s in SEASONS], ignore_index=True)
    d = d[(d["season_type"] == "REG") & (d["two_point_attempt"] != 1) & d["posteam"].notna()]
    d["game_date"] = pd.to_datetime(d["game_date"])
    pos = {}
    for s in SEASONS:
        r = pd.read_csv(CACHE / f"roster_{s}.csv", usecols=["gsis_id", "position"],
                        low_memory=False).dropna()
        pos.update(r.drop_duplicates("gsis_id").set_index("gsis_id")["position"].to_dict())
    return d, pos


def team_games(d, pos):
    d = d.copy()
    z = lambda c: d[c].fillna(0)  # noqa: E731
    d["throw"] = ((z("pass_attempt") == 1) & (z("sack") != 1) & (z("qb_spike") != 1)).astype(int)
    d["run"] = ((z("rush_attempt") == 1) & (z("qb_kneel") != 1)).astype(int)
    d["db"] = (z("qb_dropback") == 1).astype(int)
    d["play"] = ((d["throw"] + d["run"] + z("sack")) > 0).astype(int)
    d["rb_run"] = d["run"] * (d["rusher_player_id"].map(pos) == "RB").astype(int)
    d["hit"] = ((z("qb_hit") == 1) | (z("sack") == 1)).astype(int) * d["db"]
    close = (d["score_differential"].abs() <= 7) & (d["qtr"] <= 3)
    d["n_db"] = np.where(close, d["db"], 0)
    d["n_play"] = np.where(close, d["play"], 0)
    for c, on in (("epa_p", "play"), ("succ_p", "play"), ("sg", "play"), ("nh", "play")):
        src = {"epa_p": "epa", "succ_p": "success", "sg": "shotgun", "nh": "no_huddle"}[c]
        d[c] = np.where(d[on] == 1, d[src], np.nan)
    d["epa_db"] = np.where(d["db"] == 1, d["epa"], np.nan)
    d["epa_run"] = np.where(d["run"] == 1, d["epa"], np.nan)
    d["air"] = np.where(d["throw"] == 1, d["air_yards"], np.nan)
    d["cpoe_t"] = np.where(d["throw"] == 1, d["cpoe"], np.nan)
    d["cmp_t"] = np.where(d["throw"] == 1, z("complete_pass"), 0)
    g = KEY + ["home_team", "spread_line", "total_line", "roof", "wind"]
    t = d.groupby(g, dropna=False).agg(
        att=("throw", "sum"), cmp=("cmp_t", "sum"), runs=("run", "sum"),
        rb_runs=("rb_run", "sum"), plays=("play", "sum"), dropbacks=("db", "sum"),
        sacks=("sack", "sum"), scr=("qb_scramble", "sum"), hits=("hit", "sum"),
        n_db=("n_db", "sum"), n_play=("n_play", "sum"), pass_oe=("pass_oe", "mean"),
        epa=("epa_p", "mean"), pass_epa=("epa_db", "mean"), rush_epa=("epa_run", "mean"),
        succ=("succ_p", "mean"), cpoe=("cpoe_t", "mean"), adot=("air", "mean"),
        shotgun=("sg", "mean"), nohuddle=("nh", "mean")).reset_index()
    dr = d[d["fixed_drive"].notna()].groupby(["game_id", "posteam", "fixed_drive"]).agg(
        res=("fixed_drive_result", "first"), n=("play", "sum")).reset_index()
    dr["pts"] = dr["res"].map({"Touchdown": 7, "Field goal": 3}).fillna(0)
    dr = dr[dr["n"] > 0].groupby(["game_id", "posteam"]).agg(
        drives=("pts", "size"), ppd=("pts", "mean"), ppdrive=("n", "mean")).reset_index()
    t = t.merge(dr, on=["game_id", "posteam"], how="left")
    qb = d[d["throw"] == 1].groupby(["game_id", "posteam", "passer_player_id"]).agg(
        q_att=("throw", "sum"), q_cmp=("cmp_t", "sum"), q_cpoe=("cpoe_t", "mean"),
        q_adot=("air", "mean"), q_epa=("epa_db", "mean")).reset_index()
    qb = qb.sort_values("q_att").drop_duplicates(["game_id", "posteam"], keep="last")
    t = t.merge(qb, on=["game_id", "posteam"], how="left").rename(
        columns={"passer_player_id": "qb"})
    t["sack_rate"] = t["sacks"] / t["dropbacks"].clip(lower=1)
    t["press"] = t["hits"] / t["dropbacks"].clip(lower=1)
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


def starter(t):
    """Who the starter is, without asking who played last week."""
    share, gap = np.zeros(len(t)), np.full(len(t), np.nan)
    for _, g in t.groupby("posteam", sort=False):
        qbs = g["qb"].tolist()
        for i, ix in enumerate(g.index):
            past = qbs[max(0, i - 17):i]
            if past:
                share[t.index.get_loc(ix)] = past.count(qbs[i]) / len(past)
            back = [k for k in range(i - 1, -1, -1) if qbs[k] == qbs[i]]
            gap[t.index.get_loc(ix)] = (i - 1 - back[0]) if back else 40
    t["q_share"], t["q_gap"] = share, np.minimum(gap, 40)
    top = t.groupby(["posteam", "game_id"])["q_share"].transform("max")
    t["q_main"] = (t["q_share"] >= 0.5).astype(int)
    return t


def features(t):
    t = t.copy()
    off = t.groupby("posteam", sort=False)
    de = t.groupby("defteam", sort=False)
    for c in OFF:
        t[f"o_{c}_4"] = roll(off, c, 4)
        t[f"o_{c}_8"] = roll(off, c, 8)
    for c in DEF:
        t[f"d_{c}_6"] = roll(de, c, 6)
    # the opponent taken out: what he did against what that defence usually allows
    for c in ADJ:
        t[f"_oa_{c}"] = t[c] - t[f"d_{c}_6"]
        t[f"_da_{c}"] = t[c] - t[f"o_{c}_8"]
    off = t.groupby("posteam", sort=False)
    de = t.groupby("defteam", sort=False)
    for c in ADJ:
        t[f"oa_{c}_8"] = roll(off, f"_oa_{c}", 8)
        t[f"da_{c}_6"] = roll(de, f"_da_{c}", 6)
        t[f"x_{c}"] = (t[f"o_{c}_8"] + t[f"da_{c}_6"] + t[f"d_{c}_6"] + t[f"oa_{c}_8"]) / 2
    q = t.groupby("qb", sort=False)
    for c in ("q_att", "q_cmp", "q_cpoe", "q_adot", "q_epa"):
        t[f"{c}_4"] = roll(q, c, 4, 1)
        t[f"{c}_8"] = roll(q, c, 8, 1)
    t["q_rate_8"] = t["q_cmp_8"] / t["q_att_8"].clip(lower=1)
    t["q_starts"] = q.cumcount()
    t["q_new"] = (t["q_starts"] < 4).astype(int)
    t["q_same"] = (off["qb"].shift(1) == t["qb"]).astype(int)
    t["o_att_last"] = off["att"].shift(1)
    t = starter(t)
    wq = t.groupby(["posteam", "qb"], sort=False)
    for c in WITH:
        t[f"wq_{c}_8"] = roll(wq, c, 8, 1)
        t[f"wq_{c}_8"] = t[f"wq_{c}_8"].fillna(t[f"o_{c}_8"])
    t["wq_n"] = wq.cumcount()
    t["week_n"] = t["week"]
    return t


SCRIPT = ["fav", "total_line", "implied", "home", "indoor", "wind", "week_n"]
OLD = (SCRIPT + ["q_att_4", "q_att_8", "q_cmp_8", "q_rate_8", "q_starts", "q_same", "q_new",
                 "o_att_last"]
       + [f"o_{c}_{n}" for c in ("att", "cmp", "plays", "runs", "neutral", "cmp_rate",
                                 "sack_rate", "scr") for n in (4, 8)]
       + [f"d_{c}_6" for c in ("att", "cmp", "plays", "runs", "cmp_rate", "sack_rate")])
MAN = ["q_share", "q_gap", "q_main", "q_starts", "q_new", "wq_n"] + \
      [f"q_{c}_{n}" for c in ("att", "cmp", "cpoe", "adot", "epa") for n in (4, 8)] + \
      ["q_rate_8"] + [f"wq_{c}_8" for c in WITH]
DEEP = (SCRIPT + MAN + [f"o_{c}_{n}" for c in OFF for n in (4, 8)]
        + [f"d_{c}_6" for c in DEF]
        + [f"{p}_{c}_{n}" for c in ADJ for p, n in (("oa", 8), ("da", 6))]
        + [f"x_{c}" for c in ADJ])
STARTER = [c for c in OLD if c != "q_same"] + ["q_share", "q_gap", "q_main", "wq_n"] + \
          [f"wq_{c}_8" for c in ("att", "cmp", "plays", "runs", "neutral", "cmp_rate")]
LEAN = SCRIPT + MAN + [f"x_{c}" for c in ADJ] + \
       [f"o_{c}_8" for c in ("neutral", "pass_oe", "plays", "ppdrive", "shotgun", "nohuddle")] + \
       [f"d_{c}_6" for c in ("pass_oe", "plays", "press", "cmp_rate", "cpoe", "adot", "epa")]


def fit(tr, cols, target):
    a = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=250,
                                      min_samples_leaf=40, l2_regularization=1.0,
                                      random_state=0).fit(tr[cols], tr[target])
    b = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                      Ridge(alpha=30.0)).fit(tr[cols], tr[target])
    return lambda x: (a.predict(x[cols]) + b.predict(x[cols])) / 2


def walk(t, target, cols, name="pred"):
    """2025 and 2026 a week at a time, each fitted on every game before it."""
    out = []
    t = t[t[target].notna()]
    for (s, w), te in t[t["season"] >= 2025].groupby(["season", "week"]):
        tr = t[(t["season"] < s) | ((t["season"] == s) & (t["week"] < w))]
        te = te.copy()
        te[name] = fit(tr, cols, target)(te)
        out.append(te)
    return pd.concat(out)


def mae(a, b):
    return float(np.mean(np.abs(np.asarray(a, float) - np.asarray(b, float))))


def against_line(j, label):
    j = j.copy()
    j["over_side"] = j["pred"] > j["line"]
    j["right"] = np.where(j["over_side"], j["over"], 1 - j["over"])
    j["gap"] = (j["pred"] - j["line"]) / j["line"].clip(lower=0.5)
    by = ", ".join(f"{int(s)}: {x['right'].mean():.1%} on {len(x)}"
                   for s, x in j.groupby("season_x" if "season_x" in j else "season"))
    print(f"    {label}: off by {mae(j['actual'], j['pred']):.2f}, side right "
          f"{j['right'].mean():.1%} on {len(j)} ({by}); over on {j['over_side'].mean():.0%}")
    far = j[j["gap"].abs() >= j["gap"].abs().quantile(0.6)]
    byf = ", ".join(f"{int(s)}: {x['right'].mean():.1%} on {len(x)}"
                    for s, x in far.groupby("season_x" if "season_x" in far else "season"))
    print(f"      the 40% furthest from the line: {far['right'].mean():.1%} on {len(far)} "
          f"({byf}), needed {far['need'].mean():.1%}")
    return j


def passing(t, rec):
    t = t[t["q_att"] >= 10]
    for market, target in (("pass_att", "q_att"), ("pass_completions", "q_cmp")):
        print(f"\n{market.upper()}")
        r = rec[(rec["market_code"] == market) & (rec["actual"] != rec["line"])]
        runs = {}
        for label, cols in (("old features (started last week)", OLD),
                            ("old, with the real starter", STARTER),
                            ("the man, the tendencies, opponent taken out", LEAN),
                            ("everything", DEEP)):
            p = walk(t, target, cols)
            runs[label] = p
            both = ", ".join(f"{s}: {mae(x[target], x['pred']):.2f}"
                             for s, x in p.groupby("season"))
            print(f"  {label} ({len(cols)} features): off by {both} on {len(p):,} starts")
            ch = p[p["q_same"] == 0]
            print(f"    starts where a different man played last week ({len(ch)}): "
                  f"off by {mae(ch[target], ch['pred']):.2f}")
            j = r.merge(p, left_on=["gsis", "game_date"], right_on=["qb", "game_date"])
            if len(j) >= 60:
                against_line(j, "against the line")
        j = r.merge(runs["old features (started last week)"], left_on=["gsis", "game_date"],
                    right_on=["qb", "game_date"])
        if len(j):
            print(f"  on those {len(j)} props the line was off by "
                  f"{mae(j['actual'], j['line']):.2f}, our published number by "
                  f"{mae(j['actual'], j['projection']):.2f}, published side right "
                  f"{j['hit'].mean():.1%}")
    # completions as attempts times a completion rate, each modelled apart
    print("\nCOMPLETIONS BUILT IN TWO STEPS: attempts, times a completion rate")
    t = t.copy()
    t["q_rate"] = t["q_cmp"] / t["q_att"].clip(lower=1)
    a = walk(t, "q_att", LEAN, "p_att")
    b = walk(t, "q_rate", LEAN, "p_rate")
    p = a.merge(b[["game_id", "posteam", "p_rate"]], on=["game_id", "posteam"])
    p["pred"] = p["p_att"] * p["p_rate"]
    print("  off by " + ", ".join(f"{s}: {mae(x['q_cmp'], x['pred']):.2f}"
                                  for s, x in p.groupby("season")))
    r = rec[(rec["market_code"] == "pass_completions") & (rec["actual"] != rec["line"])]
    j = r.merge(p, left_on=["gsis", "game_date"], right_on=["qb", "game_date"])
    if len(j) >= 60:
        against_line(j, "against the line")


def backs(d, pos, t):
    """One row a running back a game, with his share and the work left behind."""
    ru = d[(d["rush_attempt"] == 1) & (d["qb_kneel"] != 1) & d["rusher_player_id"].notna()]
    ru = ru[ru["rusher_player_id"].map(pos) == "RB"]
    b = ru.groupby(KEY + ["rusher_player_id"]).agg(
        car=("rush_attempt", "size"), yds=("yards_gained", "sum"), epa=("epa", "mean"),
        succ=("success", "mean")).reset_index().rename(columns={"rusher_player_id": "pid"})
    b = b.merge(t[["game_id", "posteam", "rb_runs"]], on=["game_id", "posteam"])
    b["share"] = b["car"] / b["rb_runs"].clip(lower=1)
    b = b.sort_values(["game_date", "game_id"]).reset_index(drop=True)
    g = b.groupby("pid", sort=False)
    for c in ("car", "share", "yds", "epa", "succ"):
        b[f"b_{c}_4"] = g[c].transform(lambda v: v.shift(1).rolling(4, min_periods=1).mean())
        b[f"b_{c}_8"] = g[c].transform(lambda v: v.shift(1).rolling(8, min_periods=1).mean())
    b["b_car_last"] = g["car"].shift(1)
    b["b_car_max8"] = g["car"].transform(lambda v: v.shift(1).rolling(8, min_periods=1).max())
    b["b_games"] = g.cumcount()
    # the work left behind: backs who carried in the team's last three games and are not here
    vac, room = {}, {}
    for team, tg in b.groupby("posteam", sort=False):
        games = tg.drop_duplicates("game_id")["game_id"].tolist()
        hist, last = {}, {}
        for i, gid in enumerate(games):
            here = tg[tg["game_id"] == gid]
            act = set(here["pid"])
            gone = sum(np.mean(hist[p][-4:]) for p in hist
                       if p not in act and i - last[p] <= 3)
            known = sum(np.mean(hist[p][-4:]) for p in act if p in hist)
            vac[(team, gid)], room[(team, gid)] = gone, known
            for p, s in zip(here["pid"], here["share"]):
                hist.setdefault(p, []).append(s)
                last[p] = i
    k = list(zip(b["posteam"], b["game_id"]))
    b["vacated"] = [vac[x] for x in k]
    b["room"] = [room[x] for x in k]
    b["norm_share"] = b["b_share_4"] / b["room"].clip(lower=0.2)
    b["lifted"] = b["b_share_4"] * (1 + b["vacated"] / b["room"].clip(lower=0.2))
    return b


RB = ["b_car_4", "b_car_8", "b_car_last", "b_car_max8", "b_share_4", "b_share_8", "b_yds_8",
      "b_epa_8", "b_succ_8", "b_games", "vacated", "room", "norm_share", "lifted", "p_runs",
      "exp_car"] + SCRIPT + \
     [f"o_{c}_8" for c in ("rb_runs", "runs", "plays", "neutral", "pass_oe", "rush_epa", "epa",
                           "succ", "ppd", "scr")] + \
     [f"d_{c}_6" for c in ("rb_runs", "runs", "plays", "rush_epa", "succ", "pass_oe", "ppd")] + \
     [f"x_{c}" for c in ("runs", "plays", "epa", "pass_oe")] + ["q_main", "q_share", "wq_runs_8"]
RB_OLD = ["b_car_4", "b_car_8", "b_car_last", "b_share_4", "b_share_8", "b_games"] + SCRIPT


def rushing(d, pos, t, rec):
    print("\nRUSH_ATT: the team's carries, then this back's share of them")
    team = walk(t, "rb_runs", LEAN + [f"o_{c}_{n}" for c in ("rb_runs", "runs") for n in (4, 8)]
                + ["d_rb_runs_6", "d_runs_6", "d_rush_epa_6", "o_rush_epa_8"], "p_runs")
    print("  team running-back carries off by " + ", ".join(
        f"{s}: {mae(x['rb_runs'], x['p_runs']):.2f} (its 8-game average "
        f"{mae(x['rb_runs'], x['o_rb_runs_8'].fillna(x['rb_runs'].mean())):.2f})"
        for s, x in team.groupby("season")))
    b = backs(d, pos, t)
    keep = ["game_id", "posteam"] + [c for c in t.columns if c in RB or c == "q_same"]
    b = b.merge(t[list(dict.fromkeys(keep))], on=["game_id", "posteam"], how="left")
    # team carries known before the game: the walked number where there is one
    b = b.merge(team[["game_id", "posteam", "p_runs"]], on=["game_id", "posteam"], how="left")
    b["p_runs"] = b["p_runs"].fillna(b["o_rb_runs_8"])
    b["exp_car"] = b["p_runs"] * b["lifted"]
    r = rec[(rec["market_code"] == "rush_att") & (rec["actual"] != rec["line"])
            & (rec["position"] == "RB")]
    for label, cols in (("his own carries and the script", RB_OLD),
                        ("team carries, share, work left behind, efficiency", RB)):
        p = walk(b, "car", cols)
        print(f"  {label} ({len(cols)} features): off by " + ", ".join(
            f"{s}: {mae(x['car'], x['pred']):.2f}" for s, x in p.groupby("season")))
        ch = p[p["vacated"] >= 0.25]
        print(f"    games where a quarter of the carries had been left behind ({len(ch)}): "
              f"off by {mae(ch['car'], ch['pred']):.2f}, his 4-game average "
              f"{mae(ch['car'], ch['b_car_4'].fillna(0)):.2f}")
        j = r.merge(p, left_on=["gsis", "game_date"], right_on=["pid", "game_date"])
        if len(j) >= 60:
            j = against_line(j, "against the line")
            print(f"    on those props the line was off by {mae(j['actual'], j['line']):.2f}, "
                  f"our published number by {mae(j['actual'], j['projection']):.2f}, "
                  f"published side right {j['hit'].mean():.1%}")
            for nm, m in (("unders", ~j["over_side"]), ("overs", j["over_side"]),
                          ("work left behind", j["vacated"] >= 0.25)):
                x = j[m]
                if len(x) >= 30:
                    print(f"      {nm}: {x['right'].mean():.1%} on {len(x)}, needed "
                          f"{x['need'].mean():.1%}")
            j.to_pickle(CACHE / "football_rush_att.pkl")


def main():
    d, pos = load()
    t = features(team_games(d, pos))
    rec = pd.read_pickle(CACHE / "record.pkl")
    for c in ("line", "actual", "projection", "need"):
        rec[c] = rec[c].astype(float)
    rec["game_date"] = pd.to_datetime(rec["game_date"])
    print(f"{len(t):,} team games, {len(DEEP)} features")
    back = t[(t["q_same"] == 0) & (t["q_main"] == 1) & (t["season"] >= 2025)]
    print(f"starts by the de facto starter coming back after another man played: {len(back)}")
    passing(t, rec)
    rushing(d, pos, t, rec)


if __name__ == "__main__":
    sys.exit(main())
