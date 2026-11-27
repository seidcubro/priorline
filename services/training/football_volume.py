"""Pass attempts, completions and carries from the football underneath them.

The board's volume numbers came from what a player had done lately. That is
the wrong question for how often a quarterback throws or a back carries. Those
are decided by the team, the score and who is on the field, so this starts from
the team, one row a team a game, with what was known before kickoff:

    who is under centre   The starter is the man the depth chart names, and a
                          returning starter is not a backup. Each start carries
                          the share of his team's last 17 games he started, how
                          many team games since he last did, his career starts,
                          and his team's numbers in HIS starts only. When a
                          starter comes back from injury the team is read on
                          his games, not on the games of whoever filled in.
    efficiency            EPA a play, a dropback and a carry, success rate,
                          completion rate over expected, depth of target,
                          pressure rate, points a drive, plays a drive.
    tendency              pass rate over expected, pass rate in a close game,
                          shotgun and no-huddle rates.
    the script            the spread and the total, home or away, roof, wind.
    the defence           every one of those as allowed by the defence across
                          from him over its last six.
    the opponent taken    each offence's number set against what the defences
    out                   it faced usually allow, and the same for defences.

Measured on 2025 and 2026, each game predicted from the games before it
(lab/training/research_football.py), on the props the book priced:

    pass attempts      off by 6.15 where our published number was off by 6.37
    completions        off by 4.22 against 4.55
    carries            off by 3.91 against 4.04
    quarterback changed since last week (40 props): 7.23 against 8.25 on
                       attempts, 4.53 against 5.07 on completions

The book's line is still closer than either (6.02, 4.11, 3.69). This makes our
number better. It has not yet been shown to pick the side better, and the
carries number is written to a shadow table and graded there, not published.

Two runs:

    python football_volume.py           before the board: writes the attempts
                                        and completions numbers for every
                                        quarterback who could start this week
    python football_volume.py shadow    after the board: writes this model's
                                        carries number beside each priced back

Before publishing, it checks itself: this season's games are predicted a week
at a time and set against what we published on the same props before this went
live. A market whose number is not closer than the published one is left out.
FOOTBALL_VOLUME=0 turns the published numbers off.
"""

from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import pick_depth as depth

warnings.filterwarnings("ignore")
ARTIFACTS = Path(os.getenv("ARTIFACT_DIR", "/artifacts"))
NAME = "football_volume.json"
LIVE_FROM = "2026-11-27"
MARKET = {"pass_att": "q_att", "pass_completions": "q_cmp"}
KEY = ["season", "week", "game_id", "game_date", "posteam", "defteam"]
PBP = ["game_id", "season", "week", "season_type", "game_date", "posteam", "defteam",
       "home_team", "spread_line", "total_line", "roof", "wind", "pass_attempt",
       "rush_attempt", "sack", "qb_scramble", "qb_kneel", "qb_spike", "complete_pass",
       "score_differential", "qtr", "passer_player_id", "rusher_player_id",
       "two_point_attempt", "yards_gained", "epa", "success", "cpoe", "air_yards", "qb_hit",
       "qb_dropback", "pass_oe", "shotgun", "no_huddle", "fixed_drive", "fixed_drive_result"]
OFF = ["att", "cmp", "runs", "rb_runs", "plays", "dropbacks", "sack_rate", "scr", "neutral",
       "pass_oe", "epa", "pass_epa", "rush_epa", "succ", "cpoe", "adot", "press", "ppd",
       "drives", "ppdrive", "shotgun", "nohuddle", "cmp_rate"]
DEF = ["att", "cmp", "runs", "rb_runs", "plays", "pass_oe", "epa", "pass_epa", "rush_epa",
       "succ", "cpoe", "adot", "press", "ppd", "cmp_rate", "sack_rate"]
ADJ = ["att", "plays", "runs", "pass_oe", "epa", "succ", "cmp_rate", "press", "cpoe", "ppd"]
WITH = ["att", "cmp", "plays", "runs", "pass_oe", "neutral", "epa", "cmp_rate", "cpoe", "adot",
        "press", "sack_rate", "scr"]
SCRIPT = ["fav", "total_line", "implied", "home", "indoor", "wind", "week_n"]
MAN = ["q_share", "q_gap", "q_main", "q_starts", "q_new", "wq_n"] + \
      [f"q_{c}_{n}" for c in ("att", "cmp", "cpoe", "adot", "epa") for n in (4, 8)] + \
      ["q_rate_8"] + [f"wq_{c}_8" for c in WITH]
DEEP = (SCRIPT + MAN + [f"o_{c}_{n}" for c in OFF for n in (4, 8)]
        + [f"d_{c}_6" for c in DEF]
        + [f"{p}_{c}_{n}" for c in ADJ for p, n in (("oa", 8), ("da", 6))]
        + [f"x_{c}" for c in ADJ])
STARTER = (SCRIPT + ["q_att_4", "q_att_8", "q_cmp_8", "q_rate_8", "q_starts", "q_new",
                     "o_att_last", "q_share", "q_gap", "q_main", "wq_n"]
           + [f"o_{c}_{n}" for c in ("att", "cmp", "plays", "runs", "neutral", "cmp_rate",
                                     "sack_rate", "scr") for n in (4, 8)]
           + [f"d_{c}_6" for c in ("att", "cmp", "plays", "runs", "cmp_rate", "sack_rate")]
           + [f"wq_{c}_8" for c in ("att", "cmp", "plays", "runs", "neutral", "cmp_rate")])
TEAM_RUNS = (SCRIPT + ["q_share", "q_main", "q_starts", "wq_runs_8", "wq_plays_8", "wq_scr_8"]
             + [f"x_{c}" for c in ADJ]
             + [f"o_{c}_{n}" for c in ("rb_runs", "runs", "plays", "neutral", "pass_oe",
                                       "rush_epa", "ppdrive") for n in (4, 8)]
             + [f"d_{c}_6" for c in ("rb_runs", "runs", "plays", "rush_epa", "pass_oe", "epa")])
RB = ["b_car_4", "b_car_8", "b_car_last", "b_car_max8", "b_share_4", "b_share_8", "b_yds_8",
      "b_epa_8", "b_succ_8", "b_games", "vacated", "room", "norm_share", "lifted", "p_runs",
      "exp_car"] + SCRIPT + \
     [f"o_{c}_8" for c in ("rb_runs", "runs", "plays", "neutral", "pass_oe", "rush_epa", "epa",
                           "succ", "ppd", "scr")] + \
     [f"d_{c}_6" for c in ("rb_runs", "runs", "plays", "rush_epa", "succ", "pass_oe", "ppd")] + \
     [f"x_{c}" for c in ("runs", "plays", "epa", "pass_oe")] + ["q_main", "q_share", "wq_runs_8"]


def load(seasons):
    out, pos = [], {}
    for s in seasons:
        try:
            f = depth.fetch(f"pbp_{s}.csv.gz", f"{depth.BASE}/pbp/play_by_play_{s}.csv.gz")
            r = depth.fetch(f"roster_{s}.csv", f"{depth.BASE}/rosters/roster_{s}.csv")
        except Exception as err:
            print(f"  no play-by-play for {s}: {err}")
            continue
        d = pd.read_csv(f, usecols=PBP, low_memory=False)
        out.append(d[(d["season_type"] == "REG") & (d["two_point_attempt"] != 1)
                     & d["posteam"].notna()])
        ros = pd.read_csv(r, usecols=["gsis_id", "position"], low_memory=False).dropna()
        pos.update(ros.drop_duplicates("gsis_id").set_index("gsis_id")["position"].to_dict())
    d = pd.concat(out, ignore_index=True)
    d["game_date"] = pd.to_datetime(d["game_date"])
    return d, pos


def team_games(d, pos):
    """One row a team a game: what it did, and who threw for it."""
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
    on = d["play"] == 1
    d["epa_p"] = np.where(on, d["epa"], np.nan)
    d["succ_p"] = np.where(on, d["success"], np.nan)
    d["sg"] = np.where(on, d["shotgun"], np.nan)
    d["nh"] = np.where(on, d["no_huddle"], np.nan)
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
    t["future"] = 0
    return t


def roll(g, col, n, minp=2):
    return g[col].transform(lambda v: v.shift(1).rolling(n, min_periods=minp).mean())


def starter(t):
    """Who the starter is, without asking who played last week.

    q_share is the share of the team's last 17 games this man started. The de
    facto starter has most of them however long he has been out. q_gap is how
    many team games have gone by since his last start.
    """
    share, gap = np.zeros(len(t)), np.full(len(t), 40.0)
    where = {ix: n for n, ix in enumerate(t.index)}
    for _, g in t.groupby("posteam", sort=False):
        qbs = g["qb"].tolist()
        for i, ix in enumerate(g.index):
            past = qbs[max(0, i - 17):i]
            if past:
                share[where[ix]] = past.count(qbs[i]) / len(past)
            back = [k for k in range(i - 1, -1, -1) if qbs[k] == qbs[i]]
            if back:
                gap[where[ix]] = min(40, i - 1 - back[0])
    t["q_share"], t["q_gap"] = share, gap
    t["q_main"] = (t["q_share"] >= 0.5).astype(int)
    return t


def features(t, future=None):
    """Everything known before each game. `future` rows are games not yet played:
    season, week, game_id, game_date, posteam, defteam, home_team, spread_line,
    total_line, roof, wind, qb."""
    if future:
        u = pd.DataFrame(future)
        u["game_date"] = pd.to_datetime(u["game_date"])
        u["future"] = 1
        t = pd.concat([t, u], ignore_index=True)
    t = t.sort_values(["game_date", "future", "game_id"]).reset_index(drop=True)
    t["home"] = (t["posteam"] == t["home_team"]).astype(int)
    t["fav"] = np.where(t["home"] == 1, t["spread_line"], -t["spread_line"])
    t["implied"] = t["total_line"] / 2 + t["fav"] / 2
    t["indoor"] = t["roof"].isin(["dome", "closed"]).astype(int)
    t["wind"] = np.where(t["indoor"] == 1, 0, pd.to_numeric(t["wind"], errors="coerce").fillna(6))
    off = t.groupby("posteam", sort=False)
    de = t.groupby("defteam", sort=False)
    new = {}
    for c in OFF:
        new[f"o_{c}_4"] = roll(off, c, 4)
        new[f"o_{c}_8"] = roll(off, c, 8)
    for c in DEF:
        new[f"d_{c}_6"] = roll(de, c, 6)
    t = pd.concat([t, pd.DataFrame(new)], axis=1)
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
    t["o_att_last"] = off["att"].shift(1)
    t = starter(t)
    wq = t.groupby(["posteam", "qb"], sort=False)
    for c in WITH:
        t[f"wq_{c}_8"] = roll(wq, c, 8, 1).fillna(t[f"o_{c}_8"])
    t["wq_n"] = wq.cumcount()
    t["week_n"] = t["week"]
    return t


def fit(tr, cols, target):
    a = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=250,
                                      min_samples_leaf=40, l2_regularization=1.0,
                                      random_state=0).fit(tr[cols], tr[target])
    b = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                      Ridge(alpha=30.0)).fit(tr[cols], tr[target])
    return lambda x: (a.predict(x[cols]) + b.predict(x[cols])) / 2


def passing_model(tr, target):
    """The plain set with the real starter and the full set, averaged."""
    a, b = fit(tr, STARTER, target), fit(tr, DEEP, target)
    return lambda x: (a(x) + b(x)) / 2


def held_out(t, target, cols=None, folds=6):
    """Every game predicted by a model that never saw its week."""
    t = t[t[target].notna()].copy()
    wk = t["season"].astype(int) * 100 + t["week"].astype(int)
    hands = pd.Series(sorted(wk.unique())).sample(frac=1, random_state=3).reset_index(drop=True)
    t["fold"] = wk.map({w: i % folds for i, w in hands.items()})
    out = []
    for f in range(folds):
        tr, te = t[t["fold"] != f], t[t["fold"] == f].copy()
        te["pred"] = fit(tr, cols, target)(te) if cols else passing_model(tr, target)(te)
        out.append(te)
    return pd.concat(out)


def walk(t, target, season):
    """One season a week at a time, each fitted on every game before it."""
    out = []
    t = t[t[target].notna() & (t["future"] == 0)]
    for w, te in t[t["season"] == season].groupby("week"):
        tr = t[(t["season"] < season) | ((t["season"] == season) & (t["week"] < w))]
        te = te.copy()
        te["pred"] = passing_model(tr, target)(te)
        out.append(te)
    return pd.concat(out) if out else pd.DataFrame()


def backs(d, pos, t, future=None, active=None):
    """One row a running back a game, with his share and the work left behind.

    `future` rows are pid, posteam, game_id, game_date for games not yet
    played; `active` maps a team to the backs who will dress, which a played
    game answers for itself.
    """
    ru = d[(d["rush_attempt"] == 1) & (d["qb_kneel"] != 1) & d["rusher_player_id"].notna()]
    ru = ru[ru["rusher_player_id"].map(pos) == "RB"]
    b = ru.groupby(KEY + ["rusher_player_id"]).agg(
        car=("rush_attempt", "size"), yds=("yards_gained", "sum"), epa=("epa", "mean"),
        succ=("success", "mean")).reset_index().rename(columns={"rusher_player_id": "pid"})
    b = b.merge(t.loc[t["future"] == 0, ["game_id", "posteam", "rb_runs"]],
                on=["game_id", "posteam"])
    b["share"] = b["car"] / b["rb_runs"].clip(lower=1)
    b["future"] = 0
    if future:
        u = pd.DataFrame(future)
        u["game_date"] = pd.to_datetime(u["game_date"])
        u["future"] = 1
        b = pd.concat([b, u], ignore_index=True)
    b = b.sort_values(["game_date", "future", "game_id"]).reset_index(drop=True)
    g = b.groupby("pid", sort=False)
    for c in ("car", "share", "yds", "epa", "succ"):
        b[f"b_{c}_4"] = g[c].transform(lambda v: v.shift(1).rolling(4, min_periods=1).mean())
        b[f"b_{c}_8"] = g[c].transform(lambda v: v.shift(1).rolling(8, min_periods=1).mean())
    b["b_car_last"] = g["car"].shift(1)
    b["b_car_max8"] = g["car"].transform(lambda v: v.shift(1).rolling(8, min_periods=1).max())
    b["b_games"] = g.cumcount()
    vac, room = {}, {}
    for team, tg in b.groupby("posteam", sort=False):
        hist, last = {}, {}
        for i, (gid, here) in enumerate(tg.groupby("game_id", sort=False)):
            if here["future"].iloc[0] == 1:
                recent = {p for p in hist if i - last[p] <= 3}
                act = (set((active or {}).get(team, ())) & recent) | set(here["pid"])
            else:
                act = set(here["pid"])
            vac[(team, gid)] = sum(np.mean(hist[p][-4:]) for p in hist
                                   if p not in act and i - last[p] <= 3)
            room[(team, gid)] = sum(np.mean(hist[p][-4:]) for p in act if p in hist)
            if here["future"].iloc[0] == 0:
                for p, s in zip(here["pid"], here["share"]):
                    hist.setdefault(p, []).append(s)
                    last[p] = i
    k = list(zip(b["posteam"], b["game_id"]))
    b["vacated"] = [vac[x] for x in k]
    b["room"] = [room[x] for x in k]
    b["norm_share"] = b["b_share_4"] / b["room"].clip(lower=0.2)
    b["lifted"] = b["b_share_4"] * (1 + b["vacated"] / b["room"].clip(lower=0.2))
    return b


def engine():
    from sqlalchemy import create_engine
    url = os.getenv("DATABASE_URL") or (
        f"postgresql+psycopg2://{os.getenv('POSTGRES_USER', 'app')}:"
        f"{os.getenv('POSTGRES_PASSWORD', 'app')}@{os.getenv('POSTGRES_HOST', 'postgres')}:"
        f"{os.getenv('POSTGRES_PORT', '5432')}/{os.getenv('POSTGRES_DB', 'app')}")
    return create_engine(url)


def code(team):
    return depth.SAME.get(team, team)


def world(conn):
    """This week's games, who is on each team, who is ruled out, and the
    depth chart's quarterbacks in order."""
    from sqlalchemy import text
    games = pd.read_sql(text(
        "SELECT game_id, season, week, game_date, home_team, away_team, spread_line, "
        "total_line, roof, wind FROM nfl_games WHERE game_date >= CURRENT_DATE "
        "AND game_date <= CURRENT_DATE + 8 AND game_type = 'REG'"), conn)
    people = pd.read_sql(text(
        "SELECT name, external_id, team, position, status FROM players "
        "WHERE external_id IS NOT NULL AND position IN ('QB', 'RB')"), conn)
    out = pd.read_sql(text(
        "SELECT DISTINCT ON (player_id) player_id, report_status FROM injuries "
        "WHERE season = (SELECT MAX(season) FROM injuries) "
        "ORDER BY player_id, week DESC"), conn)
    gone = set(out.loc[out["report_status"].fillna("").str.lower().isin(
        ("out", "doubtful")), "player_id"])
    try:
        over = pd.read_sql(text(
            "SELECT DISTINCT ON (player_id) player_id, status FROM player_status_overrides "
            "WHERE expires_at > NOW() ORDER BY player_id, created_at DESC"), conn)
        cleared = set(over.loc[over["status"].str.lower() == "active", "player_id"])
        gone = (gone - cleared) | set(over.loc[over["status"].str.lower() != "active",
                                               "player_id"])
    except Exception:
        pass
    gone |= set(people.loc[people["status"].fillna("ACT") != "ACT", "external_id"])
    chart = pd.read_sql(text(
        "SELECT player_id, team, MIN(depth_team) AS rank FROM depth_charts "
        "WHERE depth_position = 'QB' AND depth_team IS NOT NULL "
        "AND season = (SELECT MAX(season) FROM depth_charts) "
        "AND week = (SELECT MAX(week) FROM depth_charts "
        "            WHERE season = (SELECT MAX(season) FROM depth_charts)) "
        "GROUP BY player_id, team"), conn)
    return games, people, gone, chart


def week_rows(games):
    rows = []
    for g in games.itertuples():
        for mine, other in ((g.home_team, g.away_team), (g.away_team, g.home_team)):
            rows.append({"season": int(g.season), "week": int(g.week),
                         "game_id": f"next_{g.game_id}", "game_date": g.game_date,
                         "posteam": code(mine), "defteam": code(other),
                         "home_team": code(g.home_team), "spread_line": g.spread_line,
                         "total_line": g.total_line, "roof": g.roof, "wind": g.wind})
    return rows


def candidates(rows, people, gone, chart, t):
    """Each team's quarterbacks, the depth chart's starter first.

    Every one of them gets a number as though he starts, read on his own
    games. The board only prices the man the book expects, so the others cost
    nothing, and nobody has to guess the starter right for the number to be.

    That includes a man the report still lists as out. A report runs behind
    the news, and the week a starter comes back he has to have a number ready
    whether or not the report has caught up. Injured men are only moved to the
    back of the order, which decides nothing but who is printed as the starter
    and who the backs' carries are read beside.
    """
    rank = {r.player_id: r.rank for r in chart.itertuples()}
    last = t[t["future"] == 0].groupby("qb")["game_date"].max().to_dict()
    qbs = people[(people["position"] == "QB")].copy()
    qbs["team"] = qbs["team"].map(code)
    by = {}
    for team in {r["posteam"] for r in rows}:
        mine = qbs[qbs["team"] == team]
        order = sorted(mine["external_id"], key=lambda p: (
            p in gone, rank.get(p, 9), -(last[p].toordinal() if p in last else 0)))
        by[team] = order[:6]
    return by


def numbers():
    """The attempts and completions numbers, written before the board is built."""
    from sqlalchemy import text
    if os.getenv("FOOTBALL_VOLUME", "1") == "0":
        print("football volume: off (FOOTBALL_VOLUME=0)")
        return
    # Every quarterback who could start already has a number in the file, so a
    # ruling an hour later needs no refit. Twelve hours covers the hourly runs.
    try:
        old = json.loads((ARTIFACTS / NAME).read_text(encoding="utf-8"))
        age = (time.time() - float(old.get("built_at", 0))) / 3600
        if (age < float(os.getenv("FOOTBALL_VOLUME_HOURS", "12")) and old.get("numbers")
                and os.getenv("FOOTBALL_VOLUME_DRY") != "1"):
            print(f"football volume: numbers are {age:.1f} hours old, kept")
            return
    except (OSError, ValueError):
        pass
    eng = engine()
    with eng.connect() as conn:
        games, people, gone, chart = world(conn)
        graded = pd.read_sql(text(
            "SELECT player_id AS qb, game_date, market_code, line, projection, actual "
            "FROM prop_edge_results WHERE hit IS NOT NULL AND source = 'live' "
            "AND market_code IN ('pass_att', 'pass_completions') AND game_date < :live"),
            conn, params={"live": LIVE_FROM})
    if games.empty:
        print("football volume: no games in the next eight days, nothing written")
        return
    season = int(games["season"].max())
    d, pos = load(range(season - 4, season + 1))
    base = team_games(d, pos)
    rows = week_rows(games)
    played = features(base)
    by = candidates(rows, people, gone, chart, played)
    names = people.drop_duplicates("external_id").set_index("external_id")["name"].to_dict()

    report = {"built_at": time.time(), "live_from": LIVE_FROM, "check": {}, "numbers": {},
              "starters": {}}
    graded["game_date"] = pd.to_datetime(graded["game_date"])
    ok = {}
    for market, target in MARKET.items():
        hist = played[played["q_att"] >= 10]
        w = walk(hist, target, season)
        g = graded[graded["market_code"] == market].merge(w, on=["qb", "game_date"]) \
            if len(w) else pd.DataFrame()
        if len(g) < 100:
            ok[market] = False
            report["check"][market] = {"props": int(len(g)), "accepted": False}
            print(f"  football volume, {market}: {len(g)} graded props to check against, "
                  "under 100, so the published number is left as it is")
            continue
        for c in ("line", "projection", "actual"):
            g[c] = g[c].astype(float)
        mine = float((g["actual"] - g["pred"]).abs().mean())
        ours = float((g["actual"] - g["projection"]).abs().mean())
        book = float((g["actual"] - g["line"]).abs().mean())
        ok[market] = mine < ours
        report["check"][market] = {"props": int(len(g)), "model": mine, "published": ours,
                                   "line": book, "accepted": bool(ok[market])}
        print(f"  football volume, {market}, {season} a week at a time on {len(g)} priced "
              f"props: this was off by {mine:.2f}, what we published by {ours:.2f}, the "
              f"line by {book:.2f}" + ("" if ok[market] else ". NOT closer, so not used"))

    for k in range(6):
        future = [dict(r, qb=by[r["posteam"]][k]) for r in rows if len(by[r["posteam"]]) > k]
        if not future:
            break
        t = features(base, future)
        hist = t[(t["future"] == 0) & (t["q_att"] >= 10)]
        nxt = t[t["future"] == 1]
        for market, target in MARKET.items():
            if not ok.get(market):
                continue
            pred = passing_model(hist, target)(nxt)
            for r, v in zip(nxt.itertuples(), pred):
                report["numbers"].setdefault(r.qb, {})[market] = round(float(v), 2)
                if k == 0:
                    report["starters"][r.posteam] = {
                        "qb": names.get(r.qb, r.qb), "share_of_last_17": round(r.q_share, 2),
                        "games_since_his_last_start": int(r.q_gap),
                        market: round(float(v), 1)} | report["starters"].get(r.posteam, {})
    waiting = [f"{names.get(p, p)} {v.get('pass_att', '-')}" for team in sorted(by)
               for p in by[team] if p in gone and p in report["numbers"]
               for v in [report["numbers"][p]]]
    if waiting:
        print("    listed out, with attempts ready for the week they are back: "
              + ", ".join(waiting))
    for team, s in sorted(report["starters"].items()):
        back = (" (the usual starter, back after another man started)"
                if s["games_since_his_last_start"] and s["share_of_last_17"] >= 0.5 else "")
        print(f"    {team}: {s['qb']}, started {s['share_of_last_17']:.0%} of the last 17"
              f"{back}; attempts {s.get('pass_att', '-')}, completions "
              f"{s.get('pass_completions', '-')}")
    if os.getenv("FOOTBALL_VOLUME_DRY") == "1":
        print("football volume: FOOTBALL_VOLUME_DRY=1, so nothing is written")
        return
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / NAME).write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"football volume: numbers for {len(report['numbers'])} quarterbacks written")


def published(artifact_dir) -> dict:
    """{player id: {market: number}} for the board builder, or {} if there is
    nothing fresh. A file more than two days old is last week's games."""
    if os.getenv("FOOTBALL_VOLUME", "1") == "0":
        return {}
    try:
        rep = json.loads((Path(artifact_dir) / NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if time.time() - float(rep.get("built_at", 0)) > 48 * 3600:
        return {}
    return rep.get("numbers", {})


def recentre(ladder: dict, number: float) -> dict:
    """The range with its median on the number, keeping its shape. Attempts and
    completions are close to symmetric, so the middle game is the average."""
    mid = ladder.get(0.50) if ladder else None
    if not mid or mid <= 0 or not number or number <= 0:
        return ladder
    return {q: v * (number / mid) for q, v in ladder.items()}


SHADOW = """
CREATE TABLE IF NOT EXISTS shadow_numbers (
    model TEXT, player_id TEXT, player_name TEXT, market_code TEXT, game_date DATE,
    line FLOAT, number FLOAT, side TEXT, board_number FLOAT, board_side TEXT,
    gap FLOAT, built_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (model, player_id, market_code, game_date)
)
"""


def shadow():
    """This model's carries number beside every priced running back. Graded in
    eval_rules.py; nothing on the board changes."""
    from sqlalchemy import text
    eng = engine()
    with eng.connect() as conn:
        games, people, gone, chart = world(conn)
        board = pd.read_sql(text(
            "SELECT DISTINCT ON (player_name) player_name, line, projection, "
            "recommended_side, home_team, away_team, commence_time FROM prop_edges "
            "WHERE market_code = 'rush_att' ORDER BY player_name, line"), conn)
    if games.empty or board.empty:
        print("football volume shadow: no games or no priced carries, nothing written")
        return
    season = int(games["season"].max())
    d, pos = load(range(season - 4, season + 1))
    base = team_games(d, pos)
    rows = week_rows(games)
    by = candidates(rows, people, gone, chart, features(base))
    t = features(base, [dict(r, qb=by[r["posteam"]][0]) for r in rows if by[r["posteam"]]])
    rbs = people[people["position"] == "RB"].copy()
    rbs["team"] = rbs["team"].map(code)
    who = rbs.sort_values("status").drop_duplicates("name").set_index("name")
    nxt = t[t["future"] == 1].set_index("posteam")
    future = []
    for r in board.itertuples():
        if r.player_name not in who.index:
            continue
        p = who.loc[r.player_name]
        if p["team"] not in nxt.index:
            continue
        g = nxt.loc[p["team"]]
        future.append({"pid": p["external_id"], "posteam": p["team"], "defteam": g["defteam"],
                       "season": g["season"], "week": g["week"], "game_id": g["game_id"],
                       "game_date": g["game_date"]})
    if not future:
        print("football volume shadow: no priced back matched a game, nothing written")
        return
    active = {team: set(x.loc[~x["external_id"].isin(gone), "external_id"])
              for team, x in rbs.groupby("team")}
    b = backs(d, pos, t, future, active)
    hist_t = t[t["future"] == 0]
    oof = held_out(hist_t, "rb_runs", TEAM_RUNS)[["game_id", "posteam", "pred"]] \
        .rename(columns={"pred": "p_runs"})
    nxt_runs = nxt.reset_index()[["game_id", "posteam"]].assign(
        p_runs=fit(hist_t[hist_t["rb_runs"].notna()], TEAM_RUNS, "rb_runs")(nxt.reset_index()))
    keep = list(dict.fromkeys(["game_id", "posteam"] + [c for c in t.columns if c in RB]))
    b = b.merge(t[keep], on=["game_id", "posteam"], how="left") \
         .merge(pd.concat([oof, nxt_runs]), on=["game_id", "posteam"], how="left")
    b["p_runs"] = b["p_runs"].fillna(b["o_rb_runs_8"])
    b["exp_car"] = b["p_runs"] * b["lifted"]
    model = fit(b[(b["future"] == 0) & b["car"].notna()], RB, "car")
    y = b[b["future"] == 1].copy()
    y["number"] = model(y)
    ids = who["external_id"].to_dict()
    board["pid"] = board["player_name"].map(ids)
    y = y.merge(board, on="pid")
    with eng.begin() as conn:
        conn.execute(text(SHADOW))
        for r in y.itertuples():
            side = "over" if r.number > float(r.line) else "under"
            conn.execute(text(
                "INSERT INTO shadow_numbers (model, player_id, player_name, market_code, "
                "game_date, line, number, side, board_number, board_side, gap) VALUES "
                "('football_v1', :p, :n, 'rush_att', :d, :l, :x, :s, :bn, :bs, :g) "
                "ON CONFLICT (model, player_id, market_code, game_date) DO UPDATE SET "
                "line = EXCLUDED.line, number = EXCLUDED.number, side = EXCLUDED.side, "
                "board_number = EXCLUDED.board_number, board_side = EXCLUDED.board_side, "
                "gap = EXCLUDED.gap, built_at = NOW()"),
                {"p": r.pid, "n": r.player_name, "d": pd.Timestamp(r.game_date).date(),
                 "l": float(r.line), "x": float(r.number), "s": side,
                 "bn": float(r.projection) if r.projection is not None else None,
                 "bs": r.recommended_side,
                 "g": (float(r.number) - float(r.line)) / max(float(r.line), 0.5)})
    for r in y.sort_values("vacated", ascending=False).itertuples():
        left = f", {r.vacated:.0%} of the carries left behind" if r.vacated >= 0.15 else ""
        print(f"    {r.player_name}: line {float(r.line):.1f}, this model {r.number:.1f}, "
              f"the board {float(r.projection):.1f}{left}")
    print(f"football volume shadow: carries for {len(y)} backs written, nothing on the "
          "board changed")


if __name__ == "__main__":
    shadow() if "shadow" in sys.argv[1:] else numbers()
