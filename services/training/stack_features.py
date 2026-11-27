"""Every play since 2022 as one row a player a game, with what was known before it.

The features under stack_elite.py. For each player and game: his volume and
efficiency over his last four and eight games, his share of his team's targets,
air yards and carries, his explosive plays, his success rate and his red-zone
work, a quarterback's dropbacks and sacks, how his team calls plays, and what
the defence across from him had allowed to his position over its last six.

Everything is lagged: a row never holds anything from its own game. A game
that has not been played is passed in as an empty row and comes back with the
same features, which is how the board's picks are scored.

Play-by-play and rosters come from nflverse through pick_depth's cache.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import pick_depth as depth

COLS = ["game_id", "season", "week", "season_type", "game_date", "posteam", "defteam",
        "home_team", "pass_attempt", "rush_attempt", "sack", "qb_scramble", "qb_kneel",
        "qb_dropback", "complete_pass", "yards_gained", "air_yards", "yards_after_catch",
        "yardline_100", "pass_touchdown", "success", "epa", "first_down",
        "two_point_attempt", "passer_player_id", "rusher_player_id", "receiver_player_id",
        "interception"]
KEY = ["season", "week", "game_id", "game_date", "posteam", "defteam", "home_team"]
BASE = ["tgt", "rec", "rec_yds", "air", "yac", "rz_tgt", "x_rec", "rec_succ", "rec_epa",
        "rec_fd", "car", "rush_yds", "rz_car", "x_rush", "rush_succ", "rush_epa", "scr",
        "dropbacks", "sacks", "att", "cmp", "pass_yds", "pass_td", "pass_air", "ints",
        "rz_att", "x_pass", "pass_epa", "tgt_share", "air_share", "car_share", "pass_rate",
        "t_plays", "t_succ"]
STAT = {"recs": "rec", "rec_yds": "rec_yds"}
ALLOWED = ["rec", "rec_yds", "tgt", "car", "rush_yds", "pass_yds", "pass_td", "att", "cmp",
           "x_rec", "x_rush", "sacks"]


def plays(seasons):
    out, pos = [], {}
    for s in seasons:
        try:
            f = depth.fetch(f"pbp_{s}.csv.gz", f"{depth.BASE}/pbp/play_by_play_{s}.csv.gz")
            r = depth.fetch(f"roster_{s}.csv", f"{depth.BASE}/rosters/roster_{s}.csv")
        except Exception as err:
            print(f"  no play-by-play for {s}: {err}")
            continue
        d = pd.read_csv(f, usecols=COLS, low_memory=False)
        out.append(d[(d["season_type"] == "REG") & (d["two_point_attempt"] != 1)])
        ros = pd.read_csv(r, usecols=["gsis_id", "position"], low_memory=False).dropna()
        pos.update(ros.drop_duplicates("gsis_id").set_index("gsis_id")["position"].to_dict())
    d = pd.concat(out, ignore_index=True)
    d["game_date"] = pd.to_datetime(d["game_date"])
    return d, pos


def player_games(d, pos):
    """One row a player a game: everything he did, counted."""
    th = d[(d["pass_attempt"] == 1) & (d["sack"] != 1)]
    tg = th[th["receiver_player_id"].notna()]
    red = lambda v: (v <= 20).sum()  # noqa: E731
    rec = tg.groupby(KEY + ["receiver_player_id"]).agg(
        tgt=("pass_attempt", "size"), rec=("complete_pass", "sum"),
        rec_yds=("yards_gained", "sum"), air=("air_yards", "sum"),
        yac=("yards_after_catch", "sum"), rz_tgt=("yardline_100", red),
        x_rec=("yards_gained", lambda v: (v >= 20).sum()),
        rec_succ=("success", "sum"), rec_epa=("epa", "sum"),
        rec_fd=("first_down", "sum")).reset_index().rename(
        columns={"receiver_player_id": "pid"})
    ru = d[(d["rush_attempt"] == 1) & d["rusher_player_id"].notna() & (d["qb_kneel"] != 1)]
    rush = ru.groupby(KEY + ["rusher_player_id"]).agg(
        car=("rush_attempt", "size"), rush_yds=("yards_gained", "sum"),
        rz_car=("yardline_100", red), x_rush=("yards_gained", lambda v: (v >= 10).sum()),
        rush_succ=("success", "sum"), rush_epa=("epa", "sum"),
        scr=("qb_scramble", "sum")).reset_index().rename(columns={"rusher_player_id": "pid"})
    db = d[(d["qb_dropback"] == 1) & d["passer_player_id"].notna()]
    qb = db.groupby(KEY + ["passer_player_id"]).agg(
        dropbacks=("qb_dropback", "size"), sacks=("sack", "sum")).reset_index().rename(
        columns={"passer_player_id": "pid"})
    pa = th[th["passer_player_id"].notna()].groupby(KEY + ["passer_player_id"]).agg(
        att=("pass_attempt", "size"), cmp=("complete_pass", "sum"),
        pass_yds=("yards_gained", "sum"), pass_td=("pass_touchdown", "sum"),
        pass_air=("air_yards", "sum"), ints=("interception", "sum"),
        rz_att=("yardline_100", red), x_pass=("yards_gained", lambda v: (v >= 20).sum()),
        pass_epa=("epa", "sum")).reset_index().rename(columns={"passer_player_id": "pid"})
    on = KEY + ["pid"]
    g = rec.merge(rush, on=on, how="outer").merge(qb, on=on, how="outer") \
           .merge(pa, on=on, how="outer").fillna(0)
    team = d.groupby(KEY).agg(t_plays=("epa", "size"), t_pass=("pass_attempt", "sum"),
                              t_succ=("success", "mean")).reset_index()
    t_tg = tg.groupby(KEY).agg(t_tgt=("pass_attempt", "size"),
                               t_air=("air_yards", "sum")).reset_index()
    t_car = ru.groupby(KEY).agg(t_car=("rush_attempt", "size")).reset_index()
    g = g.merge(team, on=KEY).merge(t_tg, on=KEY, how="left").merge(t_car, on=KEY, how="left")
    g["tgt_share"] = g["tgt"] / g["t_tgt"].clip(lower=1)
    g["air_share"] = g["air"] / g["t_air"].clip(lower=1)
    g["car_share"] = g["car"] / g["t_car"].clip(lower=1)
    g["pass_rate"] = g["t_pass"] / g["t_plays"].clip(lower=1)
    g["pos"] = g["pid"].map(pos)
    g["home"] = (g["posteam"] == g["home_team"]).astype(int)
    return g


def upcoming(rows, pos):
    """Empty rows for games not yet played, shaped like player_games' rows.

    `rows` has pid, posteam, defteam, game_date, home (1 or 0), game_id.
    """
    u = pd.DataFrame(rows)
    if u.empty:
        return u
    u["game_date"] = pd.to_datetime(u["game_date"])
    u["pos"] = u["pid"].map(pos)
    u["future"] = 1
    return u


def lagged(g):
    """What he had done before each game: last four, last eight, the last one."""
    g = g.sort_values(["pid", "game_date"]).reset_index(drop=True)
    grp = g.groupby("pid", sort=False)
    out = {}
    for c in BASE:
        s = grp[c]
        out[f"{c}_4"] = s.transform(lambda v: v.shift(1).rolling(4, min_periods=1).mean())
        out[f"{c}_8"] = s.transform(lambda v: v.shift(1).rolling(8, min_periods=1).mean())
    for c in ("tgt", "rec", "rec_yds", "car", "rush_yds"):
        out[f"{c}_last"] = grp[c].shift(1)
        out[f"{c}_sd8"] = grp[c].transform(
            lambda v: v.shift(1).rolling(8, min_periods=3).std())
    out["games"] = grp.cumcount()
    out["rest"] = grp["game_date"].diff().dt.days
    return pd.concat([g, pd.DataFrame(out)], axis=1)


def defence(g):
    """What each defence had allowed before each game, to each position."""
    rows = []
    for p in ("WR", "TE", "RB", "QB"):
        x = g[g["pos"] == p].groupby(["defteam", "game_id", "game_date"])[ALLOWED] \
            .sum(min_count=1).reset_index().sort_values(["defteam", "game_date"])
        for c in ALLOWED:
            x[f"d_{c}"] = x.groupby("defteam")[c].transform(
                lambda v: v.shift(1).rolling(6, min_periods=2).mean())
        x["pos"] = p
        rows.append(x[["defteam", "game_id", "pos"] + [f"d_{c}" for c in ALLOWED]])
    return pd.concat(rows, ignore_index=True)


def table(seasons, future=None):
    """Player-games with their lagged features, and the same for `future` rows."""
    d, pos = plays(seasons)
    g = player_games(d, pos)
    g["future"] = 0
    if future:
        u = upcoming(future, pos)
        if len(u):
            g = pd.concat([g, u], ignore_index=True)
    g = lagged(g)
    return g.merge(defence(g), on=["defteam", "game_id", "pos"], how="left")


def for_props(x):
    """The columns that depend on the prop: its line against his own numbers."""
    x = x.copy()
    for suf in ("_4", "_8", "_last", "_sd8"):
        x["m" + suf] = np.where(x["market_code"] == "recs", x[f"rec{suf}"], x[f"rec_yds{suf}"])
    x["line_vs_8"] = x["line"] / x["m_8"].clip(lower=0.5)
    x["line_vs_4"] = x["line"] / x["m_4"].clip(lower=0.5)
    x["proj_gap"] = (x["projection"] - x["line"]) / x["line"].clip(lower=0.5)
    x["trend"] = (x["m_4"] - x["m_8"]) / x["m_8"].clip(lower=0.5)
    we_over = x["recommended_side"] == "over"
    p = x["price_american"].astype(float)
    need = np.where(p > 0, 100 / (p + 100), -p / (-p + 100))
    x["we_over"] = we_over.astype(int)
    x["p_over_ours"] = np.where(we_over, x["win_prob"], 1 - x["win_prob"])
    x["need_over"] = np.where(we_over, need, 1.045 - need)
    x["need_under"] = 1.045 - x["need_over"]
    x["is_recs"] = (x["market_code"] == "recs").astype(int)
    for q in ("RB", "WR", "TE"):
        x[f"pos_{q}"] = (x["pos"] == q).astype(int)
    return x


def columns(x):
    """Every lagged or pre-game number. Nothing from the game itself."""
    keep = [c for c in x.columns if c.endswith(("_4", "_8", "_last", "_sd8"))
            or c.startswith(("d_", "pos_"))]
    return keep + ["games", "rest", "home", "line", "projection", "proj_gap", "line_vs_8",
                   "line_vs_4", "trend", "we_over", "p_over_ours", "need_over",
                   "need_under", "is_recs"]
