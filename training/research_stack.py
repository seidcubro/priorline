"""Can anything pick sides better than the board does, on lines it has never seen?

The whole question in one bench. Every play since 2022 becomes a row a player
a game, with what was known before kickoff: his volume and efficiency over his
last four and eight games, his share of the offence, his explosive plays and
success rate, his red-zone work, a quarterback's dropbacks and sacks, how his
team calls plays, what the defence across from him has allowed to his position,
the spread and total, the roof, the wind.

Those are joined to every prop the site has graded (pull_record.py), which has
the real line, the price, the number we published and what happened. A model
then learns "did it go over" on 2025 and is scored on 2026, a week at a time,
only ever having seen the weeks before. Three things are compared on exactly
the same props:

    published     the side the board took
    favourite     the side the price says is likelier. No model at all. This is
                  the bar: anything that cannot beat the book's own favourite
                  has learned nothing.
    stack         the model here, which is given the line, the price and our
                  published number as well as every feature above

and each is read two ways, because they answer different questions:

    right         how often the side came in
    value         that, minus what the prices needed

for all picks, and for the top fifth by the model's own confidence, which is
what an elite tier is.

    python lab/training/pull_record.py
    python lab/training/research_stack.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")
CACHE = Path(__file__).resolve().parents[2] / "marketing" / ".cache"
SEASONS = (2022, 2023, 2024, 2025, 2026)
STAT = {"recs": "rec", "rec_yds": "rec_yds", "rush_att": "car", "rush_yds": "rush_yds",
        "pass_att": "att", "pass_completions": "cmp", "pass_yds": "pass_yds",
        "pass_td": "pass_td"}
VIG = 1.045


def plays():
    cols = ["game_id", "season", "week", "season_type", "game_date", "posteam", "defteam",
            "home_team", "spread_line", "total_line", "roof", "wind", "temp",
            "pass_attempt", "rush_attempt", "sack", "qb_scramble", "qb_kneel", "qb_dropback",
            "complete_pass", "yards_gained", "air_yards", "yards_after_catch",
            "yardline_100", "pass_touchdown", "rush_touchdown", "success", "epa",
            "first_down", "two_point_attempt", "passer_player_id", "rusher_player_id",
            "receiver_player_id", "interception"]
    out, pos = [], {}
    for s in SEASONS:
        d = pd.read_csv(CACHE / f"pbp_{s}.csv.gz", usecols=cols, low_memory=False)
        out.append(d[(d["season_type"] == "REG") & (d["two_point_attempt"] != 1)])
        r = pd.read_csv(CACHE / f"roster_{s}.csv", usecols=["gsis_id", "position"],
                        low_memory=False).dropna()
        pos.update(r.drop_duplicates("gsis_id").set_index("gsis_id")["position"].to_dict())
    d = pd.concat(out, ignore_index=True)
    d["game_date"] = pd.to_datetime(d["game_date"])
    return d, pos


def player_games(d, pos):
    """One row a player a game: everything he did, counted."""
    key = ["season", "week", "game_id", "game_date", "posteam", "defteam"]
    th = d[(d["pass_attempt"] == 1) & (d["sack"] != 1)]
    tg = th[th["receiver_player_id"].notna()]
    rec = tg.groupby(key + ["receiver_player_id"]).agg(
        tgt=("pass_attempt", "size"), rec=("complete_pass", "sum"),
        rec_yds=("yards_gained", "sum"), air=("air_yards", "sum"),
        yac=("yards_after_catch", "sum"),
        rz_tgt=("yardline_100", lambda v: (v <= 20).sum()),
        x_rec=("yards_gained", lambda v: (v >= 20).sum()),
        rec_succ=("success", "sum"), rec_epa=("epa", "sum"),
        rec_fd=("first_down", "sum")).reset_index().rename(
        columns={"receiver_player_id": "pid"})
    ru = d[(d["rush_attempt"] == 1) & d["rusher_player_id"].notna() & (d["qb_kneel"] != 1)]
    rush = ru.groupby(key + ["rusher_player_id"]).agg(
        car=("rush_attempt", "size"), rush_yds=("yards_gained", "sum"),
        rz_car=("yardline_100", lambda v: (v <= 20).sum()),
        x_rush=("yards_gained", lambda v: (v >= 10).sum()),
        rush_succ=("success", "sum"), rush_epa=("epa", "sum"),
        scr=("qb_scramble", "sum")).reset_index().rename(
        columns={"rusher_player_id": "pid"})
    db = d[(d["qb_dropback"] == 1) & d["passer_player_id"].notna()]
    qb = db.groupby(key + ["passer_player_id"]).agg(
        dropbacks=("qb_dropback", "size"), sacks=("sack", "sum")).reset_index().rename(
        columns={"passer_player_id": "pid"})
    pa = th[th["passer_player_id"].notna()].groupby(key + ["passer_player_id"]).agg(
        att=("pass_attempt", "size"), cmp=("complete_pass", "sum"),
        pass_yds=("yards_gained", "sum"), pass_td=("pass_touchdown", "sum"),
        pass_air=("air_yards", "sum"), ints=("interception", "sum"),
        rz_att=("yardline_100", lambda v: (v <= 20).sum()),
        x_pass=("yards_gained", lambda v: (v >= 20).sum()),
        pass_epa=("epa", "sum")).reset_index().rename(columns={"passer_player_id": "pid"})
    g = rec.merge(rush, on=key + ["pid"], how="outer").merge(
        qb, on=key + ["pid"], how="outer").merge(pa, on=key + ["pid"], how="outer").fillna(0)
    g["pos"] = g["pid"].map(pos)
    team = d.groupby(key).agg(
        t_plays=("epa", "size"), t_pass=("pass_attempt", "sum"), t_rush=("rush_attempt", "sum"),
        t_succ=("success", "mean")).reset_index()
    t_tg = tg.groupby(key).agg(t_tgt=("pass_attempt", "size"), t_air=("air_yards", "sum")).reset_index()
    t_car = ru.groupby(key).agg(t_car=("rush_attempt", "size")).reset_index()
    g = g.merge(team, on=key).merge(t_tg, on=key, how="left").merge(t_car, on=key, how="left")
    g["tgt_share"] = g["tgt"] / g["t_tgt"].clip(lower=1)
    g["air_share"] = g["air"] / g["t_air"].clip(lower=1)
    g["car_share"] = g["car"] / g["t_car"].clip(lower=1)
    g["pass_rate"] = g["t_pass"] / g["t_plays"].clip(lower=1)
    info = d.drop_duplicates("game_id")[["game_id", "home_team", "spread_line", "total_line",
                                         "roof", "wind", "temp"]]
    g = g.merge(info, on="game_id")
    g["fav"] = np.where(g["posteam"] == g["home_team"], g["spread_line"], -g["spread_line"])
    g["home"] = (g["posteam"] == g["home_team"]).astype(int)
    g["indoor"] = g["roof"].isin(["dome", "closed"]).astype(int)
    g["wind"] = np.where(g["indoor"] == 1, 0, g["wind"].fillna(5))
    g["temp"] = np.where(g["indoor"] == 1, 70, g["temp"].fillna(60))
    return g


BASE = ["tgt", "rec", "rec_yds", "air", "yac", "rz_tgt", "x_rec", "rec_succ", "rec_epa",
        "rec_fd", "car", "rush_yds", "rz_car", "x_rush", "rush_succ", "rush_epa", "scr",
        "dropbacks", "sacks", "att", "cmp", "pass_yds", "pass_td", "pass_air", "ints",
        "rz_att", "x_pass", "pass_epa", "tgt_share", "air_share", "car_share", "pass_rate",
        "t_plays", "t_succ"]


def lagged(g):
    """What he had done before each game: last four, last eight, and the last one."""
    g = g.sort_values(["pid", "game_date"]).reset_index(drop=True)
    grp = g.groupby("pid", sort=False)
    out = {}
    for c in BASE:
        s = grp[c]
        out[f"{c}_4"] = s.transform(lambda v: v.shift(1).rolling(4, min_periods=1).mean())
        out[f"{c}_8"] = s.transform(lambda v: v.shift(1).rolling(8, min_periods=1).mean())
    for c in ("tgt", "rec", "rec_yds", "car", "rush_yds", "att", "cmp", "pass_yds", "pass_td"):
        out[f"{c}_last"] = grp[c].shift(1)
        out[f"{c}_sd8"] = grp[c].transform(lambda v: v.shift(1).rolling(8, min_periods=3).std())
    out["games"] = grp.cumcount()
    out["rest"] = grp["game_date"].diff().dt.days
    return pd.concat([g, pd.DataFrame(out)], axis=1)


def defence(g):
    """What each defence had allowed before each game, to each position."""
    rows = []
    for p in ("WR", "TE", "RB", "QB"):
        x = g[g["pos"] == p].groupby(["defteam", "game_id", "game_date"]).agg(
            a_rec=("rec", "sum"), a_rec_yds=("rec_yds", "sum"), a_tgt=("tgt", "sum"),
            a_car=("car", "sum"), a_rush_yds=("rush_yds", "sum"), a_pass_yds=("pass_yds", "sum"),
            a_pass_td=("pass_td", "sum"), a_att=("att", "sum"), a_cmp=("cmp", "sum"),
            a_x=("x_rec", "sum"), a_xr=("x_rush", "sum"), a_sacks=("sacks", "sum")).reset_index()
        x = x.sort_values(["defteam", "game_date"])
        cols = [c for c in x.columns if c.startswith("a_")]
        for c in cols:
            x[f"d_{c[2:]}"] = x.groupby("defteam")[c].transform(
                lambda v: v.shift(1).rolling(6, min_periods=2).mean())
        x["pos"] = p
        rows.append(x[["defteam", "game_id", "pos"] + [f"d_{c[2:]}" for c in cols]])
    return pd.concat(rows, ignore_index=True)


def bench():
    d, pos = plays()
    g = lagged(player_games(d, pos))
    g = g.merge(defence(g), on=["defteam", "game_id", "pos"], how="left")
    rec = pd.read_pickle(CACHE / "record.pkl")
    for c in ("line", "actual", "projection", "win_prob"):
        rec[c] = rec[c].astype(float)
    rec = rec[rec["season"].isin([2025, 2026]) & (rec["actual"] != rec["line"])
              & rec["market_code"].isin(STAT)].copy()
    rec["game_date"] = pd.to_datetime(rec["game_date"])
    x = rec.merge(g, left_on=["gsis", "game_date"], right_on=["pid", "game_date"],
                  how="inner", suffixes=("", "_g"))
    # The market's own stat, named the same for every market.
    for suf in ("_4", "_8", "_last", "_sd8"):
        x["m" + suf] = [r[f"{STAT[m]}{suf}"] for m, r in zip(x["market_code"], x.to_dict("records"))]
    x["line_vs_8"] = x["line"] / x["m_8"].clip(lower=0.5)
    x["line_vs_4"] = x["line"] / x["m_4"].clip(lower=0.5)
    x["proj_gap"] = (x["projection"] - x["line"]) / x["line"].clip(lower=0.5)
    x["trend"] = (x["m_4"] - x["m_8"]) / x["m_8"].clip(lower=0.5)
    we_over = x["recommended_side"] == "over"
    x["p_over_ours"] = np.where(we_over, x["win_prob"], 1 - x["win_prob"])
    x["need_over"] = np.where(we_over, x["need"], VIG - x["need"])
    x["need_under"] = VIG - x["need_over"]
    x["we_over"] = we_over.astype(int)
    for m in STAT:
        x[f"is_{m}"] = (x["market_code"] == m).astype(int)
    for p in ("QB", "RB", "WR", "TE"):
        x[f"pos_{p}"] = (x["position"] == p).astype(int)
    return x


def features(x):
    drop = {"over", "hit", "actual", "edge_id", "player_id", "season", "week", "need",
            "win_prob", "price_american"}
    num = [c for c in x.columns if c not in drop and pd.api.types.is_numeric_dtype(x[c])
           and not c.endswith("_g")]
    # Nothing from the game itself: only what was lagged or known before it.
    live = set(BASE) | {"t_tgt", "t_air", "t_car", "t_pass", "t_rush"}
    return [c for c in num if c not in live]


def models():
    return {
        "boosted": lambda: HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.04, max_iter=220, min_samples_leaf=60,
            l2_regularization=2.0, random_state=0),
        "linear": lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                        LogisticRegression(C=0.05, max_iter=2000)),
    }


def walk(x, cols):
    """2026 a week at a time, each week predicted from everything before it."""
    test = x[x["season"] == 2026]
    out = []
    for wk in sorted(test["week"].unique()):
        tr = x[(x["season"] == 2025) | ((x["season"] == 2026) & (x["week"] < wk))]
        te = test[test["week"] == wk]
        if len(te) < 5:
            continue
        preds = {}
        for name, make in models().items():
            m = make().fit(tr[cols], tr["over"].astype(int))
            preds[name] = m.predict_proba(te[cols])[:, 1]
        te = te.assign(**{f"p_{k}": v for k, v in preds.items()})
        te["p_stack"] = np.mean(list(preds.values()), axis=0)
        out.append(te)
    return pd.concat(out, ignore_index=True)


def swap(x, cols, market=None):
    """The check on a result: learn on 2026 and score 2025, which is the other
    way round and shares nothing with the walk above. A finding that is real
    shows up in both. With `market`, only that market's props are used."""
    if market:
        x = x[x["market_code"] == market]
    tr, te = x[x["season"] == 2026], x[x["season"] == 2025].copy()
    preds = {}
    for name, make in models().items():
        m = make().fit(tr[cols], tr["over"].astype(int))
        preds[name] = m.predict_proba(te[cols])[:, 1]
    te = te.assign(**{f"p_{k}": v for k, v in preds.items()})
    te["p_stack"] = np.mean(list(preds.values()), axis=0)
    return te


def score(t, over, name):
    over = np.asarray(over, dtype=bool)
    right = np.where(over, t["over"], 1 - t["over"])
    need = np.where(over, t["need_over"], t["need_under"])
    return f"{name:<34}{len(t):>6}{right.mean():>9.1%}{need.mean():>9.1%}{(right.mean() - need.mean()):>+9.1%}"


def report(t, title):
    print(f"\n{title}")
    print(f"{'':<34}{'picks':>6}{'right':>9}{'needed':>9}{'value':>9}")
    print(score(t, t["we_over"] == 1, "published side"))
    print(score(t, t["need_over"] > t["need_under"], "the book's favourite"))
    for k in ("boosted", "linear", "stack"):
        print(score(t, t[f"p_{k}"] > 0.5, f"model: {k}"))
    p = t["p_stack"].to_numpy()
    conf = np.maximum(p, 1 - p)
    edge = np.where(p > 0.5, p - t["need_over"], (1 - p) - t["need_under"])
    for label, rank in (("top fifth by confidence", conf), ("top fifth by value", edge)):
        cut = np.quantile(rank, 0.8)
        s = t[rank >= cut]
        print(score(s, s["p_stack"] > 0.5, f"stack, {label}"))
    fav = np.maximum(t["need_over"], t["need_under"])
    s = t[fav >= np.quantile(fav, 0.8)]
    print(score(s, s["need_over"] > s["need_under"], "favourite, top fifth by price"))
    pub = t["win_prob"].to_numpy()
    s = t[pub >= np.quantile(pub, 0.8)]
    print(score(s, s["we_over"] == 1, "published, top fifth by chance"))


def main():
    x = bench()
    cols = features(x)
    print(f"{len(x):,} graded props matched to play-by-play "
          f"({(x['season'] == 2025).sum():,} in 2025, {(x['season'] == 2026).sum():,} in 2026), "
          f"{len(cols)} features")
    t = walk(x, cols)
    t.to_pickle(CACHE / "stack_2026.pkl")
    report(t, "2026, every market, each week predicted from the weeks before it")
    for m in STAT:
        s = t[t["market_code"] == m]
        if len(s) >= 150:
            report(s, f"2026, {m}")
    back = swap(x, cols)
    report(back, "THE OTHER WAY: learned on 2026, scored on 2025, every market")
    for m in ("rec_yds", "recs"):
        report(back[back["market_code"] == m], f"the other way, {m}, model trained on every market")
        report(swap(x, cols, m), f"the other way, {m}, model trained on {m} alone")


if __name__ == "__main__":
    sys.exit(main())
