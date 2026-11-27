"""Choose the elite catches and receiving-yards picks by a model's confidence, not the gap.

The board ranks a pick by how far our number sits from the line. On receiving
props that ranking's top tier was right 53.9% of the time in 2026. A second
model, given the line, the price, our own number and everything in
stack_features.py, does not pick sides better than the board across all props.
But the picks it is most sure of come in far more often than the rest, and
that held on both seasons, each scored by a model that had never seen it:

    most confident fifth of receiving picks   2026: 59.6% to 62.1% right
                                              2025: 58.4% to 58.5% right
    and ahead of what the prices needed by 1.2 to 2.9 points in every run

The spread is across three builds of the model. Narrower slices looked better
(68% on the top tenth of catches in one build) and did not survive a change of
features, so the fifth is what is used. The same idea was tried on rushing and
passing with their own models and found nothing.

What this does, after the board is built:

    1. learns "did it go over" from every graded receiving prop
    2. checks itself: each season is predicted by a model fitted on the others,
       and the most confident fifth has to be ahead of its price in every one.
       If it is not, nothing is changed and it says so.
    3. scores the board's receiving picks. Those at or above the confidence
       that marked the top fifth become elite. Receiving picks that were elite
       by gap and are not chosen step down to strong.

It never changes a side, a number or a price, only the tier, and never touches
a pick the board has marked. STACK_ELITE=0 turns it off; STACK_ELITE_DRY=1
prints what it would do.

    python stack_elite.py
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import pick_depth as depth
import stack_features as sf

warnings.filterwarnings("ignore")
MARKETS = ("recs", "rec_yds")
TOP = 0.80
MIN_SLICE = 80
ARTIFACTS = Path(os.getenv("ARTIFACT_DIR", "/artifacts"))


def fit(x, cols):
    y = x["over"].astype(int)
    a = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.04, max_iter=220,
                                       min_samples_leaf=60, l2_regularization=2.0,
                                       random_state=0).fit(x[cols], y)
    b = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                      LogisticRegression(C=0.05, max_iter=2000)).fit(x[cols], y)
    return a, b


def chance(models, x, cols):
    return np.mean([m.predict_proba(x[cols])[:, 1] for m in models], axis=0)


def season_of(dates):
    d = pd.to_datetime(dates)
    return np.where(d.dt.month >= 3, d.dt.year, d.dt.year - 1)


def short(team):
    return str(team or "").split()[-1] if team else ""


def run(graded, board, people):
    """graded: gsis, game_date, market_code, line, projection, recommended_side,
    win_prob, actual, price_american. board: the same without actual, plus id,
    home_team, away_team, commence_time, event_id, player_name. people: name,
    external_id, team, position. Returns (ids to make elite, report dict)."""
    g = graded[graded["market_code"].isin(MARKETS)].copy()
    for c in ("line", "projection", "win_prob", "actual", "price_american"):
        g[c] = g[c].astype(float)
    g = g[(g["actual"] != g["line"]) & g["win_prob"].notna() & g["price_american"].notna()]
    g["game_date"] = pd.to_datetime(g["game_date"])
    g["over"] = g["actual"] > g["line"]
    g["season"] = season_of(g["game_date"])
    seasons = sorted(int(s) for s in g["season"].unique())
    if len(seasons) < 2:
        return [], {"accepted": False, "why": "fewer than two graded seasons"}

    b = board[board["market_code"].isin(MARKETS)].copy()
    who = people.drop_duplicates("name").set_index("name")
    future = []
    for r in b.itertuples():
        if r.player_name not in who.index:
            continue
        p = who.loc[r.player_name]
        sides = [depth.NICK.get(short(r.away_team)), depth.NICK.get(short(r.home_team))]
        mine = depth.SAME.get(p["team"], p["team"])
        if mine not in sides or sides[0] is None or sides[1] is None:
            continue
        future.append({"pid": p["external_id"], "posteam": mine,
                       "defteam": next(s for s in sides if s != mine),
                       "game_date": pd.Timestamp(r.commence_time).tz_localize(None).normalize()
                       if pd.Timestamp(r.commence_time).tzinfo
                       else pd.Timestamp(r.commence_time).normalize(),
                       "home": int(sides[1] == mine), "game_id": f"next_{r.event_id}"})
    future = [dict(t) for t in {tuple(sorted(f.items())) for f in future}]
    T = sf.table(range(seasons[0] - 1, seasons[-1] + 1), future)

    x = g.merge(T[T["future"] == 0], left_on=["gsis", "game_date"],
                right_on=["pid", "game_date"], how="inner", suffixes=("", "_t"))
    x = sf.for_props(x)
    cols = sf.columns(x)

    # Each season predicted by a model fitted on the others.
    held, table, ok = [], [], True
    for s in seasons:
        tr, te = x[x["season"] != s], x[x["season"] == s].copy()
        if len(tr) < 500 or len(te) < 300:
            continue
        te["p"] = chance(fit(tr, cols), te, cols)
        held.append(te)
    if len(held) < 2:
        return [], {"accepted": False, "why": "not enough graded props in two seasons"}
    oof = pd.concat(held)
    oof["conf"] = np.maximum(oof["p"], 1 - oof["p"])
    cut = float(oof["conf"].quantile(TOP))
    for s, te in oof.groupby("season"):
        top = te[te["conf"] >= te["conf"].quantile(TOP)]
        over = top["p"] > 0.5
        right = float(np.where(over, top["over"], ~top["over"]).mean())
        need = float(np.where(over, top["need_over"], top["need_under"]).mean())
        pub = float(np.where(top["we_over"] == 1, top["over"], ~top["over"]).mean())
        table.append({"season": int(s), "picks": int(len(top)), "right": right,
                      "needed": need, "published_side_right": pub})
        if len(top) < MIN_SLICE or right <= need:
            ok = False
    report = {"accepted": bool(ok), "cut": cut, "held_out": table,
              "trained_on": int(len(x)), "features": len(cols)}
    if not ok or b.empty or not future:
        return [], report

    models = fit(x, cols)
    b["gsis"] = b["player_name"].map(who["external_id"].to_dict())
    b["game_id"] = "next_" + b["event_id"].astype(str)
    for c in ("line", "projection", "win_prob", "price_american"):
        b[c] = b[c].astype(float)
    y = b.merge(T[T["future"] == 1], left_on=["gsis", "game_id"], right_on=["pid", "game_id"],
                how="inner", suffixes=("", "_t"))
    if y.empty:
        return [], report
    y = sf.for_props(y)
    y["p"] = chance(models, y, cols)
    y["conf"] = np.maximum(y["p"], 1 - y["p"])
    # The model has to be sure, and sure of the side the board took: this
    # changes a tier, never a side.
    y["same_side"] = (y["p"] > 0.5) == (y["recommended_side"] == "over")
    pick = y[(y["conf"] >= cut) & y["same_side"]]
    report["scored"] = int(len(y))
    report["chosen"] = [f"{r.player_name} {r.market_code} {r.recommended_side} {r.line} "
                        f"({r.conf:.0%})" for r in pick.sort_values("conf", ascending=False)
                        .drop_duplicates(["player_name", "market_code"]).itertuples()]
    return [int(i) for i in pick["id"]], report


def main():
    from sqlalchemy import create_engine, text
    if os.getenv("STACK_ELITE", "1") == "0":
        print("stack elite: off (STACK_ELITE=0)")
        return
    url = os.getenv("DATABASE_URL") or (
        f"postgresql+psycopg2://{os.getenv('POSTGRES_USER', 'app')}:"
        f"{os.getenv('POSTGRES_PASSWORD', 'app')}@{os.getenv('POSTGRES_HOST', 'postgres')}:"
        f"{os.getenv('POSTGRES_PORT', '5432')}/{os.getenv('POSTGRES_DB', 'app')}")
    engine = create_engine(url)
    with engine.connect() as conn:
        graded = pd.read_sql(text(
            "SELECT player_id AS gsis, game_date, market_code, line, projection, "
            "recommended_side, win_prob, actual, price_american FROM prop_edge_results "
            "WHERE hit IS NOT NULL AND market_code IN ('recs', 'rec_yds') "
            "AND game_date >= '2025-08-01'"), conn)
        board = pd.read_sql(text(
            "SELECT id, event_id, player_name, market_code, line, projection, "
            "recommended_side, win_prob, price_american, home_team, away_team, "
            "commence_time, edge_tier, notes FROM prop_edges "
            "WHERE market_code IN ('recs', 'rec_yds')"), conn)
        people = pd.read_sql(text(
            "SELECT name, external_id, team, position, status FROM players "
            "WHERE external_id IS NOT NULL AND position IN ('RB', 'FB', 'WR', 'TE')"), conn)
    people["rank"] = (people["status"] != "ACT").astype(int)
    people = people.sort_values("rank")
    ids, report = run(graded, board, people)
    for row in report.get("held_out", []):
        print(f"  stack elite, {row['season']} held out: its most confident fifth was right "
              f"{row['right']:.1%} on {row['picks']} (needed {row['needed']:.1%}; the "
              f"published side on the same picks {row['published_side_right']:.1%})")
    if not report.get("accepted"):
        print("stack elite: NOT applied. " + report.get(
            "why", "its most confident fifth was not ahead of its price in every season."))
    else:
        # Never a marked pick: those are published as small on purpose.
        marked = set(board.loc[board["notes"].notna(), "id"].astype(int))
        ids = [i for i in ids if i not in marked]
        print(f"stack elite: {len(report.get('chosen', []))} receiving pick(s) at or above "
              f"{report['cut']:.0%} confidence, of {report.get('scored', 0)} scored: "
              + ("; ".join(report.get("chosen", [])) or "none this slate"))
        if os.getenv("STACK_ELITE_DRY") == "1":
            print("  STACK_ELITE_DRY=1, so the board is left as it is")
        else:
            with engine.begin() as conn:
                conn.execute(text(
                    "UPDATE prop_edges SET edge_tier = 'strong' WHERE market_code IN "
                    "('recs', 'rec_yds') AND edge_tier = 'elite' AND NOT (id = ANY(:ids))"),
                    {"ids": ids})
                if ids:
                    conn.execute(text(
                        "UPDATE prop_edges SET edge_tier = 'elite' WHERE id = ANY(:ids)"),
                        {"ids": ids})
                conn.execute(text(
                    "UPDATE prop_edges_history h SET edge_tier = e.edge_tier "
                    "FROM prop_edges e WHERE h.player_name = e.player_name "
                    "AND h.market_code = e.market_code AND h.bookmaker_key = e.bookmaker_key "
                    "AND h.commence_time = e.commence_time AND h.commence_time > NOW() "
                    "AND e.market_code IN ('recs', 'rec_yds') "
                    "AND h.edge_tier IS DISTINCT FROM e.edge_tier"))
    try:
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        (ARTIFACTS / "stack_elite.json").write_text(json.dumps(report, indent=1),
                                                    encoding="utf-8")
    except OSError:
        pass


if __name__ == "__main__":
    main()
