"""Fit the role-regime correction for running backs, and write it only if it holds up.

See role_regime.py for what it does and why. This measures it on the database:
every running back's regular-season game from 2022 on, with his eight-game
window and his regime level worked out from what was known before the game.

K, the number of games' worth of window mixed into a short regime, is fitted on
every season before the last two and scored on those two, one at a time, so the
score is on games the fit never saw.

It is written as accepted only if, in BOTH held-out seasons, on the games where
a back's level differs from his window by GAP carries or more:

    the error in carries is lower than the window's, and
    the error in rushing yards, at his own yards a carry, is no worse.

Those are the games the correction changes. Scoring it on every game would
bury it under the nine in ten it leaves alone, and scoring it only on backs
stepping up would hide whether it is right about backs stepping back.

    python fit_role_regime.py            # prints, writes nothing
    WRITE=1 python fit_role_regime.py    # writes artifacts/role_regime.json
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

import role_regime as rr

DATABASE_URL = os.getenv("DATABASE_URL") or (
    f"postgresql+psycopg2://{os.getenv('POSTGRES_USER', 'app')}:"
    f"{os.getenv('POSTGRES_PASSWORD', 'app')}"
    f"@{os.getenv('POSTGRES_HOST', 'postgres')}:"
    f"{os.getenv('POSTGRES_PORT', '5432')}/{os.getenv('POSTGRES_DB', 'app')}"
)
ARTIFACTS = Path(os.getenv("ARTIFACT_DIR", "/artifacts"))
GRID = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
MIN_GAMES = 20


def mae(a, b):
    return float(np.mean(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))))


def changed(d, k):
    d = d.assign(fix=rr.blend(d, k))
    return d[~d["same"] & ((d["fix"] - d["window"]).abs() >= rr.GAP)]


def fit(d: pd.DataFrame):
    seasons = sorted(d["season"].unique())
    hold = seasons[-2:]
    train = d[~d["season"].isin(hold)]
    aff = train[~train["same"]]
    scores = {k: mae(aff["actual"], rr.blend(aff, k)) for k in GRID}
    k = min(scores, key=scores.get)
    print(f"{len(d):,} back-games, {seasons[0]} to {seasons[-1]}. K fitted on "
          f"{len(aff):,} games before {hold[0]} where the backfield had changed: "
          + ", ".join(f"{g:g}: {m:.2f}" for g, m in scores.items()) + f". K = {k:g}.")
    ok, table = True, []
    print(f"\n  {'held-out season':<18}{'':<24}{'games':>6}{'window':>9}{'level':>9}"
          f"{'yds window':>12}{'yds level':>11}")
    for s in hold:
        c = changed(d[d["season"] == s], k)
        for name, x in (("all that change", c),
                        ("stepping up", c[c["fix"] > c["window"]]),
                        ("stepping back", c[c["fix"] < c["window"]])):
            if len(x) < MIN_GAMES:
                print(f"  {s:<18}{name:<24}{len(x):>6}   too few to read")
                if name == "all that change":
                    ok = False
                continue
            a, b = mae(x["actual"], x["window"]), mae(x["actual"], x["fix"])
            ya = mae(x["actual_yds"], x["window_yds"])
            yb = mae(x["actual_yds"], x["fix"] * x["ypc"])
            print(f"  {s:<18}{name:<24}{len(x):>6}{a:>9.2f}{b:>9.2f}{ya:>12.1f}{yb:>11.1f}")
            if name == "all that change":
                table.append({"season": int(s), "games": int(len(x)), "carries_window": a,
                              "carries_level": b, "yards_window": ya, "yards_level": yb})
                if not (b < a and yb <= ya):
                    ok = False
            else:
                print(f"  {'':<18}{'':<24}{'':>6}  window said {x['window'].mean():.1f}, "
                      f"level {x['fix'].mean():.1f}, he got {x['actual'].mean():.1f}")
    return {"accepted": bool(ok), "k": float(k), "gap": rr.GAP, "holdout": table}


def main():
    engine = create_engine(DATABASE_URL)
    with engine.connect() as conn:
        d = pd.read_sql(text(
            """
            SELECT player_id, season, week, team,
                   COALESCE(carries, 0) AS carries,
                   COALESCE(rushing_yards, 0) AS rushing_yards
            FROM player_game_stats_app
            WHERE season_type = 'REG' AND team IS NOT NULL AND season >= 2022
              AND position IN ('RB', 'FB')
            """), conn)
    if d.empty:
        print("no back-games to fit on")
        return
    out = fit(rr.backtest(d))
    print("\n" + ("ACCEPTED: the level beat the window in both held-out seasons."
                  if out["accepted"] else
                  "NOT ACCEPTED: it did not beat the window in both held-out seasons, "
                  "so nothing is applied."))
    if os.getenv("WRITE") == "1":
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        (ARTIFACTS / rr.ARTIFACT).write_text(json.dumps(out, indent=1), encoding="utf-8")
        print(f"written to {ARTIFACTS / rr.ARTIFACT}")
    else:
        print("WRITE=1 to write it.")


if __name__ == "__main__":
    main()
