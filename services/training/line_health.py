"""Which teams are without a starting offensive lineman this week.

A running back's carries line does not move enough when his line is short a
starter. On every running back carries prop priced in 2025 and 2026, from week
4 on, where one of the team's five most used linemen was out by the injury
report or had already missed the last game:

    the under was right 59.1% on 303, where its prices needed 52.6%
    2025: 58.9% on 214        2026: 59.6% on 89
    ahead of its price in 16 of 22 weeks
    with every lineman available the under was right 49.6% on 474

It held whichever side the board had taken: 57.6% on the 210 where we already
said under, 62.4% on the 93 where we had said over. Rushing yards did not
follow (53.5%), so this is carries only. The first three weeks are left out
because "his usual five" is still last season's line then.

Everything here is known before kickoff. The usual five come from snap counts
through the last game played (nflverse, from Pro Football Reference). A lineman
is out if this week's report lists him Out or Doubtful, if the roster has him
on a reserve list, or if he missed the team's last game and nothing since says
he is back. A man who missed a game and then practised in full with no game
status has been cleared and is not counted: the first build flagged Trey Smith
that way on a night he was starting. Measuring it from who actually played, which is a
look at the game, read 60.9%; the pre-game version above is the honest one.
lab/training/research_absences.py has the wider search this came out of:
receivers, tight ends and defenders missing did not move any market's price.

Two more situations came out of the same search and are written here to be
graded, without touching the board (lab/training/research_situations.py):

    short line AND the opponent short in the secondary
        the back's carries under was right 64.1% on 131 (64.0% and 64.4%),
        ahead of its price in 13 of 16 weeks. The more of these that are true
        the lower the carries: with neither, the under was right 46.2% on 277.
    a different quarterback starts than started last week
        catches unders on his team were right 56.7% on 372 against 52.5%
        needed (58.6% in 2025, 53.8% in 2026). Running backs' catches most of
        all, 65.6% on 93. The catches line is set on the old quarterback.

    python line_health.py           before the board: writes the teams
    python line_health.py shadow    after the board: writes the under on every
                                    priced back on those teams to the shadow
                                    table, where eval_rules.py grades it

The board builder acts on it: a carries under on one of these teams is at
least strong, and a carries over is small. On since 27 November 2026.
LINE_OUT_RULE=0 turns it off and leaves the shadow grading running.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import football_volume as fv
import pick_depth as depth

ARTIFACTS = Path(os.getenv("ARTIFACT_DIR", "/artifacts"))
NAME = "line_health.json"
OL = {"T", "G", "C", "OL"}
DB = {"CB", "S", "DB", "FS", "SS"}
BACKTEST = {"line_out_v1": "59.1% on 303, needed 52.6%",
            "line_and_secondary_v1": "64.1% on 131, needed 52.6%",
            "new_qb_recs_v1": "56.7% on 372, needed 52.5%"}
FROM_WEEK = 4


def norm(name):
    n = re.sub(r"[^a-z ]", "", str(name).lower())
    return " ".join(w for w in n.split() if w not in ("jr", "sr", "ii", "iii", "iv", "v"))


def usual_five(season, positions=OL, col="offense_pct"):
    """Each team's five most used linemen (or defensive backs) going into its
    next game, and whether each one played in the last game.
    {team: [(pfr id, name, missed_last, week he last played this season or 0)]}"""
    frames = []
    for s in (season - 1, season):
        try:
            f = depth.fetch(f"snap_counts_{s}.csv",
                            f"{depth.BASE}/snap_counts/snap_counts_{s}.csv")
            frames.append(pd.read_csv(f, low_memory=False))
        except Exception as err:
            print(f"  no snap counts for {s}: {err}")
    if not frames:
        return {}
    s = pd.concat(frames, ignore_index=True)
    s = s[(s["game_type"] == "REG") & s["position"].isin(positions) & (s[col] > 0)]
    out = {}
    for team, tg in s.sort_values(["season", "week"]).groupby("team", sort=False):
        hist, last, name, wk = {}, {}, {}, {}
        n = 0
        for n, (_, here) in enumerate(tg.groupby("game_id", sort=False)):
            this = int(here["week"].iloc[0]) if int(here["season"].iloc[0]) == season else 0
            for p, pct, who in zip(here["pfr_player_id"], here[col], here["player"]):
                hist.setdefault(p, []).append(float(pct))
                last[p], name[p], wk[p] = n, who, this
        nxt = n + 1
        five = sorted(((float(np.mean(v[-4:])), p) for p, v in hist.items()
                       if nxt - last[p] <= 3), reverse=True)[:5]
        out[fv.code(team)] = [(p, name[p], nxt - last[p] >= 2, wk[p])
                              for sh, p in five if sh >= 0.4]
    return out


def teams():
    """Written before the board is built."""
    from sqlalchemy import text
    eng = fv.engine()
    with eng.connect() as conn:
        games = pd.read_sql(text(
            "SELECT season, week, home_team, away_team FROM nfl_games "
            "WHERE game_date >= CURRENT_DATE AND game_date <= CURRENT_DATE + 8 "
            "AND game_type = 'REG'"), conn)
        if games.empty:
            print("line health: no games in the next eight days, nothing written")
            return
        season = int(games["season"].max())
        inj = pd.read_sql(text(
            "SELECT player_id, week, report_status, practice_status FROM injuries "
            "WHERE season = :s"), conn, params={"s": season})
    week_of = {}
    for g in games.itertuples():
        week_of[fv.code(g.home_team)] = int(g.week)
        week_of[fv.code(g.away_team)] = int(g.week)
    ros = pd.read_csv(depth.fetch(f"roster_{season}.csv",
                                  f"{depth.BASE}/rosters/roster_{season}.csv"),
                      usecols=["gsis_id", "pfr_id", "full_name", "status"], low_memory=False)
    reserve = set(ros.loc[ros["status"].isin(["RES", "PUP", "NFI", "SUS", "RSN", "EXE"]),
                          "gsis_id"])
    # The league's own file as well as my table: it keeps the rows that carry a
    # practice line and no game status, which is how a cleared man reads.
    try:
        more = pd.read_csv(depth.fetch(f"injuries_{season}.csv",
                                       f"{depth.BASE}/injuries/injuries_{season}.csv"),
                           low_memory=False)
        more = more[more["game_type"] == "REG"].rename(columns={"gsis_id": "player_id"})
        inj = pd.concat([more[["player_id", "week", "report_status", "practice_status"]], inj],
                        ignore_index=True)
    except Exception as err:
        print(f"  no league injury file for {season}: {err}")
    by_pfr = ros.dropna(subset=["pfr_id"]).drop_duplicates("pfr_id").set_index("pfr_id")["gsis_id"].to_dict()
    by_name = {norm(n): g for n, g in zip(ros["full_name"], ros["gsis_id"])}
    # His latest report up to this week. One from this week is read as it
    # stands; an older one only counts if it put him on a reserve list.
    latest = {}
    for r in inj.sort_values("week", kind="stable").itertuples():
        latest[r.player_id] = (
            int(r.week), "" if pd.isna(r.report_status) else str(r.report_status).strip().lower(),
            "" if pd.isna(r.practice_status) else str(r.practice_status).strip().lower())

    def state(gsis, wk, missed_last, played):
        """Why he is out, or nothing."""
        if gsis in reserve:
            return "on a reserve list"
        w, st, practice = latest.get(gsis, (0, "", ""))
        if w > wk:
            return None
        if st.startswith(("injured reserve", "ir", "pup", "reserve")):
            return st
        if w == wk:
            return st if st.startswith(("out", "doubtful")) else None
        if not missed_last:
            return None
        if w > played and not st.startswith(("out", "doubtful")) and practice.startswith("full"):
            return None
        return "missed the last game, nothing since says he is back"
    report = {"built_at": time.time(), "season": season, "teams": {}, "secondary": {},
              "new_qb": {}, "opp": {}}
    for g in games.itertuples():
        report["opp"][fv.code(g.home_team)] = fv.code(g.away_team)
        report["opp"][fv.code(g.away_team)] = fv.code(g.home_team)
    for team, five in sorted(usual_five(season, DB, "defense_pct").items()):
        wk = week_of.get(team)
        if wk is None or wk < FROM_WEEK:
            continue
        out = [f"{who} ({st})" for pfr, who, missed_last, played in five
               for st in [state(by_pfr.get(pfr) or by_name.get(norm(who)), wk, missed_last,
                                played)] if st]
        if out:
            report["secondary"][team] = out
    try:
        report["new_qb"] = new_starters(season, set(week_of))
    except Exception as err:
        print(f"  could not read who starts at quarterback: {err}")
    for team, five in sorted(usual_five(season).items()):
        wk = week_of.get(team)
        if wk is None or wk < FROM_WEEK:
            continue
        out = []
        for pfr, who, missed_last, played in five:
            gsis = by_pfr.get(pfr) or by_name.get(norm(who))
            st = state(gsis, wk, missed_last, played)
            if st:
                out.append(f"{who} ({st})")
        if out:
            report["teams"][team] = out
            print(f"    {team}: {'; '.join(out)}")
    print(f"line health: {len(report['teams'])} of {len(week_of)} teams are without a "
          "usual starting lineman")
    both = [f"{t} (against {report['opp'][t]})" for t in report["teams"]
            if report["opp"].get(t) in report["secondary"]]
    print("  short line and the opponent short in the secondary: " + (", ".join(both) or "none"))
    print("  a different quarterback than last week: "
          + ("; ".join(f"{t}: {v}" for t, v in sorted(report["new_qb"].items())) or "none"))
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / NAME).write_text(json.dumps(report, indent=1), encoding="utf-8")


def new_starters(season, playing):
    """{team: "X starts, Y started last week"} where the depth chart's first
    healthy quarterback is not the man who took the team's first snap under
    centre last game."""
    from sqlalchemy import text
    with fv.engine().connect() as conn:
        _, people, gone, chart = fv.world(conn)
    d = pd.read_csv(depth.fetch(f"pbp_{season}.csv.gz",
                                f"{depth.BASE}/pbp/play_by_play_{season}.csv.gz"),
                    usecols=["game_id", "game_date", "play_id", "posteam", "season_type",
                             "pass_attempt", "passer_player_id"], low_memory=False)
    d = d[(d["season_type"] == "REG") & (d["pass_attempt"] == 1) & d["passer_player_id"].notna()]
    first = d.sort_values(["game_date", "game_id", "play_id"]).drop_duplicates(
        ["game_id", "posteam"])
    last = first.drop_duplicates("posteam", keep="last").set_index("posteam")["passer_player_id"]
    names = people.drop_duplicates("external_id").set_index("external_id")["name"].to_dict()
    chart = chart[~chart["player_id"].isin(gone)].copy()
    chart["team"] = chart["team"].map(fv.code)
    now = chart.sort_values("rank").drop_duplicates("team").set_index("team")["player_id"]
    out = {}
    for team in sorted(playing):
        if team in now.index and team in last.index and now[team] != last[team]:
            out[team] = (f"{names.get(now[team], now[team])} starts, "
                         f"{names.get(last[team], last[team])} started last week")
    return out


def read(artifact_dir=None) -> dict:
    try:
        rep = json.loads((Path(artifact_dir or ARTIFACTS) / NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {} if time.time() - float(rep.get("built_at", 0)) > 36 * 3600 else rep


def short(artifact_dir=None) -> dict:
    """{team: [who is out]} for this week, or {} if there is nothing fresh."""
    return read(artifact_dir).get("teams", {})


def acts() -> bool:
    return os.getenv("LINE_OUT_RULE", "1") != "0"


def shadow():
    """The unders this file believes in, written for grading."""
    from sqlalchemy import text
    rep = read()
    out, sec, qb, opp = (rep.get(k, {}) for k in ("teams", "secondary", "new_qb", "opp"))
    if not out and not qb:
        print("line health shadow: no short line and no new quarterback, nothing written")
        return
    eng = fv.engine()
    with eng.connect() as conn:
        board = pd.read_sql(text(
            "SELECT DISTINCT ON (player_name, market_code) player_name, market_code, line, "
            "projection, recommended_side, commence_time FROM prop_edges "
            "WHERE market_code IN ('rush_att', 'recs') "
            "ORDER BY player_name, market_code, line"), conn)
        people = pd.read_sql(text(
            "SELECT name, external_id, team, position FROM players "
            "WHERE external_id IS NOT NULL AND position IN ('RB', 'WR', 'TE')"), conn)
    people["team"] = people["team"].map(fv.code)
    who = people.drop_duplicates("name").set_index("name")
    n = {}
    with eng.begin() as conn:
        conn.execute(text(fv.SHADOW))
        for r in board.itertuples():
            if r.player_name not in who.index:
                continue
            p = who.loc[r.player_name]
            models = []
            if r.market_code == "rush_att" and p["position"] == "RB" and p["team"] in out:
                models.append("line_out_v1")
                if opp.get(p["team"]) in sec:
                    models.append("line_and_secondary_v1")
            if r.market_code == "recs" and p["team"] in qb:
                models.append("new_qb_recs_v1")
            when = pd.Timestamp(r.commence_time)
            for m in models:
                conn.execute(text(
                    "INSERT INTO shadow_numbers (model, player_id, player_name, market_code, "
                    "game_date, line, side, board_number, board_side) VALUES "
                    "(:m, :p, :n, :k, :d, :l, 'under', :bn, :bs) "
                    "ON CONFLICT (model, player_id, market_code, game_date) DO UPDATE SET "
                    "line = EXCLUDED.line, board_number = EXCLUDED.board_number, "
                    "board_side = EXCLUDED.board_side, built_at = NOW()"),
                    {"m": m, "p": p["external_id"], "n": r.player_name, "k": r.market_code,
                     "d": (when.tz_convert("America/New_York") if when.tzinfo else when).date(),
                     "l": float(r.line),
                     "bn": float(r.projection) if r.projection is not None else None,
                     "bs": r.recommended_side})
                n[m] = n.get(m, 0) + 1
                print(f"    {m}: {r.player_name} {r.market_code} under {float(r.line):.1f}; "
                      f"the board says {r.recommended_side}")
    print("line health shadow: " + (", ".join(f"{v} for {k}" for k, v in n.items())
                                    or "nothing priced on those teams")
          + " written; the board is unchanged by this step")


if __name__ == "__main__":
    shadow() if "shadow" in sys.argv[1:] else teams()
