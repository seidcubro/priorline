"""A running back's workload depends on who else is in the backfield. Sort his games by that.

The model projects a back from his last eight games. Emmett Johnson carried 22
and 23 times in the two games Kenneth Walker missed, and for the third game
without Walker the model said 11.4, because six of those eight games were from
when Walker was playing. The window mixes two different jobs and reports the
average of them, which is the workload of neither.

The same mistake runs the other way. When the starter comes back, the backup's
window still holds the weeks he started, and the model keeps him there.

So for a back and the game in front of him:

    the backfield   the established backs on his team: a fifth of the team's
                    carries or more over their last five games
    his regime      his earlier games this season in which the established
                    backs who are out now were out, and the ones who are
                    playing now were playing
    his level       his mean carries over the last four regime games, pulled
                    toward his eight-game window by K games' worth while the
                    regime is short

A back whose regime is his window is left alone, and so is one whose level is
within GAP carries of his window. For the rest the level replaces the model's
carries, and rushing yards and touchdowns follow at his own rate per carry.

Measured before it was built, on play-by-play from 2022 on, K fitted on the
earlier seasons and scored on 2025 and 2026 (lab/training/research_role_regime.py).
Where the backfield had changed, the error in carries fell 8.6% and 5.5% against
the window. For backs stepping up, the window said 11.6 carries, the level 15.1,
and they got 14.9. Against the model's own stored projections on 118 such games
in 2026: backs stepping up, the model said 12.9 and they got 14.9; backs
stepping back, the model said 12.8 and they got 10.5, and the level said 9.7.
The same sorting did nothing for targets, at any position, so it is not
applied to them.

fit_role_regime.py fits K on the database and writes it only if it beats the
window on seasons it did not see. Nothing here runs without that file.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

EST = 0.20          # share of team carries that makes a back established
TRAIL = 5           # games that share is measured over
WINDOW = 8          # the model's window
RECENT = 4          # regime games that count
STALE_WEEKS = 6     # an established back not seen for this long is not in the backfield
GAP = 2.0           # carries the level must differ from the window by to be used
POSITIONS = ("RB", "FB")
# Where the middle game lands against the level, on the 446 games from 2022 on
# where the level differed from the window. A back's yards have a long right
# tail, so the middle game is under the average one. The same ratio held for
# backs stepping up (0.91), stepping back (0.92) and every back with a window
# of ten carries or more (0.91).
MEDIAN = {"rush_yds": 0.91, "rush_att": 0.96}
ARTIFACT = "role_regime.json"


def load_params(artifact_dir) -> dict:
    """The fitted K, if the fit was accepted. Otherwise nothing, and nothing runs."""
    path = Path(artifact_dir) / ARTIFACT
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if data.get("accepted") else {}


def team_history(t: pd.DataFrame):
    """One team's season: who played each week, and each back's games in order.

    `t` has player_id, week, carries, rushing_yards for one team and season,
    one row a back a game he played in.
    """
    played = {int(w): set(x["player_id"]) for w, x in t.groupby("week")}
    total = t.groupby("week")["carries"].sum().to_dict()
    hist: dict = {}
    for r in t.sort_values("week").itertuples(index=False):
        tot = float(total.get(r.week, 0.0))
        hist.setdefault(r.player_id, []).append(
            (int(r.week), float(r.carries), float(r.rushing_yards),
             float(r.carries) / tot if tot > 0 else 0.0))
    return played, hist


def established(hist: dict, before_week: int) -> set:
    """The backs who carry the load, from games before `before_week`."""
    out = set()
    for pid, h in hist.items():
        h = [x for x in h if x[0] < before_week]
        if (len(h) >= 2 and np.mean([x[3] for x in h[-TRAIL:]]) >= EST
                and h[-1][0] >= before_week - STALE_WEEKS):
            out.add(pid)
    return out


def level_for(pid, hist: dict, played: dict, before_week: int,
              out_now: set, in_now: set, k: float):
    """His window, his regime level, and how many regime games it stands on."""
    h = [x for x in hist.get(pid, []) if x[0] < before_week]
    if len(h) < 2:
        return None
    win = h[-WINDOW:]
    reg = [x for x in h
           if all(o not in played.get(x[0], ()) for o in out_now)
           and all(i in played.get(x[0], ()) for i in in_now if i != pid)]
    reg = reg[-RECENT:]
    window = float(np.mean([x[1] for x in win]))
    same = [x[0] for x in reg] == [x[0] for x in win[-RECENT:]]
    if not reg or same:
        return {"window": window, "level": window, "n": len(reg), "same": True}
    raw = float(np.mean([x[1] for x in reg]))
    n = float(len(reg))
    return {"window": window, "level": (n * raw + k * window) / (n + k),
            "n": len(reg), "same": False}


def levels(games: pd.DataFrame, ruled_out: set, k: float, gap: float = GAP) -> dict:
    """For the week in front of us: every back whose level differs from his window.

    `games` is this season's back-games (player_id, team, week, carries,
    rushing_yards), `ruled_out` the players the report has out. A back's team
    is the one he last played for.
    """
    out: dict = {}
    if games.empty:
        return out
    latest = games.sort_values("week").drop_duplicates("player_id", keep="last")
    team_of = latest.set_index("player_id")["team"].to_dict()
    for team, t in games.groupby("team"):
        t = t[t["player_id"].map(team_of) == team]
        if t.empty:
            continue
        played, hist = team_history(t)
        nxt = int(t["week"].max()) + 1
        est = established(hist, nxt)
        out_now = {p for p in est if p in ruled_out}
        in_now = est - out_now
        for pid in hist:
            if pid in ruled_out:
                continue
            got = level_for(pid, hist, played, nxt, out_now, in_now, k)
            if not got or got["same"] or abs(got["level"] - got["window"]) < gap:
                continue
            got["team"] = team
            got["out"] = sorted(out_now)
            out[pid] = got
    return out


def backtest(games: pd.DataFrame) -> pd.DataFrame:
    """Every back-game with what was known before it, who was absent read off
    who actually played. `games` has season as well as the columns above."""
    rows = []
    for (season, team), t in games.groupby(["season", "team"]):
        played, hist = team_history(t)
        for w in sorted(played):
            est = established(hist, w)
            out_now = {p for p in est if p not in played[w]}
            in_now = est - out_now
            now = t[t["week"] == w].set_index("player_id")
            for pid in played[w]:
                got = level_for(pid, hist, played, w, out_now, in_now, 0.0)
                if not got:
                    continue
                h = [x for x in hist[pid] if x[0] < w]
                carried = sum(x[1] for x in h[-16:])
                rows.append({
                    "season": season, "week": w, "team": team, "player_id": pid,
                    "actual": float(now.at[pid, "carries"]),
                    "actual_yds": float(now.at[pid, "rushing_yards"]),
                    "window": got["window"], "raw": got["level"], "n": got["n"],
                    "same": got["same"],
                    "window_yds": float(np.mean([x[2] for x in h[-WINDOW:]])),
                    "ypc": (sum(x[2] for x in h[-16:]) + 60 * 4.2) / (carried + 60),
                })
    return pd.DataFrame(rows)


def blend(d: pd.DataFrame, k: float) -> np.ndarray:
    """backtest() keeps the raw regime mean; this pulls it toward the window."""
    n = d["n"].to_numpy(dtype=float)
    raw = d["raw"].to_numpy(dtype=float)
    win = d["window"].to_numpy(dtype=float)
    return np.where(d["same"].to_numpy(), win, (n * raw + k * win) / np.maximum(n + k, 1e-9))
