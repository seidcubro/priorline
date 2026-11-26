"""Does a running back's workload follow who else is in the backfield, and can we use it?

The model projects a back from his last eight games. Emmett Johnson carried 22
and 23 times in the two games Kenneth Walker missed, and for the third game
without Walker the model said 11.4, because six of those eight games were from
when Walker was playing. The window mixes two different jobs.

This asks whether sorting his past games by who was available fixes it. For a
back and an upcoming game:

    the backfield   the established backs on his team (a fifth of the carries
                    or more over their last five games)
    his regime      his earlier games this season in which the same established
                    backs were missing as are missing now, and none of the ones
                    playing now were missing

    window          his mean carries over his last eight games. What the model
                    leans on.
    regime          his mean carries over his regime games (the last four),
                    pulled toward the window by K games' worth when the regime
                    is short. With no regime games it is the window.

Scored on every back-game, and separately on the games where the two differ,
which are the only ones a fix can change. Fitted (K) on the seasons before the
last two and scored on those, so the result is on games it never saw. Carries,
then rushing yards at his own yards a carry.

Runs anywhere: it reads nflverse play-by-play, not the database.

    python lab/training/research_role_regime.py
"""

from __future__ import annotations

import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

BASE = "https://github.com/nflverse/nflverse-data/releases/download"
CACHE = Path(__file__).resolve().parents[2] / "marketing" / ".cache"
SEASONS = (2022, 2023, 2024, 2025, 2026)
HOLDOUT = (2025, 2026)
EST, TRAIL, WINDOW, RECENT = 0.20, 5, 8, 4


def fetch(name, url):
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / name
    if not f.exists() or (name.endswith("2026.csv.gz") and time.time() - f.stat().st_mtime > 43200):
        req = urllib.request.Request(url, headers={"User-Agent": "research"})
        with urllib.request.urlopen(req, timeout=180) as r:
            f.write_bytes(r.read())
    return f


def load(stat="carries"):
    rows = []
    for s in SEASONS:
        d = pd.read_csv(fetch(f"pbp_{s}.csv.gz", f"{BASE}/pbp/play_by_play_{s}.csv.gz"),
                        usecols=["game_id", "season", "week", "season_type", "posteam",
                                 "rush_attempt", "pass_attempt", "sack", "yards_gained",
                                 "rusher_player_id", "receiver_player_id",
                                 "complete_pass", "two_point_attempt", "qb_kneel"],
                        low_memory=False)
        d = d[(d["season_type"] == "REG") & (d["two_point_attempt"] != 1)]
        pos = pd.read_csv(fetch(f"roster_{s}.csv", f"{BASE}/rosters/roster_{s}.csv"),
                          usecols=["gsis_id", "position"], low_memory=False)
        pos = pos.dropna().drop_duplicates("gsis_id").set_index("gsis_id")["position"]
        if stat == "carries":
            r = d[(d["rush_attempt"] == 1) & d["rusher_player_id"].notna()
                  & (d["qb_kneel"] != 1)]
            g = r.groupby(["season", "week", "game_id", "posteam", "rusher_player_id"]).agg(
                vol=("rush_attempt", "size"), yds=("yards_gained", "sum")).reset_index()
            g = g.rename(columns={"rusher_player_id": "pid"})
        else:
            r = d[(d["pass_attempt"] == 1) & (d["sack"] != 1) & d["receiver_player_id"].notna()]
            g = r.groupby(["season", "week", "game_id", "posteam", "receiver_player_id"]).agg(
                vol=("pass_attempt", "size"), yds=("complete_pass", "sum")).reset_index()
            g = g.rename(columns={"receiver_player_id": "pid"})
        # Anyone who touched the ball at all that week played. A back with no
        # carry and a catch was there.
        touched = pd.concat([
            d.loc[d["rusher_player_id"].notna(), ["game_id", "posteam", "rusher_player_id"]]
             .rename(columns={"rusher_player_id": "pid"}),
            d.loc[d["receiver_player_id"].notna(), ["game_id", "posteam", "receiver_player_id"]]
             .rename(columns={"receiver_player_id": "pid"})]).drop_duplicates()
        touched["season"] = s
        wk = d.drop_duplicates("game_id").set_index("game_id")["week"]
        touched["week"] = touched["game_id"].map(wk)
        g = touched.merge(g, on=["season", "week", "game_id", "posteam", "pid"], how="left")
        g[["vol", "yds"]] = g[["vol", "yds"]].fillna(0)
        g["pos"] = g["pid"].map(pos)
        rows.append(g)
    return pd.concat(rows, ignore_index=True)


def build(g, position="RB"):
    """One row a back-game with what was known before it."""
    g = g[g["pos"] == position].copy()
    out = []
    for (season, team), t in g.groupby(["season", "posteam"]):
        weeks = sorted(t["week"].unique())
        by_week = {w: x.set_index("pid") for w, x in t.groupby("week")}
        tot = {w: float(by_week[w]["vol"].sum()) for w in weeks}
        hist = {}        # pid -> list of (week, vol, yds, share)
        for w in weeks:
            now = by_week[w]
            # Established backs, from games before this one.
            est = {pid for pid, h in hist.items()
                   if len(h) >= 2 and np.mean([x[3] for x in h[-TRAIL:]]) >= EST
                   and h[-1][0] >= w - 6}
            out_now = frozenset(pid for pid in est if pid not in now.index)
            in_now = frozenset(pid for pid in est if pid in now.index)
            for pid, row in now.iterrows():
                h = hist.get(pid, [])
                if len(h) >= 2:
                    win = h[-WINDOW:]
                    # Games of his in which the backs out now were out, and the
                    # backs in now (other than him) were in.
                    reg = [x for x in h
                           if all(o not in by_week[x[0]].index for o in out_now)
                           and all(i in by_week[x[0]].index for i in in_now if i != pid)]
                    reg = reg[-RECENT:]
                    carried = sum(x[1] for x in h[-16:])
                    out.append({
                        "season": season, "week": w, "team": team, "pid": pid,
                        "actual": float(row["vol"]), "actual_yds": float(row["yds"]),
                        "window": float(np.mean([x[1] for x in win])),
                        "window_yds": float(np.mean([x[2] for x in win])),
                        "regime": float(np.mean([x[1] for x in reg])) if reg else np.nan,
                        "n_regime": len(reg), "n_window": len(win),
                        "same": len(reg) >= len(win[-RECENT:]) and
                                [x[0] for x in reg] == [x[0] for x in win[-RECENT:]],
                        "ypc": (sum(x[2] for x in h[-16:]) + 60 * 4.2) / (carried + 60),
                        "n_out": len(out_now),
                    })
            for pid, row in now.iterrows():
                share = float(row["vol"]) / tot[w] if tot[w] > 0 else 0.0
                hist.setdefault(pid, []).append((w, float(row["vol"]), float(row["yds"]), share))
    return pd.DataFrame(out)


def blend(d, k):
    n = d["n_regime"].to_numpy(dtype=float)
    reg = np.where(n > 0, d["regime"].fillna(0).to_numpy(), d["window"].to_numpy())
    return (n * reg + k * d["window"].to_numpy()) / (n + k)


def mae(a, b):
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


def report(d, k, title, unit):
    d = d.copy()
    d["fix"] = blend(d, k)
    aff = d[~d["same"] & (d["n_regime"] > 0) & ((d["window"] >= 4) | (d["fix"] >= 4))]
    print(f"\n  {title}")
    print(f"    {'':<34}{'games':>7}{'window':>9}{'regime':>9}{'change':>9}")
    for name, s in (("every back-game", d[(d["window"] >= 4) | (d["fix"] >= 4)]),
                    ("where his backfield changed", aff),
                    ("  and he should get more", aff[aff["fix"] > aff["window"] + 2]),
                    ("  and he should get fewer", aff[aff["fix"] < aff["window"] - 2])):
        if len(s) < 20:
            print(f"    {name:<34}{len(s):>7}   too few")
            continue
        a, b = mae(s["actual"], s["window"]), mae(s["actual"], s["fix"])
        print(f"    {name:<34}{len(s):>7}{a:>9.2f}{b:>9.2f}{(b - a) / a:>+9.1%}")
        if name.startswith("  and"):
            print(f"    {'':<34}{'':>7}  window said {s['window'].mean():.1f}, regime "
                  f"{s['fix'].mean():.1f}, he got {s['actual'].mean():.1f} {unit}")
    return aff


def main():
    stat = sys.argv[1] if len(sys.argv) > 1 else "carries"
    position = sys.argv[2] if len(sys.argv) > 2 else "RB"
    d = build(load(stat), position)
    fit = d[~d["season"].isin(HOLDOUT)]
    aff = fit[~fit["same"] & (fit["n_regime"] > 0)]
    scores = [(mae(aff["actual"], blend(aff, k)), k) for k in (0, 0.25, 0.5, 1, 2, 4, 8)]
    best = min(scores)[1]
    print(f"{stat}, {position}: {len(d):,} player-games, {SEASONS[0]} to {SEASONS[-1]}")
    print("  K fitted on the earlier seasons, where his backfield changed: "
          + ", ".join(f"K={k:g} {m:.2f}" for m, k in scores) + f". Using K={best:g}.")
    unit = "carries" if stat == "carries" else "targets"
    for s in HOLDOUT:
        report(d[d["season"] == s], best, f"{s}, never seen by the fit", unit)
    if stat == "carries":
        y = d[d["season"].isin(HOLDOUT)].copy()
        y["fix"] = blend(y, best)
        a = y[~y["same"] & (y["n_regime"] > 0) & ((y["window"] >= 4) | (y["fix"] >= 4))]
        up = a[a["fix"] > a["window"] + 2]
        for name, s in (("where his backfield changed", a), ("  and he should get more", up)):
            print(f"\n  rushing yards, {name.strip()}: {len(s)} games. Window mean "
                  f"{mae(s['actual_yds'], s['window_yds']):.1f} off; regime carries at his "
                  f"yards a carry {mae(s['actual_yds'], s['fix'] * s['ypc']):.1f} off.")


if __name__ == "__main__":
    main()
