"""Where do rushing overs and unders actually land against the book's line?

Read from the published record (pull_record.py) joined to play-by-play, so it
runs anywhere. For every rushing prop we published a pick on, in 2025 and 2026:
how often the over came in, split by things known before the game.

    our side         what we picked, and how it did
    always under     the blunt rule, for scale
    his workload     his eight-game window of carries
    backfield        whether the backs available had changed, and which way
                     (role_regime.py), which is the fix that went live on
                     26 November 2026: this is that fix against real lines
    the spread       his team favoured or not
    the line         against his window: is the book above or below what he
                     has been doing

Each cell is props, how often the over came in, and how often the under did.
A rule is only worth anything if it leans the same way in both seasons.

    python lab/training/pull_record.py
    python lab/training/research_rushing_sides.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "services" / "training"))
import role_regime as rr  # noqa: E402

CACHE = Path(__file__).resolve().parents[2] / "marketing" / ".cache"


def backs():
    """Every back-game 2024 on, with what was known before it."""
    rows, sched = [], []
    for s in (2024, 2025, 2026):
        d = pd.read_csv(CACHE / f"pbp_{s}.csv.gz", usecols=[
            "game_id", "season", "week", "season_type", "game_date", "posteam",
            "home_team", "spread_line", "total_line", "rush_attempt", "yards_gained",
            "rusher_player_id", "receiver_player_id", "two_point_attempt", "qb_kneel"],
            low_memory=False)
        d = d[(d["season_type"] == "REG") & (d["two_point_attempt"] != 1)]
        pos = pd.read_csv(CACHE / f"roster_{s}.csv", usecols=["gsis_id", "position"],
                          low_memory=False).dropna().drop_duplicates("gsis_id") \
                .set_index("gsis_id")["position"]
        r = d[(d["rush_attempt"] == 1) & d["rusher_player_id"].notna() & (d["qb_kneel"] != 1)]
        g = r.groupby(["season", "week", "game_id", "posteam", "rusher_player_id"]).agg(
            carries=("rush_attempt", "size"), rushing_yards=("yards_gained", "sum")).reset_index()
        g = g.rename(columns={"rusher_player_id": "player_id", "posteam": "team"})
        seen = pd.concat([
            d.loc[d["rusher_player_id"].notna(), ["season", "week", "game_id", "posteam", "rusher_player_id"]]
             .rename(columns={"rusher_player_id": "player_id", "posteam": "team"}),
            d.loc[d["receiver_player_id"].notna(), ["season", "week", "game_id", "posteam", "receiver_player_id"]]
             .rename(columns={"receiver_player_id": "player_id", "posteam": "team"})]).drop_duplicates()
        g = seen.merge(g, how="left", on=["season", "week", "game_id", "team", "player_id"])
        g[["carries", "rushing_yards"]] = g[["carries", "rushing_yards"]].fillna(0)
        g = g[g["player_id"].map(pos).isin(["RB", "FB"])]
        rows.append(g)
        x = d.drop_duplicates("game_id")[["game_id", "game_date", "home_team", "spread_line", "total_line"]]
        sched.append(x)
    g = pd.concat(rows, ignore_index=True)
    b = rr.backtest(g)
    b["level"] = rr.blend(b, 1.0)
    key = g[["season", "week", "team", "game_id"]].drop_duplicates()
    b = b.merge(key, on=["season", "week", "team"]).merge(pd.concat(sched), on="game_id")
    # nflverse: spread_line is positive when the home team is favoured.
    b["favoured_by"] = np.where(b["team"] == b["home_team"], b["spread_line"], -b["spread_line"])
    b["game_date"] = pd.to_datetime(b["game_date"])
    return b


def cell(x):
    if len(x) < 30:
        return f"{len(x):>6}   too few"
    o = x["over"].mean()
    return f"{len(x):>6}{o:>8.1%}{1 - o:>8.1%}"


def table(d, title, groups):
    print(f"\n  {title}")
    print(f"    {'':<34}" + "".join(f"{s:>10}{'over':>6}{'under':>8}" for s in (2025, 2026)))
    for name, mask in groups:
        print(f"    {name:<34}" + "  ".join(cell(d[mask & (d['season'] == s)]) for s in (2025, 2026)))


def main():
    rec = pd.read_pickle(CACHE / "record.pkl")
    rec = rec[rec["season"].isin([2025, 2026]) & (rec["actual"].astype(float) != rec["line"].astype(float))]
    b = backs()
    for code in ("rush_att", "rush_yds"):
        d = rec[rec["market_code"] == code].copy()
        d["game_date"] = pd.to_datetime(d["game_date"])
        d = d.merge(b.drop(columns=["season"]), left_on=["gsis", "game_date"],
                    right_on=["player_id", "game_date"], how="inner", suffixes=("", "_b"))
        d["line"] = d["line"].astype(float)
        base = d["window"] if code == "rush_att" else d["window_yds"]
        d["line_vs_window"] = d["line"] / base.clip(lower=1)
        all_ = pd.Series(True, index=d.index)
        print(f"\n{code.upper()}: {len(d):,} running back props matched to play-by-play")
        ours = d["hit"].groupby(d["season"]).mean()
        print("    our published side was right " + ", ".join(
            f"{s}: {ours.get(s, float('nan')):.1%}" for s in (2025, 2026)))
        table(d, "every prop", [("all", all_),
                                ("we said over", d["recommended_side"] == "over"),
                                ("we said under", d["recommended_side"] == "under")])
        table(d, "by his eight-game window of carries",
              [("under 8", d["window"] < 8), ("8 to 13", (d["window"] >= 8) & (d["window"] < 13)),
               ("13 to 18", (d["window"] >= 13) & (d["window"] < 18)), ("18 and up", d["window"] >= 18)])
        table(d, "by what happened to his backfield",
              [("unchanged", d["same"]),
               ("changed, he steps up 2+", ~d["same"] & (d["level"] >= d["window"] + 2)),
               ("changed, he steps back 2+", ~d["same"] & (d["level"] <= d["window"] - 2)),
               ("changed, within 2", ~d["same"] & ((d["level"] - d["window"]).abs() < 2))])
        table(d, "by the spread",
              [("favoured by 4+", d["favoured_by"] >= 4), ("within 3.5", d["favoured_by"].abs() < 4),
               ("underdog by 4+", d["favoured_by"] <= -4)])
        table(d, "by the line against his window",
              [("line under 85% of it", d["line_vs_window"] < 0.85),
               ("85 to 100%", (d["line_vs_window"] >= 0.85) & (d["line_vs_window"] < 1.0)),
               ("100 to 115%", (d["line_vs_window"] >= 1.0) & (d["line_vs_window"] < 1.15)),
               ("over 115%", d["line_vs_window"] >= 1.15)])
        if code == "rush_att":
            lvl = d["level"]
        else:
            lvl = d["level"] * d["ypc"] * rr.MEDIAN["rush_yds"]
        d["level_side_over"] = lvl > d["line"]
        ch = ~d["same"] & ((d["level"] - d["window"]).abs() >= 2)
        print("\n  the role-regime level's side, on the props it changes")
        for s in (2025, 2026):
            x = d[ch & (d["season"] == s)]
            if len(x) < 20:
                print(f"    {s}: {len(x)} props, too few")
                continue
            lv = np.where(x["level_side_over"], x["over"], 1 - x["over"]).mean()
            print(f"    {s}: {len(x)} props. Our published side was right {x['hit'].mean():.1%}; "
                  f"the level's side {lv:.1%}; always under {1 - x['over'].mean():.1%}")


if __name__ == "__main__":
    main()
