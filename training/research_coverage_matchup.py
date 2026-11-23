"""Does a receiver's record against man and zone, set against how the defence plays, beat the line?

The idea is the one a film watcher has: the Giants play a lot of man, this
receiver gets open against man and draws a big share of the targets when he
sees it, so he should have a day. The feature store holds nothing like it.
It knows how many yards a defence allows, not how it defends.

The data exists and is free, with one catch. nflverse publishes, for every
play, whether the defence was in man or zone and which coverage it was
(charted by NGS to 2022 and by FTN from 2023), and it is released only after
the season ends. So the past seasons can be tested for nothing, and using it
during a season means either last year's habits or paying FTN for the feed.
This is the test that says whether that is worth paying for.

Built from every charted target, 2023 onward, and for each game using only
the games before it:

    his share against man    his targets on man plays over his team's targets
                             on man plays, in the games he played. The same
                             against zone.
    man lean                 the first minus the second. Above zero is a
                             receiver who draws more of the ball against man.
    defence's man rate       the share of throws against this opponent that
                             came against man, less the league's rate.
    matchup                  man lean times the defence's man rate above the
                             league. Positive when a man-beater meets a
                             man-heavy defence or a zone-beater meets a
                             zone-heavy one; negative when they are crossed.

Then against the lines for catches and receiving yards: for each number, how
often it rated a prop that went over above one that did not (0.50 is nothing),
in 2023-24 together and in each half of 2025. And the props cut into fifths
by the matchup number, and the straight film-watcher cut: the third of
receivers who lean most toward man, against the third of defences that play
the most of it.

It is worth paying for only if it leans the same way in all three periods.

Read-only on the database. Downloads the play files from nflverse the first
time (a few minutes) and keeps them.

    sh lab/run.sh research_coverage_matchup.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import backtest_gap_system as bt
import research_reproject_blanks as RB
from research_who_sorts import sorts

BASE = "https://github.com/nflverse/nflverse-data/releases/download"
SEASONS = (2023, 2024, 2025)
MARKETS = ("recs", "rec_yds")
PBP_COLS = ["game_id", "play_id", "season", "week", "game_date", "posteam",
            "defteam", "pass_attempt", "receiver_player_id", "complete_pass",
            "yards_gained", "sack"]


def fetch(kind, season):
    """One season of plays or of charting, kept after the first download."""
    RB.CACHE.mkdir(parents=True, exist_ok=True)
    keep = RB.CACHE / f"{kind}_{season}.pkl"
    if keep.exists():
        return pd.read_pickle(keep)
    if kind == "pbp":
        urls = [f"{BASE}/pbp/play_by_play_{season}.csv.gz",
                f"{BASE}/pbp/play_by_play_{season}.parquet"]
    else:
        urls = [f"{BASE}/pbp_participation/pbp_participation_{season}.csv",
                f"{BASE}/pbp_participation/pbp_participation_{season}.parquet"]
    last = None
    for u in urls:
        try:
            d = (pd.read_parquet(u) if u.endswith(".parquet")
                 else pd.read_csv(u, low_memory=False))
            break
        except Exception as e:  # the next address, or say why none worked
            last = e
    else:
        print(f"  {kind} {season}: could not be downloaded ({type(last).__name__}: {last})")
        return None
    if kind == "pbp":
        d = d[[c for c in PBP_COLS if c in d.columns]]
    else:
        d = d[["nflverse_game_id", "play_id", "defense_man_zone_type",
               "defense_coverage_type"]].rename(columns={"nflverse_game_id": "game_id"})
    d.to_pickle(keep)
    return d


def targets():
    frames = []
    for s in SEASONS:
        p, c = fetch("pbp", s), fetch("participation", s)
        if p is None or c is None:
            continue
        p = p[(p["pass_attempt"] == 1) & p["receiver_player_id"].notna()
              & (p["sack"] != 1)]
        j = p.merge(c, on=["game_id", "play_id"], how="left")
        kind = j["defense_man_zone_type"].astype(str).str.upper()
        j["cov"] = np.where(kind.str.contains("MAN"), "man",
                            np.where(kind.str.contains("ZONE"), "zone", None))
        print(f"  {s}: {len(j):,} targets, {j['cov'].notna().mean():.0%} charted, "
              f"{(j['cov'] == 'man').sum() / max(j['cov'].notna().sum(), 1):.0%} of those against man")
        frames.append(j[j["cov"].notna()])
    if not frames:
        return None
    t = pd.concat(frames, ignore_index=True)
    t["game_date"] = pd.to_datetime(t["game_date"])
    t["yards"] = np.where(t["complete_pass"] == 1, t["yards_gained"].fillna(0), 0.0)
    return t


def before(frame, key, cols):
    """Running totals of every earlier game, per key."""
    frame = frame.sort_values("game_date").copy()
    g = frame.groupby(key)
    for c in cols:
        frame[f"p_{c}"] = g[c].cumsum() - frame[c]
    return frame


def signals(t):
    wide = lambda d, by: (d.pivot_table(index=by, columns="cov", values="n",
                                        aggfunc="sum", fill_value=0)
                          .reindex(columns=["man", "zone"], fill_value=0).reset_index())
    t = t.assign(n=1)
    game = ["game_id", "game_date"]
    # His targets, and his team's, by coverage, each game.
    his = wide(t, game + ["posteam", "defteam", "receiver_player_id"]).rename(
        columns={"man": "t_man", "zone": "t_zone"})
    team = wide(t, game + ["posteam"]).rename(
        columns={"man": "tm_man", "zone": "tm_zone"})
    his = his.merge(team, on=game + ["posteam"])
    his = before(his, "receiver_player_id", ["t_man", "t_zone", "tm_man", "tm_zone"])
    ok_m, ok_z = his["p_tm_man"] >= 40, his["p_tm_zone"] >= 80
    his["share vs man"] = np.where(ok_m, his["p_t_man"] / his["p_tm_man"].clip(lower=1), np.nan)
    his["share vs zone"] = np.where(ok_z, his["p_t_zone"] / his["p_tm_zone"].clip(lower=1), np.nan)
    his["man lean"] = his["share vs man"] - his["share vs zone"]
    # What each defence has shown, and the league, before each game.
    faced = wide(t, game + ["defteam"]).rename(columns={"man": "f_man", "zone": "f_zone"})
    faced = before(faced, "defteam", ["f_man", "f_zone"])
    seen = faced["p_f_man"] + faced["p_f_zone"]
    faced["def_man"] = np.where(seen >= 120, faced["p_f_man"] / seen.clip(lower=1), np.nan)
    day = faced.groupby("game_date")[["f_man", "f_zone"]].sum().sort_index().cumsum().shift(1)
    day["lg_man"] = day["f_man"] / (day["f_man"] + day["f_zone"])
    faced = faced.merge(day[["lg_man"]], left_on="game_date", right_index=True, how="left")
    faced["defence man rate"] = faced["def_man"] - faced["lg_man"]
    his = his.merge(faced[["game_id", "defteam", "defence man rate"]],
                    on=["game_id", "defteam"], how="left")
    his["matchup"] = his["man lean"] * his["defence man rate"]
    return his.rename(columns={"receiver_player_id": "player_id", "game_date": "d"})


def props(code, sig):
    o, f = bt.lines(code), bt.features(code)
    if o is None or o.empty or f is None:
        return None
    f = f.rename(columns={"name": "player_name"})
    f["d"] = pd.to_datetime(f["as_of_game_date"])
    j = o.merge(f[["player_name", "d", "player_id", "label_actual", "season"]],
                on=["player_name", "d"], how="inner")
    j = j[(j["line"] * 2) % 2 == 1]
    j["went_over"] = j["label_actual"].astype(float) > j["line"]
    keep = ["player_id", "d", "share vs man", "share vs zone", "man lean",
            "defence man rate", "matchup"]
    j = j.merge(sig[keep].drop_duplicates(["player_id", "d"]),
                on=["player_id", "d"], how="inner")
    mid = j.loc[j["season"] == 2025, "d"].median()
    j["period"] = np.where(j["season"] < 2025, "2023-24",
                           np.where(j["d"] <= mid, "2025 first half", "2025 second half"))
    return j


def report(code, j):
    periods = ["2023-24", "2025 first half", "2025 second half"]
    print(f"\n{code.upper()}: {len(j):,} priced props with a coverage record "
          f"behind them, over came in {j['went_over'].mean():.1%}")
    print("    " + f"{'number':<22}" + "".join(f"{p:>20}" for p in periods))
    print("    " + f"{'props':<22}" + "".join(
        f"{int((j['period'] == p).sum()):>20}" for p in periods))
    for c in ("share vs man", "share vs zone", "man lean", "defence man rate", "matchup"):
        line = f"    {c:<22}"
        for p in periods:
            s = j[(j["period"] == p) & j[c].notna()]
            line += f"{sorts(s[c], s['went_over']):>20.3f}"
        print(line)
    k = j[j["matchup"].notna()].copy()
    if len(k) >= 250:
        k["fifth"] = pd.qcut(k["matchup"].rank(method="first"), 5, labels=False)
        print("\n    by fifth of the matchup number, crossed to matched: "
              "props and how often the over came in")
        for f, g in k.groupby("fifth"):
            print(f"      {int(f) + 1}   " + "   ".join(
                f"{p}: {int((g['period'] == p).sum()):>4} "
                f"{g.loc[g['period'] == p, 'went_over'].mean():>6.1%}" for p in periods))
        lean_hi = k["man lean"] >= k["man lean"].quantile(2 / 3)
        lean_lo = k["man lean"] <= k["man lean"].quantile(1 / 3)
        d_hi = k["defence man rate"] >= k["defence man rate"].quantile(2 / 3)
        d_lo = k["defence man rate"] <= k["defence man rate"].quantile(1 / 3)
        print("\n    the film-watcher cuts")
        for name, m in (("man-beater against a man-heavy defence", lean_hi & d_hi),
                        ("man-beater against a zone-heavy defence", lean_hi & d_lo),
                        ("zone-beater against a zone-heavy defence", lean_lo & d_lo),
                        ("zone-beater against a man-heavy defence", lean_lo & d_hi)):
            g = k[m]
            print(f"      {name:<42}" + "   ".join(
                f"{p}: {int((g['period'] == p).sum()):>4} "
                f"{g.loc[g['period'] == p, 'went_over'].mean():>6.1%}" for p in periods))


def main():
    t = targets()
    if t is None:
        print("No charted plays could be read. Nothing to test.")
        return
    sig = signals(t)
    print(f"{len(sig):,} receiver-games with a record before them")
    for code in MARKETS:
        j = props(code, sig)
        if j is None or len(j) < 200:
            print(f"\n{code}: too few priced props matched")
            continue
        report(code, j)
    print("\n0.50 means the number says nothing the line has not priced. Above "
          "it, a higher number goes with the over.")


if __name__ == "__main__":
    main()
