"""Every market read against the situation, with no model: who is on the field
and what kind of game it is.

The one thing that has beaten a line this month is information about who is
playing: a back's carries under behind a short offensive line. This asks the
same kind of question everywhere. Each situation is something known before
kickoff, and for each market it prints how often the over came in against what
each side's price needed, in 2025 and in 2026.

    who is out        a usual starter missing on his line, among his receivers,
                      at tight end, or on the defence across from him: out by
                      that week's report, or missed the last game with nothing
                      on the report
    the man himself   first game back after missing one, his snap share last
                      game and whether it has been rising or falling, a short
                      week
    under centre      the quarterback is not the team's usual starter, or has
                      fewer than four career starts
    the game          favourite or underdog and by how much, a high or low
                      total, wind, a roof, a division game, late in the season

A situation is only worth a rule if one side beats its price in BOTH seasons.
This reads about 300 cells, so some will look good by luck: the bar here is
two points clear in each season on 250 props or more, and anything that passes
still has to make football sense.

    python lab/training/research_situations.py
"""

from __future__ import annotations

import re
import sys
import warnings

import numpy as np
import pandas as pd

import research_football as F

warnings.filterwarnings("ignore")
GROUP = {"WR": "wr", "TE": "te", "RB": "rb", "FB": "rb", "T": "ol", "G": "ol", "C": "ol",
         "OL": "ol", "CB": "db", "S": "db", "DB": "db", "FS": "db", "SS": "db", "DL": "dl",
         "DE": "dl", "DT": "dl", "NT": "dl", "LB": "lb", "OLB": "lb", "ILB": "lb", "MLB": "lb",
         "QB": "qb"}
TOP = {"ol": 5, "wr": 3, "te": 1, "rb": 2, "db": 5, "dl": 4, "lb": 3}
MARKETS = ("rush_att", "rush_yds", "recs", "rec_yds", "pass_yds", "pass_td", "pass_att",
           "pass_completions")


def norm(n):
    n = re.sub(r"[^a-z ]", "", str(n).lower())
    return " ".join(w for w in n.split() if w not in ("jr", "sr", "ii", "iii", "iv", "v"))


def snaps():
    s = pd.concat([pd.read_csv(F.CACHE / f"snap_counts_{y}.csv") for y in F.SEASONS],
                  ignore_index=True)
    s = s[s["game_type"] == "REG"].copy()
    s["grp"] = s["position"].map(GROUP)
    s["pct"] = np.where(s["grp"].isin(["db", "dl", "lb"]), s["defense_pct"], s["offense_pct"])
    s["nm"] = s["player"].map(norm)
    return s[s["grp"].notna() & (s["pct"] > 0)]


def reports():
    inj = pd.concat([pd.read_csv(F.CACHE / f"injuries_{y}.csv", low_memory=False)
                     for y in F.SEASONS])
    inj = inj[inj["game_type"] == "REG"]
    inj["team"] = inj["team"].replace({"LAR": "LA"})
    return {(r.season, r.week, r.team, norm(r.full_name)): str(r.report_status)
            for r in inj.itertuples()}


def who_is_out(s, status):
    """A row a team a game: usual starters out, by what was known beforehand.
    And a row a player a game: his own snaps going in."""
    teams, men = [], []
    for team, tg in s.sort_values(["season", "week"]).groupby("team", sort=False):
        hist, last, name, grp = {}, {}, {}, {}
        for i, (gid, here) in enumerate(tg.groupby("game_id", sort=False)):
            season, week = int(here["season"].iloc[0]), int(here["week"].iloc[0])
            row = {"team": team, "game_id": gid, "sn_season": season, "sn_week": week}
            for g, n in TOP.items():
                top = sorted(((float(np.mean(hist[p][-4:])), p) for p in hist
                              if grp[p] == g and i - last[p] <= 3), reverse=True)[:n]
                out = 0
                for sh, p in top:
                    if sh < 0.4:
                        continue
                    st = status.get((season, week, team, name[p]))
                    if st in ("Out", "Doubtful") or (st is None and i - last[p] >= 2):
                        out += 1
                row[f"{g}_out"] = out
            teams.append(row)
            for p, g, pct, nm in zip(here["pfr_player_id"], here["grp"], here["pct"], here["nm"]):
                v = hist.get(p, [])
                men.append({"pfr": p, "game_id": gid, "sn_last": v[-1] if v else np.nan,
                            "sn_4": float(np.mean(v[-4:])) if v else np.nan,
                            "sn_prev": float(np.mean(v[-5:-1])) if len(v) >= 3 else np.nan,
                            "back": int(bool(v) and i - last[p] >= 2 and i - last[p] <= 8),
                            "sn_games": len(v)})
                hist.setdefault(p, []).append(float(pct))
                last[p], name[p], grp[p] = i, nm, g
    return pd.DataFrame(teams), pd.DataFrame(men)


def props():
    d, pos = F.load()
    t = F.features(F.team_games(d, pos))
    t["rest"] = t.groupby("posteam")["game_date"].diff().dt.days
    s = snaps()
    teams, men = who_is_out(s, reports())
    t = t.merge(teams.rename(columns={"team": "posteam"}), on=["posteam", "game_id"], how="left")
    opp = teams[["team", "game_id", "db_out", "dl_out", "lb_out"]].rename(
        columns={"team": "defteam", "db_out": "opp_db_out", "dl_out": "opp_dl_out",
                 "lb_out": "opp_lb_out"})
    t = t.merge(opp, on=["defteam", "game_id"], how="left")
    # which team each player was on that day
    on = pd.concat([d[["game_id", "game_date", "posteam", c]].rename(columns={c: "gsis"})
                    for c in ("passer_player_id", "rusher_player_id")]
                   + [pd.read_csv(F.CACHE / f"pbp_{y}.csv.gz",
                                  usecols=["game_id", "game_date", "posteam",
                                           "receiver_player_id", "season_type"],
                                  low_memory=False).query("season_type == 'REG'")
                      .rename(columns={"receiver_player_id": "gsis"})
                      [["game_id", "game_date", "posteam", "gsis"]] for y in F.SEASONS])
    on = on.dropna().drop_duplicates(["game_id", "gsis"])
    on["game_date"] = pd.to_datetime(on["game_date"])
    rec = pd.read_pickle(F.CACHE / "record.pkl")
    for c in ("line", "actual", "projection", "need"):
        rec[c] = rec[c].astype(float)
    rec["game_date"] = pd.to_datetime(rec["game_date"])
    rec = rec[(rec["actual"] != rec["line"]) & rec["market_code"].isin(MARKETS)]
    rec = rec.drop(columns=["season"]).merge(on, on=["gsis", "game_date"])
    keep = ["game_id", "posteam", "season", "week", "fav", "total_line", "implied", "home",
            "indoor", "wind", "rest", "q_main", "q_new", "q_share", "q_same", "ol_out",
            "wr_out", "te_out", "rb_out", "opp_db_out", "opp_dl_out", "opp_lb_out",
            "o_neutral_8", "o_plays_8", "d_plays_6", "d_pass_oe_6", "o_pass_oe_8",
            "d_rush_epa_6", "d_pass_epa_6", "o_epa_8", "d_epa_6", "o_ppd_8", "d_press_6",
            "o_press_8", "d_adot_6", "d_cpoe_6"]
    j = rec.merge(t[keep], on=["game_id", "posteam"])
    ros = pd.concat([pd.read_csv(F.CACHE / f"roster_{y}.csv", usecols=["gsis_id", "pfr_id"],
                                 low_memory=False) for y in F.SEASONS]).dropna() \
            .drop_duplicates("gsis_id").set_index("gsis_id")["pfr_id"].to_dict()
    j["pfr"] = j["gsis"].map(ros)
    j = j.merge(men, on=["pfr", "game_id"], how="left")
    j["need_o"] = np.where(j["recommended_side"] == "over", j["need"], 1.045 - j["need"])
    j["need_u"] = 1.045 - j["need_o"]
    return j


def situations(j):
    hi = lambda c, q: j[c] >= j.groupby("season")[c].transform(lambda v: v.quantile(q))  # noqa: E731
    lo = lambda c, q: j[c] <= j.groupby("season")[c].transform(lambda v: v.quantile(q))  # noqa: E731
    late = j["week"] >= 4
    return {
        "a starting lineman out (wk 4+)": (j["ol_out"] >= 1) & late,
        "two linemen out (wk 4+)": (j["ol_out"] >= 2) & late,
        "a top-3 receiver out (wk 4+)": (j["wr_out"] >= 1) & late,
        "the tight end out (wk 4+)": (j["te_out"] >= 1) & late,
        "a top-2 back out (wk 4+)": (j["rb_out"] >= 1) & late,
        "opp missing 2+ defensive backs (wk 4+)": (j["opp_db_out"] >= 2) & late,
        "opp missing 1+ defensive back (wk 4+)": (j["opp_db_out"] >= 1) & late,
        "opp missing a defensive lineman (wk 4+)": (j["opp_dl_out"] >= 1) & late,
        "opp missing a linebacker (wk 4+)": (j["opp_lb_out"] >= 1) & late,
        "his first game back": j["back"] == 1,
        "his snaps rising (last game 15+ above before)": (j["sn_last"] - j["sn_prev"]) >= 0.15,
        "his snaps falling (last game 15+ below before)": (j["sn_last"] - j["sn_prev"]) <= -0.15,
        "played under half the snaps last game": j["sn_last"] < 0.5,
        "played 80%+ of the snaps last game": j["sn_last"] >= 0.8,
        "short week (under 6 days)": j["rest"] < 6,
        "long rest (10+ days)": j["rest"] >= 10,
        "quarterback is not the usual starter": j["q_main"] == 0,
        "quarterback has under 4 starts": j["q_new"] == 1,
        "quarterback changed since last week": j["q_same"] == 0,
        "favoured by 6+": j["fav"] >= 6,
        "favoured by 3 to 6": (j["fav"] >= 3) & (j["fav"] < 6),
        "underdog by 3 to 6": (j["fav"] <= -3) & (j["fav"] > -6),
        "underdog by 6+": j["fav"] <= -6,
        "total 48+": j["total_line"] >= 48,
        "total under 41": j["total_line"] < 41,
        "team total 26+": j["implied"] >= 26,
        "team total under 19": j["implied"] < 19,
        "wind 14+ outdoors": j["wind"] >= 14,
        "indoors": j["indoor"] == 1,
        "home": j["home"] == 1,
        "away": j["home"] == 0,
        "week 13 on": j["week"] >= 13,
        "weeks 1 to 3": j["week"] <= 3,
        "opp allows the most plays (top fifth)": hi("d_plays_6", 0.8),
        "opp allows the fewest plays (bottom fifth)": lo("d_plays_6", 0.2),
        "opp gets thrown on most (top fifth)": hi("d_pass_oe_6", 0.8),
        "opp gets run on most (bottom fifth pass rate)": lo("d_pass_oe_6", 0.2),
        "opp worst against the run (top fifth EPA)": hi("d_rush_epa_6", 0.8),
        "opp best against the run (bottom fifth)": lo("d_rush_epa_6", 0.2),
        "opp worst against the pass (top fifth EPA)": hi("d_pass_epa_6", 0.8),
        "opp best against the pass (bottom fifth)": lo("d_pass_epa_6", 0.2),
        "opp pressures most (top fifth)": hi("d_press_6", 0.8),
        "own offence best (top fifth EPA)": hi("o_epa_8", 0.8),
        "own offence worst (bottom fifth EPA)": lo("o_epa_8", 0.2),
        "own offence throws most (top fifth)": hi("o_pass_oe_8", 0.8),
        "own offence runs most (bottom fifth)": lo("o_pass_oe_8", 0.2),
    }


def main():
    j = props()
    j = j[j["season"] >= 2025]
    j.to_pickle(F.CACHE / "situations.pkl")
    print(f"{len(j):,} priced props, 2025 and 2026")
    sit = situations(j)
    found = []
    for market in MARKETS:
        for pos in (None,) if not market.startswith(("rec", "rush")) else (None, "RB", "WR", "TE", "QB"):
            m = j["market_code"] == market
            if pos:
                m &= j["position"] == pos
            if m.sum() < 300:
                continue
            for name, mask in sit.items():
                x = j[m & mask.fillna(False)]
                if len(x) < 250:
                    continue
                cells = []
                for s, v in x.groupby("season"):
                    if len(v) < 60:
                        cells = []
                        break
                    cells.append((int(s), len(v), v["over"].mean(), v["need_o"].mean(),
                                  v["need_u"].mean()))
                if len(cells) < 2:
                    continue
                over_edge = min(c[2] - c[3] for c in cells)
                under_edge = min((1 - c[2]) - c[4] for c in cells)
                side, edge = ("over", over_edge) if over_edge > under_edge else ("under", under_edge)
                if edge >= 0.02:
                    right = x["over"].mean() if side == "over" else 1 - x["over"].mean()
                    need = x["need_o"].mean() if side == "over" else x["need_u"].mean()
                    found.append((edge, market, pos or "all", name, side, len(x), right, need,
                                  cells))
    found.sort(reverse=True)
    print(f"{len(found)} cells clear their price by two points in both seasons:\n")
    for edge, market, pos, name, side, n, right, need, cells in found:
        by = ", ".join(f"{s}: {(o if side == 'over' else 1 - o):.1%} on {k}" for s, k, o, a, b in cells)
        print(f"  {market} ({pos}), {name}: the {side} {right:.1%} on {n}, needed {need:.1%}; {by}")


if __name__ == "__main__":
    sys.exit(main())
