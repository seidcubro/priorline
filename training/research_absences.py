"""Who is missing: pass catchers, linemen and defenders, from the snap counts.

research_football.py knew the quarterback and the backs. It did not know that
a team was without its top receiver, two starting linemen, or that the defence
across from it was missing half its secondary. The book knows, but it may not
move a volume line as far as it should.

Snap counts (nflverse, from Pro Football Reference) name everyone who played in
every game with his position. For each team and game, the men who had been
playing over the team's last three games and did not take a snap in this one
are missing, weighted by how much of the offence or defence they had been on
the field for:

    offence   receivers and tight ends missing (snap share), whether the most
              used receiver is one of them, how many of the five most used
              linemen are missing, backs missing, and the share of targets
              that belonged to men who are not here
    defence   how many of the five most used defensive backs, four most used
              linemen and three most used linebackers are missing

A played game answers who was missing for itself. That is a small look ahead:
a man hurt in the first quarter counts as playing, and a healthy scratch counts
as missing. Before a real game the same thing comes from the injury report.

Each market is fitted with and without them and read against the line, and the
props are then read with no model at all: on the games where the most is
missing, did the over or the under beat its price, in each season.

    python lab/training/research_absences.py
"""

from __future__ import annotations

import sys
import warnings

import numpy as np
import pandas as pd

import research_football as F

warnings.filterwarnings("ignore")
GROUP = {"WR": "wr", "TE": "te", "RB": "rb", "FB": "rb", "T": "ol", "G": "ol", "C": "ol",
         "OL": "ol", "CB": "db", "S": "db", "DB": "db", "FS": "db", "SS": "db", "DL": "dl",
         "DE": "dl", "DT": "dl", "NT": "dl", "LB": "lb", "OLB": "lb", "ILB": "lb", "MLB": "lb"}
# Snap counts only. "Had no target today" was tried and is a look at the game
# itself: a team that throws 22 times targets fewer men, so it reads as missing
# receivers and predicts its own low attempts.
ABS = ["wr_gone", "te_gone", "skill_gone", "wr1_gone", "ol_gone", "rbs_gone",
       "opp_db_gone", "opp_dl_gone", "opp_lb_gone", "opp_def_gone", "own_def_gone"]


def snaps():
    s = pd.concat([pd.read_csv(F.CACHE / f"snap_counts_{y}.csv") for y in F.SEASONS],
                  ignore_index=True)
    s = s[s["game_type"] == "REG"]
    s["grp"] = s["position"].map(GROUP)
    s["pct"] = np.where(s["grp"].isin(["db", "dl", "lb"]), s["defense_pct"], s["offense_pct"])
    return s[s["grp"].notna() & (s["pct"] > 0)]


def missing(s):
    """One row a team a game: who had been playing and is not."""
    rows = []
    for team, tg in s.sort_values(["season", "week"]).groupby("team", sort=False):
        hist, last = {}, {}
        for i, (gid, here) in enumerate(tg.groupby("game_id", sort=False)):
            act = set(here["pfr_player_id"])
            usual = {p: (g, float(np.mean(v[-4:]))) for p, (g, v) in hist.items()
                     if i - last[p] <= 3}
            out = {"team": team, "game_id": gid}
            by = {}
            for p, (g, share) in usual.items():
                by.setdefault(g, []).append((share, p))
            for g in by:
                by[g].sort(reverse=True)
            gone = lambda g, n: sum(1 for sh, p in by.get(g, [])[:n]  # noqa: E731
                                    if p not in act and sh >= 0.4)
            share_gone = lambda g: sum(sh for sh, p in by.get(g, [])  # noqa: E731
                                       if p not in act and sh >= 0.25)
            out["wr_gone"], out["te_gone"] = share_gone("wr"), share_gone("te")
            out["rbs_gone"] = share_gone("rb")
            out["skill_gone"] = out["wr_gone"] + out["te_gone"]
            out["wr1_gone"] = gone("wr", 1)
            out["ol_gone"] = gone("ol", 5)
            out["db_gone"], out["dl_gone"], out["lb_gone"] = gone("db", 5), gone("dl", 4), gone("lb", 3)
            out["def_gone"] = out["db_gone"] + out["dl_gone"] + out["lb_gone"]
            out["known"] = int(i >= 2)
            rows.append(out)
            for p, g, pct in zip(here["pfr_player_id"], here["grp"], here["pct"]):
                hist.setdefault(p, (g, []))[1].append(pct)
                last[p] = i
    return pd.DataFrame(rows)


def targets_gone(d):
    """The share of a team's targets that belonged to men with no target today."""
    th = d[(d["pass_attempt"] == 1) & (d["sack"] != 1) & d["receiver_player_id"].notna()]
    g = th.groupby(["game_date", "game_id", "posteam", "receiver_player_id"]).size() \
          .rename("tgt").reset_index().sort_values(["game_date", "game_id"])
    g["share"] = g["tgt"] / g.groupby(["game_id", "posteam"])["tgt"].transform("sum")
    rows = []
    for team, tg in g.groupby("posteam", sort=False):
        hist, last = {}, {}
        for i, (gid, here) in enumerate(tg.groupby("game_id", sort=False)):
            act = set(here["receiver_player_id"])
            usual = {p: float(np.mean(v[-4:])) for p, v in hist.items() if i - last[p] <= 3}
            top = max(usual, key=usual.get) if usual else None
            rows.append({"posteam": team, "game_id": gid,
                         "tgt_gone": sum(v for p, v in usual.items() if p not in act),
                         "top_tgt_gone": int(top is not None and top not in act
                                             and usual[top] >= 0.18)})
            for p, sh in zip(here["receiver_player_id"], here["share"]):
                hist.setdefault(p, []).append(sh)
                last[p] = i
    return pd.DataFrame(rows)


def load_pbp():
    cols = ["game_id", "game_date", "posteam", "pass_attempt", "sack", "receiver_player_id",
            "passer_player_id", "yards_gained", "season_type", "two_point_attempt", "season"]
    d = pd.concat([pd.read_csv(F.CACHE / f"pbp_{s}.csv.gz", usecols=cols, low_memory=False)
                   for s in F.SEASONS], ignore_index=True)
    d = d[(d["season_type"] == "REG") & (d["two_point_attempt"] != 1) & d["posteam"].notna()]
    d["game_date"] = pd.to_datetime(d["game_date"])
    return d


def table():
    d, pos = F.load()
    t = F.features(F.team_games(d, pos))
    m = missing(snaps())
    t = t.merge(m.rename(columns={"team": "posteam", "def_gone": "own_def_gone"})
                .drop(columns=["db_gone", "dl_gone", "lb_gone"]),
                on=["posteam", "game_id"], how="left")
    opp = m[["team", "game_id", "db_gone", "dl_gone", "lb_gone", "def_gone"]].rename(
        columns={"team": "defteam", "db_gone": "opp_db_gone", "dl_gone": "opp_dl_gone",
                 "lb_gone": "opp_lb_gone", "def_gone": "opp_def_gone"})
    t = t.merge(opp, on=["defteam", "game_id"], how="left")
    p = load_pbp()
    t = t.merge(targets_gone(p), on=["posteam", "game_id"], how="left")
    th = p[(p["pass_attempt"] == 1) & (p["sack"] != 1)]
    qy = th.groupby(["game_id", "posteam", "passer_player_id"])["yards_gained"].sum() \
           .rename("q_yds").reset_index().rename(columns={"passer_player_id": "qb"})
    t = t.merge(qy, on=["game_id", "posteam", "qb"], how="left")
    q = t.groupby("qb", sort=False)
    t["q_yds_4"], t["q_yds_8"] = F.roll(q, "q_yds", 4, 1), F.roll(q, "q_yds", 8, 1)
    return d, pos, t


def slices(j, label):
    """No model: on the props where the most is missing, what did each side do."""
    j = j.copy()
    j["need_o"] = np.where(j["recommended_side"] == "over", j["need"], 1.045 - j["need"])
    print(f"  {label}, with no model, on the games where the most is missing:")
    for name, mask in (("every prop", j["ol_gone"] >= -1),
                       ("top receiver missing", j["wr1_gone"] >= 1),
                       ("receivers worth a full snap share missing", j["skill_gone"] >= 1.0),
                       ("no lineman missing", j["ol_gone"] == 0),
                       ("two or more linemen missing", j["ol_gone"] >= 2),
                       ("one or more linemen missing", j["ol_gone"] >= 1),
                       ("opponent missing 2+ defensive backs", j["opp_db_gone"] >= 2),
                       ("opponent missing 3+ defenders", j["opp_def_gone"] >= 3),
                       ("nothing missing on offence", (j["ol_gone"] == 0) & (j["skill_gone"] < 0.3))):
        x = j[mask.fillna(False)]
        if len(x) < 40:
            print(f"    {name}: {len(x)} props, too few")
            continue
        by = ", ".join(f"{int(s)}: over {v['over'].mean():.1%} on {len(v)}"
                       for s, v in x.groupby("season_x"))
        print(f"    {name}: {len(x)} props, over {x['over'].mean():.1%} (needed "
              f"{x['need_o'].mean():.1%}, under needed {1.045 - x['need_o'].mean():.1%}); "
              f"{by}; line off {F.mae(x['actual'], x['line']):.2f}, published off "
              f"{F.mae(x['actual'], x['projection']):.2f}, published side "
              f"{x['hit'].mean():.1%}")


def main():
    d, pos, t = table()
    rec = pd.read_pickle(F.CACHE / "record.pkl")
    for c in ("line", "actual", "projection", "need"):
        rec[c] = rec[c].astype(float)
    rec["game_date"] = pd.to_datetime(rec["game_date"])
    print(f"{len(t):,} team games; mean missing: receivers {t['skill_gone'].mean():.2f} of a "
          f"snap share, linemen {t['ol_gone'].mean():.2f}, targets {t['tgt_gone'].mean():.1%}")
    qt = t[t["q_att"] >= 10]
    for market, target, base in (("pass_att", "q_att", F.STARTER),
                                 ("pass_completions", "q_cmp", F.STARTER),
                                 ("pass_yds", "q_yds", F.STARTER + ["q_yds_4", "q_yds_8",
                                                                     "x_epa", "x_cpoe", "q_adot_8",
                                                                     "q_epa_8", "d_pass_epa_6",
                                                                     "d_adot_6"])):
        print(f"\n{market.upper()}")
        r = rec[(rec["market_code"] == market) & (rec["actual"] != rec["line"])]
        for label, cols in (("without who is missing", base), ("with who is missing", base + ABS)):
            p = F.walk(qt, target, cols)
            print(f"  {label}: off by " + ", ".join(
                f"{s}: {F.mae(x[target], x['pred']):.2f}" for s, x in p.groupby("season")))
            big = p[(p["ol_gone"] >= 2) | (p["skill_gone"] >= 1.0)]
            print(f"    starts with 2+ linemen or a full receiver share missing ({len(big)}): "
                  f"off by {F.mae(big[target], big['pred']):.2f}")
            j = r.merge(p, left_on=["gsis", "game_date"], right_on=["qb", "game_date"])
            if len(j) >= 60:
                F.against_line(j, "against the line")
        if len(j) >= 60:
            slices(j, market)

    print("\nRUSH_ATT")
    team = F.walk(t, "rb_runs", F.LEAN + ["o_rb_runs_4", "o_rb_runs_8", "d_rb_runs_6"], "p_runs")
    b = F.backs(d, pos, t)
    keep = list(dict.fromkeys(["game_id", "posteam"] + [c for c in t.columns
                                                         if c in F.RB or c in ABS]))
    b = b.merge(t[keep], on=["game_id", "posteam"], how="left") \
         .merge(team[["game_id", "posteam", "p_runs"]], on=["game_id", "posteam"], how="left")
    b["p_runs"] = b["p_runs"].fillna(b["o_rb_runs_8"])
    b["exp_car"] = b["p_runs"] * b["lifted"]
    for market, stat in (("rush_att", "car"), ("rush_yds", "yds")):
        r = rec[(rec["market_code"] == market) & (rec["actual"] != rec["line"])
                & (rec["position"] == "RB")]
        print(f" {market}")
        for label, cols in (("without who is missing", F.RB), ("with who is missing", F.RB + ABS)):
            p = F.walk(b, stat, cols)
            print(f"  {label}: off by " + ", ".join(
                f"{s}: {F.mae(x[stat], x['pred']):.2f}" for s, x in p.groupby("season")))
            j = r.merge(p, left_on=["gsis", "game_date"], right_on=["pid", "game_date"])
            if len(j) >= 60:
                F.against_line(j, "against the line")
        slices(j, market)

    for market in ("recs", "rec_yds"):
        r = rec[(rec["market_code"] == market) & (rec["actual"] != rec["line"])]
        team_of = r.merge(t[["game_date", "posteam", "game_id", "season"] + ABS]
                          .rename(columns={"season": "season_t"}), left_on="game_date",
                          right_on="game_date")
        # his own team's row: the team whose receivers include him that day
        mine = load_pbp()
        mine = mine[mine["receiver_player_id"].notna()][["game_id", "posteam",
                                                         "receiver_player_id"]].drop_duplicates()
        j = team_of.merge(mine, left_on=["game_id", "posteam", "gsis"],
                          right_on=["game_id", "posteam", "receiver_player_id"])
        j = j.rename(columns={"season": "season_x"})
        print(f"\n{market.upper()}: {len(j)} priced props on players who were targeted")
        slices(j, market)


if __name__ == "__main__":
    sys.exit(main())
