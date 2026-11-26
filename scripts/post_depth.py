"""The numbers under a pick, worked out from every play of the season.

"He went under in six of his last ten" is where a post used to stop, and it
is not a reason. A reason is how much of the offence runs through him and
what the defence across from him gives up. This reads the season's
play-by-play (nflverse, the same source the models are fed from) and works
out, for one pick:

    catches      his share of his team's targets, his targets a game lately,
                 and the catches the opponent allows to his position
    rushing      his share of his team's carries, his carries a game lately,
                 and what the opponent allows on the ground
    passing TDs  his touchdown passes a game, his throws inside the 20, and
                 the touchdown passes the opponent allows
    passing      his attempts a game, his yards a throw, and what the
                 opponent allows through the air

The three numbers for a market are fixed. They are not chosen pick by pick to
flatter the call, so sometimes one of them argues against it, and it is
printed anyway. Each opponent number carries its rank out of 32.

The file is kept in marketing/.cache and fetched again when it is half a day
old. Regular season only.
"""

from __future__ import annotations

import time
import urllib.request
from pathlib import Path

import pandas as pd

CACHE = Path(__file__).resolve().parent.parent / "marketing" / ".cache"
BASE = "https://github.com/nflverse/nflverse-data/releases/download"
COLS = ["game_id", "week", "season_type", "posteam", "defteam", "pass_attempt",
        "rush_attempt", "complete_pass", "sack", "qb_scramble", "yards_gained",
        "air_yards", "yardline_100", "pass_touchdown", "receiver_player_id",
        "rusher_player_id", "passer_player_id", "two_point_attempt"]
NICK = {
    "Cardinals": "ARI", "Falcons": "ATL", "Ravens": "BAL", "Bills": "BUF",
    "Panthers": "CAR", "Bears": "CHI", "Bengals": "CIN", "Browns": "CLE",
    "Cowboys": "DAL", "Broncos": "DEN", "Lions": "DET", "Packers": "GB",
    "Texans": "HOU", "Colts": "IND", "Jaguars": "JAX", "Chiefs": "KC",
    "Raiders": "LV", "Chargers": "LAC", "Rams": "LA", "Dolphins": "MIA",
    "Vikings": "MIN", "Patriots": "NE", "Saints": "NO", "Giants": "NYG",
    "Jets": "NYJ", "Eagles": "PHI", "Steelers": "PIT", "49ers": "SF",
    "Seahawks": "SEA", "Buccaneers": "TB", "Titans": "TEN", "Commanders": "WAS",
}
ABBR = {v: k for k, v in NICK.items()}
SAME = {"LAR": "LA", "WSH": "WAS", "JAC": "JAX"}
POS_WORD = {"WR": "wide receivers", "TE": "tight ends", "RB": "running backs"}
_DATA = {}


def fetch(name, url, hours=12):
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / name
    if not f.exists() or time.time() - f.stat().st_mtime > hours * 3600:
        req = urllib.request.Request(url, headers={"User-Agent": "post-build"})
        with urllib.request.urlopen(req, timeout=120) as r:
            f.write_bytes(r.read())
    return f


def load(season):
    if season in _DATA:
        return _DATA[season]
    d = pd.read_csv(fetch(f"pbp_{season}.csv.gz",
                          f"{BASE}/pbp/play_by_play_{season}.csv.gz"),
                    usecols=COLS, low_memory=False)
    d = d[(d["season_type"] == "REG") & (d["two_point_attempt"] != 1)]
    ros = pd.read_csv(fetch(f"roster_{season}.csv",
                            f"{BASE}/rosters/roster_{season}.csv"),
                      usecols=["gsis_id", "position"], low_memory=False)
    pos = ros.dropna().drop_duplicates("gsis_id").set_index("gsis_id")["position"]
    pos = pos.replace({"FB": "RB"})
    thrown = d[(d["pass_attempt"] == 1) & (d["sack"] != 1)].copy()
    tg = thrown[thrown["receiver_player_id"].notna()].copy()
    tg["pos"] = tg["receiver_player_id"].map(pos)
    runs = d[(d["rush_attempt"] == 1) & d["rusher_player_id"].notna()].copy()
    _DATA[season] = {"all": d, "thrown": thrown, "targets": tg, "runs": runs}
    return _DATA[season]


def ordinal(n):
    if 10 <= n % 100 <= 20:
        return f"{n}th"
    return f"{n}{ {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def ranked(per_team, team):
    """A team's figure, and where it stands: 1 is the fewest allowed."""
    if team not in per_team.index:
        return None, None, ""
    r = int(per_team.rank(method="min")[team])
    n = len(per_team)
    words = (f"{ordinal(r)} fewest" if r <= n / 2 else f"{ordinal(n - r + 1)} most")
    if r == 1:
        words = "the fewest"
    if r == n:
        words = "the most"
    return float(per_team[team]), r, words


def per_game(frame, by, value=None, how="size"):
    games = frame.groupby(by)["game_id"].nunique()
    if how == "size":
        tot = frame.groupby(by).size()
    else:
        tot = frame.groupby(by)[value].sum()
    return (tot / games).dropna()


def fmt(v, places=1):
    return f"{v:.{places}f}"


def with_him(frame, mine):
    """His team's plays in the games he played, whichever team that was.

    A player traded in mid-season has two. Counting only today's team gave a
    receiver 59% of the targets.
    """
    keys = mine[["game_id", "posteam"]].drop_duplicates()
    return frame.merge(keys, on=["game_id", "posteam"])


def catches(D, gsis, team, opp, pos, last=4):
    tg = D["targets"]
    mine = tg[tg["receiver_player_id"] == gsis]
    if mine.empty:
        return [], []
    games = list(mine.sort_values("week")["game_id"].unique())
    team_tg = with_him(tg, mine)
    share = len(mine) / max(len(team_tg), 1)
    recent = mine[mine["game_id"].isin(games[-last:])]
    per = len(recent) / max(len(games[-last:]), 1)
    rate = mine["complete_pass"].mean()
    tiles = [(f"{share * 100:.0f}%", "OF HIS TEAM'S", "TARGETS"),
             (fmt(per), "TARGETS A GAME", f"LAST {len(games[-last:])}")]
    says = [f"He draws {share * 100:.0f}% of his team's targets and has seen "
            f"{fmt(per)} a game over his last {len(games[-last:])}, catching "
            f"{rate * 100:.0f}% of what is thrown to him."]
    if pos in POS_WORD:
        allowed = per_game(tg[(tg["pos"] == pos) & (tg["complete_pass"] == 1)], "defteam")
        v, r, words = ranked(allowed, opp)
        if v is not None:
            tiles.append((fmt(v), f"CATCHES THE {ABBR.get(opp, opp).upper()} ALLOW",
                          f"TO {pos}S. {words.upper()}"))
            says.append(f"The {ABBR.get(opp, opp)} allow {fmt(v)} catches a game "
                        f"to {POS_WORD[pos]}, {words} in the league.")
    return tiles, says


def rushing(D, gsis, team, opp, pos, yards, last=4):
    runs = D["runs"]
    mine = runs[runs["rusher_player_id"] == gsis]
    if mine.empty:
        return [], []
    games = list(mine.sort_values("week")["game_id"].unique())
    team_runs = with_him(runs, mine)
    share = len(mine) / max(len(team_runs), 1)
    recent = mine[mine["game_id"].isin(games[-last:])]
    per = len(recent) / max(len(games[-last:]), 1)
    ypc = mine["yards_gained"].sum() / max(len(mine), 1)
    tiles = [(f"{share * 100:.0f}%", "OF HIS TEAM'S", "CARRIES"),
             (fmt(per), "CARRIES A GAME", f"LAST {len(games[-last:])}")]
    says = [f"He takes {share * 100:.0f}% of his team's carries, {fmt(per)} a "
            f"game over his last {len(games[-last:])}, at {fmt(ypc)} yards a carry."]
    backs = runs[runs["qb_scramble"] != 1]
    if yards:
        allowed = (backs.groupby("defteam")["yards_gained"].sum()
                   / backs.groupby("defteam").size())
        v, r, words = ranked(allowed, opp)
        if v is not None:
            tiles.append((fmt(v), f"YARDS A CARRY THE {ABBR.get(opp, opp).upper()}",
                          f"ALLOW. {words.upper()}"))
            says.append(f"The {ABBR.get(opp, opp)} allow {fmt(v)} yards a carry, "
                        f"{words} in the league.")
    else:
        allowed = per_game(backs, "defteam")
        v, r, words = ranked(allowed, opp)
        if v is not None:
            tiles.append((fmt(v), f"CARRIES A GAME AGAINST THE "
                          f"{ABBR.get(opp, opp).upper()}.", words.upper()))
            says.append(f"Teams run {fmt(v)} times a game on the "
                        f"{ABBR.get(opp, opp)}, {words} in the league.")
    return tiles, says


def passing(D, gsis, team, opp, market):
    th = D["thrown"]
    mine = th[th["passer_player_id"] == gsis]
    if mine.empty:
        return [], []
    n = mine["game_id"].nunique()
    if market == "pass_td":
        tds = mine["pass_touchdown"].sum() / n
        red = mine[mine["yardline_100"] <= 20]
        red_g = len(red) / n
        tiles = [(fmt(tds), "TOUCHDOWN PASSES", "A GAME"),
                 (fmt(red_g), "THROWS A GAME", "INSIDE THE 20")]
        says = [f"He has thrown {fmt(tds)} touchdown passes a game on "
                f"{fmt(red_g)} throws a game inside the 20."]
        allowed = per_game(th, "defteam", "pass_touchdown", "sum")
        v, r, words = ranked(allowed, opp)
        if v is not None:
            tiles.append((fmt(v), f"TD PASSES THE {ABBR.get(opp, opp).upper()} "
                          f"ALLOW", f"A GAME. {words.upper()}"))
            says.append(f"The {ABBR.get(opp, opp)} allow {fmt(v)} touchdown "
                        f"passes a game, {words} in the league.")
        return tiles, says
    att = len(mine) / n
    ypa = mine["yards_gained"].sum() / max(len(mine), 1)
    comp = mine["complete_pass"].mean()
    tiles = [(fmt(att), "THROWS", "A GAME"),
             (f"{comp * 100:.0f}%", "OF THEM", "COMPLETED")]
    says = [f"He throws {fmt(att)} times a game, completes {comp * 100:.0f}% "
            f"and gains {fmt(ypa)} yards a throw."]
    if market == "pass_yds":
        allowed = (th.groupby("defteam")["yards_gained"].sum()
                   / th.groupby("defteam").size())
        v, r, words = ranked(allowed, opp)
        lab = (f"YARDS A THROW THE {ABBR.get(opp, opp).upper()}", "ALLOW")
        say = f"The {ABBR.get(opp, opp)} allow {{}} yards a throw"
    elif market == "pass_completions":
        allowed = per_game(th, "defteam", "complete_pass", "sum")
        v, r, words = ranked(allowed, opp)
        lab = (f"COMPLETIONS THE {ABBR.get(opp, opp).upper()} ALLOW", "A GAME")
        say = f"The {ABBR.get(opp, opp)} allow {{}} completions a game"
    else:
        allowed = per_game(th, "defteam")
        v, r, words = ranked(allowed, opp)
        lab = (f"THROWS A GAME AGAINST THE {ABBR.get(opp, opp).upper()}", "")
        say = f"Teams throw {{}} times a game on the {ABBR.get(opp, opp)}"
    if v is not None:
        tiles.append((fmt(v), lab[0], f"{lab[1]}. {words.upper()}".strip(". ")
                      if lab[1] else words.upper()))
        says.append(say.format(fmt(v)) + f", {words} in the league.")
    return tiles, says


def depth(p, season):
    """Three numbers and the sentences that go with them, or nothing.

    `p` needs gsis, team, opponent (abbreviations), position and market_code.
    A failure here never stops a post: the slide is drawn without the numbers
    and says so on the console.
    """
    try:
        D = load(season)
        team = SAME.get(p.get("team"), p.get("team"))
        opp = SAME.get(p.get("opponent"), p.get("opponent"))
        gsis, pos, m = p.get("gsis"), p.get("position"), p["market_code"]
        if not (gsis and team and opp):
            return [], []
        if m in ("recs", "rec_yds"):
            return catches(D, gsis, team, opp, pos)
        if m in ("rush_yds", "rush_att"):
            return rushing(D, gsis, team, opp, pos, m == "rush_yds")
        if m.startswith("pass_"):
            return passing(D, gsis, team, opp, m)
    except Exception as err:
        print(f"  no underlying numbers for {p.get('player_name')}: {err}")
    return [], []
