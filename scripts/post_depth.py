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
        "air_yards", "yardline_100", "pass_touchdown", "rush_touchdown",
        "receiver_player_id",
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


# ------------------------------------------------- the same, told as a story
#
# A slide has room for three numbers. A video has time for the whole league:
# every one of the 32 teams drawn as a bar with the one that matters lit up.
# beats() returns the numbers a video is cut from, each with the sentence that
# is said over it.


def league(per_team, team, heading, say):
    v, r, words = ranked(per_team, team)
    if v is None:
        return None
    order = per_team.sort_values()
    return {"kind": "league", "heading": heading, "team": team,
            "bars": [(t, float(x)) for t, x in order.items()],
            "value": v, "rank": r, "words": words, "say": say(fmt(v), words)}


def beats(p, season):
    """The player, his team, and the defence across from him, in that order."""
    try:
        D = load(season)
        team = SAME.get(p.get("team"), p.get("team"))
        opp = SAME.get(p.get("opponent"), p.get("opponent"))
        gsis, pos, m = p.get("gsis"), p.get("position"), p["market_code"]
        if not (gsis and team and opp):
            return []
        last = p["player_name"].split()[-1]
        mine_n, opp_n = ABBR.get(team, team), ABBR.get(opp, opp)
        tg, runs, th = D["targets"], D["runs"], D["thrown"]
        out = []
        if m in ("recs", "rec_yds"):
            me = tg[tg["receiver_player_id"] == gsis]
            if me.empty:
                return []
            games = list(me.sort_values("week")["game_id"].unique())
            share = 100 * len(me) / max(len(with_him(tg, me)), 1)
            n = len(games[-4:])
            per = len(me[me["game_id"].isin(games[-4:])]) / n
            rate = 100 * me["complete_pass"].mean()
            out.append({"kind": "count", "kicker": "HIS ROLE", "value": share,
                        "places": 0, "suffix": "%",
                        "label": f"OF THE {mine_n.upper()} TARGETS",
                        "sub": f"{fmt(per)} A GAME OVER HIS LAST {n}",
                        "say": f"{last} gets {share:.0f} percent of the {mine_n} "
                               f"targets. That's about {fmt(per)} a game over his "
                               f"last {n}, and he catches {rate:.0f} percent of them."})
            out.append(league(per_game(th, "posteam"), team,
                              "THROWS A GAME, BY OFFENCE",
                              lambda v, w: f"The {mine_n} throw it {v} times a "
                                           f"game, {w} in the league."))
            if pos in POS_WORD:
                allowed = per_game(tg[(tg["pos"] == pos) & (tg["complete_pass"] == 1)],
                                   "defteam")
                out.append(league(allowed, opp, f"CATCHES ALLOWED TO {pos}S, A GAME",
                                  lambda v, w: f"On the other side, the {opp_n} give "
                                               f"up {v} catches a game to "
                                               f"{POS_WORD[pos]}. That's {w} in the league."))
        elif m in ("rush_yds", "rush_att"):
            me = runs[runs["rusher_player_id"] == gsis]
            if me.empty:
                return []
            games = list(me.sort_values("week")["game_id"].unique())
            share = 100 * len(me) / max(len(with_him(runs, me)), 1)
            n = len(games[-4:])
            per = len(me[me["game_id"].isin(games[-4:])]) / n
            ypc = me["yards_gained"].sum() / max(len(me), 1)
            out.append({"kind": "count", "kicker": "HIS ROLE", "value": share,
                        "places": 0, "suffix": "%",
                        "label": f"OF THE {mine_n.upper()} CARRIES",
                        "sub": f"{fmt(per)} A GAME OVER HIS LAST {n}",
                        "say": f"{last} takes {share:.0f} percent of the {mine_n} "
                               f"carries. That's {fmt(per)} a game over his last "
                               f"{n}, at {fmt(ypc)} yards a carry."})
            out.append(league(per_game(runs, "posteam"), team,
                              "RUNS A GAME, BY OFFENCE",
                              lambda v, w: f"The {mine_n} run it {v} times a game, "
                                           f"{w} in the league."))
            backs = runs[runs["qb_scramble"] != 1]
            if m == "rush_yds":
                allowed = (backs.groupby("defteam")["yards_gained"].sum()
                           / backs.groupby("defteam").size())
                out.append(league(allowed, opp, "YARDS A CARRY ALLOWED",
                                  lambda v, w: f"And the {opp_n} allow {v} yards a "
                                               f"carry. That's {w} in the league."))
            else:
                out.append(league(per_game(backs, "defteam"), opp,
                                  "CARRIES FACED A GAME, BY DEFENCE",
                                  lambda v, w: f"And teams run on the {opp_n} {v} "
                                               f"times a game, {w} in the league."))
        elif m.startswith("pass_"):
            me = th[th["passer_player_id"] == gsis]
            if me.empty:
                return []
            n = me["game_id"].nunique()
            if m == "pass_td":
                tds = me["pass_touchdown"].sum() / n
                red = len(me[me["yardline_100"] <= 20]) / n
                out.append({"kind": "count", "kicker": "THE QUARTERBACK",
                            "value": tds, "places": 1, "suffix": "",
                            "label": "TOUCHDOWN PASSES A GAME",
                            "sub": f"{fmt(red)} THROWS A GAME INSIDE THE 20",
                            "say": f"{last} is throwing {fmt(tds)} touchdown passes "
                                   f"a game, on {fmt(red)} throws a game inside "
                                   f"the 20."})
                a = D["all"]
                mine = a[a["posteam"] == team]
                by_air = float(mine["pass_touchdown"].fillna(0).sum())
                by_run = float(mine["rush_touchdown"].fillna(0).sum())
                if by_air + by_run > 0:
                    share = 100 * by_air / (by_air + by_run)
                    out.append({"kind": "split", "kicker": f"THE {mine_n.upper()}",
                                "left": ("THROUGH THE AIR", share),
                                "right": ("ON THE GROUND", 100 - share),
                                "heading": "HOW THEY SCORE TOUCHDOWNS",
                                "say": f"The {mine_n} have scored {share:.0f} "
                                       f"percent of their touchdowns through the "
                                       f"air, and {100 - share:.0f} percent on "
                                       f"the ground."})
                out.append(league(per_game(th, "defteam", "pass_touchdown", "sum"),
                                  opp, "TOUCHDOWN PASSES ALLOWED A GAME",
                                  lambda v, w: f"And the {opp_n} allow {v} "
                                               f"touchdown passes a game. That's "
                                               f"{w} in the league."))
            else:
                att = len(me) / n
                comp = 100 * me["complete_pass"].mean()
                ypa = me["yards_gained"].sum() / max(len(me), 1)
                out.append({"kind": "count", "kicker": "THE QUARTERBACK",
                            "value": att, "places": 1, "suffix": "",
                            "label": "THROWS A GAME",
                            "sub": f"{comp:.0f}% COMPLETED, {fmt(ypa)} YARDS A THROW",
                            "say": f"{last} throws it {fmt(att)} times a game, "
                                   f"completes {comp:.0f} percent, and gets "
                                   f"{fmt(ypa)} yards a throw."})
                if m == "pass_yds":
                    allowed = (th.groupby("defteam")["yards_gained"].sum()
                               / th.groupby("defteam").size())
                    head, tail = "YARDS A THROW ALLOWED", "yards a throw"
                elif m == "pass_completions":
                    allowed = per_game(th, "defteam", "complete_pass", "sum")
                    head, tail = "COMPLETIONS ALLOWED A GAME", "completions a game"
                else:
                    allowed = per_game(th, "defteam")
                    head, tail = "THROWS FACED A GAME", "throws a game"
                out.append(league(allowed, opp, head,
                                  lambda v, w: f"And the {opp_n} see {v} {tail}. "
                                               f"That's {w} in the league."))
        for b in out:
            if b and b["kind"] == "league":
                b["kicker"] = f"THE {ABBR.get(b['team'], b['team']).upper()}"
        return [b for b in out if b]
    except Exception as err:
        print(f"  no story for {p.get('player_name')}: {err}")
        return []


# Where each team plays: city, latitude, longitude, and whether there is a
# roof. A roof that opens is counted as a roof.
HOME = {
    "ARI": ("Arizona", 33.5276, -112.2626, True), "ATL": ("Atlanta", 33.7554, -84.4008, True),
    "BAL": ("Baltimore", 39.2780, -76.6227, False), "BUF": ("Buffalo", 42.7738, -78.7870, False),
    "CAR": ("Charlotte", 35.2258, -80.8528, False), "CHI": ("Chicago", 41.8623, -87.6167, False),
    "CIN": ("Cincinnati", 39.0955, -84.5161, False), "CLE": ("Cleveland", 41.5061, -81.6995, False),
    "DAL": ("Dallas", 32.7473, -97.0945, True), "DEN": ("Denver", 39.7439, -105.0201, False),
    "DET": ("Detroit", 42.3400, -83.0456, True), "GB": ("Green Bay", 44.5013, -88.0622, False),
    "HOU": ("Houston", 29.6847, -95.4107, True), "IND": ("Indianapolis", 39.7601, -86.1639, True),
    "JAX": ("Jacksonville", 30.3239, -81.6373, False), "KC": ("Kansas City", 39.0489, -94.4839, False),
    "LV": ("Las Vegas", 36.0909, -115.1833, True), "LAC": ("Los Angeles", 33.9535, -118.3392, True),
    "LA": ("Los Angeles", 33.9535, -118.3392, True), "MIA": ("Miami", 25.9580, -80.2389, False),
    "MIN": ("Minneapolis", 44.9736, -93.2575, True), "NE": ("Foxborough", 42.0909, -71.2643, False),
    "NO": ("New Orleans", 29.9511, -90.0812, True), "NYG": ("New Jersey", 40.8135, -74.0745, False),
    "NYJ": ("New Jersey", 40.8135, -74.0745, False), "PHI": ("Philadelphia", 39.9008, -75.1675, False),
    "PIT": ("Pittsburgh", 40.4468, -80.0158, False), "SF": ("Santa Clara", 37.4033, -121.9694, False),
    "SEA": ("Seattle", 47.5952, -122.3316, False), "TB": ("Tampa", 27.9759, -82.5033, False),
    "TEN": ("Nashville", 36.1665, -86.7713, False), "WAS": ("Washington", 38.9076, -76.8645, False),
}


def weather(home, kickoff):
    """The forecast for the middle of the game, or that there is a roof.

    Open-Meteo, which asks for no key. `kickoff` is an aware datetime. Returns
    None if the forecast cannot be read, and then the video says nothing about
    the weather rather than guess.
    """
    import datetime as dt
    import json
    if home not in HOME:
        return None
    city, lat, lon, roof = HOME[home]
    if roof:
        return {"city": city, "roof": True,
                "say": f"They're under a roof in {city}, so the weather isn't "
                       f"a factor."}
    try:
        when = (kickoff.astimezone(dt.timezone.utc)
                + dt.timedelta(minutes=90)).replace(minute=0, second=0, microsecond=0)
        day = when.date().isoformat()
        url = ("https://api.open-meteo.com/v1/forecast?"
               f"latitude={lat}&longitude={lon}"
               "&hourly=temperature_2m,precipitation_probability,precipitation,"
               "snowfall,wind_speed_10m&temperature_unit=fahrenheit"
               f"&wind_speed_unit=mph&timezone=UTC&start_date={day}&end_date={day}")
        req = urllib.request.Request(url, headers={"User-Agent": "post-build"})
        with urllib.request.urlopen(req, timeout=30) as r:
            h = json.load(r)["hourly"]
        i = h["time"].index(when.strftime("%Y-%m-%dT%H:00"))
        temp, wind = h["temperature_2m"][i], h["wind_speed_10m"][i]
        wet = h["precipitation_probability"][i] or 0
        snow = (h["snowfall"][i] or 0) > 0 or (wet >= 40 and temp <= 33)
    except Exception as err:
        print(f"  no forecast for {city}: {err}")
        return None
    fall = "snow" if snow else "rain"
    say = f"In {city} it should be about {temp:.0f} degrees"
    if wind >= 12:
        say += f", with wind around {wind:.0f} miles an hour"
    if wet >= 40:
        say += f", and a {wet:.0f} percent chance of {fall}"
    say += "."
    if wind >= 15 or wet >= 60:
        say += " Weather like that tends to pull passing numbers down."
    elif wind < 12 and wet < 40:
        say += " Nothing in the forecast that should change the game."
    return {"city": city, "roof": False, "temp": temp, "wind": wind,
            "wet": wet, "fall": fall, "say": say}


# ------------------------------------------------------- after the game
#
# A receipt says what happened. This says why: what he was thrown, what his
# team did with the ball, against what he had been getting before. It is the
# same for a pick that lost as for one that won.


def aftermath(p, season):
    """What the game itself looked like for one posted pick, or nothing.

    For a catches pick: his targets and catches that night, his targets a
    game before it, and how often his team threw and ran.
    """
    try:
        D = load(season)
        away, home = (NICK.get(x) for x in p["game"].split(" at "))
        gsis = p.get("gsis")
        ids = [g for g in D["all"]["game_id"].unique() if g.endswith(f"_{away}_{home}")]
        if not (gsis and ids):
            return {}
        gid = sorted(ids)[-1]
        team = SAME.get(p.get("team"), p.get("team"))
        th, runs, tg = D["thrown"], D["runs"], D["targets"]
        out = {"team": ABBR.get(team, team),
               "throws": int(len(th[(th["game_id"] == gid) & (th["posteam"] == team)])),
               "runs": int(len(runs[(runs["game_id"] == gid) & (runs["posteam"] == team)]))}
        m = p["market_code"]
        if m in ("recs", "rec_yds"):
            me = tg[tg["receiver_player_id"] == gsis]
            that, before = me[me["game_id"] == gid], me[me["game_id"] != gid]
            per = before.groupby("game_id").size()
            out.update(kind="targets", got=int(len(that)),
                       made=int(that["complete_pass"].sum()),
                       before=float(per.mean()) if len(per) else 0.0,
                       most=int(per.max()) if len(per) else 0)
        elif m in ("rush_yds", "rush_att"):
            me = runs[runs["rusher_player_id"] == gsis]
            that, before = me[me["game_id"] == gid], me[me["game_id"] != gid]
            per = before.groupby("game_id").size()
            out.update(kind="carries", got=int(len(that)),
                       made=float(that["yards_gained"].sum()),
                       before=float(per.mean()) if len(per) else 0.0,
                       most=int(per.max()) if len(per) else 0)
        elif m.startswith("pass_"):
            me = th[th["passer_player_id"] == gsis]
            that, before = me[me["game_id"] == gid], me[me["game_id"] != gid]
            per = before.groupby("game_id").size()
            out.update(kind="throws", got=int(len(that)),
                       made=float(that["pass_touchdown"].sum()),
                       before=float(per.mean()) if len(per) else 0.0,
                       most=int(per.max()) if len(per) else 0)
        return out
    except Exception as err:
        print(f"  nothing on the game for {p.get('player_name')}: {err}")
        return {}
