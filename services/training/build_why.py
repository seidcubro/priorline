"""Work out the numbers under every pick on the board, for the site to show beside it.

The board shows a projection and a line. The posts and videos show why: how
much of the offence runs through the player and what the defence across from
him allows, with its rank. The site is the product and was making the weaker
argument. This puts the same numbers on it, from the same code the posts use
(pick_depth.py), so the two cannot disagree.

For each player and market on the board it stores the sentences and the three
numbers in edge_why. The API serves that table and the board draws it under
the player.

It runs after the board is built and is deliberately apart from it: its own
table, created here if it is missing, and a failure here leaves the board
exactly as it was. Nothing on the board reads this.

    python build_why.py
"""

import json
import os

import pandas as pd
from sqlalchemy import create_engine, text

import pick_depth as depth

DATABASE_URL = os.getenv("DATABASE_URL") or (
    f"postgresql+psycopg2://{os.getenv('POSTGRES_USER', 'app')}:"
    f"{os.getenv('POSTGRES_PASSWORD', 'app')}"
    f"@{os.getenv('POSTGRES_HOST', 'postgres')}:"
    f"{os.getenv('POSTGRES_PORT', '5432')}/{os.getenv('POSTGRES_DB', 'app')}"
)
SKILL = ("QB", "RB", "FB", "WR", "TE")


def short(team):
    return str(team or "").split()[-1] if team else ""


def main():
    engine = create_engine(DATABASE_URL)
    with engine.connect() as conn:
        board = pd.read_sql(text(
            "SELECT DISTINCT player_name, market_code, home_team, away_team, "
            "commence_time FROM prop_edges"), conn)
        people = pd.read_sql(text(
            "SELECT name, external_id, team, position, status FROM players "
            "WHERE external_id IS NOT NULL"), conn)
    if board.empty:
        print("the board is empty, so there is nothing to explain")
        return
    # One player a name: a skill position before anything else, active before not.
    people = people[people["position"].isin(SKILL)].copy()
    people["rank"] = (people["status"] != "ACT").astype(int)
    people = people.sort_values("rank").drop_duplicates("name").set_index("name")

    rows, missed = [], 0
    for r in board.itertuples(index=False):
        if r.player_name not in people.index:
            missed += 1
            continue
        who = people.loc[r.player_name]
        sides = [depth.NICK.get(short(r.away_team)), depth.NICK.get(short(r.home_team))]
        mine = depth.SAME.get(who["team"], who["team"])
        if mine not in sides:
            missed += 1
            continue
        when = pd.Timestamp(r.commence_time)
        season = when.year if when.month >= 3 else when.year - 1
        p = {"player_name": r.player_name, "gsis": who["external_id"], "team": mine,
             "opponent": next(x for x in sides if x != mine),
             "position": who["position"], "market_code": r.market_code}
        tiles, says = depth.depth(p, season)
        if says:
            rows.append({"player_name": r.player_name, "market_code": r.market_code,
                         "why": json.dumps({"says": says, "tiles": tiles})})
    with engine.begin() as conn:
        conn.execute(text(
            "CREATE TABLE IF NOT EXISTS edge_why ("
            " player_name TEXT NOT NULL, market_code TEXT NOT NULL, why TEXT NOT NULL,"
            " built_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,"
            " PRIMARY KEY (player_name, market_code))"))
        conn.execute(text("DELETE FROM edge_why"))
        if rows:
            conn.execute(text(
                "INSERT INTO edge_why (player_name, market_code, why) "
                "VALUES (:player_name, :market_code, :why)"), rows)
    print(f"the numbers under {len(rows)} of {len(board)} board picks written to "
          f"edge_why; {missed} had no player or team to work from")


if __name__ == "__main__":
    main()
