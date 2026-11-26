"""Pull every graded pick the site has published into one file, through the public API.

The record lives in the database on the server. Every question about it used to
mean a script, a push, a pull and a paste. But the site publishes each player's
graded picks, with the line, the price, the side, the chance we gave it and how
it came in, so the whole record can be read from anywhere.

    python lab/training/pull_record.py

Writes marketing/.cache/record.pkl: one row a graded pick. Reads only.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

API = "https://api.priorline.io/api/v1"
OUT = Path(__file__).resolve().parents[2] / "marketing" / ".cache" / "record.pkl"


def get(path, **q):
    """The site limits how fast it can be read, so a refusal waits and tries again."""
    url = f"{API}/{path}" + ("?" + urllib.parse.urlencode(q) if q else "")
    req = urllib.request.Request(url, headers={"User-Agent": "research"})
    for attempt in range(10):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except urllib.error.HTTPError as err:
            if err.code != 429 or attempt == 9:
                raise
            time.sleep(float(err.headers.get("Retry-After") or min(30, 2 ** attempt)))
        except Exception:
            if attempt == 9:
                raise
            time.sleep(2)


def players():
    out, offset = [], 0
    while True:
        page = get("players", limit=200, offset=offset)["players"]
        out += page
        if len(page) < 200:
            return out
        offset += 200


def history(p):
    try:
        rows = get(f"players/{p['id']}/edge_history", limit=500)["history"]
    except Exception:
        return []
    for r in rows:
        r["player_id"], r["player"], r["position"] = p["id"], p["name"], p["position"]
        r["gsis"], r["player_team"] = p.get("external_id"), p.get("team")
    return rows


def main():
    ps = [p for p in players() if p.get("position") in ("QB", "RB", "WR", "TE", "FB")]
    print(f"{len(ps)} skill players")
    with ThreadPoolExecutor(2) as ex:
        rows = [r for chunk in ex.map(history, ps) for r in chunk]
    d = pd.DataFrame(rows)
    d = d[d["hit"].notna()].copy()
    d["game_date"] = pd.to_datetime(d["game_date"])
    d["season"] = (d["game_date"] - pd.Timedelta(days=60)).dt.year
    OUT.parent.mkdir(parents=True, exist_ok=True)
    d.to_pickle(OUT)
    print(f"{len(d):,} graded picks, {d['game_date'].min().date()} to "
          f"{d['game_date'].max().date()}, written to {OUT}")
    print(d.groupby(["season", "edge_tier"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    sys.exit(main())
