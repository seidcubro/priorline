"""Is there anything we know about a player that the yardage line has not already priced?

Five tests on 22 November 2026 found no way to the right side of a receiving
yards line: not the model, not the catch model's side, not the corrected
chance, not the size of the line, not the books disagreeing. Each of those
asked one idea. This asks all of them at once.

Every number the feature build writes for a yardage prop (the window averages,
target share, air yards, depth of target, snaps, the opponent, the game total,
all of it), plus a few built here from the lines themselves:

    line vs window        the line over the player's window average
    swing                 how much his games vary, as a share of his average
    book lean             the over's chance in the book's own prices, hold out
    book yards per catch  the yardage line over the catch line, against the
                          player's own yards per catch over his window. Two of
                          the book's lines and one stable habit of the player:
                          if they disagree, one of the lines is off.
    catch line vs window  the catch line over his window average of catches
    catch chance          the catch model's chance of the over on catches

For each one, in each season separately: take a prop that went over and one
that did not, and how often did the number rate the first one higher? 0.50
means the line has it fully priced. A number is kept only if it leans the same
way in both seasons and by at least 0.02 in each, which about one in 150
would do by luck.

Then, for the ones kept, the props cut into fifths by that number: how often
the over came in, each season. That shows whether an end of it clears 55%.

Whatever survives is a real lead and gets its own walk-forward test before
anything is built on it. If nothing survives, the receiving yards line has
priced everything in the feature store, and the next input has to come from
outside it.

Read-only. Reads the projections research_reproject_blanks.py kept.

    MARKETS=rec_yds,rush_yds sh lab/run.sh research_what_the_line_misses.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import research_reproject_blanks as RB
from research_side_by_price import implied
from research_who_sorts import sorts

VOLUME = {"rec_yds": "recs", "rush_yds": "rush_att"}
LEAN = 0.02
SKIP = {"line", "over", "under", "actual", "label_actual", "went_over", "hit",
        "needed", "price", "season", "top", "player_id"}


def widen(j, code):
    j = j.reset_index(drop=True).copy()
    if "extra_features" in j:
        ex = pd.DataFrame([e if isinstance(e, dict) else {}
                           for e in j["extra_features"]])
        ex = ex.apply(pd.to_numeric, errors="coerce")
        ex = ex[[c for c in ex.columns if c not in j.columns]]
        j = pd.concat([j, ex], axis=1)
    avg = pd.to_numeric(j["mean"], errors="coerce")
    j["line vs window"] = np.where(avg > 0, j["line"] / avg, np.nan)
    j["swing"] = np.where(avg > 0, pd.to_numeric(j["stddev"], errors="coerce") / avg, np.nan)
    no, nu = implied(j["over"]), implied(j["under"])
    j["book lean"] = no / (no + nu)
    j["our median vs line"] = j["q50"] / j["line"]
    j["our projection vs line"] = j["pred"] / j["line"]
    vol = RB.cached_arms(VOLUME.get(code, ""))
    if vol is not None:
        v = vol["with"][["player_name", "d", "line", "mean", "p_over"]].rename(
            columns={"line": "v_line", "mean": "v_mean", "p_over": "catch chance"})
        v = v.drop_duplicates(["player_name", "d"])
        j = j.merge(v, on=["player_name", "d"], how="left")
        vm = pd.to_numeric(j["v_mean"], errors="coerce")
        book = j["line"] / j["v_line"]
        own = np.where(vm > 0, avg / vm, np.nan)
        j["book yards per catch vs his own"] = np.where(own > 0, book / own, np.nan)
        j["catch line vs window"] = np.where(vm > 0, j["v_line"] / vm, np.nan)
        j["book yards per catch"] = book
    return j


def scan(j):
    rows = []
    seasons = [s for s in RB.SEASONS if (j["season"] == s).sum() >= 300]
    for c in j.columns:
        if c in SKIP or c.startswith("q") and c[1:].isdigit():
            continue
        x = pd.to_numeric(j[c], errors="coerce") if j[c].dtype != bool else j[c].astype(float)
        if x.notna().mean() < 0.6 or x.nunique() < 10:
            continue
        a = []
        for s in seasons:
            m = (j["season"] == s) & x.notna()
            a.append(sorts(x[m], j.loc[m, "went_over"]))
        if any(not np.isfinite(v) for v in a):
            continue
        rows.append((c, int(x.notna().sum()), a))
    return rows, seasons


def fifths(j, c, seasons):
    x = pd.to_numeric(j[c], errors="coerce")
    k = j[x.notna()].copy()
    k["x"] = x[x.notna()]
    print(f"\n    {c}, in fifths from lowest to highest: how often the over came in")
    print(f"    {'fifth':<7}{'from':>9}{'to':>9}" + "".join(
        f"{str(s) + ' props':>12}{'over':>8}" for s in seasons))
    k["fifth"] = pd.qcut(k["x"].rank(method="first"), 5, labels=False)
    for f, g in k.groupby("fifth"):
        line = f"    {int(f) + 1:<7}{g['x'].min():>9.2f}{g['x'].max():>9.2f}"
        for s in seasons:
            h = g[g["season"] == s]
            line += f"{len(h):>12}{h['went_over'].mean():>8.1%}"
        print(line)


def main():
    for code in RB.MARKETS:
        arms = RB.cached_arms(code)
        if arms is None:
            print(f"{code}: no kept projections. Run research_reproject_blanks.py first.")
            continue
        j = widen(arms["with"], code)
        rows, seasons = scan(j)
        kept = [r for r in rows
                if all(v - 0.5 >= LEAN for v in r[2])
                or all(0.5 - v >= LEAN for v in r[2])]
        kept.sort(key=lambda r: -min(abs(v - 0.5) for v in r[2]))
        print(f"\n{code.upper()}: {len(j):,} priced props, over came in "
              f"{j['went_over'].mean():.1%}. {len(rows)} numbers looked at, "
              f"{len(kept)} lean the same way by {LEAN} or more in every season "
              f"(about {len(rows) / 150:.1f} would by luck).")
        print(f"    {'number':<40}{'props':>7}" + "".join(f"{s:>9}" for s in seasons))
        for c, n, a in kept[:25]:
            print(f"    {c[:39]:<40}{n:>7}" + "".join(f"{v:>9.3f}" for v in a))
        print("\n    for scale, the ones built here whether kept or not")
        for c, n, a in rows:
            if " " in c:
                print(f"    {c[:39]:<40}{n:>7}" + "".join(f"{v:>9.3f}" for v in a))
        for c, _, _ in kept[:8]:
            fifths(j, c, seasons)
    print("\nAbove 0.50 means a higher number goes with the over, below means "
          "with the under.")


if __name__ == "__main__":
    main()
