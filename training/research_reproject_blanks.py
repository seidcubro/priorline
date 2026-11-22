"""Now that the model has seen blank games, do the half-point picks hold their price?

The elite tier's whole deficit this season is one kind of pick: over 0.5
catches on a fringe player, 52.7% right where the prices needed 64.8%. The
cause I settled on is that the game log had no row for a game a man played
and recorded nothing in, so the model was never shown one and projected these
players too high. On 22 November those games were written (3,946 of them) and
every market was retrained.

The record cannot say whether that worked. It holds the projections as they
were published, before the fix, and research_count_rank.py showed that
re-ranking those old numbers changes nothing: the half-point picks lose
however they are ordered. The only test is to make the projections again.

This does that for catches, walk forward. For each season it trains the point
model and the quantile ladder on every earlier season, projects the season,
and reads the result against the prices the books posted, both sides. It does
it twice on the same props:

    without   the blank games are left out of what the model trains on. As
              near as the feature store can now get to the model that
              published this season's picks.
    with      the blank games are in. The model on the board today.

"without" is not a perfect copy of the old model. A player's recent average
in the feature store now counts his blank games in both arms, because the
features were rebuilt. So the difference between the arms is the effect of
training on the blanks alone, and the real change on the board is that plus
the lower averages. If "with" is better here, the board is better by more.

For each size of line it prints, per arm:

    the projection against what happened, and how often the man finished on
    nothing; how often the ladder said over and how often over came in; then
    the bet itself, each side the median took, how often it was right and
    what its price needed.

And the question the tier is asked: the top fifth of the gap as a share of
the line each week, the way the board ranks, overall and on its half-point
picks alone.

The fix worked if, under "with", the half-point overs are close to their
price or ahead of it where "without" is well behind, in both seasons. If the
two arms look the same, blank games were not the cause and the half-point
problem is still open.

The 2026 season is projected by a model trained through 2025 only. The board
retrains every week, so this understates it, equally in both arms.

Read-only. A few minutes.

    docker compose run --rm training python research_reproject_blanks.py
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import stats_ci as S
from psycopg2.extras import RealDictCursor

import backtest_gap_system as bt
import build_prop_edges as bp
import gap_tier
import train_quantiles as tq

tr = bt.tr

MARKETS = [m.strip() for m in os.getenv("MARKETS", "recs").split(",") if m.strip()]
SEASONS = [int(s) for s in os.getenv("SEASONS", "2025,2026").split(",")]
QUANTILES = tq.QUANTILES
ARMS = ("without", "with")


def blank_games():
    """Every game written by add_blank_games: in the app copy, not in the raw log."""
    with tr.connect() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("""
            SELECT a.player_id, a.game_date
            FROM player_game_stats_app a
            WHERE NOT EXISTS (SELECT 1 FROM player_game_stats p
                              WHERE p.player_id = a.player_id
                                AND p.game_id = a.game_id)
        """)
        b = pd.DataFrame(cur.fetchall(), columns=["player_id", "game_date"])
    b["game_date"] = pd.to_datetime(b["game_date"])
    return set(zip(b["player_id"], b["game_date"]))


def project(d, family, keep):
    """Walk forward. `keep` marks the rows the model is allowed to train on."""
    d = d.copy()
    d["pred"] = np.nan
    for q in QUANTILES:
        d[f"q{int(q * 100)}"] = np.nan
    for s in SEASONS:
        past = d[(d["season"] < s) & keep]
        now = d.index[d["season"] == s]
        if len(past) < 2000 or not len(now):
            print(f"    {s}: skipped, {len(past)} rows to train on")
            continue
        X_tr, cols = tr._build_feature_dataframe(past.copy())
        X_te, _ = tr._build_feature_dataframe(d.loc[now].copy())
        for c in cols:
            if c not in X_te.columns:
                X_te[c] = 0.0
        y_tr = past["label_actual"].astype(float).to_numpy()
        point = tr.build_model(family)
        point.fit(X_tr[cols], y_tr)
        d.loc[now, "pred"] = np.maximum(0.0, point.predict(X_te[cols]))
        tr_pred, te_pred = {}, {}
        for q in QUANTILES:
            m = tq.build_quantile_model(q)
            m.fit(X_tr[cols], y_tr)
            tr_pred[q] = np.maximum(0.0, m.predict(X_tr[cols]))
            te_pred[q] = np.maximum(0.0, m.predict(X_te[cols]))
        cal = tq.fit_calibration(tr_pred, y_tr)
        rows = [bp.calibrated_quantiles(
            {q: float(te_pred[q][i]) for q in QUANTILES}, cal)
            for i in range(len(now))]
        for q in QUANTILES:
            d.loc[now, f"q{int(q * 100)}"] = [r[q] for r in rows]
    return d


def priced(d, lines, code):
    d = d[d["pred"].notna() & d["q50"].notna()].rename(columns={"name": "player_name"}).copy()
    d["d"] = pd.to_datetime(d["as_of_game_date"])
    j = d.merge(lines, on=["player_name", "d"], how="inner")
    j = j[j["over"].notna() & j["under"].notna()].copy()
    # A whole-number line can push, and a median of several books can land
    # between two real lines. Neither is a bet anyone was offered.
    j = j[(j["line"] * 2) % 2 == 1].copy()
    j["actual"] = j["label_actual"].astype(float)
    j["went_over"] = j["actual"] > j["line"]
    j["side"] = np.where(j["q50"] > j["line"], "over", "under")
    j["price"] = np.where(j["side"] == "over", j["over"], j["under"])
    p = j["price"].astype(float)
    j["needed"] = np.where(p > 0, 100.0 / (p + 100.0), -p / (-p + 100.0))
    j["hit"] = np.where(j["side"] == "over", j["went_over"], ~j["went_over"]).astype(float)
    j["p_over"] = tq.prob_over(
        {q: j[f"q{int(q * 100)}"].to_numpy(dtype=float) for q in QUANTILES},
        j["line"].to_numpy(dtype=float))
    j["gap"] = [gap_tier.share_gap(a, b) for a, b in zip(j["q50"], j["line"])]
    j["cluster"] = j["d"].dt.strftime("%Y-%m-%d")
    j["slate"] = (j["d"] - pd.Timedelta(days=3)).dt.strftime("%G-%V")
    j["market_code"] = code
    top = pd.Series(False, index=j.index)
    for _, g in j.groupby("slate"):
        if len(g) < gap_tier.MIN_RANKABLE:
            continue
        n = int(round(len(g) * gap_tier.ELITE_SHARE))
        top.loc[g.sort_values("gap", ascending=False, kind="stable").index[:n]] = True
    j["top"] = top
    return j


def bet(s, least=25):
    if len(s) < least:
        return f"{len(s):>5}  too few"
    diff = (s["hit"] - s["needed"]).to_numpy(dtype=float)
    r, lo, hi = S.clustered_bootstrap(
        lambda i: float(np.mean(diff[i])), s["cluster"].to_numpy(), n_boot=1500)
    tail = (f" [{lo:+.1%},{hi:+.1%}]"
            if np.isfinite(lo) and np.isfinite(hi) else "")
    return (f"{len(s):>5}{s['hit'].mean():>7.1%}{s['needed'].mean():>7.1%}"
            f"{diff.mean():>+7.1%}{tail}")


def band(ln):
    return ("0.5" if ln < 1 else "1.5" if ln < 2 else "2.5" if ln < 3
            else "3.5 to 5.5" if ln < 6 else "6.5 and up")


def report(code, arms, season):
    label = "both seasons" if season is None else str(season)
    print(f"\n{code.upper()}, {label}")
    cut = {a: (j if season is None else j[j["season"] == season])
           for a, j in arms.items()}
    for name in ("0.5", "1.5", "2.5", "3.5 to 5.5", "6.5 and up"):
        sub = {a: j[j["line"].map(band) == name] for a, j in cut.items()}
        n = len(sub["with"])
        if n < 25:
            continue
        s0 = sub["with"]
        print(f"\n  line {name}: {n} props. Finished on nothing "
              f"{(s0['actual'] == 0).mean():.1%}, went over "
              f"{s0['went_over'].mean():.1%}, averaged {s0['actual'].mean():.2f}.")
        for a in ARMS:
            s = sub[a]
            print(f"    {a:<8} projected {s['pred'].mean():.2f}, median "
                  f"{s['q50'].mean():.2f}, ladder said over "
                  f"{s['p_over'].mean():.1%} of the time, took the over on "
                  f"{(s['side'] == 'over').mean():.0%}")
        print(f"    {'':<8} {'overs taken':<44}unders taken")
        for a in ARMS:
            s = sub[a]
            print(f"    {a:<8} {bet(s[s['side'] == 'over']):<44}"
                  f"{bet(s[s['side'] == 'under'])}")
    print(f"\n  the top fifth of the gap each week, the way the board ranks")
    print(f"    {'':<8} {'all of it':<44}{'its half-point picks':<44}the rest of it")
    for a in ARMS:
        t = cut[a][cut[a]["top"]]
        half = t[t["line"] < 1]
        print(f"    {a:<8} {bet(t):<44}{bet(half):<44}{bet(t[t['line'] >= 1])}")
    print(f"    {'':<8} every priced prop, whichever side the median took")
    for a in ARMS:
        print(f"    {a:<8} {bet(cut[a])}")


def main():
    blanks = blank_games()
    print(f"{len(blanks):,} blank games in the game log (a snap on offense, "
          f"nothing recorded).")
    if not blanks:
        raise SystemExit("none found: BLANK_GAMES has not been applied, so "
                         "there is nothing to compare")
    for code in MARKETS:
        d = bt.features(code)
        if d is None or len(d) < 3000:
            print(f"{code}: too few labelled rows")
            continue
        is_blank = np.array([(p, g) in blanks for p, g in
                             zip(d["player_id"], d["as_of_game_date"])])
        fam = bt.FAMILY.get(code, "vote_v1")
        print(f"{code}: lookback {d.attrs.get('lookback')}, {len(d):,} "
              f"labelled rows, {int(is_blank.sum()):,} of them blank games "
              f"({int((is_blank & (d['season'] < max(SEASONS))).sum()):,} "
              f"before {max(SEASONS)}). Family {fam}.")
        if not is_blank.any():
            print(f"{code}: no blank game reached the feature store, so the "
                  f"two arms would be the same model. Nothing to test until "
                  f"the features carry them.")
            continue
        lines = bt.lines(code)
        if lines.empty:
            print(f"{code}: no priced lines")
            continue
        arms = {}
        for a in ARMS:
            keep = pd.Series(~is_blank if a == "without" else True, index=d.index)
            print(f"  fitting '{a}' ...")
            arms[a] = priced(project(d, fam, keep), lines, code)
            print(f"  '{a}': {len(arms[a]):,} priced props projected")
        for season in [None] + SEASONS:
            report(code, arms, season)
    print("\nEach bet cell: picks, how often right, what the price needed, "
          "the difference, its interval. 'without' and 'with' are the same "
          "props at the same prices; only what the model trained on differs.")


if __name__ == "__main__":
    main()
