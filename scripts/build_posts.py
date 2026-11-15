"""Draw the week's posts from the live board and the live record.

Four kinds, and every figure on every one is read, never typed:

    slate     a day's elite picks: a cover that leads with the best-known
              player, one slide per pick, and the slide that says why to
              trust them.
    game      the same for one game. This is the primetime post.
    pick      one pick on its own.
    receipt   what was posted, marked right or wrong, at the line it was
              posted at. Win or lose, it goes out.

The first post was a page of sentences and nobody reads a page of sentences
on a phone. So a slide here is one player and one pick, set large, and the
reason is drawn instead of written: his last ten games as bars against the
book's line, the ones on our side filled in. The window is always ten, so a
pick his own games argue against shows it.

There are no player photographs, team logos or league marks. That is a rule in
the marketing plan and a legal one before that.

Every build writes posted.json beside the images: exactly what was put out,
at what line and price. The receipt grades that file and nothing else, so a
line that moved after the post does not quietly improve the result.

    python scripts/build_posts.py slate --date 2026-11-15
    python scripts/build_posts.py game --team Steelers
    python scripts/build_posts.py pick --player "Josh Allen"
    python scripts/build_posts.py receipt --date 2026-11-12
    python scripts/build_posts.py receipt --date 2026-11-12 --to 2026-11-16

Run a pre-game build again after the last board build before kickoff, because
a late injury report changes the board. It reads only. It needs Pillow and
fontTools, the same as build_brand.py.
"""

import argparse
import datetime as dt
import json
import re
import urllib.parse
import urllib.request
from pathlib import Path
from zoneinfo import ZoneInfo

import build_brand as bb

API = "https://api.priorline.io/api/v1"
ET = ZoneInfo("America/New_York")
WINDOW = 10
POSTS = bb.OUT / "social" / "posts"

UNIT = {
    "recs": "catches", "rec_yds": "receiving yards",
    "rush_yds": "rushing yards", "rush_att": "carries",
    "pass_yds": "passing yards", "pass_att": "pass attempts",
    "pass_completions": "completions", "pass_td": "passing TDs",
}
TIGHT = {
    "recs": "rec", "rec_yds": "rec yds", "rush_yds": "rush yds",
    "rush_att": "carries", "pass_yds": "pass yds", "pass_att": "pass att",
    "pass_completions": "comp", "pass_td": "pass TD",
}
STAT = {
    "recs": "receptions", "rec_yds": "receiving_yards",
    "rush_yds": "rushing_yards", "rush_att": "carries",
    "pass_yds": "passing_yards", "pass_att": "pass_attempts",
    "pass_completions": "completions", "pass_td": "passing_tds",
}
CHECKS = (
    "Every pick is graded in public.",
    "Losses are written in red.",
    "No sportsbook pays me.",
)
COUNT = ["No", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight",
         "Nine", "Ten"]
LEGAL = bb.LEGAL


# ------------------------------------------------------------------- data


def get(api, path, **params):
    url = f"{api.rstrip('/')}/{path}"
    if params:
        url += "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "priorline-post-build"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def num(v):
    return f"{float(v):g}" if float(v) == int(float(v)) else f"{float(v):.1f}"


def price(p):
    return f"+{int(p)}" if p > 0 else f"{int(p)}"


def short(team):
    return team.split()[-1]


def call(p):
    return f"{p['side'].capitalize()} {p['line']:.1f}"


def kick(e):
    return dt.datetime.fromisoformat(e["commence_time"]).astimezone(ET)


def clock(t):
    return f"{t.strftime('%a').upper()} {t.strftime('%I:%M').lstrip('0')} ET"


def record(api):
    """This season's elite picks under the rule the board uses today.

    The same figure the cover of the site quotes, from the same place. None
    means it has not been stored, and then no slide makes a claim.
    """
    try:
        tiers = get(api, "record/replay")["tiers"]
    except Exception:
        return None
    return next((t for t in tiers if t["tier"] == "elite"), None)


def record_line(r, brief=False):
    if not r:
        return ""
    if brief:
        return (f"Elite picks this season: {r['hit'] * 100:.1f}% right. "
                f"Prices needed {r['needed'] * 100:.1f}%.")
    out = (f"Elite picks this season were right {r['hit'] * 100:.1f}% of the "
           f"time, on {r['picks']:,} picks tiered by the rule the board uses "
           f"today. Their prices needed {r['needed'] * 100:.1f}%.")
    if r["lo"] <= 0:
        out += " On this many picks a lead that size could still be luck."
    return out


def as_pick(e, games):
    """One board row, reduced to what a post states and a receipt grades."""
    t = kick(e)
    col = STAT.get(e["market_code"])
    form = [float(g[col]) for g in games[:WINDOW] if g.get(col) is not None]
    return {
        "player_id": e["player_id"], "player_name": e["player_name"],
        "market_code": e["market_code"], "side": e["recommended_side"],
        "line": float(e["line"]), "price": int(e["price_american"]),
        "prior": float(e["projection_median"]), "book": e.get("bookmaker_title"),
        "tier": e["edge_tier"], "star": fame(e),
        "event_id": e["event_id"],
        "game": f"{short(e['away_team'])} at {short(e['home_team'])}",
        "kickoff": t.isoformat(), "form": form,
        "last_played": str(games[0]["game_date"])[:10] if games else None,
        "last_week": games[0].get("week") if games else None,
        "last_season": games[0].get("season") if games else None,
    }


TIER_RANK = ["elite", "strong", "medium", "small"]
LEAD = []


def fame(e):
    """How well known he is, for choosing whose name goes on the cover.

    star_score is last season's fantasy points, and a quarterback scores
    about twice what a receiver of the same standing does. Left alone it
    put Bryce Young on a cover ahead of Justin Jefferson.
    """
    star = e.get("star_score") or 0
    return star * 0.55 if e["market_code"].startswith("pass_") else star


def board(api, keep, tiers=("elite",), players=None):
    """The board's picks that pass `keep`: best tier first, then best-known.

    One pick a player. `players` names them outright, in that order, for a
    card that is chosen by hand, and then any tier is taken.
    """
    rows = [e for e in get(api, "edges", limit=500)["edges"] if keep(e)]
    if players:
        out = []
        for want in players:
            hit = [e for e in rows if want.lower() in e["player_name"].lower()]
            if not hit:
                raise SystemExit(f"{want} has no pick on the board for that")
            hit.sort(key=lambda e: TIER_RANK.index(e["edge_tier"]))
            out.append(hit[0])
        return out
    rows = [e for e in rows if e["edge_tier"] in tiers]
    rows.sort(key=lambda e: (TIER_RANK.index(e["edge_tier"]), -fame(e)))
    seen, out = set(), []
    for e in rows:
        if e["player_name"] in seen:
            continue
        seen.add(e["player_name"])
        out.append(e)
    return out


def arrange(picks):
    """The order the slides run in: a name to open on and a name to end on.

    The cover is the best-known player in the best tier on the card. The
    next best-known goes last, and the cover says so, which is the reason to
    keep swiping. Everything between stays in rank order.
    """
    named = [p for p in picks
             if LEAD and LEAD[0] in p["player_name"].lower()]
    if len(picks) < 3:
        return named + [p for p in picks if p not in named]
    top = picks[0]["tier"]
    lead = named[0] if named else max(
        (p for p in picks if p["tier"] == top), key=lambda p: p["star"])
    rest = [p for p in picks if p is not lead]
    last = max(rest, key=lambda p: p["star"])
    return [lead] + [p for p in rest if p is not last] + [last]


SKIP = []


def hydrate(api, rows, limit):
    """The first `limit` picks that are fit to post.

    A player named in --skip is left off: that is for the man ruled out an
    hour ago whom the board has not caught up with. So is anyone with no game
    in three weeks, because a pick on a player coming back from that long out
    is a guess about his health, and a post is not the place for it. The next
    pick down takes the place.
    """
    picks = []
    for e in rows:
        if len(picks) == limit:
            break
        if any(n in e["player_name"].lower() for n in SKIP):
            print(f"  left off, as asked: {e['player_name']}")
            continue
        games = get(api, f"players/{e['player_id']}/games", limit=WINDOW)["games"]
        p = as_pick(e, games)
        if p["last_played"]:
            idle = (kick(e).date() - dt.date.fromisoformat(p["last_played"])).days
            if idle > 21:
                print(f"  left off {p['player_name']}: his last game on file "
                      f"is {p['last_played']}")
                continue
        if len(p["form"]) < 5:
            print(f"  check {p['player_name']}: only {len(p['form'])} games "
                  f"behind this pick")
        picks.append(p)
    return picks


def on_side(p, v):
    return v > p["line"] if p["side"] == "over" else v < p["line"]


def week_of(picks, given):
    if given:
        return int(given)
    season = max((p["last_season"] or 0) for p in picks)
    weeks = [p["last_week"] for p in picks
             if p["last_week"] and p["last_season"] == season]
    return max(weeks) + 1 if weeks else None


# ---------------------------------------------------------------- drawing


def fit(c, text, name, size, room):
    w = c.width(text, name, size)
    return size if w <= room else size * room / w


def arrow(c, x, y, length, colour, weight=3.0):
    s = c.S
    c.rect(x, y - weight / 2, length, weight, colour)
    tip, half = x + length + 16, 11
    c.d.polygon([(tip * s, y * s), ((tip - 20) * s, (y - half) * s),
                 ((tip - 20) * s, (y + half) * s)], fill=colour)


def strip(c, x, y, w, h, p, on, off, rule, label, ground):
    """His last ten games against the line, oldest on the left.

    A bar on our side of the line is solid and one on the other side is an
    outline. Nothing is left out to make it read better.
    """
    vals = list(reversed(p["form"]))
    if not vals:
        return
    top = max(max(vals), p["line"]) * 1.18 or 1.0
    n = len(vals)
    gap = w * 0.028
    bw = (w - gap * (n - 1)) / n
    for i, v in enumerate(vals):
        bh = max(5.0, h * v / top)
        bx = x + i * (bw + gap)
        if on_side(p, v):
            c.rect(bx, y + h - bh, bw, bh, on)
        else:
            c.frame(bx, y + h - bh, bw, bh, off, t=3)
    ly = y + h - h * p["line"] / top
    c.rect(x - 14, ly - 1.5, w + 28, 3, rule)
    # On a patch of the ground colour, so a tall last bar does not run
    # through it.
    tag = f"LINE {num(p['line'])}"
    tw = c.width(tag, bb.CAPS, 17, 0.1)
    c.rect(x + w + 14 - tw - 10, ly - 34, tw + 10, 30, ground)
    c.text(x + w + 14, ly - 12, tag, bb.CAPS, 17, label, track=0.1, align="r")
    c.rect(x - 14, y + h, w + 28, 1.5, label)


def said(p):
    hit = sum(1 for v in p["form"] if on_side(p, v))
    return (f"{p['side'].capitalize()} {num(p['line'])} in {hit} of his last "
            f"{len(p['form'])} games")


def names(p):
    parts = p["player_name"].split(" ", 1)
    return (parts[0], parts[1]) if len(parts) == 2 else ("", parts[0])


def pick_slide(path, p):
    w, h = 1080, 1350
    c, pad = bb.leaf(w, h)
    inner = w - pad * 2
    ink, dim, green = bb.rgb(bb.INK), bb.rgb(bb.INK_DIM), bb.rgb(bb.GREEN)
    t = dt.datetime.fromisoformat(p["kickoff"])
    y = pad + 34
    bb.stamp(c, pad, y, p["tier"].upper(), bb.TIER_INK[p["tier"]], 22)
    c.text(w - pad, y, f"{p['game'].upper()}  {clock(t)}", bb.CAPS, 20, dim,
           track=0.1, align="r")
    first, last = names(p)
    y += 92
    c.text(pad, y, first, bb.ITALIC, 52, dim)
    size = fit(c, last, bb.SERIF_BLACK, 168, inner)
    y += size * 0.98
    c.text(pad - 4, y, last, bb.SERIF_BLACK, size, ink)
    y += 36
    c.rect(pad, y, inner, 3, ink)
    y += 142
    c.text(pad - 3, y, call(p), bb.SERIF, 132, green)
    y += 62
    c.text(pad, y, UNIT[p["market_code"]], bb.ITALIC, 46, dim)
    y += 78
    col = inner / 3
    for i, (label, value) in enumerate((("OUR PRIOR", f"{p['prior']:.1f}"),
                                        ("THEIR LINE", f"{p['line']:.1f}"),
                                        ("PRICE", price(p["price"])))):
        c.text(pad + col * i, y, label, bb.CAPS, 19, dim, track=0.1)
        c.text(pad + col * i, y + 62, value, bb.SERIF, 58, ink)
    y += 112
    foot = h - pad - 22 * 2.5
    room = foot - y - 92
    if p["form"] and room > 110:
        strip(c, pad + 14, y, inner - 28, min(room, 200), p, green,
              bb.mix(bb.PAPER, bb.INK_DIM, 0.75), ink, dim, bb.rgb(bb.PAPER))
        c.text(pad, y + min(room, 200) + 56, said(p), bb.SANS_BOLD, 29, ink)
    bb.footer(c, pad, bb.INK, dim, size=22)
    c.save(path)


def cover(path, label, hook, more, swipe=True):
    """The first slide: the brand, one name, one pick. Cloth and gold."""
    w, h = 1080, 1350
    c = bb.cloth(w, h)
    gold, cream = bb.rgb(bb.GOLD), bb.rgb(bb.CREAM)
    dim = bb.mix(bb.CLOTH, bb.CREAM, 0.62)
    c.frame(40, 40, w - 80, h - 80, bb.mix(bb.CLOTH, bb.GOLD, 0.34), t=1.5)
    pad = 108
    inner = w - pad * 2
    bb.lockup(c, pad, pad + 12, 76, gold)
    first, last = names(hook)
    y = h * 0.36
    c.text(pad, y, label, bb.CAPS, 23, gold, track=0.14)
    y += 84
    c.text(pad, y, first, bb.ITALIC, 58, cream)
    size = fit(c, last, bb.SERIF_BLACK, 176, inner)
    y += size * 0.98
    c.text(pad - 5, y, last, bb.SERIF_BLACK, size, gold)
    line = f"{call(hook)} {UNIT[hook['market_code']]}."
    ls = fit(c, line, bb.ITALIC, 74, inner)
    y += ls * 1.5
    c.text(pad, y, line, bb.ITALIC, ls, cream)
    c.double_rule(pad, y + ls * 0.2, c.width(line, bb.ITALIC, ls), gold,
                  heavy=4, gap=4, light=1.5)
    if more:
        y += ls * 0.2 + 96
        for line in more:
            c.text(pad, y, line, bb.SANS, 34, cream)
            y += 50
    if swipe:
        y = h - pad - 150
        c.text(pad, y, "SWIPE", bb.CAPS, 24, gold, track=0.16)
        arrow(c, pad + c.width("SWIPE", bb.CAPS, 24, 0.16) + 26, y - 9, 150, gold)
    bb.footer(c, pad, gold, dim, size=22)
    c.save(path)


def trust_slide(path, r):
    w, h = 1080, 1350
    c, pad = bb.leaf(w, h)
    inner = w - pad * 2
    ink, dim, green = bb.rgb(bb.INK), bb.rgb(bb.INK_DIM), bb.rgb(bb.GREEN)
    y = bb.head(c, pad, "Why trust these", "The elite tier, this season", 62)
    y += 236
    c.text(pad - 8, y, f"{r['hit'] * 100:.1f}%", bb.SERIF_BLACK, 216, green)
    y += 84
    c.text(pad, y, "right.", bb.SERIF, 64, ink)
    y += 76
    c.text(pad, y, f"Their prices needed {r['needed'] * 100:.1f}%.",
           bb.ITALIC, 46, dim)
    y += 70
    note = (f"{r['picks']:,} elite picks, weeks {r['week_first']} to "
            f"{r['week_last']}, tiered by the rule the board uses today.")
    if r["lo"] <= 0:
        note += " On this many picks a lead that size could still be luck."
    y = c.wrap(pad, y, note, bb.SANS, 27, ink, inner, lead=1.5)
    y += 8
    c.double_rule(pad, y, inner, ink, heavy=3, gap=4, light=1.5)
    y += 74
    for line in CHECKS:
        c.text(pad, y, line, bb.SANS_BOLD, 34, ink)
        y += 58
    bb.footer(c, pad, bb.INK, dim, size=22)
    c.save(path)


def x_card(path, label, hook, others, r):
    """One landscape image for X: the lead pick, and the rest in a column."""
    w, h = 1600, 900
    c = bb.cloth(w, h)
    gold, cream = bb.rgb(bb.GOLD), bb.rgb(bb.CREAM)
    dim = bb.mix(bb.CLOTH, bb.CREAM, 0.62)
    c.frame(28, 28, w - 56, h - 56, bb.mix(bb.CLOTH, bb.GOLD, 0.34), t=1.5)
    pad = 104
    left = 700
    bb.lockup(c, pad, pad - 6, 60, gold)
    first, last = names(hook)
    y = 300
    c.text(pad, y, label, bb.CAPS, 21, gold, track=0.14)
    y += 70
    c.text(pad, y, first, bb.ITALIC, 46, cream)
    size = fit(c, last, bb.SERIF_BLACK, 132, left)
    y += size * 0.98
    c.text(pad - 4, y, last, bb.SERIF_BLACK, size, gold)
    line = f"{call(hook)} {UNIT[hook['market_code']]}."
    ls = fit(c, line, bb.ITALIC, 56, left)
    y += ls * 1.55
    c.text(pad, y, line, bb.ITALIC, ls, cream)
    c.double_rule(pad, y + ls * 0.2, c.width(line, bb.ITALIC, ls), gold,
                  heavy=3, gap=3, light=1.5)
    x = pad + left + 96
    rw = w - pad - x
    top = pad + 20
    if others:
        all_elite = all(p["tier"] == "elite" for p in [hook] + others)
        c.text(x, top, "ALSO ELITE" if all_elite else "ALSO ON THE CARD",
               bb.CAPS, 20, gold, track=0.14)
        c.rect(x, top + 22, rw, 2, gold)
        ry = top + 22
        row = min(104, (h - pad - 96 - ry) / len(others))
        hair = bb.mix(bb.CLOTH, bb.CREAM, 0.25)
        for p in others:
            if row >= 84:
                base = ry + row / 2 + 2
                ns = fit(c, p["player_name"], bb.SERIF, 36, rw * 0.56)
                c.text(x, base, p["player_name"], bb.SERIF, ns, cream)
                c.text(x, base + 32, UNIT[p["market_code"]], bb.ITALIC, 22, dim)
                c.text(x + rw, base, call(p), bb.SERIF, 36, gold, align="r")
                c.text(x + rw, base + 32, price(p["price"]), bb.CAPS, 17, dim,
                       track=0.1, align="r")
            else:
                # A long card: one line a pick.
                base = ry + row / 2 + 10
                said_ = f"{call(p)} {TIGHT[p['market_code']]}"
                size = min(29, row * 0.5)
                ns = fit(c, p["player_name"], bb.SERIF, size,
                         rw - c.width(said_, bb.SERIF, size) - 24)
                c.text(x, base, p["player_name"], bb.SERIF, ns, cream)
                c.text(x + rw, base, said_, bb.SERIF, size, gold, align="r")
            ry += row
            c.rect(x, ry, rw, 1, hair)
    elif hook["form"]:
        c.text(x, top, "HIS LAST TEN", bb.CAPS, 20, gold, track=0.14)
        strip(c, x + 14, top + 70, rw - 28, 330, hook, gold,
              bb.mix(bb.CLOTH, bb.CREAM, 0.45), cream, dim, bb.rgb(bb.CLOTH))
        c.text(x, top + 70 + 330 + 62, said(hook), bb.SANS_BOLD, 30, cream)
        c.text(x, top + 70 + 330 + 112,
               f"Our prior {hook['prior']:.1f}. Their line {hook['line']:.1f}. "
               f"Priced {price(hook['price'])}.", bb.SANS, 26, dim)
    if r:
        c.text(pad, h - pad - 78, record_line(r, brief=True), bb.SANS, 24, dim)
    bb.footer(c, pad, gold, dim, size=20)
    c.save(path)


def verdict(c, x, y, word, colour, size=21):
    wd = c.width(word, bb.CAPS, size, 0.1) + size * 1.1
    c.frame(x - wd, y - size * 1.22, wd, size * 1.72, bb.rgb(colour), t=2)
    c.text(x - wd + size * 0.55, y, word, bb.CAPS, size, colour, track=0.1)


MARK = {True: ("WON", bb.GREEN), False: ("LOST", bb.RED), None: ("PUSH", bb.INK_DIM)}


def result_rows(c, x, y, w, picks, row, u=1.0):
    for i, p in enumerate(picks):
        base = y + row / 2 - 4 * u
        ns = fit(c, p["player_name"], bb.SERIF, 42 * u, w * 0.62)
        c.text(x, base, p["player_name"], bb.SERIF, ns, bb.INK)
        got = "No result yet" if p["actual"] is None else f"Finished on {num(p['actual'])}"
        c.text(x, base + 38 * u,
               f"{call(p)} {UNIT[p['market_code']]}. {got}.", bb.ITALIC, 25 * u,
               bb.INK_DIM)
        if p["actual"] is not None:
            word, colour = MARK[p["hit"]]
            verdict(c, x + w, base + 14 * u, word, colour, 21 * u)
        y += row
        if i < len(picks) - 1:
            c.rect(x, y, w, 1, bb.rgb(bb.RULE))
    c.double_rule(x, y, w, bb.rgb(bb.INK), heavy=3, gap=4, light=1.5)
    return y


def tally(picks):
    done = [p for p in picks if p["actual"] is not None and p["hit"] is not None]
    return sum(1 for p in done if p["hit"]), len(done)


def receipt_slide(path, title, sub, picks, r, page=None):
    w, h = 1080, 1350
    c, pad = bb.leaf(w, h)
    inner = w - pad * 2
    ink, dim = bb.rgb(bb.INK), bb.rgb(bb.INK_DIM)
    y = bb.head(c, pad, title, sub, 62)
    if page is None or page[0] == 0:
        won, n = page[2] if page else tally(picks)
        y += 196
        c.text(pad - 6, y, f"{won} of {n}", bb.SERIF_BLACK, 190,
               bb.rgb(bb.GREEN) if won * 2 > n else ink)
        y += 66
        c.text(pad, y, "posted picks were right.", bb.ITALIC, 44, dim)
        y += 44
        c.rect(pad, y, inner, 3, ink)
    else:
        y += 20
    foot = h - pad - 22 * 2.5
    tail = 150 if r and (page is None or page[0] == page[1] - 1) else 30
    row = min(128, (foot - y - tail) / max(len(picks), 1))
    y = result_rows(c, pad, y, inner, picks, row)
    if tail > 30:
        c.wrap(pad, y + 52, record_line(r, brief=True) + " The whole record is "
               "at priorline.io.", bb.SANS, 25, ink, inner, lead=1.5)
    bb.footer(c, pad, bb.INK, dim, size=22)
    c.save(path)


def receipt_card(path, title, sub, picks, r, score):
    w, h = 1600, 900
    pad = 96
    c = bb.paper(w, h, margin=44)
    ink, dim = bb.rgb(bb.INK), bb.rgb(bb.INK_DIM)
    left = 520
    won, n = score
    c.text(pad, pad + 58, title, bb.SERIF, 58, ink)
    for i, s in enumerate(sub):
        c.text(pad, pad + 100 + i * 36, s, bb.ITALIC, 27, dim)
    c.rect(pad, pad + 100 + len(sub) * 36 - 12, left, 3, ink)
    c.text(pad - 5, pad + 400, f"{won} of {n}", bb.SERIF_BLACK, 168,
           bb.rgb(bb.GREEN) if won * 2 > n else ink)
    c.text(pad, pad + 462, "posted picks were right.", bb.ITALIC, 36, dim)
    if r:
        c.wrap(pad, pad + 540, record_line(r, brief=True), bb.SANS, 23, ink,
               left, lead=1.5)
    x = pad + left + 70
    top = pad - 14
    foot = h - pad - 20 * 2.5
    shown = picks[:6]
    c.rect(x, top, w - pad - x, 3, ink)
    row = min(112, (foot - top - 40) / len(shown))
    y = result_rows(c, x, top, w - pad - x, shown, row, 0.9)
    if len(picks) > len(shown):
        c.text(x, y + 44, f"and {len(picks) - len(shown)} more at priorline.io",
               bb.ITALIC, 24, dim)
    bb.footer(c, pad, bb.INK, dim, size=20)
    c.save(path)


# --------------------------------------------------------------- captions

TAGS = "#NFL #PlayerProps #NFLProps #SportsAnalytics"


def write_caption(out, x, ig):
    if len(x) > 280:
        print(f"  the X caption is {len(x)} characters, over the 280 limit: trim it")
    (out / "caption.txt").write_text(
        f"X\n-\n{x}\n\n\nINSTAGRAM\n---------\n{ig}\n", encoding="utf-8")


def pregame_captions(out, lead, picks, r, tag):
    brief = [f"{p['player_name']} {call(p).lower()} {TIGHT[p['market_code']]}"
             for p in picks]
    kind = "Elite" if all(p["tier"] == "elite" for p in picks) else "Top"
    x = "\n".join([f"{lead} {kind} pick{'s' if len(picks) > 1 else ''}:", "",
                   *brief, "",
                   record_line(r, brief=True), "", "priorline.io. 21+"])
    if len(x) > 280:
        # A long card does not fit a post. The image carries the picks.
        x = "\n".join([f"{lead} {COUNT[len(picks)]} {kind.lower()} picks, "
                       f"led by {picks[0]['player_name']}. All of them are on "
                       f"the card.", "", record_line(r, brief=True), "",
                       "priorline.io. 21+"])
    lines = [f"{p['player_name']}: {call(p).lower()} {UNIT[p['market_code']]} "
             f"({price(p['price'])}). {said(p)}." if p["form"] else
             f"{p['player_name']}: {call(p).lower()} {UNIT[p['market_code']]} "
             f"({price(p['price'])})." for p in picks]
    ig = "\n".join([
        f"{lead} " + ("The picks my model ranks highest." if len(picks) > 1
                      else "One of the picks my model ranks highest."), "",
        "Bars are his last ten games. Solid ones landed on our side of the "
        "line.", "", *lines, "", record_line(r), "",
        "Graded in public on Tuesday, win or lose. Full board and the whole "
        "record at priorline.io (link in bio).", "", LEGAL, "",
        f"{TAGS} {tag}".strip(),
    ])
    write_caption(out, x, ig)


# ------------------------------------------------------------------ builds


def slug(s):
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def pregame(api, rows, label, lead, folder, limit, tag, week):
    if not rows:
        raise SystemExit("the board has no elite picks for that, so there is no post")
    picks = arrange(hydrate(api, rows, limit))
    wk = week_of(picks, week)
    if wk:
        label = f"WEEK {wk}. {label}"
    r = record(api)
    if not r:
        print("  the record has no stored figure, so the trust slide is left out")
    out = POSTS / folder
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.png"):
        old.unlink()
    hook, rest = picks[0], picks[1:]
    more = []
    if rest:
        word = "elite pick" if all(p["tier"] == "elite" for p in picks) else "pick"
        more = [f"Plus {COUNT[len(rest)].lower()} more {word}"
                f"{'s' if len(rest) > 1 else ''}."]
        if len(rest) > 1:
            more.append(f"{rest[-1]['player_name']} is last.")
    n = 1
    if rest:
        cover(out / f"ig-{n:02d}-cover.png", label, hook, more)
        n += 1
    for p in picks:
        pick_slide(out / f"ig-{n:02d}-{slug(p['player_name'])}.png", p)
        n += 1
    if r:
        trust_slide(out / f"ig-{n:02d}-why-trust-these.png", r)
    x_card(out / "x-card.png", label, hook, rest, r)
    pregame_captions(out, lead, picks, r, tag)
    (out / "posted.json").write_text(json.dumps(
        {"built": dt.datetime.now(ET).isoformat(timespec="minutes"),
         "label": label, "picks": picks}, indent=1), encoding="utf-8")
    print(f"{len(picks)} pick(s), led by {hook['player_name']}, written to {out}")
    for p in picks:
        print(f"   {p['tier']:<8}{p['player_name']:<22}{call(p):<11}"
              f"{UNIT[p['market_code']]:<17}{price(p['price']):>5}  {p['game']}")


def grade(api, p):
    """What he finished on, from his game log, at the line that was posted."""
    day = dt.datetime.fromisoformat(p["kickoff"]).date()
    col = STAT[p["market_code"]]
    actual = None
    for g in get(api, f"players/{p['player_id']}/games", limit=6)["games"]:
        if abs((dt.date.fromisoformat(str(g["game_date"])[:10]) - day).days) <= 1:
            actual = g.get(col)
            break
    if actual is None:
        # He played and recorded nothing, which the game log has no row for
        # and the graded record does.
        for h in get(api, f"players/{p['player_id']}/edge_history", limit=40)["history"]:
            if (h["market_code"] == p["market_code"] and h["actual"] is not None
                    and abs((dt.date.fromisoformat(str(h["game_date"])[:10])
                             - day).days) <= 1):
                actual = h["actual"]
                break
    p["actual"] = None if actual is None else float(actual)
    p["hit"] = (None if actual is None or p["actual"] == p["line"]
                else on_side(p, p["actual"]))
    return p


def receipt(api, start, end):
    picks, labels = [], []
    for f in sorted(POSTS.glob("*/posted.json")):
        data = json.loads(f.read_text(encoding="utf-8"))
        for p in data["picks"]:
            day = dt.datetime.fromisoformat(p["kickoff"]).date()
            if start <= day <= end:
                picks.append(p)
                labels.append(p["game"])
    seen, uniq = set(), []
    for p in picks:
        key = (p["player_name"], p["market_code"], p["line"], p["side"])
        if key not in seen:
            seen.add(key)
            uniq.append(grade(api, p))
    if not uniq:
        raise SystemExit("nothing was posted in that range, so there is no receipt")
    pending = [p["player_name"] for p in uniq if p["actual"] is None]
    if pending:
        print("  no result yet for: " + ", ".join(pending))
    uniq.sort(key=lambda p: (p["kickoff"], -p["star"]))
    score = tally(uniq)
    r = record(api)
    games = list(dict.fromkeys(labels))
    one = start == end
    title = "The receipt"
    when = start.strftime("%A") if one else (
        f"{start.strftime('%A')} to {end.strftime('%A')}")
    sub = f"{when}. {games[0]}" if len(games) == 1 else f"{when}. Every pick I posted"
    out = POSTS / (f"{end.isoformat()}-receipt" if one else
                   f"{start.isoformat()}-to-{end.isoformat()}-receipt")
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.png"):
        old.unlink()
    per = 6
    pages = [uniq[i:i + per] for i in range(0, len(uniq), per)]
    for i, page in enumerate(pages):
        receipt_slide(out / f"ig-{i + 1:02d}-receipt.png", title, sub, page, r,
                      page=(i, len(pages), score))
    receipt_card(out / "x-card.png", title, [when, games[0] if len(games) == 1
                                             else "Every pick I posted"],
                 uniq, r, score)
    won, n = score
    marks = [f"{'W' if p['hit'] else 'L' if p['hit'] is False else '-'}  "
             f"{p['player_name']} {call(p).lower()} {TIGHT[p['market_code']]}"
             f"{'' if p['actual'] is None else ' (' + num(p['actual']) + ')'}"
             for p in uniq]
    x = "\n".join([f"The receipt. {won} of {n} posted picks were right.", "",
                   *marks[:7], "", "Whole record: priorline.io. 21+"])
    ig = "\n".join([
        f"The receipt. {won} of {n} posted picks were right.", "",
        "Graded at the line I posted, not the one it moved to.", "", *marks, "",
        record_line(r), "",
        "It goes out win or lose. The whole record is at priorline.io (link "
        "in bio).", "", LEGAL, "", TAGS,
    ])
    write_caption(out, x, ig)
    print(f"{won} of {n} right, written to {out}")


def main():
    ap = argparse.ArgumentParser(description="Draw the week's posts.")
    ap.add_argument("kind", choices=["slate", "game", "pick", "receipt"])
    ap.add_argument("--api", default=API)
    ap.add_argument("--date", help="YYYY-MM-DD, in Eastern time")
    ap.add_argument("--to", help="receipt only: the last day of the range")
    ap.add_argument("--team", help="game only: either team's name, e.g. Steelers")
    ap.add_argument("--player", help="pick only: the player's name")
    ap.add_argument("--players", help="game only: the card chosen by hand, "
                    "names separated by commas, in slide order")
    ap.add_argument("--lead", help="whose name goes on the cover, if not "
                    "the one it would choose")
    ap.add_argument("--skip", help="names to leave off, separated by commas: "
                    "a player ruled out since the board was built")
    ap.add_argument("--from-hour", type=int, default=0,
                    help="slate only: earliest kickoff hour, Eastern (13 "
                    "leaves out a morning game)")
    ap.add_argument("--max", type=int, default=5, help="picks on a carousel")
    ap.add_argument("--week", help="week number, if the guess is wrong")
    a = ap.parse_args()
    if a.lead:
        LEAD.append(a.lead.strip().lower())
    if a.skip:
        SKIP.extend(n.strip().lower() for n in a.skip.split(",") if n.strip())

    if a.kind == "receipt":
        start = dt.date.fromisoformat(a.date)
        receipt(a.api, start, dt.date.fromisoformat(a.to) if a.to else start)
        return

    if a.kind == "slate":
        day = dt.date.fromisoformat(a.date)
        # The night game gets a post of its own.
        rows = board(a.api, lambda e: kick(e).date() == day
                     and a.from_hour <= kick(e).hour < 19)
        name = day.strftime("%A")
        pregame(a.api, rows, name.upper(), f"{name}'s slate.",
                f"{day.isoformat()}-{name.lower()}-slate", a.max,
                f"#NFL{name}", a.week)
    elif a.kind == "game":
        want = a.team.lower()
        named = [n.strip() for n in a.players.split(",")] if a.players else None
        # A game card is the best of that game, whatever tier that is. One
        # game rarely has four elite picks.
        rows = board(a.api, lambda e: want in e["home_team"].lower()
                     or want in e["away_team"].lower(),
                     tiers=("elite", "strong", "medium"), players=named)
        if not rows:
            raise SystemExit("the board has no picks for that team's game")
        t = kick(rows[0])
        game = f"{short(rows[0]['away_team'])} at {short(rows[0]['home_team'])}"
        night = f"{t.strftime('%A')} night" if t.hour >= 19 else t.strftime("%A")
        tag = {"Sunday night": "#SNF", "Monday night": "#MNF",
               "Thursday night": "#TNF"}.get(night, "")
        pregame(a.api, rows, f"{night.upper()}. {game.upper()}",
                f"{game}, {night.lower()}.",
                f"{t.date().isoformat()}-{slug(game)}", a.max, tag, a.week)
    else:
        want = a.player.lower()
        rows = board(a.api, lambda e: want in e["player_name"].lower(),
                     tiers=("elite", "strong", "medium"))
        if not rows:
            raise SystemExit("that player has no pick on the board")
        t = kick(rows[0])
        pregame(a.api, rows, t.strftime("%A").upper(),
                f"{rows[0]['player_name']}, {t.strftime('%A')}.",
                f"{t.date().isoformat()}-{slug(rows[0]['player_name'])}", 1,
                "", a.week)


if __name__ == "__main__":
    main()
