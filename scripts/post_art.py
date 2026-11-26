"""The look of a post: one player, his team's colour, one call, set dead centre.

The first posts were ledger pages. They were honest and nobody stopped for
them: flush left, a lot of small type, no face and no colour. The accounts
people do stop for put a player on a bright ground and say one thing in type
you can read from across a room. This keeps what is ours (the mark, Besley and
Archivo, the notice, every figure read from the board) and takes that lesson.

Every slide is built on one centre line. The ground is the colour of the
player's team at the top, falling to black where the type sits. The player is
cut out over it. Under him: his name, the call, and his recent games drawn as
tiles or bars, solid where they landed on our side and outlined where they
did not. Nothing is left out to make a pick look better.

The photographs are the league's own headshots, the same ones the site shows.
They are not ours. --no-photos draws every slide without them.
"""

from __future__ import annotations

import io
import urllib.request

from PIL import Image, ImageChops

import build_brand as bb

NIGHT = "#08090b"
WHITE = "#f4f1e8"

# Ground and accent for each team. The accent has to read on black, so where
# a team's second colour is too dark the light one is used.
TEAM = {
    "ARI": ("#97233f", "#ffb612"), "ATL": ("#a71930", "#f2f2f2"),
    "BAL": ("#2a1a80", "#d4af37"), "BUF": ("#00338d", "#f04a5a"),
    "CAR": ("#0085ca", "#f2f2f2"), "CHI": ("#0b1f47", "#f0601e"),
    "CIN": ("#fb4f14", "#ff6a2b"), "CLE": ("#ff3c00", "#ff5a26"),
    "DAL": ("#003594", "#c3c9ce"), "DEN": ("#fb4f14", "#ff6a2b"),
    "DET": ("#0076b6", "#c3c9ce"), "GB": ("#203731", "#ffb612"),
    "HOU": ("#0a2a4a", "#f04a5a"), "IND": ("#002c5f", "#f2f2f2"),
    "JAX": ("#006778", "#d7a22a"), "KC": ("#e31837", "#ffb81c"),
    "LV": ("#3a3f44", "#d0d4d7"), "LAC": ("#0080c6", "#ffc20e"),
    "LA": ("#003594", "#ffd100"), "MIA": ("#008e97", "#fc6a2a"),
    "MIN": ("#4f2683", "#ffc62f"), "NE": ("#002244", "#f04a5a"),
    "NO": ("#8a7a4f", "#d3bc8d"), "NYG": ("#0b2265", "#f04a5a"),
    "NYJ": ("#125740", "#f2f2f2"), "PHI": ("#004c54", "#c3c9ce"),
    "PIT": ("#ffb612", "#ffc233"), "SF": ("#aa0000", "#d9b872"),
    "SEA": ("#002244", "#69be28"), "TB": ("#d50a0a", "#f2f2f2"),
    "TEN": ("#4b92db", "#6aa9ea"), "WAS": ("#5a1414", "#ffb612"),
}
ALIAS = {"LAR": "LA", "WSH": "WAS", "OAK": "LV", "SD": "LAC", "STL": "LA",
         "JAC": "JAX", "GNB": "GB", "KAN": "KC", "NWE": "NE", "NOR": "NO",
         "SFO": "SF", "TAM": "TB"}
PHOTOS = [True]
_SHOTS = {}


def colours(team):
    t = ALIAS.get(team or "", team or "")
    return TEAM.get(t, (bb.CLOTH, bb.GOLD))


def headshot(url):
    """The cut-out of the player, trimmed to his head and shoulders.

    The bottom of the league's picture has its shield on the collar. The
    picture is cut above it, so no league mark reaches a post.
    """
    if not url or not PHOTOS[0]:
        return None
    if url in _SHOTS:
        return _SHOTS[url]
    shot = None
    try:
        big = url.replace("f_auto,q_auto", "f_png,q_auto,w_1400")
        req = urllib.request.Request(big, headers={"User-Agent": "post-build"})
        with urllib.request.urlopen(req, timeout=30) as r:
            im = Image.open(io.BytesIO(r.read())).convert("RGBA")
        box = im.getchannel("A").getbbox()
        if box and im.getchannel("A").getextrema()[0] == 0:
            left, top, right, _ = box
            shot = im.crop((left, top, right, int(im.height * 0.80)))
    except Exception as err:
        print(f"  no photograph ({err}), drawn without one")
    _SHOTS[url] = shot
    return shot


def ground(w, h, team, fall=0.66, glow=(0.5, 0.30), across=False):
    """Team colour at the top falling to black, with a light behind the head.

    `across` runs it from the left edge instead, for a wide card with the
    player on the left and the type on black to his right.
    """
    c = bb.Sheet(w, h, NIGHT)
    s = c.S
    bg, _ = colours(team)
    if across:
        ramp = Image.linear_gradient("L").rotate(270, expand=True).resize(
            (int(w * s * fall), h * s))
    else:
        ramp = Image.linear_gradient("L").rotate(180).resize(
            (w * s, int(h * s * fall)))
    ramp = ramp.point(lambda v: int(255 * (v / 255) ** 0.85))
    mask = Image.new("L", c.im.size, 0)
    mask.paste(ramp, (0, 0))
    c.im.paste(Image.new("RGB", c.im.size, bb.rgb(bg)), (0, 0), mask)
    r = int(w * s * 0.62)
    halo = ImageChops.invert(Image.radial_gradient("L")).resize((r * 2, r * 2))
    # The gradient only reaches white in its corners. Left alone, the
    # sides of the square show as a seam.
    halo = halo.point(lambda v: int(max(0, v - 84) * 0.7))
    light = Image.new("RGB", halo.size, bb.mix(bg, "#ffffff", 0.55))
    c.im.paste(light, (int(w * s * glow[0]) - r, int(h * s * glow[1]) - r), halo)
    return c


def place(c, shot, cx, top, height, fade=0.34):
    """Stand the player on the centre line and melt his shoulders into the dark."""
    if shot is None:
        return
    s = c.S
    k = height * s / shot.height
    im = shot.resize((int(shot.width * k), int(height * s)), Image.LANCZOS)
    ramp = Image.linear_gradient("L").rotate(180).resize(
        (im.width, int(im.height * fade)))
    veil = Image.new("L", im.size, 255)
    veil.paste(ramp, (0, im.height - ramp.height))
    im.putalpha(ImageChops.multiply(im.getchannel("A"), veil))
    c.im.paste(im, (int(cx * s - im.width / 2), int(top * s)), im)


def mid(c, y, text, name, size, fill, track=0.0, room=None, cx=None):
    if room:
        wide = c.width(text, name, size, track)
        if wide > room:
            size = size * room / wide
    c.text(c.w / 2 if cx is None else cx, y, text, name, size, fill,
           track=track, align="c")
    return size


def brand(c, y, fill, size=40, cx=None):
    wide = size * 0.98 + c.width("PriorLine", bb.SERIF, size * 0.62)
    bb.lockup(c, (c.w / 2 if cx is None else cx) - wide / 2, y, size, fill)


def notice(c, fill):
    mid(c, c.h - 78, "PRIORLINE.IO", bb.CAPS, 21, bb.rgb(WHITE), track=0.18)
    mid(c, c.h - 44, bb.LEGAL, bb.SANS, 17, fill)


def tiles(c, cx, y, values, hits, accent, size=96, gap=22):
    """His games as square tiles on the centre line.

    Solid in the team colour where he landed on our side, outlined where he
    did not.
    """
    n = len(values)
    x = cx - (n * size + (n - 1) * gap) / 2
    for v, hit in zip(values, hits):
        if hit:
            c.rect(x, y, size, size, accent)
            ink = bb.rgb(NIGHT)
        else:
            c.frame(x, y, size, size, bb.mix(NIGHT, WHITE, 0.8), t=3)
            ink = bb.rgb(WHITE)
        fs = size * (0.5 if len(v) < 3 else 0.38)
        c.text(x + size / 2, y + size / 2 + fs * 0.36, v, bb.SERIF_BLACK, fs,
               ink, align="c")
        x += size + gap


def bars(c, cx, y, w, h, values, hits, line, accent, tag):
    """His last ten games against the line, oldest on the left."""
    if not values:
        return
    top = max(max(values), line) * 1.2 or 1.0
    n = len(values)
    gap = w * 0.03
    bw = (w - gap * (n - 1)) / n
    x0 = cx - w / 2
    off = bb.mix(NIGHT, WHITE, 0.5)
    for i, (v, hit) in enumerate(zip(values, hits)):
        bh = max(6.0, h * v / top)
        bx = x0 + i * (bw + gap)
        if hit:
            c.rect(bx, y + h - bh, bw, bh, accent)
        else:
            c.frame(bx, y + h - bh, bw, bh, off, t=3)
    ly = y + h - h * line / top
    white = bb.rgb(WHITE)
    c.rect(x0 - 18, ly - 1.5, w + 36, 3, white)
    tw = c.width(tag, bb.CAPS, 17, 0.12)
    c.rect(x0 + w + 18 - tw - 16, ly - 34, tw + 16, 30, bb.rgb(NIGHT))
    c.text(x0 + w + 18 - tw - 8, ly - 12, tag, bb.CAPS, 17, white, track=0.12)


def pill(c, cx, y, text, accent, size=24):
    wide = c.width(text, bb.CAPS, size, 0.16) + 150
    h = size * 2.5
    x = cx - wide / 2
    c.rect(x, y, wide, h, accent)
    ink = bb.rgb(NIGHT)
    c.text(x + 38, y + h / 2 + size * 0.36, text, bb.CAPS, size, ink, track=0.16)
    ax = x + wide - 96
    ay = y + h / 2
    c.rect(ax, ay - 2, 42, 4, ink)
    s = c.S
    c.d.polygon([((ax + 60) * s, ay * s), ((ax + 40) * s, (ay - 11) * s),
                 ((ax + 40) * s, (ay + 11) * s)], fill=ink)


LEAVES = ("#e0a030", "#c8591c", "#cda64e", "#8f2f14", "#f0d9a0")


def season(c, theme, low=0.50):
    """Dress a slide for the day. Thanksgiving is falling leaves in both top
    corners, mirrored on the centre line, and nothing over the player."""
    if theme != "thanksgiving":
        return
    import math
    import random
    rnd = random.Random(1126)
    s = c.S
    layer = Image.new("RGBA", c.im.size, (0, 0, 0, 0))
    from PIL import ImageDraw
    d = ImageDraw.Draw(layer)
    for _ in range(26):
        x = rnd.uniform(18, c.w * 0.27)
        y = rnd.uniform(150, c.h * low)
        if x > c.w * 0.27 * (1 - (y - 150) / (c.h * low)) + 40:
            continue
        size = rnd.uniform(16, 40)
        turn = rnd.uniform(0, math.pi)
        col = bb.rgb(rnd.choice(LEAVES)) + (rnd.randint(150, 235),)
        for cx in (x, c.w - x):
            t = turn if cx == x else math.pi - turn
            pts = []
            for k in range(24):
                a = 2 * math.pi * k / 24
                # A leaf: pointed at both ends, widest in the middle.
                px, py = math.cos(a) * size, math.sin(a) ** 3 * size * 0.42
                pts.append(((cx + px * math.cos(t) - py * math.sin(t)) * s,
                            (y + px * math.sin(t) + py * math.cos(t)) * s))
            d.polygon(pts, fill=col)
    c.im.paste(layer, (0, 0), layer)
