"""Turn a built post into a narrated breakdown video for each pick.

A short explainer, the kind you would watch on purpose: a voice talks through
how we see the player, and what it says is drawn as it is said. The order is
the order of the argument:

    the question     the player, the line, over or under
    the player       his number, counting up
    his team         how the offence leans (every team in the league as a bar,
                     his lit up, or how they split their touchdowns)
    the defence      what the team across from him allows, all 32 again
    the weather      the forecast for the middle of the game, with the rain
                     falling if rain is coming, or that there is a roof
    his last ten     against the line, one bar at a time
    the pick         our number against the book's, then the call
    where to go      the site and the Instagram account

Every figure comes from posted.json (what the post said) and post_depth.py
(every play of the season), and the forecast from Open-Meteo. Nothing is
typed. If a number is against the pick it is said anyway.

The voice is Microsoft's read-aloud voice through the edge-tts package: the
script text goes to that service and speech comes back with the time of every
word, which is what the captions follow. --no-voice builds it silent.

    python scripts/build_video.py marketing/social/posts/2026-11-26-thanksgiving
    python scripts/build_video.py <folder> --player "Josh Allen" --occasion "on Thanksgiving night"

1080 by 1920, 30 frames a second, 45 to 60 seconds. Everything that matters
sits clear of the buttons Instagram, TikTok and Shorts draw over the top and
bottom. Needs Pillow, imageio-ffmpeg and edge-tts.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import math
import random
import re
import subprocess
import tempfile
import wave
from pathlib import Path

import imageio_ffmpeg
from PIL import Image, ImageDraw

import build_brand as bb
import build_posts as bp
import post_art as art
import post_depth as deep

W, H, FPS = 1080, 1920, 30
RATE = 44100
VOICE = "en-US-AndrewNeural"
TOP, FOOT = 730, 1390          # the band the drawings live in
CAP_Y = 1478                   # captions
SAY_UNIT = {
    "recs": "catches", "rec_yds": "receiving yards", "rush_yds": "rushing yards",
    "rush_att": "rush attempts", "pass_yds": "passing yards",
    "pass_att": "pass attempts", "pass_completions": "completions",
    "pass_td": "touchdown passes",
}
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


# ----------------------------------------------------------------- helpers


def clamp(v, lo=0.0, hi=1.0):
    return max(lo, min(hi, v))


def ease(t):
    t = clamp(t)
    return 1 - (1 - t) ** 3


def back(t):
    """Ease out, going a little past and settling. For things that land."""
    t = clamp(t)
    c = 1.70158
    return 1 + (c + 1) * (t - 1) ** 3 + c * (t - 1) ** 2


def at(t, start, length):
    return ease((t - start) / length)


def font(name, size):
    return bb.font(name, max(1, round(size)))


def col(rgb, a=1.0):
    return tuple(rgb) + (int(255 * clamp(a)),)


def text(d, cx, y, s, name, size, rgb, a=1.0, track=0.0, room=None, anchor="m"):
    """Baseline text, centred on cx unless anchor is 'l' or 'r'."""
    if a <= 0.01 or not s:
        return size
    f = font(name, size)
    wide = f.getlength(s) + track * size * max(len(s) - 1, 0)
    if room and wide > room:
        size = size * room / wide
        f = font(name, size)
        wide = f.getlength(s) + track * size * max(len(s) - 1, 0)
    x = cx - wide / 2 if anchor == "m" else cx - wide if anchor == "r" else cx
    if not track:
        d.text((x, y), s, font=f, fill=col(rgb, a), anchor="ls")
        return size
    for ch in s:
        d.text((x, y), ch, font=f, fill=col(rgb, a), anchor="ls")
        x += f.getlength(ch) + track * size
    return size


def kicker(d, t, word, accent, dim):
    """The small heading of a part, with a rule that draws out from the middle."""
    a = at(t, 0.0, 0.3)
    text(d, W / 2, TOP + 34, word, bb.CAPS, 26, accent, a, track=0.24)
    half = 70 * at(t, 0.05, 0.4)
    d.rectangle([W / 2 - half, TOP + 54, W / 2 + half, TOP + 57], fill=col(accent, a))


# ------------------------------------------------------------------- voice


async def _speak(line, path, voice):
    import edge_tts
    c = edge_tts.Communicate(line, voice, rate="+10%", boundary="WordBoundary")
    marks = []
    with open(path, "wb") as f:
        async for ch in c.stream():
            if ch["type"] == "audio":
                f.write(ch["data"])
            elif ch["type"] == "WordBoundary":
                marks.append((ch["offset"] / 1e7, ch["duration"] / 1e7, ch["text"]))
    return marks


def speak(line, tmp, i, voice):
    """One line said aloud: its sound as 16-bit mono, and when each word falls."""
    mp3 = Path(tmp) / f"s{i}.mp3"
    marks = asyncio.run(_speak(line, mp3, voice))
    raw = subprocess.run(
        [FFMPEG, "-loglevel", "error", "-i", str(mp3), "-f", "s16le", "-ac", "1",
         "-ar", str(RATE), "-"], capture_output=True, check=True).stdout
    return raw, marks


def silent(line):
    """No voice: give each word a third of a second so the captions still run."""
    ws = line.split()
    return b"\x00\x00" * int(RATE * (0.34 * len(ws) + 0.2)), [
        (0.1 + 0.34 * i, 0.3, w) for i, w in enumerate(ws)]


def chunks(marks, limit=26):
    """Words gathered into caption lines short enough to read at a glance."""
    out, cur = [], []
    for m in marks:
        trial = " ".join(w for _, _, w in cur + [m])
        if cur and (len(trial) > limit or cur[-1][2][-1:] in ".?,"):
            out.append(cur)
            cur = []
        cur.append(m)
    if cur:
        out.append(cur)
    return out


def captions(d, t, lines, accent):
    """The words being said, a line at a time, the current word lit."""
    white = bb.rgb(art.WHITE)
    for ws in lines:
        start, end = ws[0][0] - 0.06, ws[-1][0] + ws[-1][1] + 0.18
        if not (start <= t < end):
            continue
        f = font(bb.SANS_BOLD, 50)
        space = f.getlength(" ")
        wide = sum(f.getlength(w) for _, _, w in ws) + space * (len(ws) - 1)
        size = 50
        if wide > W - 170:
            size = 50 * (W - 170) / wide
            f = font(bb.SANS_BOLD, size)
            space = f.getlength(" ")
            wide = sum(f.getlength(w) for _, _, w in ws) + space * (len(ws) - 1)
        x = W / 2 - wide / 2
        d.rounded_rectangle([x - 26, CAP_Y - size * 1.02, x + wide + 26,
                             CAP_Y + size * 0.42], radius=14, fill=(0, 0, 0, 170))
        for w0, wd, word in ws:
            on = t >= w0 - 0.03
            d.text((x, CAP_Y), word, font=f, anchor="ls",
                   fill=col(accent if w0 - 0.03 <= t < w0 + wd + 0.08 else white,
                            1.0 if on else 0.45))
            x += f.getlength(word) + space
        return


# ------------------------------------------------------------- the grounds


def grounds(p, theme, question):
    """Two stills everything is drawn over: the opening, and the working frame
    with the player small at the top."""
    keep = bb.Sheet.S
    bb.Sheet.S = 1
    try:
        shot = art.headshot(p.get("headshot"))
        team = p.get("team")
        white = bb.rgb(art.WHITE)
        accent = bb.rgb(art.colours(team)[1])
        dim = bb.mix(art.NIGHT, art.WHITE, 0.62)

        a = art.ground(W, H, team, fall=0.62, glow=(0.5, 0.33))
        art.season(a, theme, low=0.44, top=320)
        art.place(a, shot, W / 2, 330, 640)
        da = ImageDraw.Draw(a.im, "RGBA")
        text(da, W / 2, 1056, p["player_name"], bb.SERIF, 78, white, room=W - 160)
        t = dt.datetime.fromisoformat(p["kickoff"])
        text(da, W / 2, 1112, f"{p['game'].upper()}    {bp.clock(t)}", bb.CAPS, 24,
             dim, track=0.18)
        text(da, W / 2, 1272, question[0], bb.SERIF_BLACK, 124, accent, room=W - 130)
        text(da, W / 2, 1338, question[1], bb.CAPS, 34, white, track=0.3)

        b = art.ground(W, H, team, fall=0.40, glow=(0.5, 0.20))
        art.season(b, theme, low=0.30, top=320)
        art.place(b, shot, W / 2, 292, 330)
        db = ImageDraw.Draw(b.im, "RGBA")
        text(db, W / 2, 672, p["player_name"], bb.SERIF, 50, white, room=W - 200)
        text(db, W / 2, 708, f"{bp.num(p['line'])} {bp.UNIT[p['market_code']].upper()}",
             bb.CAPS, 21, accent, track=0.24)
    finally:
        bb.Sheet.S = keep
    for im in (a.im, b.im):
        d = ImageDraw.Draw(im, "RGBA")
        mark = 40
        wide = mark * 0.98 + font(bb.SERIF, mark * 0.62).getlength("PriorLine")
        m = bb.mark_mask(mark)
        im.paste(Image.new("RGB", m.size, white), (int(W / 2 - wide / 2), 256), m)
        d.text((W / 2 - wide / 2 + mark * 0.98, 256 + mark * 0.69), "PriorLine",
               font=font(bb.SERIF, mark * 0.62), fill=white, anchor="ls")
        text(d, W / 2, 1590, bb.LEGAL, bb.SANS, 17, dim)
    return a.im, b.im, accent, white, dim


# ------------------------------------------------------------ the drawings


def draw_count(d, t, b, accent, white, dim):
    kicker(d, t, b["kicker"], accent, dim)
    k = at(t, 0.1, 1.0)
    shown = f"{b['value'] * k:.{b['places']}f}{b['suffix']}"
    full = f"{b['value']:.{b['places']}f}{b['suffix']}"
    f = font(bb.SERIF_BLACK, 290)
    x = W / 2 - f.getlength(full) / 2
    d.text((x, TOP + 380), shown, font=f, fill=col(white, at(t, 0.05, 0.25)), anchor="ls")
    a = at(t, 0.5, 0.35)
    text(d, W / 2, TOP + 462 + (1 - a) * 22, b["label"], bb.CAPS, 36, white, a,
         track=0.14, room=W - 160)
    a = at(t, 1.0, 0.35)
    wide = 300 * a
    d.rectangle([W / 2 - wide, TOP + 498, W / 2 + wide, TOP + 501], fill=col(accent, a))
    text(d, W / 2, TOP + 556 + (1 - a) * 22, b["sub"], bb.CAPS, 30, accent, a,
         track=0.12, room=W - 160)


def draw_split(d, t, b, accent, white, dim):
    kicker(d, t, b["kicker"], accent, dim)
    text(d, W / 2, TOP + 130, b["heading"], bb.CAPS, 28, white, at(t, 0.15, 0.3),
         track=0.16)
    x0, y0, w, h = 110, TOP + 300, W - 220, 120
    left, right = b["left"][1], b["right"][1]
    k = at(t, 0.3, 1.0)
    cut = x0 + w * left / 100
    d.rectangle([x0, y0, x0 + w, y0 + h], fill=col(bb.mix(art.NIGHT, art.WHITE, 0.14)))
    d.rectangle([x0, y0, x0 + (cut - x0) * k, y0 + h], fill=col(accent))
    d.rectangle([x0 + w - (x0 + w - cut) * k, y0, x0 + w, y0 + h],
                fill=col(bb.mix(art.NIGHT, art.WHITE, 0.82)))
    d.rectangle([cut - 3, y0 - 14, cut + 3, y0 + h + 14], fill=col(art.NIGHT and bb.rgb(art.NIGHT)))
    text(d, x0, y0 - 30, f"{left * k:.0f}%", bb.SERIF_BLACK, 110, accent,
         at(t, 0.3, 0.3), anchor="l")
    text(d, x0 + w, y0 - 30, f"{right * k:.0f}%", bb.SERIF_BLACK, 110, white,
         at(t, 0.3, 0.3), anchor="r")
    a = at(t, 1.0, 0.35)
    text(d, x0, y0 + h + 56, b["left"][0], bb.CAPS, 28, accent, a, track=0.14, anchor="l")
    text(d, x0 + w, y0 + h + 56, b["right"][0], bb.CAPS, 28, white, a, track=0.14,
         anchor="r")


def draw_league(d, t, b, accent, white, dim):
    """All 32 teams as bars, fewest on the left, and the one that matters lit."""
    kicker(d, t, b["kicker"], accent, dim)
    text(d, W / 2, TOP + 122, b["heading"], bb.CAPS, 27, white, at(t, 0.15, 0.3),
         track=0.14, room=W - 160)
    bars = b["bars"]
    n = len(bars)
    x0, w, base, tall = 96, W - 192, TOP + 590, 300
    gap = 5
    bw = (w - gap * (n - 1)) / n
    top = max(v for _, v in bars) or 1.0
    low = min(v for _, v in bars)
    floor = low * 0.55          # so the difference between teams can be seen
    plain = bb.mix(art.NIGHT, art.WHITE, 0.36)
    hx = hy = None
    for i, (team, v) in enumerate(bars):
        k = back((t - 0.25 - i * 0.022) / 0.45) if t > 0.25 + i * 0.022 else 0.0
        bh = max(4.0, tall * (v - floor) / (top - floor)) * max(k, 0.0)
        bx = x0 + i * (bw + gap)
        mine = team == b["team"]
        lit = mine and t > 1.25
        d.rectangle([bx, base - bh, bx + bw, base], fill=col(accent if lit else plain))
        if mine:
            hx, hy = bx + bw / 2, base - bh
    a = at(t, 1.25, 0.3)
    if hx is not None and a > 0:
        pop = back((t - 1.25) / 0.4)
        tag = f"{b['value']:.1f}"
        d.rectangle([hx - 2, hy - 6 - max(20 * pop, 0), hx + 2, hy - 6],
                    fill=col(accent, a))
        side = "l" if hx < W / 2 else "r"
        tx = clamp(hx, 150, W - 150)
        text(d, tx, hy - 40 * pop, tag, bb.SERIF_BLACK, 76 * max(pop, 0.2), accent, a)
        text(d, W / 2, TOP + 222, b["words"].upper(), bb.SERIF_BLACK, 74, white, a,
             room=W - 200)
        del side
    a = at(t, 0.5, 0.4)
    text(d, x0, base + 44, "FEWEST", bb.CAPS, 20, dim, a, track=0.16, anchor="l")
    text(d, x0 + w, base + 44, "MOST", bb.CAPS, 20, dim, a, track=0.16, anchor="r")
    text(d, W / 2, base + 44, "ALL 32 TEAMS", bb.CAPS, 20, dim, a, track=0.16)


def draw_weather(d, t, b, accent, white, dim, seed=7):
    kicker(d, t, "THE WEATHER", accent, dim)
    if b["roof"]:
        k = at(t, 0.2, 0.7)
        box = [W / 2 - 300, TOP + 190, W / 2 + 300, TOP + 790]
        d.arc(box, 180, 180 + 180 * k, fill=col(accent), width=12)
        d.rectangle([W / 2 - 300, TOP + 488, W / 2 - 300 + 600 * k, TOP + 494],
                    fill=col(white))
        text(d, W / 2, TOP + 420, "INDOORS", bb.SERIF_BLACK, 104, white, at(t, 0.5, 0.3))
        text(d, W / 2, TOP + 560, f"UNDER A ROOF IN {b['city'].upper()}", bb.CAPS, 28,
             accent, at(t, 0.8, 0.3), track=0.16)
        return
    wet = b["wet"] >= 40
    if wet:
        rnd = random.Random(seed)
        for _ in range(70):
            x = rnd.uniform(60, W - 60)
            sp = rnd.uniform(520, 820)
            ln = rnd.uniform(22, 46)
            y = TOP + 80 + ((rnd.uniform(0, 620) + t * sp) % 620)
            if b["fall"] == "snow":
                r = rnd.uniform(3, 6)
                y = TOP + 80 + ((rnd.uniform(0, 620) + t * sp * 0.25) % 620)
                d.ellipse([x - r, y - r, x + r, y + r], fill=col(white, 0.5 * at(t, 0, 0.5)))
            else:
                d.line([x, y, x - ln * 0.22, y + ln], fill=col(white, 0.30 * at(t, 0, 0.5)),
                       width=3)
    tiles = [(f"{b['temp'] * at(t, 0.2, 0.9):.0f}°", "DEGREES"),
             (f"{b['wind'] * at(t, 0.45, 0.9):.0f}", "MPH WIND"),
             (f"{b['wet'] * at(t, 0.7, 0.9):.0f}%", f"CHANCE OF {b['fall'].upper()}")]
    cw, gap = 290, 24
    x = W / 2 - (cw * 3 + gap * 2) / 2
    for i, (v, lab) in enumerate(tiles):
        a = at(t, 0.15 + i * 0.25, 0.35)
        y = TOP + 200 + (1 - a) * 30
        d.rectangle([x, y, x + cw, y + 300], fill=col(bb.rgb(art.NIGHT), 0.72 * a))
        d.rectangle([x, y, x + cw, y + 7], fill=col(accent, a))
        text(d, x + cw / 2, y + 178, v, bb.SERIF_BLACK, 112, white, a, room=cw - 30)
        text(d, x + cw / 2, y + 244, lab, bb.CAPS, 22, accent, a, track=0.12, room=cw - 30)
        x += cw + gap
    text(d, W / 2, TOP + 590, f"{b['city'].upper()}, AROUND HALF-TIME", bb.CAPS, 24, dim,
         at(t, 1.0, 0.4), track=0.18)


def draw_ten(d, t, p, accent, white, dim):
    kicker(d, t, "HIS LAST TEN", accent, dim)
    vals = list(reversed(p["form"][:bp.WINDOW]))
    if not vals:
        return
    x0, y0, w, h = 130, TOP + 250, W - 260, 300
    top = max(max(vals), p["line"]) * 1.22 or 1.0
    gap = w * 0.03
    bw = (w - gap * (len(vals) - 1)) / len(vals)
    off = bb.mix(art.NIGHT, art.WHITE, 0.55)
    for i, v in enumerate(vals):
        k = back((t - 0.25 - i * 0.09) / 0.4) if t > 0.25 + i * 0.09 else 0.0
        bh = max(8.0, h * v / top) * max(k, 0.0)
        bx = x0 + i * (bw + gap)
        if bh < 2:
            continue
        if bp.on_side(p, v):
            d.rectangle([bx, y0 + h - bh, bx + bw, y0 + h], fill=col(accent))
        else:
            d.rectangle([bx, y0 + h - bh, bx + bw, y0 + h], outline=col(off), width=5)
        if k > 0.9:
            text(d, bx + bw / 2, y0 + h + 40, bp.num(v), bb.CAPS, 24, dim, 1.0)
    ly = y0 + h - h * p["line"] / top
    k = at(t, 0.2, 0.8)
    d.rectangle([x0 - 24, ly - 2, x0 - 24 + (w + 48) * k, ly + 2], fill=col(white))
    text(d, x0 + w + 24, ly - 14, f"LINE {bp.num(p['line'])}", bb.CAPS, 22, white,
         at(t, 0.9, 0.3), track=0.12, anchor="r")
    hit = sum(1 for v in vals if bp.on_side(p, v))
    a = at(t, 1.5, 0.35)
    text(d, W / 2, TOP + 190, f"{p['side'].upper()} IN {hit} OF {len(vals)}",
         bb.SERIF_BLACK, 76, white, a, room=W - 200)


def stamp(p, accent):
    """The call, drawn once at twice the size so it can land without blurring."""
    im = Image.new("RGBA", (W * 2, 520), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    text(d, W, 400, bp.call(p).upper(), bb.SERIF_BLACK, 340, accent, room=W * 2 - 240)
    return im


def draw_pick(frame, d, t, p, accent, white, dim, sprite, when):
    kicker(d, t, "THE PICK", accent, dim)
    rows = (("OUR NUMBER", p["prior"], white), ("THEIR LINE", p["line"], accent))
    big = max(p["prior"], p["line"]) or 1.0
    x0, w = 130, W - 260
    fade = 1 - at(t, when - 0.1, 0.3)
    for i, (lab, v, c) in enumerate(rows):
        y = TOP + 150 + i * 150
        k = at(t, 0.2 + i * 0.3, 0.8)
        text(d, x0, y, lab, bb.CAPS, 24, dim, at(t, 0.1 + i * 0.3, 0.3) * fade,
             track=0.16, anchor="l")
        d.rectangle([x0, y + 18, x0 + w, y + 74],
                    fill=col(bb.mix(art.NIGHT, art.WHITE, 0.12), fade))
        d.rectangle([x0, y + 18, x0 + w * 0.86 * (v / big) * k, y + 74], fill=col(c, fade))
        text(d, x0 + w, y, f"{v * k:.1f}", bb.SERIF_BLACK, 60, c, fade, anchor="r")
    if t < when:
        return
    u = t - when
    flash = max(0.0, 1 - u / 0.22)
    if flash > 0:
        d.rectangle([0, 0, W, H], fill=col(white, 0.5 * flash))
    k = 2.2 - 1.2 * back(u / 0.36)
    size = (max(2, int(sprite.width * k / 2)), max(2, int(sprite.height * k / 2)))
    s = sprite.resize(size, Image.LANCZOS)
    frame.paste(s, (int(W / 2 - s.width / 2), int(TOP + 330 - s.height * 0.62)), s)
    a = at(u, 0.3, 0.3)
    text(d, W / 2, TOP + 470, bp.UNIT[p["market_code"]].upper(), bb.CAPS, 44, white, a,
         track=0.3)
    text(d, W / 2, TOP + 552, f"{p['tier'].upper()} PICK     PRICE {bp.price(p['price'])}",
         bb.CAPS, 26, dim, a, track=0.18)


def end_still():
    c = Image.new("RGB", (W, H), bb.rgb(bb.CLOTH))
    return c


def draw_end(d, t, r, handle):
    gold, cream = bb.rgb(bb.GOLD), bb.rgb(bb.CREAM)
    dim = bb.mix(bb.CLOTH, bb.CREAM, 0.62)
    a = at(t, 0.0, 0.4)
    mark = 100
    wide = mark * 0.98 + font(bb.SERIF, mark * 0.62).getlength("PriorLine")
    m = bb.mark_mask(mark)
    d._image.paste(Image.new("RGB", m.size, gold), (int(W / 2 - wide / 2), 470), m)
    d.text((W / 2 - wide / 2 + mark * 0.98, 470 + mark * 0.69), "PriorLine",
           font=font(bb.SERIF, mark * 0.62), fill=col(gold, a), anchor="ls")
    half = 260 * at(t, 0.2, 0.6)
    d.rectangle([W / 2 - half, 640, W / 2 + half, 646], fill=col(gold))
    d.rectangle([W / 2 - half, 654, W / 2 + half, 656], fill=col(gold))
    y = 790
    for i, line in enumerate(bp.CHECKS):
        k = at(t, 0.5 + i * 0.35, 0.4)
        text(d, W / 2, y + (1 - k) * 26, line.replace("pays me", "pays us"),
             bb.SANS_BOLD, 48, cream, k)
        y += 90
    k = at(t, 1.7, 0.45)
    text(d, W / 2, 1210 + (1 - k) * 26, "priorline.io", bb.SERIF_BLACK, 112, gold, k)
    text(d, W / 2, 1272, "THE FULL BOARD. THE WHOLE RECORD.", bb.CAPS, 25, cream, k,
         track=0.18)
    k = at(t, 2.3, 0.45)
    text(d, W / 2, 1380 + (1 - k) * 20, f"@{handle.upper()} ON INSTAGRAM", bb.CAPS, 34, gold, k,
         track=0.16)
    if r:
        text(d, W / 2, 1090, f"ELITE PICKS THIS SEASON: {r['hit'] * 100:.1f}% RIGHT. "
             f"PRICES NEEDED {r['needed'] * 100:.1f}%.", bb.CAPS, 21, dim,
             at(t, 1.5, 0.4), track=0.12)
    text(d, W / 2, 1590, bb.LEGAL, bb.SANS, 17, dim)


def progress(d, t, cuts, accent):
    x0, y, w, gap = 90, 226, W - 180, 8
    seg = (w - gap * (len(cuts) - 1)) / len(cuts)
    start = 0.0
    rest = bb.mix(art.NIGHT, art.WHITE, 0.32)
    for i, end in enumerate(cuts):
        x = x0 + i * (seg + gap)
        d.rectangle([x, y, x + seg, y + 5], fill=col(rest))
        k = clamp((t - start) / (end - start))
        if k > 0:
            d.rectangle([x, y, x + seg * k, y + 5], fill=col(accent))
        start = end


# ------------------------------------------------------------------ the cut


def script(p, r, occasion, handle, season):
    """The parts in order: what is drawn, and what is said over it."""
    name, unit = p["player_name"], SAY_UNIT[p["market_code"]]
    line, side = bp.num(p["line"]), p["side"]
    parts = [("open", None, f"Here's how we see {name} {occasion}. The book has "
              f"him at {line} {unit}. So, over or under?")]
    for b in deep.beats(p, season):
        parts.append((b["kind"], b, b["say"]))
    home = deep.NICK.get(p["game"].split(" at ")[-1])
    w = deep.weather(home, dt.datetime.fromisoformat(p["kickoff"]))
    if w:
        say = w["say"]
        if not p["market_code"].startswith(("pass_", "rec")):
            say = say.replace(" Weather like that tends to pull passing numbers down.", "")
        parts.append(("weather", w, say))
    vals = p["form"][:bp.WINDOW]
    if vals:
        hit = sum(1 for v in vals if bp.on_side(p, v))
        parts.append(("ten", None, f"And over his last {len(vals)} games, he's "
                      f"gone {side} {line} in {hit} of them."))
    parts.append(("pick", None, f"Our number for him is {p['prior']:.1f}. The "
                  f"line is {line}. So we're going {side} {line} {unit}."))
    parts.append(("end", None, "Every pick we post gets graded in public, win or "
                  "lose. For the full board and more breakdowns like this, head "
                  f"to priorline dot io, and follow {handle} on Instagram. "
                  "Twenty-one and up, and this is a projection, not betting advice."))
    return parts


def render(path, p, r, theme, occasion, handle, voice, season):
    parts = script(p, r, occasion, handle, season)
    unit = bp.UNIT[p["market_code"]].upper()
    with tempfile.TemporaryDirectory() as tmp:
        sound, cuts, clock = b"", [], 0.0
        timed = []
        for i, (kind, data, say) in enumerate(parts):
            raw, marks = speak(say, tmp, i, voice) if voice else silent(say)
            pad = 1.3 if kind == "end" else 0.5 if kind == "pick" else 0.32
            dur = len(raw) / 2 / RATE + pad
            sound += raw + b"\x00\x00" * int(RATE * pad)
            pick_at = None
            if kind == "pick":
                # The call lands as the voice gets to "So we're going".
                so = [m[0] for m in marks if m[2].lower().startswith("so")]
                pick_at = so[-1] if so else dur * 0.55
            timed.append((kind, data, clock, dur, chunks(marks), pick_at))
            clock += dur
            cuts.append(clock)
        total = clock
        A, B, accent, white, dim = grounds(
            p, theme, (f"OVER OR UNDER {bp.num(p['line'])}?", unit))
        sprite = stamp(p, accent)
        E = end_still()
        silent_mp4 = Path(tmp) / "v.mp4"
        out = imageio_ffmpeg.write_frames(
            str(silent_mp4), (W, H), fps=FPS, codec="libx264", quality=7,
            pix_fmt_in="rgb24", pix_fmt_out="yuv420p", macro_block_size=8,
            ffmpeg_log_level="error")
        out.send(None)
        gold = bb.rgb(bb.GOLD)
        for n in range(int(total * FPS)):
            t = n / FPS
            i = next(j for j, c in enumerate(cuts) if t < c)
            kind, data, start, dur, lines, pick_at = timed[i]
            u = t - start
            if kind == "open":
                # A slow push in on the player while the question is asked.
                z = 1 + 0.05 * u / dur
                cw, ch = int(W / z), int(H / z)
                frame = A.crop(((W - cw) // 2, (H - ch) // 2 - 20,
                                (W - cw) // 2 + cw, (H - ch) // 2 - 20 + ch)
                               ).resize((W, H), Image.BILINEAR)
            elif kind == "end":
                frame = Image.blend(B, E, at(u, 0, 0.35)) if u < 0.35 else E.copy()
            else:
                k = at(u, 0, 0.35) if timed[i - 1][0] == "open" else 1.0
                frame = Image.blend(A, B, k) if k < 1 else B.copy()
            d = ImageDraw.Draw(frame, "RGBA")
            if kind == "count":
                draw_count(d, u, data, accent, white, dim)
            elif kind == "split":
                draw_split(d, u, data, accent, white, dim)
            elif kind == "league":
                draw_league(d, u, data, accent, white, dim)
            elif kind == "weather":
                draw_weather(d, u, data, accent, white, dim)
            elif kind == "ten":
                draw_ten(d, u, p, accent, white, dim)
            elif kind == "pick":
                draw_pick(frame, d, u, p, accent, white, dim, sprite, pick_at)
            elif kind == "end":
                draw_end(d, u, r, handle)
            captions(d, u, lines, gold if kind == "end" else accent)
            progress(d, t, cuts, gold if kind == "end" else accent)
            out.send(frame.tobytes())
        out.close()
        wav = Path(tmp) / "a.wav"
        with wave.open(str(wav), "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(RATE)
            f.writeframes(sound)
        subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", str(silent_mp4),
                        "-i", str(wav), "-c:v", "copy", "-c:a", "aac", "-b:a",
                        "160k", "-shortest", str(path)], check=True)
    said = "\n\n".join(s for _, _, s in parts)
    path.with_suffix(".txt").write_text(said + "\n", encoding="utf-8")
    return total


def main():
    ap = argparse.ArgumentParser(description="Draw a narrated video for each pick of a post.")
    ap.add_argument("folder", help="a post folder holding posted.json")
    ap.add_argument("--player", help="only this player's video")
    ap.add_argument("--api", default=bp.API)
    ap.add_argument("--theme", choices=["thanksgiving"])
    ap.add_argument("--occasion", help="how the day is said, e.g. 'on Thanksgiving "
                    "night'. Left out, it is the day of the game")
    ap.add_argument("--instagram", default="priorline",
                    help="the Instagram account named at the end, without the @")
    ap.add_argument("--voice", default=VOICE)
    ap.add_argument("--no-voice", action="store_true")
    ap.add_argument("--no-photos", action="store_true")
    a = ap.parse_args()
    if a.no_photos:
        art.PHOTOS[0] = False
    folder = Path(a.folder)
    data = json.loads((folder / "posted.json").read_text(encoding="utf-8"))
    r = bp.record(a.api)
    done = 0
    for i, p in enumerate(data["picks"], 1):
        if a.player and a.player.lower() not in p["player_name"].lower():
            continue
        t = dt.datetime.fromisoformat(p["kickoff"])
        occasion = a.occasion or (f"on {t.strftime('%A')} night" if t.hour >= 19
                                  else f"on {t.strftime('%A')}")
        slug = re.sub(r"[^a-z0-9]+", "-", p["player_name"].lower()).strip("-")
        path = folder / f"video-{i:02d}-{slug}.mp4"
        secs = render(path, p, r, a.theme, occasion, a.instagram,
                      None if a.no_voice else a.voice, p.get("last_season") or t.year)
        print(f"  {path.name}: {secs:.0f} seconds")
        done += 1
    print(f"{done} video(s) written to {folder}")


if __name__ == "__main__":
    main()
