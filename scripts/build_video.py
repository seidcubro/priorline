"""Turn a built post into a narrated breakdown video for each pick.

A short explainer, the kind you would watch on purpose. A voice talks through
how we see the player over a beat, and what it says is drawn as it says it.
The picture moves on the beat: the ground breathes, the player bobs, numbers
land with a bounce, each part arrives behind a swipe, and the call drops with
a hit and a burst of confetti. The order is the order of the argument:

    the question     the player, the line, over or under
    the player       his number, counting up
    his team         how the offence leans
    the defence      all 32 teams as bars, the one across from him lit
    the weather      the forecast for the middle of the game, rain falling if
                     rain is coming, or that there is a roof
    his last ten     against the line, one bar at a time
    the pick         our number against the book's, then the call
    where to go      the site, the Instagram account, the slogan

Every figure comes from posted.json (what the post said) and post_depth.py
(every play of the season), and the forecast from Open-Meteo. Nothing is
typed. If a number is against the pick it is said anyway.

The voice is one of Microsoft's read-aloud voices through the edge-tts
package: the script goes to that service and speech comes back with the time
of every word, which is what the captions follow. The beat is made in
post_sound.py and belongs to nobody else; --music puts a track of your own
under it instead.

    python scripts/build_video.py marketing/social/posts/2026-11-26-thanksgiving
    python scripts/build_video.py <folder> --player "Josh Allen" --occasion "on Thanksgiving"
    python scripts/build_video.py <folder> --voices      # samples of each voice

1080 by 1920, 30 frames a second, 45 to 60 seconds. Everything that matters
sits clear of the buttons Instagram, TikTok and Shorts draw over the top and
bottom. Needs Pillow, numpy, imageio-ffmpeg and edge-tts.
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
import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageEnhance

import build_brand as bb
import build_posts as bp
import post_art as art
import post_depth as deep
import post_sound as snd

W, H, FPS = 1080, 1920, 30
RATE = snd.RATE
# Three ways to get a voice, best first.
#   eleven   ElevenLabs. The one that sounds like a person. Needs an account
#            and its key in the ELEVENLABS_API_KEY environment variable.
#   kokoro   Kokoro, an open model that runs on this machine. Free, no account,
#            nothing leaves the computer. The default.
#   edge     Microsoft's read-aloud voices. Free, and the flattest of the three.
VOICE = {"kokoro": "am_michael", "eleven": "iP95p4xoKVk53GoZ742B",
         "edge": "en-US-BrianMultilingualNeural"}
VOICES = ("am_michael", "am_fenrir", "am_puck", "am_liam", "am_eric", "am_adam",
          "am_echo", "am_onyx", "af_heart", "af_bella")
ENGINE = ["kokoro"]
MODEL = Path(__file__).resolve().parent.parent / "marketing" / ".cache"
_KOKORO = []
TOP, FOOT = 730, 1390          # the band the drawings live in
CAP_Y = 1520                   # captions
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


def back(t, c=1.9):
    """Ease out, going past and settling. For things that land."""
    t = clamp(t)
    return 1 + (c + 1) * (t - 1) ** 3 + c * (t - 1) ** 2


def at(t, start, length):
    return ease((t - start) / length)


def pop(t, start, length=0.42):
    """0 before it arrives, then a bounce up to 1."""
    return 0.0 if t < start else max(0.0, back((t - start) / length))


def font(name, size):
    return bb.font(name, max(1, round(size)))


def col(rgb, a=1.0):
    return tuple(rgb[:3]) + (int(255 * clamp(a)),)


def text(d, cx, y, s, name, size, rgb, a=1.0, track=0.0, room=None, anchor="m",
         stroke=0):
    """Baseline text, centred on cx unless anchor is 'l' or 'r'."""
    if a <= 0.01 or not s or size < 1:
        return size
    f = font(name, size)
    wide = f.getlength(s) + track * size * max(len(s) - 1, 0)
    if room and wide > room:
        size = size * room / wide
        f = font(name, size)
        wide = f.getlength(s) + track * size * max(len(s) - 1, 0)
    x = cx - wide / 2 if anchor == "m" else cx - wide if anchor == "r" else cx
    kw = {"stroke_width": stroke, "stroke_fill": (0, 0, 0, int(255 * a))} if stroke else {}
    if not track:
        d.text((x, y), s, font=f, fill=col(rgb, a), anchor="ls", **kw)
        return size
    for ch in s:
        d.text((x, y), ch, font=f, fill=col(rgb, a), anchor="ls", **kw)
        x += f.getlength(ch) + track * size
    return size


def box(d, x0, y0, x1, y1, fill):
    if x1 > x0 and y1 > y0:
        d.rectangle([x0, y0, x1, y1], fill=fill)


def rrect(d, xy, radius=0, **kw):
    """A rounded box that is still a box while it is too small to round."""
    x0, y0, x1, y1 = xy
    if x1 - x0 < 2 or y1 - y0 < 2:
        return
    d.rounded_rectangle([x0, y0, x1, y1],
                        radius=max(0, min(radius, (x1 - x0) / 2 - 1, (y1 - y0) / 2 - 1)),
                        **kw)


def kicker(d, t, word, accent, dim):
    """The small heading of a part: a filled tab that bounces in."""
    k = pop(t, 0.12, 0.4)
    if k <= 0:
        return
    f = font(bb.CAPS, 28)
    wide = f.getlength(word) + 0.2 * 28 * (len(word) - 1) + 56
    h = 56 * k
    box(d, W / 2 - wide * k / 2, TOP + 26 - h / 2, W / 2 + wide * k / 2, TOP + 26 + h / 2,
        col(accent))
    text(d, W / 2, TOP + 36, word, bb.CAPS, 28, bb.rgb(art.NIGHT), at(t, 0.2, 0.2),
         track=0.2)


# ------------------------------------------------------------------- voice


async def _speak(line, path, voice):
    import edge_tts
    c = edge_tts.Communicate(line, voice, rate="+13%", pitch="+4Hz",
                             boundary="WordBoundary")
    marks = []
    with open(path, "wb") as f:
        async for ch in c.stream():
            if ch["type"] == "audio":
                f.write(ch["data"])
            elif ch["type"] == "WordBoundary":
                marks.append((ch["offset"] / 1e7, ch["duration"] / 1e7, ch["text"]))
    return marks


def decode(path):
    raw = subprocess.run(
        [FFMPEG, "-loglevel", "error", "-i", str(path), "-f", "s16le", "-ac", "1",
         "-ar", str(RATE), "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768.0


def spread(line, sound):
    """When each word falls, for a voice that does not say.

    The speech is found between the silence at either end and shared out
    among the words by their length, with a breath after each stop.
    """
    words = line.split()
    loud = np.where(np.abs(sound) > 0.02)[0]
    start = loud[0] / RATE if len(loud) else 0.0
    end = loud[-1] / RATE if len(loud) else len(sound) / RATE
    weight = [len(re.sub(r"[^A-Za-z0-9]", "", w)) + 1.5
              + (3.5 if w[-1:] in ".!?" else 1.5 if w[-1:] == "," else 0) for w in words]
    total = sum(weight) or 1.0
    marks, t = [], start
    for w, k in zip(words, weight):
        d = (end - start) * k / total
        marks.append((t, d * 0.9, w.strip(".,!?")))
        t += d
    return marks


def kokoro(line, voice):
    if not _KOKORO:
        from kokoro_onnx import Kokoro
        _KOKORO.append(Kokoro(str(MODEL / "kokoro-v1.0.onnx"),
                              str(MODEL / "voices-v1.0.bin")))
    x, sr = _KOKORO[0].create(line, voice=voice, speed=1.12, lang="en-us")
    x = np.asarray(x, dtype=np.float64)
    n = int(len(x) * RATE / sr)
    x = np.interp(np.arange(n) * sr / RATE, np.arange(len(x)), x)
    return x / (np.abs(x).max() + 1e-9) * 0.7


def eleven(line, voice, tmp, i):
    """ElevenLabs, with the time of every letter, which gives the time of every word."""
    import base64
    import os
    import urllib.request
    # The key is read from the environment, or from a file in the cache folder
    # that git never sees. It is never written anywhere by this script.
    key = os.environ.get("ELEVENLABS_API_KEY")
    kept = MODEL / "elevenlabs.key"
    if not key and kept.exists():
        key = kept.read_text(encoding="utf-8-sig").strip()
    if not key:
        raise SystemExit("no ElevenLabs key: set ELEVENLABS_API_KEY, or save the "
                         f"key on one line in {kept}")
    body = json.dumps({"text": line, "model_id": "eleven_multilingual_v2",
                       "voice_settings": {"stability": 0.35, "similarity_boost": 0.8,
                                          "style": 0.55, "use_speaker_boost": True}})
    req = urllib.request.Request(
        f"https://api.elevenlabs.io/v1/text-to-speech/{voice}/with-timestamps",
        data=body.encode(), headers={"xi-api-key": key, "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        got = json.load(r)
    mp3 = Path(tmp) / f"s{i}.mp3"
    mp3.write_bytes(base64.b64decode(got["audio_base64"]))
    al = got.get("alignment") or {}
    chars = al.get("characters") or []
    starts = al.get("character_start_times_seconds") or []
    ends = al.get("character_end_times_seconds") or []
    marks, word, w0, w1 = [], "", None, None
    for ch, a, b in zip(chars, starts, ends):
        if ch.isspace():
            if word:
                marks.append((w0, w1 - w0, word.strip(".,!?")))
            word, w0 = "", None
            continue
        if w0 is None:
            w0 = a
        word, w1 = word + ch, b
    if word:
        marks.append((w0, w1 - w0, word.strip(".,!?")))
    sound = decode(mp3)
    return sound, marks or spread(line, sound)


def speak(line, tmp, i, voice):
    """One line said aloud: its sound, and when each word falls."""
    if ENGINE[0] == "kokoro":
        sound = kokoro(line, voice)
        return sound, spread(line, sound)
    if ENGINE[0] == "eleven":
        return eleven(line, voice, tmp, i)
    mp3 = Path(tmp) / f"s{i}.mp3"
    marks = asyncio.run(_speak(line, mp3, voice))
    return decode(mp3), marks


def silent(line):
    ws = line.split()
    return np.zeros(int(RATE * (0.3 * len(ws) + 0.2))), [
        (0.1 + 0.3 * i, 0.26, w) for i, w in enumerate(ws)]


def chunks(marks, limit=15):
    """Two or three words at a time: enough to read in the moment they are said."""
    out, cur = [], []
    for m in marks:
        trial = " ".join(w for _, _, w in cur + [m])
        if cur and (len(trial) > limit or len(cur) >= 3):
            out.append(cur)
            cur = []
        cur.append(m)
    if cur:
        out.append(cur)
    return out


def captions(d, t, lines, accent):
    """The words as they are said: big capitals with a black edge, each word
    arriving on its own beat and the one being spoken in the team colour."""
    white = bb.rgb(art.WHITE)
    for n, ws in enumerate(lines):
        start = ws[0][0] - 0.04
        end = lines[n + 1][0][0] - 0.04 if n + 1 < len(lines) else ws[-1][0] + ws[-1][1] + 0.3
        if not (start <= t < end):
            continue
        size = 84
        words = [w.upper() for _, _, w in ws]
        f = font(bb.SANS_BOLD, size)
        space = f.getlength(" ") * 1.7
        wide = sum(f.getlength(w) for w in words) + space * (len(words) - 1)
        if wide > W - 150:
            size *= (W - 150) / wide
            f = font(bb.SANS_BOLD, size)
            space = f.getlength(" ") * 1.7
            wide = sum(f.getlength(w) for w in words) + space * (len(words) - 1)
        x = W / 2 - wide / 2
        for (w0, wd, _), word in zip(ws, words):
            ww = f.getlength(word)
            if t >= w0 - 0.04:
                live = w0 - 0.04 <= t < w0 + wd + 0.06
                k = 1 + 0.12 * max(0.0, 1 - (t - w0 + 0.04) / 0.16)
                fs = size * k
                d.text((x + ww / 2, CAP_Y), word, font=font(bb.SANS_BOLD, fs),
                       anchor="ms", fill=col(accent if live else white),
                       stroke_width=max(4, int(fs * 0.1)), stroke_fill=(0, 0, 0, 255))
            x += ww + space
        return


# ------------------------------------------------------------- the grounds


def stripes(accent):
    """Slanted bands that drift up behind everything. Barely there."""
    im = Image.new("RGBA", (W, H + 400), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    for i in range(-10, 30):
        y = i * 200
        d.polygon([(0, y + 300), (W, y), (W, y + 70), (0, y + 370)], fill=col(accent, 0.055))
    return im


def cutout(shot, height, accent, fade=0.30):
    """The player, with a flat shadow of himself in the team colour behind."""
    if shot is None:
        return None
    k = height / shot.height
    im = shot.resize((int(shot.width * k), int(height)), Image.LANCZOS)
    ramp = Image.linear_gradient("L").rotate(180).resize((im.width, int(im.height * fade)))
    veil = Image.new("L", im.size, 255)
    veil.paste(ramp, (0, im.height - ramp.height))
    alpha = ImageChops.multiply(im.getchannel("A"), veil)
    im.putalpha(alpha)
    out = Image.new("RGBA", (im.width + 40, im.height), (0, 0, 0, 0))
    ghost = Image.new("RGBA", im.size, tuple(accent) + (0,))
    ghost.putalpha(alpha.point(lambda v: int(v * 0.85)))
    out.paste(ghost, (0, 0), ghost)
    out.paste(ghost, (40, 0), ghost)
    out.paste(im, (20, 0), im)
    return out


def grounds(p, theme):
    """The two stills everything moves over, and a brighter copy of each for
    the beat to flash to."""
    keep = bb.Sheet.S
    bb.Sheet.S = 1
    try:
        team = p.get("team")
        white = bb.rgb(art.WHITE)
        accent = bb.rgb(art.colours(team)[1])
        dim = bb.mix(art.NIGHT, art.WHITE, 0.62)
        a = art.ground(W, H, team, fall=0.62, glow=(0.5, 0.33))
        art.season(a, theme, low=0.44, top=320)
        b = art.ground(W, H, team, fall=0.40, glow=(0.5, 0.20))
        art.season(b, theme, low=0.30, top=320)
    finally:
        bb.Sheet.S = keep
    out = []
    for im in (a.im, b.im):
        d = ImageDraw.Draw(im, "RGBA")
        mark = 40
        wide = mark * 0.98 + font(bb.SERIF, mark * 0.62).getlength("PriorLine")
        m = bb.mark_mask(mark)
        im.paste(Image.new("RGB", m.size, white), (int(W / 2 - wide / 2), 256), m)
        d.text((W / 2 - wide / 2 + mark * 0.98, 256 + mark * 0.69), "PriorLine",
               font=font(bb.SERIF, mark * 0.62), fill=white, anchor="ls")
        text(d, W / 2, 1640, bb.LEGAL, bb.SANS, 17, dim)
        out.append((im, ImageEnhance.Brightness(im).enhance(1.28)))
    return out[0], out[1], accent, white, dim


# ------------------------------------------------------------ the drawings


def draw_open(d, t, p, accent, white, dim):
    """The question: OVER and UNDER trade places on the beat until it is asked."""
    k = pop(t, 0.15, 0.45)
    text(d, W / 2, 1056, p["player_name"], bb.SERIF, 78 * k, white, room=W - 160)
    when = dt.datetime.fromisoformat(p["kickoff"])
    text(d, W / 2, 1112, f"{p['game'].upper()}    {bp.clock(when)}", bb.CAPS, 24, dim,
         at(t, 0.4, 0.3), track=0.18)
    beat = int(t / snd.BEAT)
    hot = beat % 2
    for i, word in enumerate(("OVER", "UNDER")):
        cx = W / 2 + (-300 if i == 0 else 300)
        k = pop(t, 0.45 + i * 0.12, 0.4)
        if k <= 0:
            continue
        on = hot == i
        s = k * (1 + (0.07 * snd.pulse(t) if on else 0))
        ww, hh = 300 * s, 116 * s
        if on:
            rrect(d, [cx - ww / 2, 1250 - hh / 2, cx + ww / 2, 1250 + hh / 2],
                                radius=18, fill=col(accent))
        else:
            rrect(d, [cx - ww / 2, 1250 - hh / 2, cx + ww / 2, 1250 + hh / 2],
                                radius=18, outline=col(white, 0.7), width=4)
        text(d, cx, 1250 + 22 * s, word, bb.SERIF_BLACK, 62 * s,
             bb.rgb(art.NIGHT) if on else white)
    k = pop(t, 0.7, 0.45)
    text(d, W / 2, 1296, bp.num(p["line"]), bb.SERIF_BLACK, 136 * k, white)
    text(d, W / 2, 1396, bp.UNIT[p["market_code"]].upper(), bb.CAPS, 34, accent,
         at(t, 0.9, 0.3), track=0.3)


def footballs(d, t, value, line, accent, white, dim, y):
    """A count you can see: one ball a touchdown, part of a ball for the rest,
    and a mark where the book's line sits."""
    n = max(2, int(math.ceil(max(value, line))) + 1)
    bw, bh, gap = 120, 70, 26
    x0 = W / 2 - (n * bw + (n - 1) * gap) / 2
    fill = value * at(t, 0.6, 1.0)
    for i in range(n):
        k = pop(t, 0.45 + i * 0.1, 0.4)
        if k <= 0:
            continue
        cx, cy = x0 + i * (bw + gap) + bw / 2, y
        w2, h2 = bw * k / 2, bh * k / 2
        d.ellipse([cx - w2, cy - h2, cx + w2, cy + h2], outline=col(dim), width=4)
        part = clamp(fill - i)
        if part > 0:
            ball = Image.new("RGBA", (bw, bh), (0, 0, 0, 0))
            bd = ImageDraw.Draw(ball)
            bd.ellipse([0, 0, bw - 1, bh - 1], fill=col(accent))
            bd.line([bw * 0.3, bh / 2, bw * 0.7, bh / 2], fill=col(bb.rgb(art.NIGHT)), width=4)
            for lx in (0.4, 0.5, 0.6):
                bd.line([bw * lx, bh * 0.36, bw * lx, bh * 0.64],
                        fill=col(bb.rgb(art.NIGHT)), width=4)
            ball = ball.crop((0, 0, max(1, int(bw * part)), bh))
            d._image.paste(ball, (int(cx - bw / 2), int(cy - bh / 2)), ball)
    a = at(t, 1.5, 0.3)
    lx = x0 + line * (bw + gap) - gap / 2 if line == int(line) else \
        x0 + int(line) * (bw + gap) + bw * (line - int(line))
    box(d, lx - 3, y - 62, lx + 3, y + 62, col(white, a))
    text(d, lx, y + 96, f"THE LINE: {bp.num(line)}", bb.CAPS, 22, white, a, track=0.14)


def draw_count(d, t, b, accent, white, dim, p):
    kicker(d, t, b["kicker"], accent, dim)
    k = at(t, 0.25, 0.9)
    shown = f"{b['value'] * k:.{b['places']}f}{b['suffix']}"
    cy = TOP + 300
    if b["suffix"] == "%":
        # A ring that fills to his share, the number inside it.
        r = 230
        d.ellipse([W / 2 - r, cy - r, W / 2 + r, cy + r],
                  outline=col(bb.mix(art.NIGHT, art.WHITE, 0.2)), width=34)
        d.arc([W / 2 - r, cy - r, W / 2 + r, cy + r], -90, -90 + 3.6 * b["value"] * k,
              fill=col(accent), width=34)
        s = pop(t, 0.2, 0.45)
        text(d, W / 2, cy + 62 * s, shown, bb.SERIF_BLACK, 176 * s, white)
        a = at(t, 0.7, 0.35)
        text(d, W / 2, TOP + 600 + (1 - a) * 22, b["label"], bb.CAPS, 34, white, a,
             track=0.14, room=W - 160)
        a = at(t, 1.1, 0.35)
        text(d, W / 2, TOP + 648 + (1 - a) * 22, b["sub"], bb.CAPS, 28, accent, a,
             track=0.12, room=W - 160)
        return
    s = pop(t, 0.2, 0.45)
    text(d, W / 2, TOP + 330 - (1 - s) * 40, shown, bb.SERIF_BLACK, 270 * s, white)
    a = at(t, 0.6, 0.35)
    text(d, W / 2, TOP + 400 + (1 - a) * 22, b["label"], bb.CAPS, 36, white, a,
         track=0.14, room=W - 160)
    if p["market_code"] == "pass_td" and b["value"] <= 5:
        footballs(d, t, b["value"], p["line"], accent, white, dim, TOP + 508)
        return
    a = at(t, 1.0, 0.35)
    wide = 300 * a
    box(d, W / 2 - wide, TOP + 440, W / 2 + wide, TOP + 444, col(accent, a))
    text(d, W / 2, TOP + 500 + (1 - a) * 22, b["sub"], bb.CAPS, 30, accent, a,
         track=0.12, room=W - 160)


def draw_split(d, t, b, accent, white, dim):
    kicker(d, t, b["kicker"], accent, dim)
    text(d, W / 2, TOP + 130, b["heading"], bb.CAPS, 28, white, at(t, 0.3, 0.3), track=0.16)
    x0, y0, w, h = 110, TOP + 320, W - 220, 130
    left, right = b["left"][1], b["right"][1]
    k = at(t, 0.4, 1.0)
    cut = x0 + w * left / 100
    box(d, x0, y0, x0 + w, y0 + h, col(bb.mix(art.NIGHT, art.WHITE, 0.14)))
    box(d, x0, y0, x0 + (cut - x0) * k, y0 + h, col(accent))
    box(d, x0 + w - (x0 + w - cut) * k, y0, x0 + w, y0 + h,
        col(bb.mix(art.NIGHT, art.WHITE, 0.86)))
    big = pop(t, 0.4, 0.45)
    text(d, x0, y0 - 34, f"{left * k:.0f}%", bb.SERIF_BLACK, 120 * big, accent, anchor="l")
    text(d, x0 + w, y0 - 34, f"{right * k:.0f}%", bb.SERIF_BLACK, 120 * big, white,
         anchor="r")
    a = at(t, 1.1, 0.35)
    text(d, x0, y0 + h + 60, b["left"][0], bb.CAPS, 30, accent, a, track=0.14, anchor="l")
    text(d, x0 + w, y0 + h + 60, b["right"][0], bb.CAPS, 30, white, a, track=0.14,
         anchor="r")
    # The bigger half gets the arrow.
    a = pop(t, 1.5, 0.4)
    if a > 0:
        cx = x0 + w - (x0 + w - cut) / 2 if right >= left else x0 + (cut - x0) / 2
        tip = y0 + h + 110
        d.polygon([(cx, tip), (cx - 26 * a, tip + 40 * a), (cx + 26 * a, tip + 40 * a)],
                  fill=col(accent if left > right else white))


def draw_league(d, t, b, accent, white, dim):
    """All 32 teams as bars, fewest on the left, and the one that matters lit."""
    kicker(d, t, b["kicker"], accent, dim)
    text(d, W / 2, TOP + 122, b["heading"], bb.CAPS, 27, white, at(t, 0.3, 0.3),
         track=0.14, room=W - 160)
    bars = b["bars"]
    n = len(bars)
    x0, w, base, tall = 96, W - 192, TOP + 600, 300
    gap = 5
    bw = (w - gap * (n - 1)) / n
    top = max(v for _, v in bars) or 1.0
    floor = min(v for _, v in bars) * 0.55     # so the gaps between teams show
    plain = bb.mix(art.NIGHT, art.WHITE, 0.36)
    lit_at = 1.35
    hx = hy = None
    for i, (team, v) in enumerate(bars):
        k = pop(t, 0.35 + i * 0.024, 0.45)
        bh = max(4.0, tall * (v - floor) / (top - floor)) * k
        bx = x0 + i * (bw + gap)
        mine = team == b["team"]
        lit = mine and t > lit_at
        if lit:
            bh *= 1 + 0.10 * max(0.0, 1 - (t - lit_at) / 0.3)
        box(d, bx, base - bh, bx + bw, base, col(accent if lit else plain))
        if mine:
            hx, hy = bx + bw / 2, base - bh
    u = t - lit_at
    if hx is not None and u > 0:
        # A ring bursts off the bar, and its number drops in above it.
        r = 30 + 240 * ease(u / 0.5)
        ra = max(0.0, 1 - u / 0.5)
        if ra > 0:
            d.ellipse([hx - r, hy - r, hx + r, hy + r], outline=col(accent, ra), width=6)
        k = pop(t, lit_at, 0.4)
        tx = clamp(hx, 150, W - 150)
        box(d, hx - 2, hy - 6 - 20 * k, hx + 2, hy - 6, col(accent))
        text(d, tx, hy - 40 * k, f"{b['value']:.1f}", bb.SERIF_BLACK, 80 * k, accent)
        k = pop(t, lit_at + 0.12, 0.45)
        text(d, W / 2, TOP + 230, b["words"].upper(), bb.SERIF_BLACK, 80 * k, white,
             room=W - 200)
    a = at(t, 0.6, 0.4)
    text(d, x0, base + 44, "FEWEST", bb.CAPS, 20, dim, a, track=0.16, anchor="l")
    text(d, x0 + w, base + 44, "MOST", bb.CAPS, 20, dim, a, track=0.16, anchor="r")
    text(d, W / 2, base + 44, "ALL 32 TEAMS", bb.CAPS, 20, dim, a, track=0.16)


def cloud(d, cx, cy, s, rgb, a):
    for dx, dy, r in ((-70, 14, 46), (-22, -16, 62), (40, -4, 54), (84, 18, 40)):
        d.ellipse([cx + (dx - r) * s, cy + (dy - r) * s, cx + (dx + r) * s, cy + (dy + r) * s],
                  fill=col(rgb, a))
    rrect(d, [cx - 116 * s, cy + 6 * s, cx + 124 * s, cy + 60 * s],
                        radius=26 * s, fill=col(rgb, a))


def draw_weather(d, t, b, accent, white, dim, seed=7):
    kicker(d, t, "THE WEATHER", accent, dim)
    if b["roof"]:
        k = at(t, 0.3, 0.7)
        d.arc([W / 2 - 300, TOP + 190, W / 2 + 300, TOP + 790], 180, 180 + 180 * k,
              fill=col(accent), width=14)
        box(d, W / 2 - 300, TOP + 488, W / 2 - 300 + 600 * k, TOP + 496, col(white))
        text(d, W / 2, TOP + 420, "INDOORS", bb.SERIF_BLACK, 108 * pop(t, 0.5, 0.45), white)
        text(d, W / 2, TOP + 566, f"UNDER A ROOF IN {b['city'].upper()}", bb.CAPS, 28,
             accent, at(t, 0.9, 0.3), track=0.16)
        return
    wet = b["wet"] >= 40
    if wet:
        rnd = random.Random(seed)
        for _ in range(90):
            x = rnd.uniform(40, W - 40)
            sp = rnd.uniform(560, 900)
            ln = rnd.uniform(24, 52)
            if b["fall"] == "snow":
                r = rnd.uniform(3, 7)
                y = TOP + 150 + ((rnd.uniform(0, 520) + t * sp * 0.25) % 520)
                d.ellipse([x - r, y - r, x + r, y + r], fill=col(white, 0.55 * at(t, 0.2, 0.5)))
            else:
                y = TOP + 150 + ((rnd.uniform(0, 520) + t * sp) % 520)
                d.line([x, y, x - ln * 0.22, y + ln], fill=col(white, 0.34 * at(t, 0.2, 0.5)),
                       width=3)
        bob = 8 * math.sin(t * 2.2)
        cloud(d, W / 2 - 250, TOP + 150 + bob, pop(t, 0.2, 0.5), bb.mix(art.NIGHT, art.WHITE, 0.78), 1)
        cloud(d, W / 2 + 230, TOP + 130 - bob, pop(t, 0.32, 0.5) * 0.8,
              bb.mix(art.NIGHT, art.WHITE, 0.6), 1)
    tiles = [(f"{b['temp'] * at(t, 0.3, 0.9):.0f}°", "DEGREES"),
             (f"{b['wind'] * at(t, 0.5, 0.9):.0f}", "MPH WIND"),
             (f"{b['wet'] * at(t, 0.7, 0.9):.0f}%", f"CHANCE OF {b['fall'].upper()}")]
    cw, gap = 290, 24
    x = W / 2 - (cw * 3 + gap * 2) / 2
    for i, (v, lab) in enumerate(tiles):
        k = pop(t, 0.3 + i * 0.2, 0.45)
        if k > 0:
            y = TOP + 250
            hh = 290 * k
            rrect(d, [x, y + 145 - hh / 2, x + cw, y + 145 + hh / 2], radius=16,
                                fill=col(bb.rgb(art.NIGHT), 0.82))
            box(d, x + 16, y + 145 - hh / 2, x + cw - 16, y + 145 - hh / 2 + 8, col(accent))
            text(d, x + cw / 2, y + 186, v, bb.SERIF_BLACK, 116 * k, white, room=cw - 30)
            text(d, x + cw / 2, y + 250, lab, bb.CAPS, 22, accent, at(t, 0.5 + i * 0.2, 0.3),
                 track=0.12, room=cw - 30)
        x += cw + gap
    text(d, W / 2, TOP + 620, f"{b['city'].upper()}, AROUND HALF-TIME", bb.CAPS, 24, dim,
         at(t, 1.1, 0.4), track=0.18)


def draw_ten(d, t, p, accent, white, dim):
    kicker(d, t, "HIS LAST TEN", accent, dim)
    vals = list(reversed(p["form"][:bp.WINDOW]))
    if not vals:
        return
    x0, y0, w, h = 130, TOP + 270, W - 260, 290
    top = max(max(vals), p["line"]) * 1.22 or 1.0
    gap = w * 0.03
    bw = (w - gap * (len(vals) - 1)) / len(vals)
    off = bb.mix(art.NIGHT, art.WHITE, 0.55)
    seen = 0
    for i, v in enumerate(vals):
        k = pop(t, 0.35 + i * 0.09, 0.4)
        if k <= 0:
            continue
        bh = max(10.0, h * v / top) * k
        bx = x0 + i * (bw + gap)
        if bp.on_side(p, v):
            seen += 1
            box(d, bx, y0 + h - bh, bx + bw, y0 + h, col(accent))
        else:
            d.rectangle([bx, y0 + h - max(bh, 10), bx + bw, y0 + h], outline=col(off), width=5)
        text(d, bx + bw / 2, y0 + h + 42, bp.num(v), bb.CAPS, 24, dim, at(t, 0.5 + i * 0.09, 0.2))
    ly = y0 + h - h * p["line"] / top
    k = at(t, 0.3, 0.8)
    box(d, x0 - 24, ly - 2, x0 - 24 + (w + 48) * k, ly + 2, col(white))
    text(d, x0 + w + 24, ly - 14, f"LINE {bp.num(p['line'])}", bb.CAPS, 22, white,
         at(t, 1.0, 0.3), track=0.12, anchor="r")
    k = pop(t, 0.3, 0.4)
    text(d, W / 2, TOP + 200, f"{p['side'].upper()} IN {seen} OF {len(vals)}",
         bb.SERIF_BLACK, 80 * k, white, room=W - 200)


def stamp(p, accent):
    """The call, drawn once at twice the size so it can land without blurring."""
    im = Image.new("RGBA", (W * 2, 520), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    text(d, W, 400, bp.call(p).upper(), bb.SERIF_BLACK, 340, accent, room=W * 2 - 240)
    return im


def confetti(accent, seed=3):
    rnd = random.Random(seed)
    shades = [accent, bb.rgb(art.WHITE), bb.rgb(bb.GOLD)]
    return [(rnd.uniform(-1, 1) * 560, -rnd.uniform(520, 1250), rnd.uniform(8, 20),
             rnd.uniform(0, 6.28), rnd.uniform(-9, 9), rnd.choice(shades))
            for _ in range(120)]


def draw_pick(frame, d, t, p, accent, white, dim, sprite, bits, when):
    kicker(d, t, "THE PICK", accent, dim)
    rows = (("OUR NUMBER", p["prior"], white), ("THEIR LINE", p["line"], accent))
    big = max(p["prior"], p["line"]) or 1.0
    x0, w = 130, W - 260
    fade = 1 - at(t, when - 0.12, 0.25)
    for i, (lab, v, c) in enumerate(rows):
        y = TOP + 170 + i * 160
        k = at(t, 0.3 + i * 0.3, 0.8)
        text(d, x0, y, lab, bb.CAPS, 26, dim, at(t, 0.2 + i * 0.3, 0.3) * fade,
             track=0.16, anchor="l")
        box(d, x0, y + 18, x0 + w, y + 82, col(bb.mix(art.NIGHT, art.WHITE, 0.12), fade))
        box(d, x0, y + 18, x0 + w * 0.86 * (v / big) * k, y + 82, col(c, fade))
        text(d, x0 + w, y, f"{v * k:.1f}", bb.SERIF_BLACK, 66, c, fade, anchor="r")
    if t < when:
        return
    u = t - when
    flash = max(0.0, 1 - u / 0.2)
    if flash > 0:
        d.rectangle([0, 0, W, H], fill=col(white, 0.55 * flash))
    r = 80 + 900 * ease(u / 0.6)
    ra = max(0.0, 1 - u / 0.6)
    if ra > 0:
        d.ellipse([W / 2 - r, TOP + 260 - r, W / 2 + r, TOP + 260 + r],
                  outline=col(accent, ra), width=10)
    for vx, vy, size, rot, spin, shade in bits:
        x = W / 2 + vx * u
        y = TOP + 300 + vy * u + 1300 * u * u
        if y > H or u > 2.6:
            continue
        a = rot + spin * u
        dx, dy = math.cos(a) * size, math.sin(a) * size
        d.polygon([(x - dx, y - dy), (x - dy * 0.5, y + dx * 0.5), (x + dx, y + dy),
                   (x + dy * 0.5, y - dx * 0.5)], fill=col(shade))
    k = 2.4 - 1.4 * back(u / 0.34, 2.4)
    size = (max(2, int(sprite.width * k / 2)), max(2, int(sprite.height * k / 2)))
    s = sprite.resize(size, Image.LANCZOS)
    frame.paste(s, (int(W / 2 - s.width / 2), int(TOP + 350 - s.height * 0.62)), s)
    text(d, W / 2, TOP + 490, bp.UNIT[p["market_code"]].upper(), bb.CAPS, 46, white,
         at(u, 0.25, 0.3), track=0.3)
    k = pop(u, 0.45, 0.4)
    if k > 0:
        tag = f"{p['tier'].upper()} PICK    {bp.price(p['price'])}"
        f = font(bb.CAPS, 28)
        wide = (f.getlength(tag) + 0.16 * 28 * (len(tag) - 1) + 60) * k
        rrect(d, [W / 2 - wide / 2, TOP + 540, W / 2 + wide / 2, TOP + 600],
                            radius=30, outline=col(white, 0.8), width=3)
        text(d, W / 2, TOP + 580, tag, bb.CAPS, 28, white, at(u, 0.55, 0.2), track=0.16)


def draw_end(d, t, r, handle, slogan_at):
    gold, cream = bb.rgb(bb.GOLD), bb.rgb(bb.CREAM)
    dim = bb.mix(bb.CLOTH, bb.CREAM, 0.62)
    k = pop(t, 0.1, 0.5)
    mark = 110 * max(k, 0.01)
    wide = mark * 0.98 + font(bb.SERIF, mark * 0.62).getlength("PriorLine")
    if mark >= 30:
        m = bb.mark_mask(int(mark))
        d._image.paste(Image.new("RGB", m.size, gold), (int(W / 2 - wide / 2), int(520 - mark / 2)), m)
        d.text((W / 2 - wide / 2 + mark * 0.98, 520 - mark / 2 + mark * 0.69), "PriorLine",
               font=font(bb.SERIF, mark * 0.62), fill=col(gold), anchor="ls")
    y = 760
    for i, line in enumerate(bp.CHECKS):
        k = at(t, 0.5 + i * 0.3, 0.4)
        text(d, W / 2, y + (1 - k) * 30, line.replace("pays me", "pays us"),
             bb.SANS_BOLD, 46, cream, k)
        y += 84
    k = pop(t, 1.6, 0.45)
    text(d, W / 2, 1140, "priorline.io", bb.SERIF_BLACK, 116 * k, gold)
    k = pop(t, 2.2, 0.45)
    if k > 0:
        tag = f"@{handle.upper()} ON INSTAGRAM"
        f = font(bb.CAPS, 32)
        w2 = (f.getlength(tag) + 0.14 * 32 * (len(tag) - 1) + 70) * k / 2
        rrect(d, [W / 2 - w2, 1196, W / 2 + w2, 1264], radius=34, fill=col(gold))
        text(d, W / 2, 1242, tag, bb.CAPS, 32, bb.rgb(bb.CLOTH), at(t, 2.3, 0.2), track=0.14)
    # The slogan arrives as it is said.
    u = t - slogan_at
    if u > 0:
        k = pop(u, 0.0, 0.45)
        text(d, W / 2 - 14, 1400, bb.TAG_A, bb.SERIF_BLACK, 76 * k, gold, anchor="r")
        k = pop(u, 0.5, 0.45)
        text(d, W / 2 + 14, 1400, bb.TAG_B, bb.ITALIC, 76 * k, cream, anchor="l")
        half = 330 * at(u, 0.9, 0.5)
        box(d, W / 2 - half, 1428, W / 2 + half, 1434, col(gold))
        box(d, W / 2 - half, 1442, W / 2 + half, 1444, col(gold))
    text(d, W / 2, 1640, bb.LEGAL, bb.SANS, 17, dim)


def swipe(d, u, accent, white, length=0.34):
    """Two slanted slabs cross the frame as one part gives way to the next."""
    if u >= length:
        return
    k = u / length
    x = -700 + (W + 1400) * ease(k)
    for off, wide, shade in ((0, 520, accent), (-560, 90, white)):
        a, b = x + off, x + off + wide
        d.polygon([(a, TOP - 30), (b, TOP - 30), (b - 220, FOOT + 60), (a - 220, FOOT + 60)],
                  fill=col(shade))


def progress(d, t, cuts, accent):
    x0, y, w, gap = 90, 226, W - 180, 8
    seg = (w - gap * (len(cuts) - 1)) / len(cuts)
    start = 0.0
    rest = bb.mix(art.NIGHT, art.WHITE, 0.32)
    for i, end in enumerate(cuts):
        x = x0 + i * (seg + gap)
        box(d, x, y, x + seg, y + 6, col(rest))
        k = clamp((t - start) / (end - start))
        box(d, x, y, x + seg * k, y + 6, col(accent))
        start = end


# ------------------------------------------------------------------ the cut

LEAD_IN = {"count": "First up. ", "split": "Now check this out. ",
           "league": ""}


def weather_line(w, market):
    if w["roof"]:
        return f"And the weather? They're indoors in {w['city']}, so that's a non-factor."
    line = "And the weather? "
    if w["wet"] >= 80:
        line += (f"It's going to be pouring {w['fall']} in {w['city']}! About "
                 f"{w['temp']:.0f} degrees")
    else:
        line += f"In {w['city']}, it's about {w['temp']:.0f} degrees"
        if w["wet"] >= 40:
            line += f", with a {w['wet']:.0f} percent chance of {w['fall']}"
    if w["wind"] >= 12:
        line += f", and wind around {w['wind']:.0f} miles an hour"
    line += "."
    rough = w["wind"] >= 15 or w["wet"] >= 60
    if rough and market.startswith(("pass_", "rec")):
        line += " That is not passing weather."
    elif not rough:
        line += " Nothing there to worry about."
    return line


def script(p, occasion, handle, season):
    """The parts in order: what is drawn, and what is said over it."""
    name, unit = p["player_name"], SAY_UNIT[p["market_code"]]
    line, side = bp.num(p["line"]), p["side"]
    parts = [("open", None, f"{name}, {occasion}! The book says {line} {unit}. "
              f"Over, or under? Let's break it down.")]
    story = deep.beats(p, season)
    for n, b in enumerate(story):
        lead = "First up. " if n == 0 else ""
        if b["kind"] == "split":
            lead = "Now check this out. "
        if n == len(story) - 1 and n > 0:
            lead = "And here's the big one. "
        say = b["say"]
        if lead and say.startswith(("And ", "On the other side, ")):
            say = say.split(" ", 1)[1] if say.startswith("And ") else say[len("On the other side, "):]
            say = say[0].upper() + say[1:]
        parts.append((b["kind"], b, lead + say))
    home = deep.NICK.get(p["game"].split(" at ")[-1])
    w = deep.weather(home, dt.datetime.fromisoformat(p["kickoff"]))
    if w:
        parts.append(("weather", w, weather_line(w, p["market_code"])))
    vals = p["form"][:bp.WINDOW]
    if vals:
        hit = sum(1 for v in vals if bp.on_side(p, v))
        parts.append(("ten", None, f"And get this. He's gone {side} {line} in {hit} "
                      f"of his last {len(vals)} games."))
    parts.append(("pick", None, f"So here's the play. Our number is {p['prior']:.1f}, "
                  f"the line is {line}. We're going {side} {line} {unit}!"))
    said = " ".join(handle.replace("priorline", "prior line ").split())
    said = re.sub(r"\bio$", "I O", said)
    parts.append(("end", None, "We grade every single pick in public, win or lose. "
                  f"Want more like this? Head to prior line dot I O, and follow "
                  f"{said} on Instagram. PriorLine. Our prior. Their line."))
    return parts


def render(path, p, r, theme, occasion, handle, voice, season, own):
    parts = script(p, occasion, handle, season)
    with tempfile.TemporaryDirectory() as tmp:
        voice_track, cuts, clock, timed, fx = [], [], 0.0, [], []
        for i, (kind, data, say) in enumerate(parts):
            raw, marks = speak(say, tmp, i, voice) if voice else silent(say)
            pad = 1.6 if kind == "end" else 0.7 if kind == "pick" else 0.3
            dur = len(raw) / RATE + pad
            voice_track.append(np.concatenate([raw, np.zeros(int(RATE * pad))]))
            mark = None
            if kind == "pick":
                # The call lands as the voice gets to "We're going".
                hits = [m[0] for m in marks if m[2].lower().strip(".,!") == "going"]
                mark = (hits[-1] + 0.12) if hits else dur * 0.6
                fx.append((snd.riser(), clock + mark - 0.9, 0.5))
                fx.append((snd.impact(), clock + mark, 1.0))
            elif kind == "end":
                hits = [m[0] for m in marks if m[2].lower().strip(".,!") == "our"]
                mark = hits[-1] if hits else dur * 0.7
            elif kind == "league":
                fx.append((snd.ding(), clock + 1.35, 0.45))
            elif kind == "count":
                fx.append((snd.ding(1318.5), clock + 1.15, 0.3))
            if i:
                fx.append((snd.whoosh(), clock - 0.05, 0.55))
            timed.append((kind, data, clock, dur, chunks(marks), mark))
            clock += dur
            cuts.append(clock)
        total = clock
        (A, A2), (B, B2), accent, white, dim = grounds(p, theme)
        band = stripes(accent)
        shot = art.headshot(p.get("headshot"))
        big, small = cutout(shot, 640, accent), cutout(shot, 330, accent)
        sprite, bits = stamp(p, accent), confetti(accent)
        E = Image.new("RGB", (W, H), bb.rgb(bb.CLOTH))
        E2 = ImageEnhance.Brightness(E).enhance(1.5)
        gold = bb.rgb(bb.GOLD)
        silent_mp4 = Path(tmp) / "v.mp4"
        out = imageio_ffmpeg.write_frames(
            str(silent_mp4), (W, H), fps=FPS, codec="libx264", quality=7,
            pix_fmt_in="rgb24", pix_fmt_out="yuv420p", macro_block_size=8,
            ffmpeg_log_level="error")
        out.send(None)
        try:
            for n in range(int(total * FPS)):
                t = n / FPS
                i = next(j for j, c in enumerate(cuts) if t < c)
                kind, data, start, dur, lines, mark = timed[i]
                u = t - start
                beat = snd.pulse(t)
                move = at(t, cuts[0], 0.45)        # 0 in the opening, 1 after
                if kind == "end":
                    k = at(u, 0, 0.3)
                    frame = Image.blend(E, E2, 0.25 * beat)
                    if k < 1:
                        frame = Image.blend(Image.blend(B, B2, 0.5 * beat), frame, k)
                else:
                    a = Image.blend(A, A2, 0.55 * beat)
                    if move > 0:
                        b2 = Image.blend(B, B2, 0.55 * beat)
                        a = b2 if move >= 1 else Image.blend(a, b2, move)
                    frame = a
                    off = int((t * 46) % 200)
                    piece = band.crop((0, off, W, off + H))
                    frame.paste(piece, (0, 0), piece)
                    if big is not None:
                        # The player drops in, bobs, and nods on the beat; when
                        # the question is asked he moves up out of the way.
                        k = pop(t, 0.0, 0.5)
                        src, hgt, top_y = (big, 640, 330) if move < 0.5 else (small, 330, 292)
                        if 0 < move < 1:
                            hgt = 640 + (330 - 640) * move
                            top_y = 330 + (292 - 330) * move
                            src = big
                        s = max(0.05, k) * (1 + 0.022 * beat)
                        hh = int(hgt * s)
                        ww = int(src.width * hh / src.height)
                        im = src if (ww, hh) == src.size else src.resize((ww, hh), Image.BILINEAR)
                        bob = 7 * math.sin(t * 2.4)
                        frame.paste(im, (int(W / 2 - ww / 2),
                                         int(top_y + (hgt - hh) / 2 + bob)), im)
                d = ImageDraw.Draw(frame, "RGBA")
                if kind != "open" and kind != "end":
                    a = at(t, cuts[0] + 0.15, 0.3)
                    text(d, W / 2, 672, p["player_name"], bb.SERIF, 50, white, a, room=W - 200)
                    text(d, W / 2, 708, f"{bp.num(p['line'])} "
                         f"{bp.UNIT[p['market_code']].upper()}", bb.CAPS, 21, accent, a,
                         track=0.24)
                if kind == "open":
                    draw_open(d, u, p, accent, white, dim)
                elif kind == "count":
                    draw_count(d, u, data, accent, white, dim, p)
                elif kind == "split":
                    draw_split(d, u, data, accent, white, dim)
                elif kind == "league":
                    draw_league(d, u, data, accent, white, dim)
                elif kind == "weather":
                    draw_weather(d, u, data, accent, white, dim)
                elif kind == "ten":
                    draw_ten(d, u, p, accent, white, dim)
                elif kind == "pick":
                    draw_pick(frame, d, u, p, accent, white, dim, sprite, bits, mark)
                elif kind == "end":
                    draw_end(d, u, r, handle, mark)
                if i and kind != "end":
                    swipe(d, u, accent, white)
                if kind != "end":
                    # The closing card already shows what is being said, and the
                    # voice spells the address out in a way nobody should read.
                    captions(d, u, lines, accent)
                progress(d, t, cuts, gold if kind == "end" else accent)
                if kind == "pick" and 0 <= u - mark < 0.3:
                    s = (u - mark) / 0.3
                    frame = ImageChops.offset(frame, int(18 * math.sin(s * 40) * (1 - s)),
                                              int(12 * math.cos(s * 34) * (1 - s)))
                out.send(frame.tobytes())
        finally:
            out.close()
        sound = snd.mix(np.concatenate(voice_track), total, fx, own)
        wav = Path(tmp) / "a.wav"
        with wave.open(str(wav), "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(RATE)
            f.writeframes((np.clip(sound, -1, 1) * 32000).astype(np.int16).tobytes())
        subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", str(silent_mp4),
                        "-i", str(wav), "-c:v", "copy", "-c:a", "aac", "-b:a",
                        "192k", "-shortest", str(path)], check=True)
    path.with_suffix(".txt").write_text(
        "\n\n".join(s for _, _, s in parts) + "\n", encoding="utf-8")
    return total


def samples(folder, line):
    """The same sentence in each voice, to choose one by ear."""
    out = Path(folder) / "voices"
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.mp3"):
        old.unlink()
    for v in VOICES:
        x = kokoro(line, v)
        raw = (np.clip(x, -1, 1) * 32000).astype(np.int16).tobytes()
        subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-f", "s16le", "-ar",
                        str(RATE), "-ac", "1", "-i", "-", str(out / f"{v}.mp3")],
                       input=raw, check=True)
        print(f"  {v}.mp3")
    print(f"{len(VOICES)} voices written to {out}")


def main():
    ap = argparse.ArgumentParser(description="Draw a narrated video for each pick of a post.")
    ap.add_argument("folder", help="a post folder holding posted.json")
    ap.add_argument("--player", help="only this player's video")
    ap.add_argument("--api", default=bp.API)
    ap.add_argument("--theme", choices=["thanksgiving"])
    ap.add_argument("--occasion", help="how the day is said, e.g. 'on Thanksgiving'. "
                    "Left out, it is the day of the game")
    ap.add_argument("--instagram", default="priorlineio",
                    help="the Instagram account named at the end, without the @")
    ap.add_argument("--engine", choices=["kokoro", "eleven", "edge"], default="kokoro",
                    help="where the voice comes from (see the top of this file)")
    ap.add_argument("--voice", help="a voice of that engine, if not its usual one")
    ap.add_argument("--voices", action="store_true",
                    help="write a sample of each voice and stop")
    ap.add_argument("--music", help="a track of your own to put under the voice "
                    "in place of the beat made here")
    ap.add_argument("--no-voice", action="store_true")
    ap.add_argument("--no-photos", action="store_true")
    a = ap.parse_args()
    if a.no_photos:
        art.PHOTOS[0] = False
    ENGINE[0] = a.engine
    a.voice = a.voice or VOICE[a.engine]
    folder = Path(a.folder)
    data = json.loads((folder / "posted.json").read_text(encoding="utf-8"))
    if a.voices:
        p = data["picks"][0]
        samples(folder, f"{p['player_name']}, on game day! The book says "
                f"{bp.num(p['line'])} {SAY_UNIT[p['market_code']]}. Over, or under? "
                f"Let's break it down. And here's the big one. We're going "
                f"{p['side']} {bp.num(p['line'])}!")
        return
    own = decode(a.music) if a.music else None
    if own is not None:
        bpm, first = snd.tempo(own)
        snd.BEAT, snd.OFFSET = 60.0 / bpm, first
        print(f"  the track runs at about {bpm:.0f} beats a minute; the picture "
              f"moves on that")
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
                      None if a.no_voice else a.voice, p.get("last_season") or t.year, own)
        print(f"  {path.name}: {secs:.0f} seconds")
        done += 1
    print(f"{done} video(s) written to {folder}")


if __name__ == "__main__":
    main()
