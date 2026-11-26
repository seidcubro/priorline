"""Turn a built post into short vertical videos, one a pick.

A carousel asks for a swipe. A video does the swiping: the player and the call
are on screen from the first frame, and then the numbers under the call arrive
one at a time, each alone and large enough to read on a phone held at arm's
length. His share of the offence, his recent volume, what the defence across
from him allows and where that ranks, his last ten games against the line, our
number against the book's, and the address.

It reads posted.json from a post folder, so a video says exactly what the
slides said and the receipt grades both. Nothing is typed here.

    python scripts/build_video.py marketing/social/posts/2026-11-26-thanksgiving
    python scripts/build_video.py <folder> --player "Josh Allen"

Each video is 1080 by 1920 at 30 frames a second, about 22 seconds, with no
sound: add the audio in the app, where a trending track helps reach. Beside
each one is a .txt with the words to read over it, if you want a voice.
Everything that matters sits inside the middle of the frame, clear of the
buttons Instagram and TikTok draw over the top and bottom.

Needs Pillow and imageio-ffmpeg.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from pathlib import Path

import imageio_ffmpeg
from PIL import Image, ImageDraw

import build_brand as bb
import build_posts as bp
import post_art as art

W, H, FPS = 1080, 1920, 30
PANEL = (1090, 1500)      # the band the changing part lives in
IN, OUT = 0.32, 0.22


def font(name, size):
    return bb.font(name, size)


def layer():
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    return im, ImageDraw.Draw(im)


def put(d, y, text, name, size, fill, track=0.0, room=W - 150, cx=W / 2):
    f = font(name, size)
    wide = f.getlength(text) + track * size * max(len(text) - 1, 0)
    if wide > room:
        size = size * room / wide
        f = font(name, size)
        wide = f.getlength(text) + track * size * max(len(text) - 1, 0)
    if not track:
        d.text((cx, y), text, font=f, fill=fill, anchor="ms")
        return size
    x = cx - wide / 2
    for ch in text:
        d.text((x, y), ch, font=f, fill=fill, anchor="ls")
        x += f.getlength(ch) + track * size
    return size


def base(p, theme):
    """What stays on screen the whole time: ground, player, name, call."""
    art.bb.Sheet.S, keep = 1, art.bb.Sheet.S
    try:
        c = art.ground(W, H, p.get("team"), fall=0.60, glow=(0.5, 0.30))
        art.season(c, theme, low=0.42, top=300)
        art.place(c, art.headshot(p.get("headshot")), W / 2, 330, 520)
    finally:
        art.bb.Sheet.S = keep
    im = c.im.convert("RGBA")
    d = ImageDraw.Draw(im)
    white = bb.rgb(art.WHITE)
    accent = bb.rgb(art.colours(p.get("team"))[1])
    dim = bb.mix(art.NIGHT, art.WHITE, 0.62)
    mark = 44
    wide = mark * 0.98 + font(bb.SERIF, mark * 0.62).getlength("PriorLine")
    m = bb.mark_mask(mark)
    im.paste(Image.new("RGB", m.size, white), (int(W / 2 - wide / 2), 262), m)
    d.text((W / 2 - wide / 2 + mark * 0.98, 262 + mark * 0.69), "PriorLine",
           font=font(bb.SERIF, mark * 0.62), fill=white, anchor="ls")
    put(d, 902, p["player_name"], bb.SERIF, 62, white, room=W - 220)
    put(d, 1050, bp.call(p).upper(), bb.SERIF_BLACK, 168, accent)
    put(d, 1096, bp.UNIT[p["market_code"]].upper(), bb.CAPS, 28, white, track=0.32)
    put(d, 1560, bb.LEGAL, bb.SANS, 17, dim)
    return im, accent, white, dim


def stat_layer(v, a, b, accent, white, dim, kicker):
    im, d = layer()
    put(d, 1180, kicker, bb.CAPS, 21, dim, track=0.18)
    put(d, 1390, v, bb.SERIF_BLACK, 220, white)
    put(d, 1446, a, bb.CAPS, 30, white, track=0.12)
    if b:
        put(d, 1492, b, bb.CAPS, 30, accent, track=0.12)
    return im


def bars_layer(p, accent, white, dim):
    im, d = layer()
    put(d, 1180, "HIS LAST TEN GAMES", bb.CAPS, 21, dim, track=0.18)
    vals = list(reversed(p["form"][:bp.WINDOW]))
    if vals:
        x0, y0, w, h = 150, 1215, 780, 200
        top = max(max(vals), p["line"]) * 1.2 or 1.0
        gap = w * 0.03
        bw = (w - gap * (len(vals) - 1)) / len(vals)
        off = bb.mix(art.NIGHT, art.WHITE, 0.5)
        for i, v in enumerate(vals):
            bh = max(6.0, h * v / top)
            bx = x0 + i * (bw + gap)
            box = [bx, y0 + h - bh, bx + bw, y0 + h]
            if bp.on_side(p, v):
                d.rectangle(box, fill=accent)
            else:
                d.rectangle(box, outline=off, width=4)
        ly = y0 + h - h * p["line"] / top
        d.rectangle([x0 - 20, ly - 2, x0 + w + 20, ly + 2], fill=white)
    put(d, 1482, bp.said(p), bb.SANS_BOLD, 36, white)
    return im


def number_layer(p, accent, white, dim):
    im, d = layer()
    put(d, 1180, f"{p['tier'].upper()} PICK", bb.CAPS, 21, dim, track=0.18)
    cols = (("OUR NUMBER", f"{p['prior']:.1f}", white),
            ("THEIR LINE", bp.num(p["line"]), accent),
            ("PRICE", bp.price(p["price"]), white))
    for i, (lab, val, col) in enumerate(cols):
        cx = W / 2 + (i - 1) * 310
        put(d, 1370, val, bb.SERIF_BLACK, 120, col, room=290, cx=cx)
        put(d, 1428, lab, bb.CAPS, 22, dim, track=0.14, room=290, cx=cx)
    return im


def open_layer(p, accent, white, dim):
    im, d = layer()
    line = "HERE IS WHY"
    if len(bp.against(p)) >= 3:
        line = bp.run_words(p).upper() + "."
        put(d, 1250, line, bb.CAPS, 34, white, track=0.1)
        put(d, 1320, f"I STILL SAY {p['side'].upper()}.", bb.CAPS, 34, accent, track=0.1)
        put(d, 1420, "HERE IS WHY", bb.CAPS, 24, dim, track=0.2)
        return im
    if bp.fact(p):
        put(d, 1270, bp.fact(p), bb.CAPS, 34, accent, track=0.1)
        put(d, 1370, "AND THAT IS ONE OF THREE REASONS", bb.CAPS, 24, dim, track=0.2)
        return im
    put(d, 1300, line, bb.CAPS, 40, white, track=0.2)
    return im


def end_card(r):
    c = Image.new("RGBA", (W, H), bb.rgb(bb.CLOTH) + (255,))
    d = ImageDraw.Draw(c)
    gold, cream = bb.rgb(bb.GOLD), bb.rgb(bb.CREAM)
    dim = bb.mix(bb.CLOTH, bb.CREAM, 0.62)
    mark = 96
    wide = mark * 0.98 + font(bb.SERIF, mark * 0.62).getlength("PriorLine")
    m = bb.mark_mask(mark)
    c.paste(Image.new("RGB", m.size, gold), (int(W / 2 - wide / 2), 560), m)
    d.text((W / 2 - wide / 2 + mark * 0.98, 560 + mark * 0.69), "PriorLine",
           font=font(bb.SERIF, mark * 0.62), fill=gold, anchor="ls")
    y = 860
    for line in bp.CHECKS:
        put(d, y, line, bb.SANS_BOLD, 46, cream)
        y += 84
    if r:
        put(d, y + 60, f"ELITE PICKS THIS SEASON: {r['hit'] * 100:.1f}% RIGHT. "
            f"PRICES NEEDED {r['needed'] * 100:.1f}%.", bb.CAPS, 22, dim, track=0.12)
    put(d, 1330, "priorline.io", bb.SERIF_BLACK, 84, gold)
    put(d, 1400, "THE FULL BOARD. THE WHOLE RECORD.", bb.CAPS, 24, cream, track=0.18)
    put(d, 1560, bb.LEGAL, bb.SANS, 17, dim)
    return c


def ease(t):
    t = max(0.0, min(1.0, t))
    return 1 - (1 - t) ** 3


def faded(im, k):
    if k >= 0.999:
        return im
    out = im.copy()
    out.putalpha(im.getchannel("A").point(lambda v: int(v * k)))
    return out


def progress(d, t, cuts, total, accent):
    """The thin bar along the top, one piece a part, filling as it plays."""
    x0, y, w, gap = 90, 228, W - 180, 8
    marks = list(cuts) + [total]
    seg = (w - gap * (len(marks) - 1)) / len(marks)
    start = 0.0
    back = bb.mix(art.NIGHT, art.WHITE, 0.32)
    for i, end in enumerate(marks):
        x = x0 + i * (seg + gap)
        d.rectangle([x, y, x + seg, y + 5], fill=back)
        k = max(0.0, min(1.0, (t - start) / (end - start)))
        if k > 0:
            d.rectangle([x, y, x + seg * k, y + 5], fill=accent)
        start = end


def words(p, r):
    """What to read over it. The same sentences the caption uses."""
    name = p["player_name"]
    out = [f"{name}. {bp.call(p)} {bp.UNIT[p['market_code']]}. Here is why."]
    out += p.get("says") or []
    if p["form"]:
        out.append(f"He has gone {bp.said(p)[0].lower() + bp.said(p)[1:]}.")
    out.append(f"My number for him is {p['prior']:.1f}. The book's line is "
               f"{bp.num(p['line'])}.")
    out.append("Every pick is graded in public, win or lose. The full board "
               "is at priorline dot io.")
    return "\n\n".join(out) + "\n\n" + bb.LEGAL + "\n"


def render(path, p, r, theme):
    im, accent, white, dim = base(p, theme)
    parts = [(open_layer(p, accent, white, dim), 2.6)]
    kick = ("THE PLAYER", "HIS WORKLOAD", "THE DEFENCE")
    for i, (v, a, b) in enumerate(p.get("tiles") or []):
        parts.append((stat_layer(v, a, b, accent, white, dim,
                                 kick[min(i, 2)]), 3.4))
    if p["form"]:
        parts.append((bars_layer(p, accent, white, dim), 3.4))
    parts.append((number_layer(p, accent, white, dim), 3.2))
    end = end_card(r)
    cuts, t = [], 0.0
    for _, dur in parts:
        t += dur
        cuts.append(t)
    total = t + 3.2
    out = imageio_ffmpeg.write_frames(
        str(path), (W, H), fps=FPS, codec="libx264", quality=7,
        pix_fmt_in="rgb24", pix_fmt_out="yuv420p", macro_block_size=8,
        ffmpeg_log_level="error")
    out.send(None)
    for n in range(int(total * FPS)):
        t = n / FPS
        if t >= cuts[-1]:
            k = ease((t - cuts[-1]) / 0.4)
            frame = Image.alpha_composite(im, faded(end, k)) if k < 1 else end.copy()
        else:
            frame = im.copy()
            i = next(j for j, c in enumerate(cuts) if t < c)
            start = cuts[i - 1] if i else 0.0
            lay, dur = parts[i]
            a = ease((t - start) / IN) if i or t > 0 else 1.0
            if i == 0:
                a = 1.0          # the first frame is the thumbnail: all there
            z = 1 - max(0.0, (t - (start + dur - OUT)) / OUT)
            shift = int((1 - a) * 34)
            moved = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            moved.paste(lay, (0, shift))
            frame = Image.alpha_composite(frame, faded(moved, min(a, max(z, 0.0))))
        d = ImageDraw.Draw(frame)
        progress(d, t, cuts, total, accent if t < cuts[-1] else bb.rgb(bb.GOLD))
        out.send(frame.convert("RGB").tobytes())
    out.close()
    return total


def main():
    ap = argparse.ArgumentParser(description="Draw a video for each pick of a post.")
    ap.add_argument("folder", help="a post folder holding posted.json")
    ap.add_argument("--player", help="only this player's video")
    ap.add_argument("--api", default=bp.API)
    ap.add_argument("--theme", choices=["thanksgiving"])
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
        slug = re.sub(r"[^a-z0-9]+", "-", p["player_name"].lower()).strip("-")
        path = folder / f"video-{i:02d}-{slug}.mp4"
        secs = render(path, p, r, a.theme)
        path.with_suffix(".txt").write_text(words(p, r), encoding="utf-8")
        print(f"  {path.name}: {secs:.0f} seconds")
        done += 1
    print(f"{done} video(s) written to {folder}")


if __name__ == "__main__":
    main()
