"""The sound under a video: a beat made here, and the small noises that go with motion.

A video with a voice and nothing under it sounds like a voicemail. A track
from a record cannot go in an advert without a licence, and a platform mutes
it when it finds one. So the beat is built here from nothing, one drum at a
time: a kick, a clap, swung hats, a sub bass that bounces off the kick, an
electric piano on the off-beats and a small bell line. Ninety-four beats a
minute, four chords, light. It belongs to nobody else, and because it is made
here the picture knows exactly where every beat falls and can move on it.

Also here: the whoosh under a cut, the hit when the call lands, the rise
before it, and a bell when a number arrives.

mix() lays the voice over the beat and turns the beat down while it speaks.
A track of your own can be given in its place.
"""

from __future__ import annotations

import numpy as np

RATE = 44100
BPM = 94.0
BEAT = 60.0 / BPM
STEP = BEAT / 4            # a sixteenth
SWING = 0.028

_rng = np.random.default_rng(94)


def _t(seconds):
    return np.arange(int(RATE * seconds)) / RATE


def band(x, lo, hi):
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / RATE)
    X[(f < lo) | (f > hi)] = 0
    return np.fft.irfft(X, len(x))


def kick():
    t = _t(0.42)
    f = 46 + 120 * np.exp(-t * 30)
    body = np.sin(2 * np.pi * np.cumsum(f) / RATE) * np.exp(-t * 7.5)
    click = band(_rng.standard_normal(len(t)), 1500, 6000) * np.exp(-t * 260) * 0.25
    return body + click


def clap():
    t = _t(0.28)
    n = band(_rng.standard_normal(len(t)), 900, 4200)
    env = np.exp(-t * 20)
    for d in (0.0, 0.011, 0.023):
        env += np.where(t >= d, np.exp(-(t - d) * 260), 0) * 0.8
    return n * env * 0.5 + np.sin(2 * np.pi * 190 * t) * np.exp(-t * 40) * 0.25


def hat(open_=False):
    t = _t(0.26 if open_ else 0.07)
    n = band(_rng.standard_normal(len(t)), 7000, 16000)
    return n * np.exp(-t * (13 if open_ else 75))


def bass(freq, length):
    t = _t(length)
    f = freq * (1 + 0.5 * np.exp(-t * 40))
    ph = 2 * np.pi * np.cumsum(f) / RATE
    x = np.sin(ph) + 0.25 * np.sin(2 * ph)
    env = np.minimum(1, t * 200) * np.exp(-t * 2.6)
    return np.tanh(x * 1.6) * env


def keys(freqs, length=0.5):
    t = _t(length)
    x = np.zeros(len(t))
    for f in freqs:
        x += (np.sin(2 * np.pi * f * t) + 0.32 * np.sin(4 * np.pi * f * t)
              + 0.1 * np.sin(6 * np.pi * f * t))
    trem = 1 + 0.12 * np.sin(2 * np.pi * 5.2 * t)
    return x / len(freqs) * np.minimum(1, t * 400) * np.exp(-t * 6.5) * trem


def bell(freq, length=0.45):
    t = _t(length)
    return ((np.sin(2 * np.pi * freq * t) + 0.4 * np.sin(2 * np.pi * freq * 4.01 * t)
             * np.exp(-t * 14)) * np.exp(-t * 7))


def note(n):
    """Frequency of a MIDI note number."""
    return 440.0 * 2 ** ((n - 69) / 12)


def lay(buf, sound, at, gain=1.0):
    i = int(at * RATE)
    if i >= len(buf) or i < 0:
        return
    j = min(len(buf), i + len(sound))
    buf[i:j] += sound[:j - i] * gain


def loop():
    """Four bars: C, G, A minor, F. Sixteen beats, about ten seconds."""
    bars = 4
    out = np.zeros(int(RATE * BEAT * 4 * bars) + RATE)
    k, c, h, oh = kick(), clap(), hat(), hat(True)
    chords = [(48, [60, 64, 67, 71]), (43, [59, 62, 67, 69]),
              (45, [57, 60, 64, 67]), (41, [57, 60, 65, 69])]
    lead = {2: [(0, 76), (2, 79), (4, 81), (7, 79), (10, 76), (12, 74)],
            3: [(0, 72), (3, 76), (6, 74), (8, 72), (11, 69), (14, 72)]}
    for b in range(bars):
        t0 = b * BEAT * 4

        def at(step):
            return t0 + step * STEP + (SWING if step % 2 else 0.0)

        for s in (0, 7, 10):
            lay(out, k, at(s), 0.95)
        for s in (4, 12):
            lay(out, c, at(s), 0.6)
        for s in range(0, 16, 2):
            lay(out, h, at(s), 0.2)
        for s in (3, 9, 15):
            lay(out, h, at(s), 0.11)
        lay(out, oh, at(14), 0.16)
        root, tones = chords[b]
        for s, ln in ((0, 0.55), (7, 0.3), (10, 0.5)):
            lay(out, bass(note(root - 12), ln), at(s), 0.62)
        for s in (2, 5, 8, 11, 14):
            lay(out, keys([note(n) for n in tones]), at(s), 0.3)
        for s, n in lead.get(b, []):
            lay(out, bell(note(n)), at(s), 0.13)
    n = int(RATE * BEAT * 4 * bars)
    # What rings past the end is folded back to the start, so the loop joins.
    out[:len(out) - n] += out[n:]
    return out[:n]


def music(seconds):
    one = loop()
    reps = int(np.ceil(seconds * RATE / len(one))) + 1
    x = np.tile(one, reps)[:int(seconds * RATE)]
    # The first bar comes in without its low end, then the whole beat drops.
    n = int(RATE * BEAT * 2)
    if len(x) > n:
        x[:n] = band(np.concatenate([x[:n], np.zeros(64)]), 500, 16000)[:n] * 0.9
    return x


def whoosh(length=0.36):
    t = _t(length)
    n = band(_rng.standard_normal(len(t)), 600, 9000)
    return n * np.sin(np.pi * t / length) ** 2 * 0.5


def riser(length=0.9):
    t = _t(length)
    n = band(_rng.standard_normal(len(t)), 2000, 14000)
    tone = np.sin(2 * np.pi * np.cumsum(300 + 900 * (t / length) ** 2) / RATE)
    return (n * 0.5 + tone * 0.25) * (t / length) ** 2.2


def impact():
    k = kick() * 1.3
    c = clap()
    out = np.zeros(max(len(k), len(c)))
    out[:len(k)] += k
    out[:len(c)] += c * 0.9
    t = _t(len(out) / RATE)
    return out + np.sin(2 * np.pi * 55 * t) * np.exp(-t * 5) * 0.6


def ding(freq=1568.0):
    return bell(freq, 0.5) * 0.9 + bell(freq * 1.5, 0.5) * 0.35


def envelope(x, window=0.12):
    n = int(RATE * window)
    return np.convolve(np.abs(x), np.ones(n) / n, mode="same")


def mix(voice, seconds, effects=(), own=None, level=0.34):
    """Voice over beat, the beat ducked while the voice speaks.

    `voice` is float samples. `effects` is (sound, time, gain). `own` is a
    track of your own as float samples, used in place of the beat.
    """
    n = int(seconds * RATE)
    v = np.zeros(n)
    v[:min(n, len(voice))] = voice[:n]
    if own is not None and len(own):
        m = np.tile(own, int(np.ceil(n / len(own))))[:n]
    else:
        m = music(seconds)
    talk = np.clip(envelope(v) / 0.03, 0, 1)
    m = m * (level * (1.0 - 0.52 * talk))
    fade = int(RATE * 1.2)
    m[-fade:] *= np.linspace(1, 0, fade)
    fx = np.zeros(n)
    for sound, at, gain in effects:
        lay(fx, sound, at, gain)
    out = v * 1.25 + m + fx
    return np.tanh(out * 1.1) / np.tanh(1.1)


def pulse(t, sharp=0.2):
    """1 on the beat, falling to nothing before the next. What the picture moves on."""
    return float(np.exp(-(t % BEAT) / sharp))
