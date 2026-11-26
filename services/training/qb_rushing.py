"""Put the middle of a quarterback's rushing range where his middle game lands.

On the published record through 25 November 2026, our side on a quarterback's
rushing props was right 48.1% and 55.4% (attempts, 2025 and 2026) and 49.1% and
52.0% (yards). Taking every under would have done better in all four.

The projection was not the problem. We projected 16.8 rushing yards and
quarterbacks averaged 16.6. The problem is that the average is not where a
quarterback's game usually lands. One scramble for 30, or one game plan built
on designed runs, drags the average up, and the usual game sits well under it:
the middle game was 12 yards against that average of 16.6. The book hangs its
line near the average. Our range had its median near the average as well, so
the board, which takes its side from the median, was flipping a coin on a prop
that leans under.

The fix is to the range, not to the side. The median of a quarterback's
rushing range is moved to RATIO times the point projection, and the whole
range is scaled with it so it keeps its shape. The side, the chance and the
gap then come off a range centred where his games actually centre.

RATIO is the middle of (what he did / what we projected), measured on 1,014
quarterback rushing props: attempts 0.92 in 2025 and 0.89 in 2026, yards 0.75
and 0.74. Each season's ratio was then applied to the other season, which it
had not seen (lab/training/research_qb_rushing.py):

    attempts  2026: 62.0% right against the published 55.4%, on 92
              2025: 52.5% against 48.1%, on 345
    yards     2026: 57.7% against 52.0%, on 196
              2025: 53.5% against 49.1%, on 381

Together about 54.7% against 49.9%. It still calls an over when the projection
is far enough above the line, about a quarter of attempts props and a third of
yards props, which is why this is a correction and not a rule that says under.

Running backs are not touched. Their middle game is 0.91 of their average and
their range already says so.
"""

from __future__ import annotations

RATIO = {"rush_att": 0.91, "rush_yds": 0.75}


def recentre(ladder: dict, point: float, position, market_code: str) -> dict:
    """The range with its median at RATIO x point, for a quarterback's rushing.

    Anyone else, any other market, or a range with no usable median comes back
    as it was given.
    """
    ratio = RATIO.get(market_code)
    if ratio is None or str(position or "").upper() != "QB" or not ladder:
        return ladder
    mid = ladder.get(0.50)
    if not mid or mid <= 0 or not point or point <= 0:
        return ladder
    k = (point * ratio) / mid
    return {q: v * k for q, v in ladder.items()}
