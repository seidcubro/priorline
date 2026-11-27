"""Does the over on a quarterback with a high passing-yards line hold on a third season?

On the published record, the over on a passing-yards prop came in more often
on the highest lines: in the top quarter of lines it was 60.6% on 99 in 2025
and 56.8% on 37 in 2026. Two related cuts leaned the same way (teams favoured
by four or more, and props where our projection was well above the line). That
is 136 props, which is not enough to build on.

The database holds 2024 passing-yards prices as well, which the record pulled
through the site does not. This reads all three seasons from the same place and
prints, for each season and each quarter of that season's lines: props, how
often the over came in, what the over's price needed, and the difference. If
the top quarter is ahead in 2024 too, it is three seasons and worth a rule. If
it is not, it was two seasons of luck.

Prices are the best any book showed. Read-only.

    SEASONS=2024,2025,2026 sh lab/run.sh research_pass_highline.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import research_pass_form as F
from research_side_by_price import implied


def main():
    j = F.props("pass_yds")
    if j is None or len(j) < 200:
        print("pass_yds: nothing to read")
        return
    j = j.copy()
    j["need_over"] = implied(j["over"])
    j["need_under"] = implied(j["under"])
    j["q"] = j.groupby("season")["line"].transform(
        lambda v: pd.qcut(v.rank(method="first"), 4, labels=False))
    print(f"PASS_YDS: {len(j):,} priced props")
    print(f"  {'season':<8}{'quarter of lines':<20}{'lines':<16}{'props':>6}{'over':>8}"
          f"{'needed':>9}{'value':>8}")
    for s in sorted(j["season"].unique()):
        for q in range(4):
            x = j[(j["season"] == s) & (j["q"] == q)]
            if len(x) < 20:
                continue
            o, n = x["went_over"].mean(), x["need_over"].mean()
            print(f"  {int(s):<8}{['lowest', 'second', 'third', 'highest'][q]:<20}"
                  f"{x['line'].min():.1f} to {x['line'].max():<8.1f}{len(x):>6}{o:>8.1%}"
                  f"{n:>9.1%}{o - n:>+8.1%}")
    top = j[j["q"] == 3]
    o, n = top["went_over"].mean(), top["need_over"].mean()
    print(f"\n  highest quarter, every season: {len(top)} props, over {o:.1%}, "
          f"needed {n:.1%}, value {o - n:+.1%}")
    by = top.groupby("season")["went_over"].agg(["size", "mean"])
    print("  by season: " + ", ".join(f"{int(s)}: {r['mean']:.1%} on {int(r['size'])}"
                                       for s, r in by.iterrows()))
    print("  A rule needs the highest quarter ahead of its price in all three seasons.")


if __name__ == "__main__":
    main()
