"""The numbers under a pick. The work is in services/training/pick_depth.py.

It lives there because the site shows the same numbers and is built in the
training container, which only carries that folder. This is the door the post
and video builders come through, so they keep importing post_depth.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "services" / "training"))

from pick_depth import *  # noqa: E402,F401,F403
from pick_depth import (  # noqa: E402,F401
    ABBR, HOME, NICK, SAME, aftermath, beats, depth, load, weather,
)
