"""Copy what a build made into a folder that syncs to my phone.

Every post and video used to reach the phone by email. This copies the
finished files (slides, the X card, videos, captions) into one folder, and
whatever keeps that folder in step with the cloud does the rest. Google Drive
for desktop, OneDrive, it does not matter: this script only ever writes files
to a folder on this machine. It holds no account, no key and no login, and
cannot read or delete anything in the drive.

Where the folder is, first one found:

    the PRIORLINE_OUTBOX environment variable
    one line in marketing/.cache/outbox.txt
    a folder called "PriorLine Outbox" in Google Drive for desktop's My Drive

If none of those exists nothing is copied and the build says so once.

    python scripts/post_outbox.py marketing/social/posts/2026-11-26-thanksgiving
"""

from __future__ import annotations

import os
import shutil
import string
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KEPT = ROOT / "marketing" / ".cache" / "outbox.txt"
NAME = "PriorLine Outbox"
SEND = (".png", ".mp4", ".txt")
SKIP = ("posted.json",)


def where() -> Path | None:
    env = os.environ.get("PRIORLINE_OUTBOX")
    if env:
        return Path(env)
    if KEPT.exists():
        line = KEPT.read_text(encoding="utf-8-sig").strip()
        if line:
            return Path(line)
    for letter in string.ascii_uppercase[3:]:
        drive = Path(f"{letter}:/My Drive")
        if drive.is_dir():
            return drive / NAME
    return None


def send(folder) -> int:
    """Copy a post folder's finished files into the outbox, under its own name."""
    folder = Path(folder)
    box = where()
    if box is None:
        print("  no outbox folder set, so nothing was copied to the phone "
              "(see scripts/post_outbox.py)")
        return 0
    try:
        out = box / folder.name
        out.mkdir(parents=True, exist_ok=True)
        n = 0
        for f in sorted(folder.iterdir()):
            # The words to read over a video are for building it, not posting it.
            script = f.suffix == ".txt" and f.stem.startswith("video-")
            if f.is_file() and f.suffix in SEND and f.name not in SKIP and not script:
                target = out / f.name
                if (not target.exists() or target.stat().st_size != f.stat().st_size
                        or target.stat().st_mtime < f.stat().st_mtime):
                    shutil.copy2(f, target)
                    n += 1
        print(f"  {n} file(s) copied to {out}")
        return n
    except OSError as err:
        print(f"  could not copy to the outbox at {box}: {err}")
        return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: post_outbox.py <post folder> [more folders]")
    for arg in sys.argv[1:]:
        send(arg)
