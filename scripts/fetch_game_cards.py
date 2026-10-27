"""Fetch Commons traditional-color art from its official thumbnail host.

960px-wide source images are retained byte-for-byte. The client masks only
the outer frame. Downloads are spaced, cached, and stop on HTTP errors.
"""

import hashlib
import json
import shutil
import struct
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "game/assets/cards"
CACHE = ROOT / "external/traditional-hanafuda"
USER_AGENT = "KoiKoiGame/1.0 (+https://github.com/sheetau/koikoi)"


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    index = json.loads((DEST / "sources.json").read_text(encoding="utf-8"))
    manifest = []
    for item in index:
        tail = item["download"].split("/commons/", 1)[1]
        url = (
            "https://thumb.wikimedia.org/wikipedia/commons/thumb/"
            + tail
            + "/960px-"
            + tail.rsplit("/", 1)[1]
            + ".png"
        )
        original = CACHE / (Path(item["file"]).stem + ".thumb960.png")
        if not original.is_file():
            time.sleep(2)
            subprocess.run(
                [
                    "curl.exe",
                    "-A",
                    USER_AGENT,
                    "-4",
                    "-f",
                    "-s",
                    "-L",
                    "--max-time",
                    "30",
                    url,
                    "-o",
                    str(original),
                ],
                check=True,
            )
        content = original.read_bytes()
        if content[:8] != b"\x89PNG\r\n\x1a\n":
            raise ValueError(f"Not PNG: {original}")
        width, height = struct.unpack(">II", content[16:24])
        if width != 960 or abs(width / height - 976 / 1600) > 0.002:
            raise ValueError(f"Unexpected bounds: {original} {width}x{height}")
        shutil.copyfile(original, DEST / item["file"])
        manifest.append(
            dict(
                item,
                thumbnail=url,
                width=width,
                height=height,
                sha256=hashlib.sha256(content).hexdigest(),
            )
        )
        print(item["file"], flush=True)
    if len(manifest) != 48:
        raise ValueError(f"Expected 48 cards, got {len(manifest)}")
    (DEST / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print("48 traditional-color images verified and bundled")


if __name__ == "__main__":
    main()
