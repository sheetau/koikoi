"""Scoped development checks, with explicit content-addressed result reuse.

No training or strength evaluation. Cached passes are reported as reused, never
as new measurements. The full learned-opponent match always executes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GODOT = ROOT / "external/godot/Godot_v4.7.2-stable_win64_console.exe"
OUTPUT = ROOT / ".dashboard/checks"
SUITES = {
    "management": ["dashboard", "legacy"],
    "quick": ["motion_math", "audio_return", "backlog", "migration", "dealer_return", "pace"],
    "motion": ["motion_math", "role_motion", "capture_motion", "impact"],
    "visual": ["concept", "addendum", "layout", "grouped_roles", "review", "flow", "polish", "sky", "choice"],
    "full": ["match"],
}


def digest(paths):
    result = hashlib.sha256()
    for path in sorted(set(paths)):
        result.update(str(path.relative_to(ROOT)).encode())
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                result.update(block)
    return result.hexdigest()


def inputs(name):
    common = [Path(__file__), ROOT / "pyproject.toml"]
    if name == "dashboard":
        return common + [ROOT / ".dashboard" / p for p in ("app.py", "test_app.py")]
    if name == "legacy":
        return common + [
            ROOT / p for p in ("scripts/project_work.py", "tests/test_project_work.py")
        ]
    paths = (
        common
        + list((ROOT / "game").glob("*.gd"))
        + list((ROOT / "game").glob("*.gdshader"))
    )
    paths += list((ROOT / "game").glob("*.tscn")) + [ROOT / "game/project.godot", GODOT]
    paths += list((ROOT / "game/tests").glob("*.gd")) + list(
        (ROOT / "game/tests").glob("*.json")
    )
    paths += [p for p in (ROOT / "game/assets").rglob("*") if p.is_file()]
    return paths


def command(name):
    python = [sys.executable, "-X", "utf8"]
    if name == "dashboard":
        return python + [
            "-m",
            "unittest",
            "discover",
            "-s",
            ".dashboard",
            "-p",
            "test_app.py",
            "-q",
        ]
    if name == "legacy":
        return python + ["-m", "pytest", "tests/test_project_work.py", "-q"]
    if name == "match":
        return python + [
            "-m",
            "scripts.game_ui_check",
            "--headless",
            "--godot",
            str(GODOT),
        ]
    return [
        str(GODOT),
        "--path",
        "game",
        *([] if name in SUITES["visual"] else ["--headless"]),
        "--script",
        f"res://tests/{name}_check.gd",
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        action="append",
        choices=SUITES,
        help="Repeat to combine suites; default quick",
    )
    parser.add_argument(
        "--reuse",
        action="store_true",
        help="Explicitly reuse identical passing inputs, except full match",
    )
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    names = list(
        dict.fromkeys(n for suite in (args.suite or ["quick"]) for n in SUITES[suite])
    )
    if args.list:
        print(
            json.dumps(
                {name: command(name) for name in names}, ensure_ascii=False, indent=2
            )
        )
        return 0
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cache_path = OUTPUT / "passes.json"
    try:
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        cache = {}
    outcomes = []
    fingerprints = {}
    for name in names:
        cmd = command(name)
        category = name if name in {"dashboard", "legacy"} else "game"
        if category not in fingerprints:
            fingerprints[category] = digest(inputs(name))
        signature = hashlib.sha256(
            (fingerprints[category] + json.dumps(cmd) + sys.version).encode()
        ).hexdigest()
        previous = cache.get(name, {})
        if args.reuse and name != "match" and previous.get("signature") == signature:
            print(
                f"REUSED {name}: identical inputs; not rerun (previous {previous['seconds']:.2f}s)",
                flush=True,
            )
            outcomes.append(
                {"check": name, "reused": True, "previous_seconds": previous["seconds"]}
            )
            continue
        started = time.perf_counter()
        log = OUTPUT / f"{name}.log"
        env = dict(os.environ)
        env.pop(
            "KOIKOI_QA_FAST", None
        )  # Real display-time checks keep their timing contract.
        with log.open("w", encoding="utf-8") as stream:
            process = subprocess.Popen(
                cmd,
                cwd=ROOT,
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            try:
                code = process.wait(timeout=360)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait()
                code = 124
        seconds = time.perf_counter() - started
        output = log.read_text(encoding="utf-8", errors="replace")
        failed = code != 0 or "SCRIPT ERROR:" in output or "ERROR:" in output
        print(
            f"{'FAIL' if failed else 'PASS'} {name}: {seconds:.2f}s; {log.relative_to(ROOT)}",
            flush=True,
        )
        outcomes.append(
            {
                "check": name,
                "reused": False,
                "seconds": round(seconds, 3),
                "exit_code": code,
                "passed": not failed,
            }
        )
        if failed:
            cache.pop(name, None)
            print(output[-4000:])
        else:
            cache[name] = {"signature": signature, "seconds": seconds}
        temporary = cache_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(cache, indent=2) + "\n", encoding="utf-8")
        temporary.replace(cache_path)
    (OUTPUT / "latest.json").write_text(
        json.dumps(outcomes, indent=2) + "\n", encoding="utf-8"
    )
    return int(any(not o.get("passed", True) for o in outcomes))


if __name__ == "__main__":
    raise SystemExit(main())
