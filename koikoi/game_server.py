"""Local authenticated game bridge and Godot launcher; no remote hosting."""

from __future__ import annotations

import argparse
import json
import logging
import os
import secrets
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .game import GameSession

ROOT = Path(__file__).resolve().parents[1]
GODOT = ROOT / "external/godot/Godot_v4.7.2-stable_win64.exe"
logger = logging.getLogger(__name__)


class GameService:
    def __init__(self, loader=None):
        if loader is None:
            from .game_opponent import GameOpponent

            loader = GameOpponent
        self.loader, self.session = loader, None
        self.revision = 0
        self.agents = {}
        self.lock = threading.Lock()

    def request(self, path, payload):
        with self.lock:
            if path == "/state":
                result = (
                    self.session.response()
                    if self.session
                    else {"view": {}, "events": []}
                )
                if self.session:
                    result["events"] = [{"kind": "sync", "view": result["view"]}]
                return {**result, "revision": self.revision}
            if (
                type(payload.get("revision")) is not int
                or payload["revision"] != self.revision
            ):
                raise ValueError("Stale request; reload current game")
            if path == "/start":
                name = payload.get("opponent", "trained")
                flags = payload.get("flags", 0)
                if name not in ("trained", "rl-wp", "rl-point", "greedy"):
                    raise ValueError("Unknown opponent")
                if type(flags) is not int or flags not in (0, 8, 16, 24):
                    raise ValueError("Unknown rules")
                if name not in self.agents:
                    self.agents[name] = self.loader(name)
                self.session = GameSession(self.agents[name], flags=flags)
            elif self.session is None:
                raise ValueError("Start a game first")
            elif path == "/action":
                action = payload.get("action")
                if type(action) is not int:
                    raise ValueError("An integer action is required")
                self.session.act(action)
            elif path == "/ai":
                self.session.act()
            elif path == "/next":
                self.session.next_round()
            else:
                raise ValueError("Unknown request")
            self.revision += 1
            return {**self.session.response(), "revision": self.revision}


def make_server(service, token):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            if self.headers.get("Authorization") != f"Bearer {token}":
                self.send_error(403)
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 4096:
                    raise ValueError("Invalid request size")
                body = json.loads(self.rfile.read(size))
                if not isinstance(body, dict):
                    raise TypeError("Expected object")
                result = service.request(self.path, body)
                status = 200
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                status, result = 400, {"error": str(exc), "revision": service.revision}
            except Exception as exc:
                logger.exception("Game request failed")
                status, result = 500, {"error": str(exc), "revision": service.revision}
            data = json.dumps(result, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    return server


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--godot", type=Path, default=GODOT)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--autoplay", action="store_true")
    parser.add_argument("--qa", action="store_true")
    parser.add_argument("--ui-test", action="store_true")
    args = parser.parse_args()
    if not args.godot.is_file():
        raise SystemExit("Godot is missing. Run scripts/setup_game.ps1 first.")
    token = secrets.token_urlsafe(32)
    server = make_server(GameService(), token)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    env = dict(
        os.environ,
        KOIKOI_URL=f"http://127.0.0.1:{server.server_port}",
        KOIKOI_TOKEN=token,
    )
    command = [str(args.godot), "--path", str(ROOT / "game")]
    if args.ui_test:
        command += ["--script", "res://tests/ui_smoke.gd"]
    if args.headless:
        command += ["--headless"]
    command += ["--"]
    if args.autoplay:
        command += ["--autoplay"]
    if args.qa:
        command += ["--qa"]
    if args.ui_test:
        command += ["--ui-test"]
    try:
        if sys.stdout is None:
            log = ROOT / ".koikoi-work/game.log"
            log.parent.mkdir(exist_ok=True)
            with log.open("w", encoding="utf-8") as output:
                return subprocess.call(
                    command, env=env, cwd=ROOT, stdout=output, stderr=output
                )
        return subprocess.call(command, env=env, cwd=ROOT)
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    raise SystemExit(main())
