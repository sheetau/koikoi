"""Run the rendered game input checks with a repeatable public game seed.

This tests presentation and input routing, not playing strength.
"""

import sys
from functools import partial

from koikoi import game_server


def main():
    game_server.GameSession = partial(game_server.GameSession, seed=197)
    if "--ui-test" not in sys.argv:
        sys.argv.append("--ui-test")
    return game_server.main()


if __name__ == "__main__":
    raise SystemExit(main())
