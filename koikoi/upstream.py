"""Pinned original rules/models. Historical upstream feature bugs are preserved."""

import importlib
from functools import lru_cache
import pickle
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from .layout import *

UPSTREAM_URL = "https://github.com/guansanghai/KoiKoi-AI.git"
UPSTREAM_REV = "89d789565c7c36c84b74662e43001d960bf80901"
UPSTREAM_PATH = Path(__file__).resolve().parents[1] / "external/KoiKoi-AI"
PHASES = ("discard", "discard-pick", "draw-pick", "koikoi", "round-over")


class LegacyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch.nn.modules.linear" and name == "_LinearWithBias":
            return torch.nn.Linear
        return super().find_class(module, name)


class LegacyPickle:
    __name__ = "pickle"
    Unpickler = LegacyUnpickler
    load = staticmethod(pickle.load)


@lru_cache(maxsize=1)
def load_source():
    if not (UPSTREAM_PATH / "koikoigame.py").exists():
        raise FileNotFoundError("Run python -m koikoi.upstream --fetch first")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=UPSTREAM_PATH, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=UPSTREAM_PATH, text=True).strip()
    if revision != UPSTREAM_REV or dirty:
        raise RuntimeError("Upstream must be clean and pinned to " + UPSTREAM_REV)
    if str(UPSTREAM_PATH) not in sys.path:
        sys.path.insert(0, str(UPSTREAM_PATH))
    return importlib.import_module("koikoigame")


def inject_deal(round_state, deck):
    def cards(ids):
        return [[int(c)//4+1, int(c)%4+1] for c in ids]
    r = round_state
    r.hand = {1: sorted(cards(deck[:8])), 2: sorted(cards(deck[8:16]))}
    r.field_slot = sorted(cards(deck[16:24])) + [[0, 0] for _ in range(10)]
    r.stock = cards(deck[24:])
    r.log["basic"].update(initHand1=r.hand[1].copy(), initHand2=r.hand[2].copy(),
                          initBoard=r.field.copy(), initPile=r.stock.copy(), Dealer=r.dealer)


def skip_forced(r):
    while not r.wait_action and not r.round_over:
        r.step(None)


def original_action(phase, action):
    return action == CONTINUE if phase == KOI else [int(action)//4+1, int(action)%4+1]


def reconstruct(reference):
    """Replay public within-round actions to reconstruct the original 300x48 input."""
    source = load_source()
    s = reference.state
    if s[ROUNDS] != 8 or not s[BANKRUPT]:
        raise ValueError("Published models are only valid for the paper8 profile")
    g = source.KoiKoiGameState(round_num=int(s[ROUND])+1, round_total=8,
                              init_point=s[POINTS:POINTS+2].tolist(), init_dealer=int(s[DEALER])+1)
    r = g.round_state
    inject_deal(r, s[48:96])
    for _ in range(100):
        skip_forced(r)
        if r.turn_16 == s[TURN]+1 and r.state == PHASES[s[PHASE]]:
            return g
        if r.round_over:
            break
        h = HISTORY + (r.turn_16-1)*5
        phase = PHASES.index(r.state)
        offset = {DISCARD: 0, PICK_HAND: 1, PICK_DRAW: 3, KOI: 4}[phase]
        encoded = int(s[h+offset])
        if encoded <= 0:
            raise ValueError("Incomplete public history")
        action = STOP+encoded-1 if phase == KOI else encoded-1
        r.step(original_action(phase, action))
    raise ValueError("Could not reconstruct upstream decision")


class PublishedAgent:
    def __init__(self, variant="wp"):
        if variant not in ("wp", "point"):
            raise ValueError("Expected wp or point")
        load_source()
        net = importlib.import_module("koikoinet2L")
        self.models = {}
        for phase, name, cls in ((DISCARD, "discard", net.DiscardModel),
                                 (PICK_HAND, "pick", net.PickModel), (KOI, "koikoi", net.KoiKoiModel)):
            path = UPSTREAM_PATH / f"model_agent/{name}_rl_{variant}.pt"
            # Pinned, MIT-licensed upstream pickle; never accept arbitrary model paths here.
            old = torch.load(path, map_location="cpu", weights_only=False, pickle_module=LegacyPickle)
            model = cls()
            model.load_state_dict(old.state_dict())
            self.models[phase] = model.eval()
        self.models[PICK_DRAW] = self.models[PICK_HAND]

    @torch.no_grad()
    def act(self, reference):
        g = reconstruct(reference)
        phase = int(reference.state[PHASE])
        logits = self.models[phase](g.feature_tensor.unsqueeze(0)).squeeze(0)
        mask = torch.from_numpy(g.round_state.action_mask).bool()
        action = int(logits.masked_fill(~mask, -torch.inf).argmax())
        return action + 48 if phase == KOI else action


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch", action="store_true")
    args = parser.parse_args()
    if args.fetch and not UPSTREAM_PATH.exists():
        subprocess.run(["git", "clone", UPSTREAM_URL, str(UPSTREAM_PATH)], check=True)
        subprocess.run(["git", "checkout", UPSTREAM_REV], cwd=UPSTREAM_PATH, check=True)
    load_source()
    print(f"Verified {UPSTREAM_URL} @ {UPSTREAM_REV}")
