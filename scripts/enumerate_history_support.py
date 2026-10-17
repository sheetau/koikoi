"""Exact known-opponent support at the nine fixed, small-hand-space roots."""

import argparse
import hashlib
import math
import os
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

from koikoi.agents import EndgameAgent, solved_koi_action
from koikoi.layout import DEALER, HISTORY, PLAYER, SEED, STOCK
from koikoi.upstream import PublishedAgent, reconstruct
from scripts.audit_history_support import worlds_for
from scripts.diagnose_history_support import (
    ROOTS,
    read,
    reference,
    replay,
    sha,
    write,
)
from scripts.diagnose_history_support import (
    metadata as history_metadata,
)

OUTPUT = Path("runs/history_support_enumeration.json")
PREFLIGHT = Path("runs/history_support_enumeration_preflight.json")
PROCESS = Path("runs/history_support_enumeration_process.json")
SUPPORT = Path("runs/history_support_diagnostic.json")
TRUE = Path("runs/history_support_preflight.json")


def context(state):
    env = reference(state)
    obs = env.observe()
    unknown = np.flatnonzero(obs[:528].reshape(48, 11)[:, 4]).tolist()
    count = round(float(obs[541]) * 8)
    p, dealer = int(state[PLAYER]), int(state[DEALER])
    played = [
        int(state[HISTORY + t * 5]) - 1
        for t in range(16)
        if (dealer + t) % 2 != p and state[HISTORY + t * 5] > 0
    ]
    assert count + len(played) == 8
    return unknown, count, played


def canonical_world(state, hand):
    """Construct simulator state; canonical stock must never be used for values."""
    unknown, count, played = context(state)
    hand = tuple(hand)
    assert len(hand) == len(set(hand)) == count and set(hand) <= set(unknown)
    initial = sorted(played + list(hand))
    if np.bincount(np.array(initial) // 4, minlength=12).max() == 4:
        return None
    s = np.array(state, dtype=np.int32, copy=True)
    p = int(s[PLAYER])
    s[unknown] = -1
    s[list(hand)] = 1 - p
    s[48 + 8 * (1 - p) : 56 + 8 * (1 - p)] = initial
    s[72 : 49 + int(s[STOCK])] = sorted(set(unknown) - set(hand))
    s[SEED] = 0
    np.testing.assert_array_equal(reference(s).observe(), reference(state).observe())
    return s


class InvariantOpponent:
    def __init__(self, teacher):
        self.agent = EndgameAgent(teacher)
        self.checks = 0
        self.extra_teacher_calls = 0

    @torch.no_grad()
    def act(self, env):
        action = self.agent.act(env)
        alternate = reference(env.state)
        stock_end = 49 + int(alternate.state[STOCK])
        alternate.state[72:stock_end] = alternate.state[72:stock_end][::-1]
        alternate.state[SEED] = int(alternate.state[SEED]) ^ 0x3FFFFFFF
        np.testing.assert_array_equal(env.observe(), alternate.observe())
        np.testing.assert_array_equal(env.legal(), alternate.legal())
        torch.testing.assert_close(
            reconstruct(env).feature_tensor,
            reconstruct(alternate).feature_tensor,
            rtol=0,
            atol=0,
        )
        assert action == self.agent.act(alternate)
        self.checks += 1
        self.extra_teacher_calls += int(solved_koi_action(env) is None)
        return action


def metadata():
    assert (
        sha(SUPPORT)
        == "ec8d8656e855f413779f0337bd3e207f745ce324d983598d474e53f06f5c8259"
    )
    assert (
        sha(TRUE) == "799827876fb40f537685b945ad521c611116c2e31c3e0acbf072dc2c3c4a8a95"
    )
    return {
        "script_sha256": sha(__file__),
        "history_metadata": history_metadata(),
        "world_helper_sha256": sha("scripts/audit_history_support.py"),
        "support_sha256": sha(SUPPORT),
        "true_history_sha256": sha(TRUE),
    }


def main(mode):
    torch.set_num_threads(2)
    meta = metadata()
    output = PREFLIGHT if mode == "preflight" else OUTPUT
    assert not output.exists()
    roots = read(ROOTS)["roots"]
    selected = [
        (index, root)
        for index, root in enumerate(roots)
        if math.comb(len(context(root["state"])[0]), context(root["state"])[1]) <= 1000
    ]
    assert len(selected) == 9
    assert (
        sum(
            math.comb(len(context(r["state"])[0]), context(r["state"])[1])
            for _, r in selected
        )
        == 4792
    )
    if mode == "run":
        preflight = read(PREFLIGHT)
        assert preflight["metadata"] == meta and preflight["status"] == "complete"
        assert len(preflight["results"]) == 9
    teacher = InvariantOpponent(PublishedAgent("wp"))
    report = {
        "metadata": meta,
        "mode": mode,
        "status": "running",
        "results": [],
        "note": "Exact support only for the pinned deterministic opponent and within-round history. Canonical stock is for past-action support, never future value. No rollouts, training, final seeds or strength inference.",
    }
    manifest = {"pid": os.getpid(), "mode": mode, "status": "running", "metadata": meta}
    if mode == "run":
        report["preflight_sha256"] = manifest["preflight_sha256"] = sha(PREFLIGHT)
    write(PROCESS, manifest)
    for index, root in selected:
        started = time.perf_counter()
        state = np.array(root["state"], dtype=np.int32)
        assert hashlib.sha256(state.tobytes()).hexdigest() == root["state_sha256"]
        unknown, count, played = context(state)
        actual = tuple(np.flatnonzero(state[:48] == 1 - state[PLAYER]).tolist())
        row = {
            "root_index": index,
            "identity": root["identity"],
            "unknown": unknown,
            "opponent_count": count,
            "played": played,
            "candidate_count": math.comb(len(unknown), count),
            "hands": [],
        }
        candidates = [actual] if mode == "preflight" else combinations(unknown, count)
        for hand in candidates:
            world = canonical_world(state, hand)
            if world is None:
                row["hands"].append({"hand": list(hand), "valid": False})
                continue
            before_checks, before_calls = teacher.checks, teacher.extra_teacher_calls
            evidence = replay(world, teacher)
            entry = {
                "hand": list(hand),
                "valid": True,
                "replay": evidence,
                "stock_rng_feature_checks": teacher.checks - before_checks,
                "extra_teacher_calls": teacher.extra_teacher_calls - before_calls,
            }
            row["hands"].append(entry)
            if hand == actual:
                assert evidence["compatible"]
                assert (
                    evidence["events"]
                    == read(TRUE)["results"][index]["actual"]["events"]
                )
        lookup = {tuple(h["hand"]): h for h in row["hands"]}
        assert lookup[actual]["replay"]["compatible"]
        row["valid_count"] = sum(h["valid"] for h in row["hands"])
        row["compatible_count"] = sum(
            h["valid"] and h["replay"]["compatible"] for h in row["hands"]
        )
        if mode == "run":
            assert len(lookup) == row["candidate_count"]
            ident = root["identity"]
            worlds = worlds_for(
                state, ["seed1", "seed10001"].index(ident["model"]), ident
            )
            previous = read(SUPPORT)["results"][index]
            assert (
                hashlib.sha256(worlds.tobytes()).hexdigest()
                == previous["worlds_sha256"]
            )
            for world, prior in zip(worlds, previous["worlds"]):
                hand = tuple(np.flatnonzero(world[:48] == 1 - state[PLAYER]).tolist())
                assert lookup[hand]["valid"]
                assert lookup[hand]["replay"]["events"] == prior["events"]
                assert lookup[hand]["replay"]["compatible"] == prior["compatible"]
            row["existing_worlds_checked"] = len(worlds)
        row["seconds"] = time.perf_counter() - started
        report["results"].append(row)
        print(
            {
                "mode": mode,
                "roots": len(report["results"]),
                "valid": row["valid_count"],
                "compatible": row["compatible_count"],
                "seconds": row["seconds"],
            },
            flush=True,
        )
    assert metadata() == meta
    report["status"] = "complete"
    write(output, report)
    manifest.update(status="complete", output=str(output), output_sha256=sha(output))
    write(PROCESS, manifest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["preflight", "run"])
    main(parser.parse_args().mode)
