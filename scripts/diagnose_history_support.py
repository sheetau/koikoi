"""Fixed historical-action support diagnostic; no training or new rollouts."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from koikoi.agents import EndgameAgent, solved_koi_action
from koikoi.layout import HISTORY, KOI, PHASE, PLAYER, ROUND, STOP, TURN
from koikoi.reference import ReferenceEnv
from koikoi.search import sample_hidden_batch
from koikoi.train import provenance
from koikoi.upstream import UPSTREAM_PATH, UPSTREAM_REV, PublishedAgent, reconstruct

ROOTS = Path("runs/rollout_opponent_roots.json")
VALUES = Path("runs/rollout_opponent_diagnostic.json")
PREFLIGHT = Path("runs/history_support_preflight.json")
OUTPUT = Path("runs/history_support_diagnostic.json")
MANIFEST = Path("runs/history_support_process.json")
PACKAGE = "b5fb6911d7a0fee266f98eca186408fad9148e7d481e743e5fb71312ce6d307f"
PINS = {
    str(ROOTS): "f7d2023685927b884adc9a54e674493b2c6614cffa26c0190e1af63d36fa9dc3",
    str(VALUES): "91180a92268ca04a7622c09b6056891edd71fb4ea58b1c98f3946cef3830f9b0",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, data):
    Path(path).write_bytes((json.dumps(data, indent=2) + "\n").encode("utf-8"))


def reference(state):
    env = ReferenceEnv.__new__(ReferenceEnv)
    env.state = np.array(state, dtype=np.int32, copy=True)
    return env


def metadata():
    for path, expected in PINS.items():
        assert sha(path) == expected
    source = provenance()
    assert source["source_sha256"] == PACKAGE
    prior = read(VALUES)
    hashes = {
        name: sha(UPSTREAM_PATH / f"model_agent/{name}_rl_wp.pt")
        for name in ("discard", "pick", "koikoi")
    }
    assert hashes == prior["published_hashes"]
    assert prior["upstream_revision"] == UPSTREAM_REV
    return {
        "script_sha256": sha(__file__),
        "source_sha256": PACKAGE,
        "input_hashes": PINS,
        "published_hashes": hashes,
        "upstream_revision": UPSTREAM_REV,
    }


@torch.no_grad()
def replay(target, opponent, invariance=False):
    """Replay observations, never replace an observed action with a prediction."""
    target = np.asarray(target, dtype=np.int32)
    env = reference(target)
    env.deal(deck=target[48:96].copy())
    counts = {
        "replay_decisions": 0,
        "candidate_decisions": 0,
        "opponent_decisions": 0,
        "teacher_calls": 0,
        "guarded_decisions": 0,
        "feature_checks": 0,
        "feature_teacher_calls": 0,
    }
    events, seen = [], set()
    while (env.state[TURN], env.state[PHASE]) != (target[TURN], target[PHASE]):
        assert env.state[ROUND] == target[ROUND]
        assert counts["replay_decisions"] < 64
        phase, turn = int(env.state[PHASE]), int(env.state[TURN])
        encoded = int(target[HISTORY + turn * 5 + {0: 0, 1: 1, 2: 3, 3: 4}[phase]])
        assert encoded > 0
        observed = STOP + encoded - 1 if phase == KOI else encoded - 1
        assert env.legal()[observed]
        if env.state[PLAYER] != target[PLAYER]:
            guarded = solved_koi_action(env) is not None
            predicted = opponent.act(env)
            assert env.legal()[predicted]
            counts["opponent_decisions"] += 1
            counts["guarded_decisions" if guarded else "teacher_calls"] += 1
            events.append(
                {
                    "turn": turn,
                    "phase": phase,
                    "observed": observed,
                    "predicted": predicted,
                    "guarded": guarded,
                }
            )
            if invariance and phase not in seen:
                alternate = reference(
                    sample_hidden_batch(env, np.random.default_rng(93001 + phase), 1)[0]
                )
                np.testing.assert_array_equal(env.observe(), alternate.observe())
                np.testing.assert_array_equal(env.legal(), alternate.legal())
                torch.testing.assert_close(
                    reconstruct(env).feature_tensor,
                    reconstruct(alternate).feature_tensor,
                    rtol=0,
                    atol=0,
                )
                assert predicted == opponent.act(alternate)
                counts["feature_checks"] += 1
                counts["feature_teacher_calls"] += int(not guarded)
                seen.add(phase)
        else:
            counts["candidate_decisions"] += 1
        result = env.step(observed)
        assert not result[1], "Historical replay crossed the current round"
        counts["replay_decisions"] += 1
    np.testing.assert_array_equal(env.state, target)
    mismatches = [i for i, e in enumerate(events) if e["observed"] != e["predicted"]]
    return {
        "compatible": not mismatches,
        "first_mismatch": mismatches[0] if mismatches else None,
        "events": events,
        "counts": counts,
        "reconstructed_state_sha256": hashlib.sha256(env.state.tobytes()).hexdigest(),
    }


def conditional(values, compatible):
    count = int(sum(compatible))
    if not count:
        return None
    v = np.asarray(values, dtype=np.float64)[:, compatible]
    return {
        "arm_means": v.mean(1).tolist(),
        "self_gain": float((v[1] - v[0]).mean()),
        "published_gain": float((v[3] - v[2]).mean()),
        "gain_shift": float((v[3] - v[2] - v[1] + v[0]).mean()),
    }


def main(mode):
    torch.set_num_threads(2)
    meta = metadata()
    output = PREFLIGHT if mode == "preflight" else OUTPUT
    assert not output.exists(), "Preserve existing evidence"
    roots, prior = read(ROOTS)["roots"], read(VALUES)["results"]
    assert len(roots) == len(prior) == 40
    if mode == "run":
        preflight = read(PREFLIGHT)
        assert preflight["metadata"] == meta and preflight["status"] == "complete"
        assert len(preflight["results"]) == 40
        assert all(r["actual"]["compatible"] for r in preflight["results"])
    report = {
        "metadata": meta,
        "mode": mode,
        "status": "running",
        "results": [],
        "note": "Known deterministic opponent, within-round history only. Fixed selected correlated roots. No strength inference, new rollouts, training, or final seeds. Empty support is null; never resample.",
    }
    manifest = {"pid": os.getpid(), "mode": mode, "status": "running", "metadata": meta}
    if mode == "run":
        report["preflight_sha256"] = sha(PREFLIGHT)
        manifest["preflight_sha256"] = sha(PREFLIGHT)
    write(MANIFEST, manifest)
    opponent = EndgameAgent(PublishedAgent("wp"))
    for cached, previous in zip(roots, prior):
        started = time.perf_counter()
        ident = cached["identity"]
        assert ident == previous["identity"]
        state = np.array(cached["state"], dtype=np.int32)
        assert hashlib.sha256(state.tobytes()).hexdigest() == cached["state_sha256"]
        row = {"identity": ident}
        if mode == "preflight":
            row["actual"] = replay(state, opponent, invariance=True)
            assert row["actual"]["compatible"], ident
        else:
            model_index = ["seed1", "seed10001"].index(ident["model"])
            rng = np.random.default_rng(
                np.random.SeedSequence(
                    [
                        92009,
                        model_index,
                        ident["seed"],
                        ident["seat"],
                        ident["candidate_decision"],
                    ]
                )
            )
            worlds = sample_hidden_batch(reference(state), rng, 128)
            row["worlds_sha256"] = hashlib.sha256(worlds.tobytes()).hexdigest()
            assert row["worlds_sha256"] == previous["worlds_sha256"]
            row["worlds"] = [replay(world, opponent) for world in worlds]
            compatible = [w["compatible"] for w in row["worlds"]]
            row["compatible_count"] = sum(compatible)
            row["conditional"] = conditional(previous["values"], compatible)
            row["unconditional"] = conditional(previous["values"], [True] * 128)
            row["phases"] = {
                str(phase): {
                    "comparisons": sum(
                        e["phase"] == phase for w in row["worlds"] for e in w["events"]
                    ),
                    "mismatches": sum(
                        e["phase"] == phase and e["observed"] != e["predicted"]
                        for w in row["worlds"]
                        for e in w["events"]
                    ),
                }
                for phase in range(4)
            }
        row["seconds"] = time.perf_counter() - started
        report["results"].append(row)
        print(
            json.dumps(
                {
                    "mode": mode,
                    "roots": len(report["results"]),
                    "compatible": row.get("compatible_count"),
                    "seconds": row["seconds"],
                }
            ),
            flush=True,
        )
    assert metadata() == meta
    report["status"] = "complete"
    write(output, report)
    manifest.update(status="complete", output=str(output), output_sha256=sha(output))
    write(MANIFEST, manifest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["preflight", "run"])
    main(parser.parse_args().mode)
