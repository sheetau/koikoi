"""Audit fixed support using independent world reconstruction and native replay.

Does not import the diagnostic or its replay/statistics helpers. Reuses the pinned
PublishedAgent inference adapter; it is not an independent teacher implementation.
"""

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from koikoi.env import BatchEnv
from koikoi.layout import (
    BEFORE,
    DEALER,
    HISTORY,
    KOIS,
    PENDING,
    PHASE,
    PLAYER,
    SEED,
    STOCK,
    TURN,
)
from koikoi.reference import ReferenceEnv
from koikoi.train import provenance
from koikoi.upstream import UPSTREAM_REV, PublishedAgent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def worlds_for(state, model_index, identity):
    # Same independent algorithm as the previously audited world reconstruction.
    env = ReferenceEnv.__new__(ReferenceEnv)
    env.state = state.copy()
    observation = env.observe()
    unknown = np.flatnonzero(observation[:528].reshape(48, 11)[:, 4])
    count = round(observation[541] * 8)
    player, dealer = state[PLAYER], state[DEALER]
    played = [
        int(state[HISTORY + t * 5]) - 1
        for t in range(16)
        if (dealer + t) % 2 != player and state[HISTORY + t * 5] > 0
    ]
    rng = np.random.default_rng(
        np.random.SeedSequence(
            [
                92009,
                model_index,
                identity["seed"],
                identity["seat"],
                identity["candidate_decision"],
            ]
        )
    )
    worlds = []
    for _ in range(128):
        s = state.copy()
        while True:
            hand = rng.choice(unknown, count, replace=False)
            initial = played + hand.tolist()
            if max(np.bincount(np.array(initial, dtype=int) // 4, minlength=12)) < 4:
                break
        s[unknown] = -1
        s[hand] = 1 - player
        stock = np.array(
            [c for c in unknown if c not in set(hand.tolist())], dtype=np.int32
        )
        rng.shuffle(stock)
        s[72 : 49 + s[STOCK]] = stock
        start = 48 + 8 * (1 - player)
        s[start : start + 8] = sorted(initial)
        s[SEED] = rng.integers(0, 2**31 - 1)
        worlds.append(s)
    return np.stack(worlds)


@torch.no_grad()
def audit_native_history(worlds, records, teacher):
    """Check every recorded prediction, including those after first mismatch."""
    n = len(worlds)
    env = BatchEnv(n, "cpu", "paper8", debug=True)
    initial = worlds.copy()
    initial[:, :48] = -1
    for s in initial:
        s[s[48:56]] = 0
        s[s[56:64]] = 1
        s[s[64:72]] = 2
    initial[:, PLAYER] = initial[:, DEALER]
    initial[:, TURN] = initial[:, PHASE] = initial[:, BEFORE] = 0
    initial[:, PENDING] = -1
    initial[:, STOCK] = 47
    initial[:, KOIS : KOIS + 2] = 0
    initial[:, HISTORY:] = 0
    env.state.copy_(torch.from_numpy(initial))
    steps, event_index = 0, 0
    calls, guards = np.zeros(n, dtype=int), np.zeros(n, dtype=int)
    seat = int(worlds[0, PLAYER])
    while tuple(env.state[0, [TURN, PHASE]].tolist()) != tuple(
        worlds[0, [TURN, PHASE]]
    ):
        assert steps < 64
        states = env.state.numpy()
        phase, turn, player = (int(states[0, i]) for i in (PHASE, TURN, PLAYER))
        assert np.all(states[:, [PHASE, TURN, PLAYER]] == [phase, turn, player])
        offset = (0, 1, 3, 4)[phase]
        codes = worlds[:, HISTORY + 5 * turn + offset]
        assert np.all(codes > 0)
        actions = codes - 1 + (48 if phase == 3 else 0)
        _, legal = env.observe()
        assert legal[np.arange(n), actions].all()
        if player != seat:
            solved = env.solved_actions().numpy()
            for i, record in enumerate(records):
                event = record["events"][event_index]
                assert (event["phase"], event["turn"], event["observed"]) == (
                    phase,
                    turn,
                    int(actions[i]),
                )
                assert event["guarded"] == bool(solved[i] >= 0)
                if solved[i] >= 0:
                    prediction = int(solved[i])
                    guards[i] += 1
                else:
                    ref = ReferenceEnv.__new__(ReferenceEnv)
                    ref.state = states[i].copy()
                    prediction = teacher.act(ref)
                    calls[i] += 1
                assert event["predicted"] == prediction
                assert legal[i, prediction]
            event_index += 1
        result = env.step(torch.from_numpy(actions.copy()))
        assert not result[:, 1].any()
        steps += 1
    np.testing.assert_array_equal(env.state.numpy(), worlds)
    for i, record in enumerate(records):
        assert len(record["events"]) == event_index
        c = record["counts"]
        assert c["replay_decisions"] == steps
        assert c["candidate_decisions"] == steps - event_index
        assert c["opponent_decisions"] == event_index
        assert c["teacher_calls"] == calls[i]
        assert c["guarded_decisions"] == guards[i]
        assert (
            record["reconstructed_state_sha256"]
            == hashlib.sha256(worlds[i].tobytes()).hexdigest()
        )
        mismatches = [
            j for j, e in enumerate(record["events"]) if e["predicted"] != e["observed"]
        ]
        assert record["compatible"] == (len(mismatches) == 0)
        assert record["first_mismatch"] == (mismatches[0] if mismatches else None)
    return {
        "native_transitions": n * steps,
        "teacher_calls": int(calls.sum()),
        "guarded_decisions": int(guards.sum()),
    }


def audit_means(recorded, values, mask):
    if not mask.any():
        assert recorded is None
        return
    arm = values[:, mask].mean(axis=1)
    np.testing.assert_array_equal(recorded["arm_means"], arm)
    # All existing arrays are exact integer terminal utilities.
    assert recorded["self_gain"] == (values[1, mask] - values[0, mask]).mean()
    assert recorded["published_gain"] == (values[3, mask] - values[2, mask]).mean()
    assert (
        recorded["gain_shift"]
        == (
            values[3, mask] - values[2, mask] - values[1, mask] + values[0, mask]
        ).mean()
    )


def main():
    torch.set_num_threads(2)
    path = Path("runs/history_support_diagnostic.json")
    report, preflight = read(path), read("runs/history_support_preflight.json")
    manifest = read("runs/history_support_process.json")
    assert report["status"] == preflight["status"] == manifest["status"] == "complete"
    assert (
        report["mode"] == manifest["mode"] == "run" and preflight["mode"] == "preflight"
    )
    assert report["metadata"] == preflight["metadata"] == manifest["metadata"]
    assert sha(path) == manifest["output_sha256"]
    assert (
        report["preflight_sha256"]
        == manifest["preflight_sha256"]
        == sha("runs/history_support_preflight.json")
    )
    meta = report["metadata"]
    assert meta["script_sha256"] == sha("scripts/diagnose_history_support.py")
    assert (
        meta["source_sha256"]
        == provenance()["source_sha256"]
        == "b5fb6911d7a0fee266f98eca186408fad9148e7d481e743e5fb71312ce6d307f"
    )
    assert meta["upstream_revision"] == UPSTREAM_REV
    for p, h in meta["input_hashes"].items():
        assert sha(p) == h
    for name, h in meta["published_hashes"].items():
        assert sha(f"external/KoiKoi-AI/model_agent/{name}_rl_wp.pt") == h
    roots = read("runs/rollout_opponent_roots.json")["roots"]
    prior = read("runs/rollout_opponent_diagnostic.json")["results"]
    assert (
        len(roots)
        == len(prior)
        == len(report["results"])
        == len(preflight["results"])
        == 40
    )
    teacher = PublishedAgent("wp")
    budgets = {key: Counter() for key in ("preflight", "diagnostic", "audit")}
    summary = {}
    for root, previous, row, true in zip(
        roots, prior, report["results"], preflight["results"]
    ):
        ident = root["identity"]
        assert ident == previous["identity"] == row["identity"] == true["identity"]
        state = np.array(root["state"], dtype=np.int32)
        assert hashlib.sha256(state.tobytes()).hexdigest() == root["state_sha256"]
        assert true["actual"]["compatible"]
        budgets["audit"].update(
            audit_native_history(state[None], [true["actual"]], teacher)
        )
        budgets["preflight"].update(true["actual"]["counts"])
        phases_seen = {e["phase"] for e in true["actual"]["events"]}
        assert true["actual"]["counts"]["feature_checks"] == len(phases_seen)
        model_index = ["seed1", "seed10001"].index(ident["model"])
        worlds = worlds_for(state, model_index, ident)
        assert (
            hashlib.sha256(worlds.tobytes()).hexdigest()
            == row["worlds_sha256"]
            == previous["worlds_sha256"]
        )
        assert len(row["worlds"]) == 128
        budgets["audit"].update(audit_native_history(worlds, row["worlds"], teacher))
        for w in row["worlds"]:
            assert (
                w["counts"]["feature_checks"]
                == w["counts"]["feature_teacher_calls"]
                == 0
            )
            budgets["diagnostic"].update(w["counts"])
        mask = np.array([w["compatible"] for w in row["worlds"]], dtype=bool)
        assert row["compatible_count"] == int(mask.sum())
        for phase in range(4):
            events = [
                e for w in row["worlds"] for e in w["events"] if e["phase"] == phase
            ]
            assert row["phases"][str(phase)] == {
                "comparisons": len(events),
                "mismatches": sum(e["observed"] != e["predicted"] for e in events),
            }
        values = np.array(previous["values"], dtype=np.float64)
        assert values.shape == (4, 128) and np.isin(values, [-1, 0, 1]).all()
        audit_means(row["conditional"], values, mask)
        audit_means(row["unconditional"], values, np.ones(128, dtype=bool))
        summary.setdefault(ident["model"], []).append(
            {
                "count": int(mask.sum()),
                "opponent_decisions": len(true["actual"]["events"]),
                "conditional": row["conditional"],
                "unconditional": row["unconditional"],
            }
        )
        print(
            json.dumps(
                {
                    "model": ident["model"],
                    "audited_roots": sum(len(v) for v in summary.values()),
                }
            ),
            flush=True,
        )
    aggregates = {}
    for label, rows in summary.items():
        counts = [r["count"] for r in rows]
        nonempty = [r for r in rows if r["count"]]
        informed = [r for r in nonempty if r["opponent_decisions"]]
        aggregates[label] = {
            "roots": len(rows),
            "compatible_counts": counts,
            "zero_support_roots": counts.count(0),
            "one_to_four_support_roots": sum(1 <= n <= 4 for n in counts),
            "no_opponent_history_roots": sum(
                r["opponent_decisions"] == 0 for r in rows
            ),
            "supported_with_opponent_history_roots": len(informed),
            "total_compatible_worlds": sum(counts),
        }
        for name, group in [
            ("nonempty", nonempty),
            ("nonempty_with_history", informed),
        ]:
            aggregates[label][name] = {
                mode: {
                    key: float(np.mean([r[mode][key] for r in group]))
                    if group
                    else None
                    for key in ("self_gain", "published_gain", "gain_shift")
                }
                for mode in ("conditional", "unconditional")
            }
    audit = {
        "passed": True,
        "report_sha256": sha(path),
        "auditor_sha256": sha(__file__),
        "preflight_sha256": sha("runs/history_support_preflight.json"),
        "budgets": {k: dict(v) for k, v in budgets.items()},
        "summary": aggregates,
        "note": "All 40 actual and 5120 hypothetical histories independently replayed through native CPU transitions; every teacher prediction rechecked with existing pinned PublishedAgent, native solved guard. Independent world reconstruction, event counts and raw-value arithmetic. Not independent teacher implementation or rollout rerun. Selected-root means descriptive; no strength inference.",
    }
    output = Path("runs/history_support_independent_audit.json")
    assert not output.exists()
    output.write_bytes((json.dumps(audit, indent=2) + "\n").encode("utf-8"))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
