"""Independent enumeration completeness, native replay and support audit."""

import copy
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from koikoi.env import BatchEnv
from koikoi.layout import DEALER, HISTORY, PLAYER, SEED, STOCK
from koikoi.train import provenance
from koikoi.upstream import PublishedAgent
from scripts.audit_history_support import audit_native_history, read, sha, worlds_for


def all_subsets(cards, size):
    # Separate recursive enumeration, without the production combinations iterator.
    if size == 0:
        yield ()
        return
    for i in range(len(cards) - size + 1):
        for tail in all_subsets(cards[i + 1 :], size - 1):
            yield (cards[i],) + tail


def main():
    torch.set_num_threads(2)
    path = Path("runs/history_support_enumeration.json")
    report = read(path)
    preflight = read("runs/history_support_enumeration_preflight.json")
    manifest = read("runs/history_support_enumeration_process.json")
    assert report["status"] == preflight["status"] == manifest["status"] == "complete"
    assert (
        report["mode"] == manifest["mode"] == "run" and preflight["mode"] == "preflight"
    )
    assert manifest["output_sha256"] == sha(path)
    assert report["metadata"] == preflight["metadata"] == manifest["metadata"]
    assert (
        report["preflight_sha256"]
        == manifest["preflight_sha256"]
        == sha("runs/history_support_enumeration_preflight.json")
    )
    meta = report["metadata"]
    assert meta["script_sha256"] == sha("scripts/enumerate_history_support.py")
    assert meta["world_helper_sha256"] == sha("scripts/audit_history_support.py")
    assert (
        meta["support_sha256"]
        == sha("runs/history_support_diagnostic.json")
        == "ec8d8656e855f413779f0337bd3e207f745ce324d983598d474e53f06f5c8259"
    )
    assert (
        meta["true_history_sha256"]
        == sha("runs/history_support_preflight.json")
        == "799827876fb40f537685b945ad521c611116c2e31c3e0acbf072dc2c3c4a8a95"
    )
    hmeta = meta["history_metadata"]
    assert hmeta["script_sha256"] == sha("scripts/diagnose_history_support.py")
    assert (
        hmeta["source_sha256"]
        == provenance()["source_sha256"]
        == "b5fb6911d7a0fee266f98eca186408fad9148e7d481e743e5fb71312ce6d307f"
    )
    for p, h in hmeta["input_hashes"].items():
        assert sha(p) == h
    for name, h in hmeta["published_hashes"].items():
        assert sha(f"external/KoiKoi-AI/model_agent/{name}_rl_wp.pt") == h
    roots = read("runs/rollout_opponent_roots.json")["roots"]
    prior = read("runs/history_support_diagnostic.json")["results"]
    actual_prior = read("runs/history_support_preflight.json")["results"]
    native = BatchEnv(40, "cpu", "paper8", debug=True)
    native.state.copy_(torch.tensor([r["state"] for r in roots], dtype=torch.int32))
    observations = native.observe()[0].numpy()
    unknowns = [
        np.flatnonzero(obs[:528].reshape(48, 11)[:, 4]).tolist() for obs in observations
    ]
    sizes = [round(float(obs[541]) * 8) for obs in observations]
    selected = [i for i in range(40) if math.comb(len(unknowns[i]), sizes[i]) <= 1000]
    assert (
        [r["root_index"] for r in report["results"]]
        == [r["root_index"] for r in preflight["results"]]
        == selected
    )
    assert len(selected) == 9
    assert sum(math.comb(len(unknowns[i]), sizes[i]) for i in selected) == 4792
    budgets = {key: Counter() for key in ("enumeration", "preflight", "audit")}
    teacher = PublishedAgent("wp")
    summary = []
    for row, pre in zip(report["results"], preflight["results"]):
        index = row["root_index"]
        root, previous = roots[index], prior[index]
        assert (
            row["identity"]
            == pre["identity"]
            == root["identity"]
            == previous["identity"]
        )
        state = np.array(root["state"], dtype=np.int32)
        assert hashlib.sha256(state.tobytes()).hexdigest() == root["state_sha256"]
        player, dealer = int(state[PLAYER]), int(state[DEALER])
        played = [
            int(state[HISTORY + 5 * t]) - 1
            for t in range(16)
            if (dealer + t) % 2 != player and state[HISTORY + 5 * t] > 0
        ]
        assert row["unknown"] == unknowns[index] and row["played"] == played
        assert row["opponent_count"] == sizes[index]
        candidates = list(all_subsets(unknowns[index], sizes[index]))
        assert len(candidates) == row["candidate_count"] == len(set(candidates))
        assert [tuple(h["hand"]) for h in row["hands"]] == candidates
        worlds, records, lookup = [], [], {}
        for hand, entry in zip(candidates, row["hands"]):
            dealt = set(played) | set(hand)
            assert len(dealt) == 8
            valid = not any(
                set(range(4 * month, 4 * month + 4)) <= dealt for month in range(12)
            )
            assert entry["valid"] == valid
            lookup[hand] = entry
            if not valid:
                assert set(entry) == {"hand", "valid"}
                continue
            world = state.copy()
            for card in unknowns[index]:
                world[card] = 1 - player if card in hand else -1
            start = 48 + 8 * (1 - player)
            world[start : start + 8] = sorted(dealt)
            world[72 : 49 + state[STOCK]] = [
                c for c in unknowns[index] if c not in hand
            ]
            world[SEED] = 0
            evidence = entry["replay"]
            assert (
                entry["stock_rng_feature_checks"]
                == evidence["counts"]["opponent_decisions"]
            )
            assert entry["extra_teacher_calls"] == evidence["counts"]["teacher_calls"]
            budgets["enumeration"].update(evidence["counts"])
            budgets["enumeration"].update(
                {
                    key: entry[key]
                    for key in ("stock_rng_feature_checks", "extra_teacher_calls")
                }
            )
            worlds.append(world)
            records.append(evidence)
        worlds = np.stack(worlds)
        budgets["audit"].update(audit_native_history(worlds, records, teacher))
        # Independently check every support decision with a different future stock/seed.
        alternate = worlds.copy()
        end = 49 + int(state[STOCK])
        alternate[:, 72:end] = alternate[:, 72:end][:, ::-1]
        alternate[:, SEED] = 0x1FFFFFFF
        alternate_records = copy.deepcopy(records)
        for w, record in zip(alternate, alternate_records):
            record["reconstructed_state_sha256"] = hashlib.sha256(
                w.tobytes()
            ).hexdigest()
        budgets["audit"].update(
            audit_native_history(alternate, alternate_records, teacher)
        )
        assert row["valid_count"] == len(worlds)
        support = sum(e["compatible"] for e in records)
        assert row["compatible_count"] == support
        actual_hand = tuple(np.flatnonzero(state[:48] == 1 - player).tolist())
        assert lookup[actual_hand]["replay"]["compatible"]
        assert (
            lookup[actual_hand]["replay"]["events"]
            == actual_prior[index]["actual"]["events"]
        )
        assert pre["hands"] == [lookup[actual_hand]]
        for h in pre["hands"]:
            budgets["preflight"].update(h["replay"]["counts"])
            budgets["preflight"].update(
                {
                    key: h[key]
                    for key in ("stock_rng_feature_checks", "extra_teacher_calls")
                }
            )
        ident = row["identity"]
        sampled = worlds_for(state, ["seed1", "seed10001"].index(ident["model"]), ident)
        assert (
            hashlib.sha256(sampled.tobytes()).hexdigest() == previous["worlds_sha256"]
        )
        assert row["existing_worlds_checked"] == 128
        for world, old in zip(sampled, previous["worlds"]):
            hand = tuple(np.flatnonzero(world[:48] == 1 - player).tolist())
            assert lookup[hand]["valid"]
            assert lookup[hand]["replay"]["events"] == old["events"]
            assert lookup[hand]["replay"]["compatible"] == old["compatible"]
        summary.append(
            {
                "root_index": index,
                "model": ident["model"],
                "candidates": len(candidates),
                "valid_hands": len(worlds),
                "compatible_hands": support,
                "support_fraction": support / len(worlds),
                "old_sample_survivors": previous["compatible_count"],
                "expected_survivors_in_128": 128 * support / len(worlds),
            }
        )
        print(summary[-1], flush=True)
    audit = {
        "passed": True,
        "report_sha256": sha(path),
        "auditor_sha256": sha(__file__),
        "native_helper_sha256": sha("scripts/audit_history_support.py"),
        "budgets": {key: dict(c) for key, c in budgets.items()},
        "summary": summary,
        "note": "Independent complete hand enumeration and native replay of every valid hand under two stock/seed settings. Existing pinned teacher adapter shared. No future rollouts or general opponent/strength claim.",
    }
    output = Path("runs/history_support_enumeration_audit.json")
    assert not output.exists()
    output.write_bytes((json.dumps(audit, indent=2) + "\n").encode("utf-8"))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
