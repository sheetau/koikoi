"""Full, unpruned likelihood replay and independent belief-error arithmetic."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from koikoi.agents import EndgameAgent
from koikoi.layout import PHASE, PLAYER, ROUND
from koikoi.reference import ReferenceEnv
from koikoi.upstream import PublishedAgent
from scripts.diagnose_history_support import replay
from scripts.enumerate_history_support import canonical_world, context
from scripts.run_arbitration_study import read, sha, write


def run(output):
    audit_path = output / "independent_replay.json"
    if audit_path.exists():
        raise ValueError("Audit already exists")
    report = read(output / "report.json")
    assert report["status"] == "complete"
    assert sha(output / "protocol.json") == report["protocol_sha256"]
    assert sha(output / "records.jsonl") == report["records_sha256"]
    inputs = dict(report["inputs"])
    inputs.update(
        {
            str(p): sha(p)
            for p in (Path(__file__), output / "report.json", output / "records.jsonl")
        }
    )
    for path, digest in inputs.items():
        assert sha(path) == digest, path
    records = list(
        map(
            json.loads,
            (output / "records.jsonl").read_text(encoding="utf-8").splitlines(),
        )
    )
    sources = {
        p: list(
            map(
                json.loads,
                Path(f"runs/final_round_training_study/parent{p}/source.jsonl")
                .read_text(encoding="utf-8")
                .splitlines(),
            )
        )
        for p in (1, 2)
    }
    torch.set_num_threads(2)
    models = {name: EndgameAgent(PublishedAgent(name)) for name in ("wp", "point")}
    work, source_steps, tested_hands, verified_roots = {}, 0, 0, 0
    brier = {name: [] for name in ("uniform", "joint", "identified")}
    for row in records:
        if row["status"] != "eligible":
            continue
        original = sources[row["parent"]][row["case_index"]]
        env = ReferenceEnv(original["seed"], "paper8")
        for action in original["actions"][: row["decision"]]:
            assert env.legal()[action]
            env.step(action)
            source_steps += 1
        identity = row["public_identification"]
        assert identity["available_after_decision"] <= row["decision"]
        assert identity["round"] < env.state[ROUND] == row["round"]
        assert env.state[PLAYER] == row["seat"] and env.state[PHASE] == 0
        unknown, size, played = context(env.state)
        batch = row["proposals"]
        assert batch["mode"] == "sampled" and batch["proposed_draws"] == 128
        rng = np.random.default_rng(
            96010001 + 10000 * row["parent"] + row["case_index"]
        )
        hands, invalid = [], 0
        for _ in range(128):
            hand = sorted(int(c) for c in rng.choice(unknown, size, replace=False))
            initial = np.array(played + hand)
            if np.bincount(initial // 4, minlength=12).max() == 4:
                invalid += 1
            else:
                hands.append(hand)
        assert hands == batch["hands"] and invalid == batch["invalid_draws"]
        compatibility = []
        for hand in hands:
            world = canonical_world(env.state, hand)
            matched = []
            for name in batch["model_names"]:
                result = replay(world, models[name])
                matched.append(result["compatible"])
                for key, value in result["counts"].items():
                    work[key] = work.get(key, 0) + value
            compatibility.append(matched)
            tested_hands += 1
        assert compatibility == batch["compatible"]
        truth = {int(c) for c in np.flatnonzero(env.state[:48] == 1 - row["seat"])}
        index = batch["model_names"].index(identity["model"])
        for name in brier:
            weights = [
                1 if name == "uniform" else sum(c) if name == "joint" else int(c[index])
                for c in compatibility
            ]
            estimate = row["beliefs"][name]
            total = sum(weights)
            if not total:
                assert estimate["status"] == "zero_sampled_mass"
                continue
            cards = [
                sum(w for hand, w in zip(hands, weights) if card in hand) / total
                for card in range(48)
            ]
            np.testing.assert_allclose(
                cards, estimate["card_probabilities"], atol=1e-14, rtol=0
            )
            error = sum((cards[c] - int(c in truth)) ** 2 for c in unknown) / len(
                unknown
            )
            assert abs(error - estimate["card_brier"]) < 1e-14
            brier[name].append(error)
        verified_roots += 1
        print(json.dumps({"verified_roots": verified_roots}), flush=True)
    assert verified_roots == report["summary"]["eligible_roots"]
    for name, errors in brier.items():
        summary = report["summary"]["beliefs"][name]
        assert len(errors) == summary["estimated"]
        assert abs(sum(errors) / len(errors) - summary["mean_card_brier"]) < 1e-14
    for path, digest in inputs.items():
        assert sha(path) == digest, path
    write(
        audit_path,
        {
            "status": "passed",
            "inputs": inputs,
            "verified_roots": verified_roots,
            "verified_valid_hands": tested_hands,
            "source_prefix_replay_decisions": source_steps,
            "unpruned_hypothetical_replay": work,
            "new_evaluation_deals": 0,
            "future_rollouts": 0,
            "scope": "Every eligible root and valid proposal checked with full unpruned history replay; shared canonical constructor and published-model adapter, independent weighting arithmetic. Ineligible selection coverage is not independently reimplemented.",
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
