"""Independent continuation-root, stopping-time and complete-match study audit."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from koikoi.layout import PHASE, PLAYER, POINTS, ROUND, TERMINAL, TURN
from koikoi.published_actor import PublishedActor
from koikoi.reference import ReferenceEnv
from scripts.audit_final_round_training_study import tensor_equal
from scripts.audit_revealed_opponent_identification import elimination_ledger
from scripts.check_revealed_opponent_identification import verify_oracle
from scripts.run_arbitration_study import PARENTS, read, sha, write


def audit_endpoint(root, parent, mode):
    folder = root / f"parent{parent}_{mode}"
    paths = [
        folder / n for n in ("initial.pt", "final.pt", "metrics.jsonl", "config.json")
    ]
    initial, final = [
        torch.load(p, map_location="cpu", weights_only=False) for p in paths[:2]
    ]
    path, digest = PARENTS[parent - 1]
    assert sha(path) == digest
    original = torch.load(path, map_location="cpu", weights_only=False)
    assert final["format"] == initial["format"] == "public_memory_continuation_v1"
    cfg = final["config"]
    assert cfg == initial["config"] == read(folder / "config.json")
    assert cfg["seed"] == 92110001 + 10000 * (parent - 1)
    assert cfg["public_memory"] == (mode == "public")
    for key, expected in {
        "envs": 128,
        "horizon": 448,
        "epochs": 3,
        "minibatch": 512,
        "critic_warmup_updates": 20,
        "lr": 3e-6,
        "value_lr": 1e-4,
        "score_temperature": 0.05,
        "device": "cuda",
    }.items():
        assert cfg[key] == expected
    assert final["provenance"]["weight_initialization"]["sha256"] == digest
    for payload in (initial, final):
        tensor_equal(payload["prefix_actor"], original["actor"])
    tensor_equal(initial["value"], original["value"])
    for phase, weights in original["actor"]["current"]["weights"].items():
        for name, value in weights.items():
            expected = value.clone()
            if name == "encoder_block.f1.weight":
                expected[:, 2:4] = 0
            assert torch.equal(
                initial["actor"]["current"]["weights"][phase][name], expected
            )
    tensor_equal(final["opponents"], initial["opponents"])
    bank_path = Path(f"runs/public_memory_continuation_bank/parent{parent}.pt")
    bank = torch.load(bank_path, map_location="cpu", weights_only=False)
    assert bank["format"] == "public_memory_continuation_bank_v1"
    for payload in (initial, final):
        tensor_equal(
            payload["bank"], tuple(bank[k] for k in ("states", "seats", "opponents"))
        )
        tensor_equal(payload["memory"], bank["memory"])
    c = final["counters"]
    assert (
        c["updates"] == 520 and c["policy_updates"] == 500 and c["warmup_updates"] == 20
    )
    assert c["conditional_tail_completions"] == 66560
    assert 66560 <= c["conditional_round_endings"] <= 66560 * 7
    assert c["matches"] == c["rounds"] == 0
    assert c["decisions"] == c["learner_decisions"] + c["opponent_decisions"]
    actor = PublishedActor(final["actor"], "cpu")
    optimizer = final["optimizer"]
    names = list(actor.named_parameters())
    assert len(names) == len(optimizer["param_groups"][0]["params"])
    for (name, parameter), index in zip(names, optimizer["param_groups"][0]["params"]):
        phase = {"0": "discard", "1": "pick", "3": "koi"}[name.split(".")[1]]
        assert int(optimizer["state"][index]["step"]) == c["optimizer_steps_" + phase]
        if mode == "masked" and name.endswith("encoder_block.f1.weight"):
            assert not bool(parameter[:, 2:4].any())
    for index in optimizer["param_groups"][1]["params"]:
        assert int(optimizer["state"][index]["step"]) == c["optimizer_steps_value"]
    assert all(
        bool(torch.isfinite(v).all())
        for entry in optimizer["state"].values()
        for v in entry.values()
    )
    metrics = [json.loads(s) for s in paths[2].read_text(encoding="utf-8").splitlines()]
    assert [r["update"] for r in metrics] == list(range(1, 521))
    assert metrics[-1]["counters"] == c
    for index, row in enumerate(metrics, 1):
        assert row["metrics"]["critic_only"] == (index <= 20)
        assert row["counters"]["conditional_tail_completions"] == index * 128
    return {
        "status": "passed",
        "counters": c,
        "finite_adam_states": len(optimizer["state"]),
        "inputs": {str(p): sha(p) for p in paths + [bank_path, Path(path)]},
    }


def audit_matched_warmup(root, parent):
    for endpoint in ("initial.pt", "update_00000020.pt"):
        left, right = [
            torch.load(
                root / f"parent{parent}_{mode}" / endpoint,
                map_location="cpu",
                weights_only=False,
            )
            for mode in ("public", "masked")
        ]
        for key in (
            "actor",
            "value",
            "optimizer",
            "bank",
            "bank_rng",
            "memory",
            "counters",
            "torch_rng",
            "cuda_rng",
            "python_rng",
        ):
            tensor_equal(left[key], right[key])
        assert left["numpy_rng"][0] == right["numpy_rng"][0]
        np.testing.assert_array_equal(left["numpy_rng"][1], right["numpy_rng"][1])
        assert left["numpy_rng"][2:] == right["numpy_rng"][2:]


def audit_roots(parent):
    """Reconstruct every bank row independently from earlier public timing."""
    bank = torch.load(
        f"runs/public_memory_continuation_bank/parent{parent}.pt",
        map_location="cpu",
        weights_only=False,
    )
    identities = {
        r["case_index"]: r
        for r in map(
            json.loads,
            Path("runs/revealed_opponent_identification/records.jsonl")
            .read_text(encoding="utf-8")
            .splitlines(),
        )
        if r["parent"] == parent
    }
    rows = [
        json.loads(s)
        for s in Path(f"runs/final_round_training_study/parent{parent}/source.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    root_index = 0
    work = dict.fromkeys(("decisions", "round_endings", "completed_matches"), 0)
    for case_index, row in enumerate(rows):
        identity = identities[case_index]
        assert all(
            identity[k] == row[k] for k in ("parent", "seat", "seed", "opponent")
        )
        _, first, _, _ = elimination_ledger(identity["events"])
        assert first == identity["first_identification"]
        env = ReferenceEnv(row["seed"], "paper8")
        found = False
        for decision in range(len(row["actions"]) + 1):
            if (
                not found
                and first is not None
                and decision >= first["available_after_decision"]
                and int(env.state[ROUND]) > first["round"]
                and env.state[TURN] == 0
                and env.state[PHASE] == 0
            ):
                np.testing.assert_array_equal(bank["states"][root_index], env.state)
                assert int(bank["seats"][root_index]) == row["seat"]
                assert (
                    int(bank["opponents"][root_index])
                    == {"wp": 0, "point": 1}[row["opponent"]]
                )
                assert (
                    int(bank["memory"][root_index])
                    == {"wp": 0, "point": 1}[first["model"]]
                )
                root_index += 1
                found = True
            if decision == len(row["actions"]):
                break
            result = env.step(row["actions"][decision])
            work["decisions"] += 1
            work["round_endings"] += int(result[1])
            work["completed_matches"] += int(result[2])
        np.testing.assert_array_equal(env.state, row["endpoint"])
    assert (
        len(rows) == 1024
        and root_index == len(bank["states"]) == {1: 855, 2: 876}[parent]
    )
    return work


def replay_match(row):
    env = ReferenceEnv(row["seed"], "paper8")
    memory = row["memory"]
    events = (
        {}
        if memory is None
        else {e["available_after_decision"]: e for e in memory["events"]}
    )
    first = None
    active_candidates = {"wp", "point"}
    updates = {}
    if memory is not None:
        assert len(events) == len(memory["events"])
        candidates, first, calls, updates = elimination_ledger(memory["events"])
        assert sorted(candidates) == memory["candidates"]
        assert calls == memory["prediction_calls"]
        assert first == memory["first_identification"]
    hands, snapshots, states = {}, {}, {}
    seen, checks = set(), 0
    activation = None
    work = dict.fromkeys(("decisions", "round_endings", "completed_matches"), 0)
    for decision, action in enumerate(row["actions"]):
        assert env.state[PHASE] != TERMINAL and env.legal()[action]
        round_index, actor, phase = [int(env.state[k]) for k in (ROUND, PLAYER, PHASE)]
        if decision in updates:
            active_candidates = updates[decision]
        if (
            memory is not None
            and activation is None
            and first is not None
            and decision >= first["available_after_decision"]
            and round_index > first["round"]
            and env.state[TURN] == 0
            and phase == 0
            and active_candidates == {first["model"]}
        ):
            activation = {
                "decision": decision,
                "round": round_index,
                "candidate": first["model"],
                "code": {"wp": 0, "point": 1}[first["model"]],
            }
        revealed = False
        if (
            memory is not None
            and activation is None
            and round_index < 7
            and actor != row["seat"]
        ):
            snapshots.setdefault(round_index, []).append((decision, phase, action))
            states[decision] = env.state.copy()
            if phase == 0:
                hand = hands.setdefault(round_index, [])
                assert action not in hand
                hand.append(action)
                revealed = len(hand) == 8
        result = env.step(action)
        work["decisions"] += 1
        work["round_endings"] += int(result[1])
        work["completed_matches"] += int(result[2])
        if revealed:
            event = events[decision + 1]
            seen.add(decision + 1)
            assert (
                event["round"] == round_index and event["actor_seat"] == 1 - row["seat"]
            )
            assert event["match_key"] == f"match-{row['seed']}-seat-{row['seat']}"
            assert event["initial_opponent_hand"] == sorted(hands[round_index])
            assert [
                (c["decision"], c["phase"], c["observed"]) for c in event["checks"]
            ] == snapshots[round_index]
            verify_oracle(event, states)
            checks += len(event["checks"])
        else:
            assert decision + 1 not in events
    assert seen == set(events)
    assert env.state[PHASE] == TERMINAL and work["completed_matches"] == 1
    assert row["terminal_round"] == int(env.state[ROUND])
    assert row["terminal_points"] == env.state[POINTS : POINTS + 2].tolist()
    assert row["work"] == work
    if memory is not None:
        assert memory["activation"] == activation
    prefix = (
        row["actions"]
        if activation is None
        else row["actions"][: activation["decision"]]
    )
    return prefix, work, checks


def interval(values):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(93310001)
    results = []
    for start in range(0, 10000, 128):
        n = min(128, 10000 - start)
        selections = rng.integers(
            values.shape[1], size=(n, values.shape[0], values.shape[1])
        )
        strata = np.zeros((n, values.shape[0]))
        for group in range(values.shape[0]):
            strata[:, group] = values[group][selections[:, group]].mean(axis=1)
        results.extend(strata.mean(axis=1))
    return np.percentile(results, [2.5, 97.5]).tolist()


def audit(root):
    output = root / "independent_audit.json"
    if output.exists():
        raise ValueError("Refusing to overwrite independent audit")
    plan, produced = read(root / "protocol.json"), read(root / "analysis.json")
    assert read(root / "process.json")["status"] == produced["status"] == "complete"
    assert (
        plan["updates"],
        plan["pairs_per_opponent"],
        plan["bootstrap_seed"],
        plan["bootstrap_replicates"],
    ) == (520, 1024, 93310001, 10000)
    inputs = {
        **plan["inputs"],
        **produced["inputs"],
        str(root / "analysis.json"): sha(root / "analysis.json"),
    }
    for path, digest in inputs.items():
        assert sha(path) == digest, path
    work = dict.fromkeys(("decisions", "round_endings", "completed_matches"), 0)
    training, reproduced, checks = [], [], 0
    root_work = dict.fromkeys(work, 0)
    for parent in (1, 2):
        for key, value in audit_roots(parent).items():
            root_work[key] += value
        for mode in ("public", "masked"):
            training.append(audit_endpoint(root, parent, mode)["counters"])
        audit_matched_warmup(root, parent)
        points = np.zeros((3, 2, 1024, 2))
        for opp_index, opponent in enumerate(("wp", "point")):
            seen, prefixes, memories = set(), {}, {}
            with (root / f"eval_parent{parent}_{opponent}.jsonl").open(
                encoding="utf-8"
            ) as stream:
                for line in stream:
                    row = json.loads(line)
                    assert row["parent"] == parent and row["opponent"] == opponent
                    i, j, k = (
                        ("parent", "public", "masked").index(row["policy"]),
                        row["replicate"],
                        row["seat"],
                    )
                    assert (i, j, k) not in seen and 0 <= j < 1024 and k in (0, 1)
                    assert row["seed"] == 93110001 + 10000 * opp_index + j
                    seen.add((i, j, k))
                    prefix, counts, count_checks = replay_match(row)
                    checks += count_checks
                    for name in work:
                        work[name] += counts[name]
                    if i == 0:
                        prefixes[j, k] = prefix
                        assert row["memory"] is None
                    else:
                        assert row["memory"] is not None
                        assert prefix == prefixes[j, k][: len(prefix)]
                        if row["memory"]["activation"] is None:
                            assert prefix == prefixes[j, k]
                        # Both arms have the same frozen parent and public stopping rule.
                        evidence = {
                            x: row["memory"][x]
                            for x in (
                                "events",
                                "candidates",
                                "prediction_calls",
                                "first_identification",
                                "activation",
                            )
                        }
                        if i == 1:
                            memories[j, k] = evidence
                        else:
                            assert evidence == memories[j, k]
                    points[i, opp_index, j, k] = (
                        row["terminal_points"][k] - row["terminal_points"][1 - k]
                    )
            assert len(seen) == 6144
        for comparison, candidate, baseline in (
            ("public-minus-masked", 1, 2),
            ("public-minus-parent", 1, 0),
            ("masked-minus-parent", 2, 0),
        ):
            for scope, groups in (("mixture", [0, 1]), ("wp", [0]), ("point", [1])):
                record = {"parent": parent, "comparison": comparison, "scope": scope}
                for metric, outcomes in (
                    ("win", (np.sign(points) + 1) / 2),
                    ("point", points),
                ):
                    pair_changes = (
                        outcomes[candidate, groups] - outcomes[baseline, groups]
                    ).sum(axis=2) / 2
                    record[metric] = {
                        "change": float(pair_changes.mean()),
                        "interval95": interval(pair_changes),
                        "candidate_mean": float(outcomes[candidate, groups].mean()),
                        "baseline_mean": float(outcomes[baseline, groups].mean()),
                    }
                reproduced.append(record)
    assert reproduced == produced["contrasts"]
    success = all(
        r["win"]["interval95"][0] > 0
        for r in reproduced
        if r["scope"] == "mixture" and r["comparison"] != "masked-minus-parent"
    )
    assert success == produced["replicated_gain"]
    write(
        output,
        {
            "status": "passed",
            "inputs": inputs,
            "contrasts": reproduced,
            "replicated_gain": success,
            "training_counters": training,
            "evaluation_replay_work": work,
            "root_replay_work": root_work,
            "strict_input_checks": checks,
            "new_neural_inferences": 0,
            "new_independent_matches": 0,
        },
    )


if __name__ == "__main__":
    torch.set_num_threads(2)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    audit(parser.parse_args().output)
