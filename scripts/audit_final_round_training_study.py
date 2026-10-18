"""Independent action replay and raw-pair audit of the frozen tail study."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from koikoi.layout import PHASE, POINTS, ROUND, TERMINAL, TURN
from koikoi.published_actor import PublishedActor
from koikoi.reference import ReferenceEnv
from scripts.run_arbitration_study import PARENTS, read, sha, write


def replay(row, *, prefix):
    env = ReferenceEnv(row["seed"], "paper8")
    early_actions = []
    rounds = matches = 0
    for action in row["actions"]:
        assert env.state[PHASE] != TERMINAL and env.legal()[action]
        if env.state[ROUND] < 7:
            early_actions.append(action)
        result = env.step(action)
        rounds += int(result[1])
        matches += int(result[2])
    assert row["work"] == {
        "decisions": len(row["actions"]),
        "round_endings": rounds,
        "completed_matches": matches,
    }
    if prefix:
        np.testing.assert_array_equal(env.state, row["endpoint"])
        reached = env.state[PHASE] != TERMINAL
        assert row["reached_final_round"] == reached
        assert len(early_actions) == len(row["actions"])
        if reached:
            assert (
                env.state[ROUND] == 7 and env.state[TURN] == 0 and env.state[PHASE] == 0
            )
    else:
        assert env.state[PHASE] == TERMINAL and matches == 1
        assert row["terminal_round"] == int(env.state[ROUND])
        assert row["terminal_points"] == env.state[POINTS : POINTS + 2].tolist()
    return env.state, early_actions


def interval(values):
    rng = np.random.default_rng(89310001)
    means = []
    remaining = 10000
    while remaining:
        size = min(128, remaining)
        means.extend(
            values[rng.integers(0, len(values), (size, len(values)))].mean(axis=1)
        )
        remaining -= size
    return np.quantile(means, [0.025, 0.975]).tolist()


def tensor_equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            tensor_equal(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for left, right in zip(a, b):
            tensor_equal(left, right)
    else:
        assert a == b


def grid(parent, source):
    base, n = (88110001, 256) if source else (89110001, 1024)
    return [
        (parent, name, seat, rep, base + 10000 * i + rep)
        for i, name in enumerate(("wp", "point"))
        for seat in (0, 1)
        for rep in range(n)
    ]


def key(row):
    return tuple(row[k] for k in ("parent", "opponent", "seat", "replicate", "seed"))


def audit(root):
    output = root / "independent_audit.json"
    if output.exists():
        raise ValueError("Refusing to overwrite an independent audit")
    plan = read(root / "protocol.json")
    producer = read(root / "analysis.json")
    assert read(root / "process.json")["status"] == producer["status"] == "complete"
    assert plan["updates"] == 520 and plan["evaluation_pairs_per_cell"] == 1024
    assert plan["source_cases_per_parent"] == 1024
    assert (plan["bootstrap_seed"], plan["bootstrap_replicates"]) == (89310001, 10000)
    inputs = {
        **plan["inputs"],
        **producer["inputs"],
        str(root / "analysis.json"): sha(root / "analysis.json"),
    }
    for path, digest in inputs.items():
        assert sha(path) == digest, path
    source_work = dict.fromkeys(("decisions", "round_endings", "completed_matches"), 0)
    evaluation_work = source_work.copy()
    training_work = dict.fromkeys(
        (
            "decisions",
            "learner_decisions",
            "opponent_decisions",
            "conditional_tail_completions",
            "conditional_round_endings",
        ),
        0,
    )
    survivors = []
    outcomes = []
    for parent in (1, 2):
        folder = root / f"parent{parent}"
        checkpoint = torch.load(
            folder / "final.pt", map_location="cpu", weights_only=False
        )
        initial = torch.load(
            folder / "initial.pt", map_location="cpu", weights_only=False
        )
        parent_path, parent_sha = PARENTS[parent - 1]
        assert sha(parent_path) == parent_sha
        original = torch.load(parent_path, map_location="cpu", weights_only=False)
        assert checkpoint["config"] == plan["parents"][parent - 1]["config"]
        expected_config = {
            "device": "cuda",
            "envs": 128,
            "horizon": 64,
            "epochs": 3,
            "minibatch": 512,
            "lr": 3e-6,
            "value_lr": 1e-4,
            "score_temperature": 0.05,
            "critic_warmup_updates": 20,
            "seed": 88210001 + 10000 * (parent - 1),
        }
        assert all(checkpoint["config"][k] == v for k, v in expected_config.items())
        tensor_equal(checkpoint["prefix_actor"], original["actor"])
        tensor_equal(initial["actor"], original["actor"])
        tensor_equal(initial["value"], original["value"])
        tensor_equal(checkpoint["opponents"], initial["opponents"])
        bank = torch.load(folder / "bank.pt", map_location="cpu", weights_only=False)
        tensor_equal(checkpoint["bank"], bank)
        rows = [
            json.loads(line)
            for line in (folder / "source.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        assert [key(row) for row in rows] == grid(parent, True)
        index = 0
        for row in rows:
            state, _ = replay(row, prefix=True)
            for name in source_work:
                source_work[name] += row["work"][name]
            if row["reached_final_round"]:
                assert row["root_index"] == index
                np.testing.assert_array_equal(state, bank[0][index])
                assert int(bank[1][index]) == row["seat"]
                assert int(bank[2][index]) == ("wp", "point").index(row["opponent"])
                index += 1
            else:
                assert row["root_index"] is None
        assert index == len(bank[0])
        survivors.append(index)
        counters = checkpoint["counters"]
        assert counters["updates"] == 520 and counters["policy_updates"] == 500
        assert (
            counters["warmup_updates"] == 20
            and counters["matches"] == counters["rounds"] == 0
        )
        assert (
            counters["conditional_tail_completions"]
            == counters["conditional_round_endings"]
            == 66560
        )
        assert (
            counters["decisions"]
            == counters["learner_decisions"] + counters["opponent_decisions"]
        )
        for name in training_work:
            training_work[name] += counters[name]
        actor = PublishedActor(checkpoint["actor"], "cpu")
        opt = checkpoint["optimizer"]
        parameters = list(actor.named_parameters())
        assert len(parameters) == len(opt["param_groups"][0]["params"])
        for (name, _), index in zip(parameters, opt["param_groups"][0]["params"]):
            phase = {"0": "discard", "1": "pick", "3": "koi"}[name.split(".")[1]]
            assert (
                int(opt["state"][index]["step"]) == counters["optimizer_steps_" + phase]
            )
        for index in opt["param_groups"][1]["params"]:
            assert int(opt["state"][index]["step"]) == counters["optimizer_steps_value"]
        assert all(
            bool(torch.isfinite(v).all())
            for entry in opt["state"].values()
            for v in entry.values()
        )
        metrics = [
            json.loads(line)
            for line in (folder / "metrics.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        assert [row["update"] for row in metrics] == list(range(1, 521))
        assert metrics[-1]["counters"] == counters
        for update, row in enumerate(metrics, 1):
            assert row["counters"]["conditional_tail_completions"] == 128 * update
            assert row["metrics"]["critic_only"] == (update <= 20)
        for opponent in ("wp", "point"):
            path = root / f"eval_parent{parent}_{opponent}.jsonl"
            rows = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
            ]
            expected_grid = [k for k in grid(parent, False) if k[1] == opponent]
            assert [(r["policy"], key(r)) for r in rows] == [
                (p, k) for p in ("parent", "tail") for k in expected_grid
            ]
            prefixes = {}
            points = {"parent": np.zeros((1024, 2)), "tail": np.zeros((1024, 2))}
            for row in rows:
                state, early = replay(row, prefix=False)
                pair = (row["replicate"], row["seat"])
                if row["policy"] == "parent":
                    prefixes[pair] = early
                else:
                    assert prefixes[pair] == early
                seat = row["seat"]
                points[row["policy"]][pair] = (
                    state[POINTS + seat] - state[POINTS + 1 - seat]
                )
                for name in evaluation_work:
                    evaluation_work[name] += row["work"][name]
            entry = {"parent": parent, "opponent": opponent}
            for label in ("win", "point"):
                values = {
                    k: ((np.sign(v) + 1) / 2 if label == "win" else v)
                    for k, v in points.items()
                }
                difference = (values["tail"] - values["parent"]).mean(axis=1)
                entry[label] = {
                    "parent_mean": float(values["parent"].mean()),
                    "tail_mean": float(values["tail"].mean()),
                    "change": float(difference.mean()),
                    "interval95": interval(difference),
                }
            outcomes.append(entry)
    assert evaluation_work["completed_matches"] == 16384
    assert outcomes == producer["contrasts"]
    assert producer["replicated_gain"] == all(
        row["win"]["interval95"][0] > 0 for row in outcomes
    )
    inputs[str(Path(__file__))] = sha(__file__)
    write(
        output,
        {
            "status": "passed",
            "inputs": inputs,
            "source_attempts": 2048,
            "source_unique_deals": 512,
            "evaluation_unique_deals": 2048,
            "source_survivors": survivors,
            "source_work": source_work,
            "training_work": training_work,
            "evaluation_work": evaluation_work,
            "contrasts": outcomes,
            "replicated_gain": producer["replicated_gain"],
            "audit_replay_work": {
                k: source_work[k] + evaluation_work[k] for k in source_work
            },
            "scope": "Independent recorded-action replay and arithmetic; no repeated neural inference/training; no new independent deals.",
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    audit(args.run)


if __name__ == "__main__":
    main()
