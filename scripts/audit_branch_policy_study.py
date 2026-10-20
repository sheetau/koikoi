"""Independent CPU transition and NumPy statistics audit of the fixed screen."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from koikoi.env import BatchEnv
from koikoi.layout import EPISODE, PHASE, PLAYER, POINTS, SEED, TERMINAL
from koikoi.reference import ReferenceEnv
from koikoi.train import provenance
from koikoi.upstream_batch import parameter_digest
from scripts.train_branch_policy_gradient import sha, write


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


class Replay:
    def __init__(self):
        self.env = BatchEnv(512, "cpu", debug=True)
        self.storage = self.env.state
        self.work = dict.fromkeys(("decisions", "round_endings", "completions"), 0)

    def legal_at(self, states):
        self.env.count = len(states)
        self.env.state = self.storage[: len(states)]
        self.env.state.copy_(states)
        _, legal = self.env.observe()
        solved = self.env.solved_actions()
        return legal & (
            (solved[:, None] < 0) | (torch.arange(50)[None] == solved[:, None])
        )

    def __call__(self, initial, data, source=False):
        env = self.env
        env.count = len(initial)
        env.state = self.storage[: len(initial)]
        env.state.copy_(initial)
        work = dict.fromkeys(self.work, 0)
        counts = torch.zeros(env.count, dtype=torch.long)
        for step, actions in enumerate(data["actions"].long()):
            active = env.state[:, PHASE] != TERMINAL
            _, legal = env.observe()
            solved = env.solved_actions()
            legal &= (solved[:, None] < 0) | (torch.arange(50)[None] == solved[:, None])
            assert legal[active].gather(1, actions[active, None]).all()
            if source:
                chosen = data["selected_step"] == step
                assert torch.equal(env.state[chosen], data["states"][chosen])
                counts += active & (legal.sum(-1) > 1)
            result = env.step(actions, freeze_done=True)
            work["decisions"] += int(active.sum())
            work["round_endings"] += int(result[:, 1].sum())
            work["completions"] += int(result[:, 2].sum())
        assert torch.equal(env.state, data["final"])
        assert work == data["work"]
        if source:
            assert torch.equal(counts, data["eligible_decisions"])
        for key in work:
            self.work[key] += work[key]


def checked_training(base):
    plan, complete = read(base / "protocol.json"), read(base / "complete.json")
    assert complete["status"] == "complete"
    assert complete["protocol_sha256"] == sha(base / "protocol.json")
    assert provenance()["source_sha256"] == plan["source"]["source_sha256"]
    for name, digest in plan["bindings"].items():
        assert sha(name) == digest
    for parent in plan["parents"]:
        assert sha(parent["path"]) == parent["sha256"]
    return plan, complete


def training(base):
    base = Path(base)
    output = base / "audit.json"
    if output.exists():
        raise ValueError("Training audit already exists")
    plan, complete = checked_training(base)
    replay = Replay()
    evidence = {str(base / "complete.json"): sha(base / "complete.json")}
    for name, case in plan["cases"].items():
        directory = base / name
        retained = complete["cases"][name]
        endpoint = directory / "update_0128.pt"
        assert sha(endpoint) == retained["endpoint_sha256"]
        assert sha(directory / "metrics.jsonl") == retained["metrics_sha256"]
        metrics = [
            json.loads(line)
            for line in (directory / "metrics.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        assert len(metrics) == plan["updates_per_arm"]
        counters = {
            "updates": 0,
            "eligible_decisions": 0,
            "source": dict.fromkeys(replay.work, 0),
            "branches": dict.fromkeys(replay.work, 0),
        }
        for step, metric in enumerate(metrics, 1):
            path = directory / f"evidence_{step:04}.pt"
            assert sha(path) == retained["evidence"][path.name]
            evidence[str(path)] = retained["evidence"][path.name]
            data = load(path)
            source, branch = data["source"], data["branch"]
            for lane, state in enumerate(source["initial"].numpy()):
                assert state[SEED] == case["config"]["seed"] + lane
                assert state[EPISODE] == step - 1
                expected = ReferenceEnv(int(state[SEED]), episode=step - 1).state
                assert np.array_equal(expected, state)
            replay(source["initial"], source, source=True)
            assert source["work"] == metric["source_work"]
            assert source["work"]["completions"] == case["config"]["matches"]
            counters["updates"] += 1
            counters["eligible_decisions"] += int(source["eligible_decisions"].sum())
            for key in replay.work:
                counters["source"][key] += source["work"][key]
            if case["config"]["mode"] == "sampled_return":
                assert branch is None and metric["branch_work"] is None
            else:
                assert branch["work"] == metric["branch_work"]
                assert torch.equal(replay.legal_at(source["states"]), branch["legal"])
                roots, actions = torch.where(branch["legal"])
                assert torch.equal(roots, branch["root_indices"])
                assert torch.equal(actions, branch["first_actions"])
                assert torch.equal(actions, branch["actions"][0].long())
                replay(source["states"][roots], branch)
                points = (
                    branch["final"][:, POINTS] - branch["final"][:, POINTS + 1]
                ) * (1 - 2 * source["states"][roots, PLAYER])
                assert torch.equal(points, branch["points"])
                q = torch.zeros_like(branch["q"])
                q[roots, actions] = points.sign().float()
                assert torch.equal(q, branch["q"])
                for key in replay.work:
                    counters["branches"][key] += branch["work"][key]
            assert metric["update"] == step
            assert np.isfinite(metric["gradient_norm_before_clip"])
            assert np.isfinite(metric["root_kl_after_update"])
        payload = load(endpoint)
        assert payload["format"] == "branch_policy_gradient_v1"
        assert payload["config"] == case["config"]
        assert payload["counters"] == retained["counters"] == counters
        current = payload["actor"]["current"]
        assert parameter_digest(current["weights"]) == current["parameters_sha256"]
        evidence[str(endpoint)] = sha(endpoint)
        evidence[str(directory / "metrics.jsonl")] = sha(directory / "metrics.jsonl")
        print(json.dumps({"verified": name, "counters": counters}), flush=True)
    write(
        output,
        {
            "status": "passed",
            "protocol_sha256": sha(base / "protocol.json"),
            "auditor_sha256": sha(__file__),
            "evidence": evidence,
            "replayed_work": replay.work,
            "scope": "All source and branch transitions, initial deals, roots, outcome labels and counters; checkpoint identity. No neural training replay.",
        },
    )


def evaluation(base):
    base = Path(base)
    plan, _ = checked_training(base)
    directory = base / "evaluation"
    output = directory / "audit.json"
    if output.exists():
        raise ValueError("Evaluation audit already exists")
    protocol, summary = (
        read(directory / "protocol.json"),
        read(directory / "summary.json"),
    )
    assert summary["status"] == "complete"
    assert summary["protocol_sha256"] == sha(directory / "protocol.json")
    assert summary["points_sha256"] == sha(directory / "points.npy")
    assert protocol["settings"] == plan["evaluation"]
    for name, digest in protocol["bindings"].items():
        assert sha(name) == digest
    for model in protocol["models"].values():
        assert sha(model["path"]) == model["sha256"]
    points = np.load(directory / "points.npy")
    assert points.shape == (8, 2, plan["evaluation"]["pairs_per_opponent"], 2)
    assert set(summary["model_order"]) == set(protocol["models"])
    replay = Replay()
    settings = plan["evaluation"]
    for i, model in enumerate(summary["model_order"]):
        for j, kind in enumerate(("wp", "point")):
            for offset in range(
                0, settings["pairs_per_opponent"], settings["batch"] // 2
            ):
                path = directory / f"{model}_{kind}" / f"chunk_{offset:04}.pt"
                assert (
                    sha(path) == summary["artifacts"][str(path.relative_to(directory))]
                )
                data = load(path)
                for lane, state in enumerate(data["initial"].numpy()):
                    expected = ReferenceEnv(
                        settings[f"{kind}_seed"] + offset + lane // 2
                    ).state
                    assert np.array_equal(expected, state)
                replay(data["initial"], data)
                # Explicit seat signs avoid relying on the evaluator's formula.
                signs = np.array([1, -1] * (settings["batch"] // 2))
                margins = (
                    data["final"][:, POINTS].numpy()
                    - data["final"][:, POINTS + 1].numpy()
                ) * signs
                np.testing.assert_array_equal(
                    margins.reshape(-1, 2), data["points"].numpy()
                )
                np.testing.assert_array_equal(
                    margins.reshape(-1, 2),
                    points[i, j, offset : offset + settings["batch"] // 2],
                )
        print(f"Replayed evaluation {model}", flush=True)
    assert replay.work == summary["work"]
    order = summary["model_order"]
    for i, model in enumerate(order):
        for j, kind in enumerate(("wp", "point")):
            margins = points[i, j]
            item = summary["secondary_means"][model][kind]
            assert item["point_margin"] == float(margins.mean())
            assert item["win"] == float(((margins > 0) + 0.5 * (margins == 0)).mean())
    rows = []
    for pair in range(settings["pairs_per_opponent"]):
        row = []
        for parent in (1, 2):
            shared = order.index(f"parent{parent}_shared_actions")
            for comparator in (
                f"parent{parent}",
                f"parent{parent}_independent_actions",
                f"parent{parent}_sampled_return",
            ):
                other = order.index(comparator)
                difference = 0.0
                for opponent in range(2):
                    for seat in range(2):
                        a, b = (
                            int(points[shared, opponent, pair, seat]),
                            int(points[other, opponent, pair, seat]),
                        )
                        difference += ((a > 0) + 0.5 * (a == 0)) - (
                            (b > 0) + 0.5 * (b == 0)
                        )
                row.append(difference / 4)
        rows.append(row)
    rows = np.array(rows)
    means = rows.mean(0)
    rng = np.random.default_rng(settings["bootstrap_seed"])
    maxima = np.empty(settings["bootstrap_replicates"])
    for rep in range(len(maxima)):
        sample = rng.integers(len(rows), size=len(rows))
        maxima[rep] = np.max(np.abs(rows[sample].mean(0) - means))
    radius = np.quantile(maxima, 0.95)
    for index, item in enumerate(summary["primary"]):
        assert item["win_change"] == means[index]
        np.testing.assert_array_equal(
            item["simultaneous_95"], [means[index] - radius, means[index] + radius]
        )
    assert summary["progression_criterion"] == bool(np.all(means - radius > 0))
    write(
        output,
        {
            "status": "passed",
            "protocol_sha256": sha(directory / "protocol.json"),
            "summary_sha256": sha(directory / "summary.json"),
            "auditor_sha256": sha(__file__),
            "replayed_work": replay.work,
            "scope": "All recorded trajectories, paired deals, outcomes, work and independent primary simultaneous statistics; no neural inference replay",
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("training", "evaluation"))
    parser.add_argument("--input", default="runs/branch_policy_gradient_training")
    args = parser.parse_args()
    (training if args.kind == "training" else evaluation)(args.input)
