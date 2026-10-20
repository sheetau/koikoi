"""Fixed eight-arm root-allocation/step-size screen, with complete evaluation."""

import argparse
import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from koikoi.env import BatchEnv
from koikoi.layout import EPISODE, PLAYER, POINTS, SEED
from koikoi.published_actor import PublishedActor
from koikoi.reference import ReferenceEnv
from koikoi.train import provenance
from koikoi.upstream_batch import parameter_digest
from scripts.audit_branch_policy_study import Replay
from scripts.evaluate_branch_policy_gradient import play_chunk
from scripts.priority_branch_trainer import Config, Trainer
from scripts.run_arbitration_study import PARENTS
from scripts.train_branch_policy_gradient import setup, sha, write


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def make_plan():
    files = (
        "scripts/priority_branch_roots.py",
        "scripts/priority_branch_trainer.py",
        "scripts/priority_branch_study.py",
        "scripts/branch_policy_gradient.py",
        "scripts/train_branch_policy_gradient.py",
        "scripts/run_arbitration_study.py",
        "scripts/evaluate_branch_policy_gradient.py",
        "scripts/audit_branch_policy_study.py",
        "docs/PRIORITY_BRANCH_GRADIENT.rst",
        "tests/test_priority_branch.py",
    )
    return {
        "run_kind": "development",
        "source": provenance(),
        "bindings": {name: sha(name) for name in files},
        "updates": 128,
        "save_interval": 16,
        "parents": [{"path": path, "sha256": digest} for path, digest in PARENTS],
        "cases": {
            f"parent{index + 1}_{sampler}_{rate}": {
                "parent_index": index,
                "config": asdict(
                    Config(
                        "shared_actions",
                        seed,
                        sampler="sensitivity" if sampler == "priority" else "uniform",
                        lr=lr,
                    )
                ),
            }
            for index, seed in enumerate((114110001, 114210001))
            for sampler, rate, lr in (
                ("uniform", "low", 1e-5),
                ("priority", "low", 1e-5),
                ("uniform", "high", 1e-4),
                ("priority", "high", 1e-4),
            )
        },
        "evaluation": {
            "device": "cuda",
            "batch": 128,
            "pairs": 1024,
            "wp_seed": 118110001,
            "point_seed": 118210001,
            "bootstrap_seed": 118310001,
            "bootstrap_replicates": 10000,
            "primary": [
                [f"parent{i}_priority_high", opponent]
                for i in (1, 2)
                for opponent in (f"parent{i}", f"parent{i}_uniform_high")
            ],
            "secondary": [
                [f"parent{i}_{a}", f"parent{i}_{b}"]
                for i in (1, 2)
                for a, b in (
                    ("priority_low", "uniform_low"),
                    ("uniform_high", "uniform_low"),
                    ("priority_high", "priority_low"),
                    ("priority_high", "uniform_low"),
                )
            ],
            "criterion": "All four simultaneous 95% primary lower bounds exceed zero; no canonical promotion.",
        },
    }


def check_plan(base):
    plan = read(base / "protocol.json")
    assert provenance()["source_sha256"] == plan["source"]["source_sha256"]
    for path, digest in plan["bindings"].items():
        assert sha(path) == digest, path
    for parent in plan["parents"]:
        assert sha(parent["path"]) == parent["sha256"]
    if plan["run_kind"] == "development":
        assert plan["updates"] == 128 and len(plan["cases"]) == 8
        assert all(c["config"]["matches"] == 64 for c in plan["cases"].values())
        assert (
            plan["evaluation"]["pairs"] == 1024 and plan["evaluation"]["batch"] == 128
        )
    else:
        assert plan["run_kind"] == "capability"
    return plan


def train(base):
    plan = check_plan(base)
    start = time.perf_counter()
    finished = {}
    for name, case in plan["cases"].items():
        directory = base / name
        directory.mkdir()
        trainer = Trainer(
            plan["parents"][case["parent_index"]]["path"], Config(**case["config"])
        )
        evidence = {}
        for step in range(1, plan["updates"] + 1):
            metrics, data = trainer.update()
            path = directory / f"evidence_{step:04}.pt"
            torch.save(data, path)
            evidence[path.name] = sha(path)
            with (directory / "metrics.jsonl").open(
                "a", encoding="utf-8", newline="\n"
            ) as log:
                log.write(json.dumps(metrics, allow_nan=False) + "\n")
            if step % plan["save_interval"] == 0 or step == plan["updates"]:
                trainer.save(directory / f"update_{step:04}.pt")
                print(json.dumps({"case": name, "update": step}), flush=True)
        endpoint = directory / f"update_{plan['updates']:04}.pt"
        finished[name] = {
            "counters": trainer.counters,
            "endpoint_sha256": sha(endpoint),
            "evidence": evidence,
            "metrics_sha256": sha(directory / "metrics.jsonl"),
        }
        write(base / "progress.json", finished)
        del trainer
    check_plan(base)
    write(
        base / "complete.json",
        {
            "status": "complete",
            "protocol_sha256": sha(base / "protocol.json"),
            "cases": finished,
            "elapsed_seconds": time.perf_counter() - start,
        },
    )


def verify_reservoir(source):
    """Scalar reconstruction from recorded priorities/uniforms, independent of collector."""
    weights = source["priorities"].numpy()
    uniforms = source["selection_uniforms"].numpy()
    for lane in range(weights.shape[1]):
        total, selected = 0.0, -1
        for step in range(weights.shape[0]):
            weight = float(weights[step, lane])
            assert math.isfinite(weight) and weight >= 0
            total += weight
            if weight and float(uniforms[step, lane]) < weight / total:
                selected = step
        assert selected == int(source["selected_step"][lane])
        assert int(np.count_nonzero(weights[:, lane])) == int(
            source["eligible_decisions"][lane]
        )
        expected = (
            0 if selected < 0 else math.fsum(weights[:, lane]) / weights[selected, lane]
        )
        assert math.isclose(
            float(source["importance"][lane]), expected, rel_tol=1e-12, abs_tol=1e-12
        )
        assert math.isclose(float(source["priority_total"][lane]), total, rel_tol=1e-12)
        assert float(source["selected_priority"][lane]) == (
            0 if selected < 0 else weights[selected, lane]
        )
    if source["sampler"] == "uniform":
        assert np.isin(weights, [0, 1]).all()
        assert torch.equal(source["importance"].long(), source["eligible_decisions"])
    else:
        assert source["sampler"] == "sensitivity"
        assert ((weights == 0) | ((weights >= 0.05) & (weights <= 1.05))).all()


def audit_training(base):
    plan = check_plan(base)
    complete = read(base / "complete.json")
    assert complete["protocol_sha256"] == sha(base / "protocol.json")
    if (base / "audit.json").exists():
        raise ValueError("Training audit is already complete")
    replay = Replay()
    for name, case in plan["cases"].items():
        directory = base / name
        retained = complete["cases"][name]
        endpoint = directory / f"update_{plan['updates']:04}.pt"
        assert sha(endpoint) == retained["endpoint_sha256"]
        assert sha(directory / "metrics.jsonl") == retained["metrics_sha256"]
        metrics = [
            json.loads(line)
            for line in (directory / "metrics.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        assert len(metrics) == plan["updates"]
        counters = {
            "updates": 0,
            "eligible_decisions": 0,
            "source": dict.fromkeys(replay.work, 0),
            "branches": dict.fromkeys(replay.work, 0),
        }
        for step, metric in enumerate(metrics, 1):
            path = directory / f"evidence_{step:04}.pt"
            assert sha(path) == retained["evidence"][path.name]
            data = load(path)
            source, branch = data["source"], data["branch"]
            for lane, state in enumerate(source["initial"].numpy()):
                assert state[SEED] == case["config"]["seed"] + lane
                assert state[EPISODE] == step - 1
                assert np.array_equal(
                    state, ReferenceEnv(int(state[SEED]), episode=step - 1).state
                )
            verify_reservoir(source)
            replay(source["initial"], source, source=True)
            assert torch.equal(replay.legal_at(source["states"]), branch["legal"])
            roots, actions = torch.where(branch["legal"])
            assert torch.equal(roots, branch["root_indices"])
            assert torch.equal(actions, branch["first_actions"])
            assert torch.equal(actions, branch["actions"][0].long())
            replay(source["states"][roots], branch)
            points = (branch["final"][:, POINTS] - branch["final"][:, POINTS + 1]) * (
                1 - 2 * source["states"][roots, PLAYER]
            )
            assert torch.equal(points, branch["points"])
            q = torch.zeros_like(branch["q"])
            q[roots, actions] = points.sign().float()
            assert torch.equal(q, branch["q"])
            assert source["work"] == metric["source_work"]
            assert branch["work"] == metric["branch_work"]
            assert metric["update"] == step
            counters["updates"] += 1
            counters["eligible_decisions"] += int(source["eligible_decisions"].sum())
            for key in replay.work:
                counters["source"][key] += source["work"][key]
                counters["branches"][key] += branch["work"][key]
        payload = load(endpoint)
        assert payload["format"] == "priority_branch_gradient_v1"
        assert payload["config"] == case["config"]
        assert payload["counters"] == retained["counters"] == counters
        current = payload["actor"]["current"]
        assert parameter_digest(current["weights"]) == current["parameters_sha256"]
        print(json.dumps({"training_audit_passed": name}), flush=True)
    write(
        base / "audit.json",
        {
            "status": "passed",
            "protocol_sha256": sha(base / "protocol.json"),
            "complete_sha256": sha(base / "complete.json"),
            "auditor_sha256": sha(__file__),
            "replayed_work": replay.work,
            "scope": "All transitions, initial deals, selected roots, recorded reservoir probabilities/corrections and outcome labels. No full neural training replay.",
        },
    )


def comparison_rows(points, order, pairs):
    wins = (np.sign(points) + 1) / 2
    return np.stack(
        [
            (wins[order.index(a)] - wins[order.index(b)]).mean(axis=(0, 2))
            for a, b in pairs
        ],
        axis=1,
    )


def report(points, order, settings):
    rows = comparison_rows(points, order, settings["primary"])
    means = rows.mean(0)
    rng = np.random.default_rng(settings["bootstrap_seed"])
    maxima = []
    for offset in range(0, settings["bootstrap_replicates"], 100):
        n = min(100, settings["bootstrap_replicates"] - offset)
        indices = rng.integers(len(rows), size=(n, len(rows)))
        maxima.extend(np.abs(rows[indices].mean(1) - means).max(-1))
    radius = float(np.quantile(maxima, 0.95))
    return {
        "primary": [
            {
                "models": pair,
                "win_change": float(value),
                "simultaneous_95": [float(value - radius), float(value + radius)],
            }
            for pair, value in zip(settings["primary"], means)
        ],
        "progression_criterion": bool((means - radius > 0).all()),
        "secondary_contrast_means": [
            {"models": pair, "win_change": float(value)}
            for pair, value in zip(
                settings["secondary"],
                comparison_rows(points, order, settings["secondary"]).mean(0),
            )
        ],
        "secondary_matchup_means": {
            name: {
                kind: {
                    "win": float(((np.sign(points[i, j]) + 1) / 2).mean()),
                    "point_margin": float(points[i, j].mean()),
                }
                for j, kind in enumerate(("wp", "point"))
            }
            for i, name in enumerate(order)
        },
    }


def evaluate(base):
    plan = check_plan(base)
    complete, training_audit = read(base / "complete.json"), read(base / "audit.json")
    assert training_audit["status"] == "passed"
    assert training_audit["complete_sha256"] == sha(base / "complete.json")
    settings = plan["evaluation"]
    directory = base / "evaluation"
    directory.mkdir()
    setup(settings["bootstrap_seed"])
    models = {
        name: str(base / name / f"update_{plan['updates']:04}.pt")
        for name in plan["cases"]
    }
    models.update({f"parent{i + 1}": p["path"] for i, p in enumerate(plan["parents"])})
    for name, row in complete["cases"].items():
        assert sha(models[name]) == row["endpoint_sha256"]
    opponents = {
        kind: PublishedActor.from_published(kind, settings["device"]).requires_grad_(
            False
        )
        for kind in ("wp", "point")
    }
    write(
        directory / "protocol.json",
        {
            "training_protocol_sha256": sha(base / "protocol.json"),
            "training_audit_sha256": sha(base / "audit.json"),
            "settings": settings,
            "source": provenance(),
            "evaluator_sha256": sha(__file__),
            "models": {
                name: {"path": path, "sha256": sha(path)}
                for name, path in models.items()
            },
            "opponents": {
                kind: actor.checkpoint()["current"]["parameters_sha256"]
                for kind, actor in opponents.items()
            },
        },
    )
    order = list(models)
    points = np.empty((len(order), 2, settings["pairs"], 2), dtype=np.int32)
    env = BatchEnv(settings["batch"], settings["device"], debug=True)
    assert settings["pairs"] % (env.count // 2) == 0
    work = dict.fromkeys(("decisions", "round_endings", "completions"), 0)
    artifacts = {}
    for index, name in enumerate(order):
        actor = PublishedActor(
            load(models[name])["actor"], settings["device"]
        ).requires_grad_(False)
        for j, (kind, opponent) in enumerate(opponents.items()):
            target = directory / f"{name}_{kind}"
            target.mkdir()
            for offset in range(0, settings["pairs"], env.count // 2):
                data = play_chunk(
                    env, actor, opponent, settings[f"{kind}_seed"] + offset
                )
                points[index, j, offset : offset + env.count // 2] = data[
                    "points"
                ].numpy()
                path = target / f"chunk_{offset:04}.pt"
                torch.save(data, path)
                artifacts[str(path.relative_to(directory))] = sha(path)
                for key in work:
                    work[key] += data["work"][key]
            print(
                json.dumps(
                    {
                        "evaluated": name,
                        "opponent": kind,
                        "matches": work["completions"],
                    }
                ),
                flush=True,
            )
        del actor
    np.save(directory / "points.npy", points)
    summary = report(points, order, settings)
    summary.update(
        {
            "status": "complete",
            "model_order": order,
            "work": work,
            "protocol_sha256": sha(directory / "protocol.json"),
            "points_sha256": sha(directory / "points.npy"),
            "artifacts": artifacts,
        }
    )
    write(directory / "summary.json", summary)


def audit_evaluation(base):
    plan = check_plan(base)
    directory = base / "evaluation"
    if (directory / "audit.json").exists():
        raise ValueError("Evaluation audit already exists")
    protocol, summary = (
        read(directory / "protocol.json"),
        read(directory / "summary.json"),
    )
    assert summary["protocol_sha256"] == sha(directory / "protocol.json")
    assert summary["points_sha256"] == sha(directory / "points.npy")
    assert protocol["evaluator_sha256"] == sha(__file__)
    for model in protocol["models"].values():
        assert sha(model["path"]) == model["sha256"]
    settings = plan["evaluation"]
    assert protocol["settings"] == settings
    order = summary["model_order"]
    assert set(order) == set(protocol["models"])
    points = np.load(directory / "points.npy")
    assert points.shape == (10, 2, settings["pairs"], 2)
    replay = Replay()
    for i, model in enumerate(order):
        for j, kind in enumerate(("wp", "point")):
            for offset in range(0, settings["pairs"], settings["batch"] // 2):
                path = directory / f"{model}_{kind}" / f"chunk_{offset:04}.pt"
                assert (
                    sha(path) == summary["artifacts"][str(path.relative_to(directory))]
                )
                data = load(path)
                for lane, state in enumerate(data["initial"].numpy()):
                    expected = ReferenceEnv(
                        settings[f"{kind}_seed"] + offset + lane // 2
                    ).state
                    assert np.array_equal(state, expected)
                replay(data["initial"], data)
                signs = np.array([1, -1] * (settings["batch"] // 2))
                margins = (
                    data["final"][:, POINTS].numpy()
                    - data["final"][:, POINTS + 1].numpy()
                ) * signs
                np.testing.assert_array_equal(
                    data["points"].numpy(), margins.reshape(-1, 2)
                )
                np.testing.assert_array_equal(
                    points[i, j, offset : offset + len(margins) // 2],
                    margins.reshape(-1, 2),
                )
            cell = summary["secondary_matchup_means"][model][kind]
            values = points[i, j]
            assert cell["point_margin"] == float(values.mean())
            assert cell["win"] == float(((values > 0) + 0.5 * (values == 0)).mean())
    assert replay.work == summary["work"]
    # Independently construct every primary and secondary pair-level coordinate.
    comparisons = settings["primary"] + settings["secondary"]
    rows = np.zeros((settings["pairs"], len(comparisons)))
    for col, (a, b) in enumerate(comparisons):
        for pair in range(settings["pairs"]):
            total = 0.0
            for opponent in range(2):
                for seat in range(2):
                    x = int(points[order.index(a), opponent, pair, seat])
                    y = int(points[order.index(b), opponent, pair, seat])
                    total += (x > 0) + 0.5 * (x == 0) - (y > 0) - 0.5 * (y == 0)
            rows[pair, col] = total / 4
    means = rows.mean(0)
    rng = np.random.default_rng(settings["bootstrap_seed"])
    maxima = []
    for _ in range(settings["bootstrap_replicates"]):
        sample = rng.integers(len(rows), size=len(rows))
        maxima.append(np.abs(rows[sample, :4].mean(0) - means[:4]).max())
    radius = np.quantile(maxima, 0.95)
    for i, item in enumerate(summary["primary"]):
        assert item["models"] == comparisons[i]
        assert item["win_change"] == means[i]
        np.testing.assert_array_equal(
            item["simultaneous_95"], [means[i] - radius, means[i] + radius]
        )
    for i, item in enumerate(summary["secondary_contrast_means"], 4):
        assert item["models"] == comparisons[i] and item["win_change"] == means[i]
    assert summary["progression_criterion"] == bool((means[:4] - radius > 0).all())
    write(
        directory / "audit.json",
        {
            "status": "passed",
            "protocol_sha256": sha(directory / "protocol.json"),
            "summary_sha256": sha(directory / "summary.json"),
            "auditor_sha256": sha(__file__),
            "replayed_work": replay.work,
            "scope": "All recorded transitions, initial deals, work, outcome/secondary means and independent primary simultaneous bootstrap; no neural inference replay.",
        },
    )


def run(base, plan=None):
    base = Path(base)
    if base.exists():
        raise ValueError("Use a fresh study directory; never restart closed runs")
    base.mkdir(parents=True)
    write(base / "protocol.json", make_plan() if plan is None else plan)
    try:
        for stage, operation in (
            ("training", train),
            ("training_audit", audit_training),
            ("evaluation", evaluate),
            ("evaluation_audit", audit_evaluation),
        ):
            write(
                base / "process.json",
                {"pid": os.getpid(), "stage": stage, "status": "running"},
            )
            operation(base)
        write(
            base / "finished.json",
            {
                "status": "complete",
                "protocol_sha256": sha(base / "protocol.json"),
                "training_audit_sha256": sha(base / "audit.json"),
                "evaluation_sha256": sha(base / "evaluation" / "summary.json"),
                "evaluation_audit_sha256": sha(base / "evaluation" / "audit.json"),
            },
        )
        write(base / "process.json", {"pid": os.getpid(), "status": "complete"})
    except BaseException as error:
        write(
            base / "process.json",
            {"pid": os.getpid(), "status": "failed", "error": repr(error)},
        )
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="runs/priority_branch_study")
    run(parser.parse_args().output)
