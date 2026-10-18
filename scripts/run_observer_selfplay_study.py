"""Fixed observer-versus-signed self-play continuation comparison."""

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from koikoi.arbitration import read_checkpoint
from koikoi.evaluate import duel, make_agent, paired_interval
from koikoi.layout import EPISODE
from koikoi.train import provenance
from koikoi.upstream_batch import parameter_digest
from scripts.analyze_arbitration_study import interval, outcome, tensor_equal
from scripts.observer_selfplay import ObserverConfig, ObserverTrainer
from scripts.observer_selfplay_agent import ObserverAgent
from scripts.run_arbitration_study import read, sha, write
from scripts.run_response_study import audit_training as original_training_audit
from scripts.run_response_study import checked as original_checked
from scripts.run_response_study import stop_owned_workers

SOURCE = Path("runs/response_study")
CAPABILITY = Path("runs/observer_selfplay_capability_cuda/audit.json")


def make_plan(smoke=False):
    prior = original_checked(SOURCE)
    assert read(SOURCE / "process.json")["status"] == "complete"
    parents = prior["parents"][:1] if smoke else prior["parents"]
    capability = read(CAPABILITY)
    assert capability["status"] == "passed"
    assert capability["production_source_sha256"] == provenance()["source_sha256"]
    assert (
        capability["matched_warmup_states"]["observer"]
        == capability["matched_warmup_states"]["signed"]
    )
    inputs = dict(prior["inputs"]) | capability["inputs"]
    paths = [
        Path(__file__),
        SOURCE / "protocol.json",
        CAPABILITY,
        Path("docs/OBSERVER_SELFPLAY_STUDY.rst"),
        Path("scripts/observer_selfplay_agent.py"),
        Path("tests/test_observer_selfplay_agent.py"),
        Path("tests/test_observer_selfplay_study.py"),
    ]
    for mode, row in capability["conditions"].items():
        assert row["config"]["device"] == "cuda" and row["counters"]["updates"] == 22
        assert row["counters"]["warmup_updates"] == 20
        for suffix, key in (
            ("final.pt", "checkpoint_sha256"),
            ("warmup.pt", "warmup_checkpoint_sha256"),
        ):
            path = CAPABILITY.parent / f"{mode}_{suffix}"
            assert sha(path) == row[key]
            paths.append(path)
    cases, controls = {}, {}
    for index, parent in enumerate(parents, 1):
        old_case = f"parent{index}_selfplay"
        original_training_audit(SOURCE, old_case, prior)
        folder = SOURCE / old_case
        paths += [
            folder / name
            for name in (
                "initial.pt",
                "final.pt",
                "training_complete.json",
                "training_audit.json",
            )
        ]
        audit_path = Path("runs/response_optimizer_audits") / f"{old_case}.json"
        audit = read(audit_path)
        assert audit["status"] == "passed" and audit["case"] == old_case
        assert all(sha(path) == expected for path, expected in audit["inputs"].items())
        paths.append(audit_path)
        controls[f"parent{index}"] = {
            "path": parent["path"],
            "format": "published_ppo_v1",
        }
        controls[old_case] = {
            "path": str(folder / "final.pt"),
            "format": "published_ppo_v1",
        }
        for arm in ("observer", "signed"):
            config = dict(prior["cases"][old_case]["config"])
            config.update(policy_returns=arm, returns_backend="torch")
            if smoke:
                config.update(
                    device="cpu",
                    envs=4,
                    horizon=16,
                    minibatch=32,
                    epochs=1,
                    critic_warmup_updates=1,
                    seed=26060001,
                )
            cases[f"parent{index}_{arm}"] = {
                "arm": arm,
                "parent": parent,
                "config": asdict(ObserverConfig(**config)),
                "original_initial": str(folder / "initial.pt"),
            }
    for path, expected in inputs.items():
        assert sha(path) == expected, path
    inputs.update({str(path): sha(path) for path in paths})
    return {
        "kind": "observer_selfplay_smoke" if smoke else "observer_selfplay_development",
        "source": provenance(),
        "inputs": inputs,
        "parents": parents,
        "cases": cases,
        "controls": controls,
        "updates": 2 if smoke else 520,
        "source_study": str(SOURCE),
        "source_training_cases": list(prior["cases"]),
        "source_protocol_sha256": sha(SOURCE / "protocol.json"),
        "evaluation": {
            "first_seed": 26061000 if smoke else 27000000,
            "pairs": 2 if smoke else 5000,
            "threads": 2,
            "opponents": prior["evaluation"]["opponents"],
        },
        "resources": {"trainers": 1, "evaluators": 1 if smoke else 3},
        "criterion": "All eight paired win lower95 bounds positive: observer minus parent and observer minus matched signed control, both parents against both opponents. Ordinary selfplay and points secondary. No ranking, extensions or promotion.",
        "scope": "Matched policy-return contrast shares per-observer critic fitting; historical selfplay is a separate full-intervention control. Development comparison,not final confirmation or a general unbiased finite-observation PPO theorem.",
    }


def train(base, case):
    p = checked(base)
    if not prior_ready(p, "train"):
        raise RuntimeError("Prior GPU training is not complete")
    spec, folder = p["cases"][case], base / case
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "initial.pt").exists():
        raise ValueError("Training already exists")
    trainer = ObserverTrainer.from_parent(
        ObserverConfig(**spec["config"]), spec["parent"]["path"]
    )
    trainer.save(folder / "initial.pt")
    if p["kind"] != "observer_selfplay_smoke":
        initial, _ = read_checkpoint(folder / "initial.pt")
        original, _ = read_checkpoint(spec["original_initial"])
        assert all(
            tensor_equal(initial[key], original[key])
            for key in ("actor", "env", "torch_rng", "cuda_rng")
        )
    start = time.perf_counter()
    with (folder / "metrics.jsonl").open("x", encoding="utf-8", newline="\n") as log:
        for _ in range(p["updates"]):
            row = (
                trainer.update()
                | trainer.counters
                | {"elapsed_seconds": time.perf_counter() - start}
            )
            log.write(json.dumps(row) + "\n")
            log.flush()
            if trainer.counters["updates"] == spec["config"]["critic_warmup_updates"]:
                trainer.save(folder / "warmup.pt")
                if spec["arm"] == "signed":
                    index = p["parents"].index(spec["parent"]) + 1
                    audit_warmup(base, p, [index])
            if trainer.counters["updates"] % 20 == 0:
                print(json.dumps(row), flush=True)
                checked(base)
                trainer.save(folder / "latest.pt")
    trainer.save(folder / "final.pt")
    checked(base)
    write(
        folder / "training_complete.json",
        {"checkpoint_sha256": sha(folder / "final.pt"), "counters": trainer.counters},
    )
    write(folder / "training_audit.json", audit_training(base, case, p))


def audit_training(base, case, p):
    spec, folder = p["cases"][case], base / case
    current, checkpoint_sha = read_checkpoint(folder / "final.pt")
    initial, _ = read_checkpoint(folder / "initial.pt")
    warmup, warmup_sha = read_checkpoint(folder / "warmup.pt")
    parent, parent_sha = read_checkpoint(spec["parent"]["path"])
    receipt = read(folder / "training_complete.json")
    c, counts = spec["config"], current["counters"]
    assert (
        checkpoint_sha == receipt["checkpoint_sha256"] and counts == receipt["counters"]
    )
    assert parent_sha == spec["parent"]["sha256"]
    assert current["config"] == initial["config"] == warmup["config"] == c
    assert (
        current["format"]
        == initial["format"]
        == warmup["format"]
        == "observer_selfplay_v1"
    )
    assert tensor_equal(initial["actor"], parent["actor"])
    for key, value in parent["value"].items():
        actual = initial["value"][key]
        assert torch.equal(actual[:, :560] if key == "0.weight" else actual, value)
    assert not initial["value"]["0.weight"][:, 560].any()
    assert (
        all(value == 0 for value in initial["counters"].values())
        and not initial["optimizer"]["state"]
    )
    for payload in (initial, warmup, current):
        assert payload["provenance"]["source_sha256"] == p["source"]["source_sha256"]
        assert payload["provenance"]["initialized_from"]["sha256"] == parent_sha
        for name, expected in payload["provenance"]["observer_sources"].items():
            assert expected == p["inputs"][str(Path("scripts") / name)]
    assert counts["updates"] == p["updates"]
    batch = c["envs"] * c["horizon"]
    assert counts["decisions"] == p["updates"] * batch
    assert counts["value_rows"] == 2 * counts["decisions"]
    assert counts["matches"] == int(current["env"][:, EPISODE].sum())
    assert counts["warmup_updates"] == c["critic_warmup_updates"]
    assert counts["policy_updates"] == p["updates"] - c["critic_warmup_updates"]
    logs = [
        json.loads(line)
        for line in (folder / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(logs) == p["updates"]
    for update, row in enumerate(logs, 1):
        assert row["updates"] == update and row["decisions"] == update * batch
        assert row["value_rows"] == 2 * row["decisions"]
        assert row["critic_only"] == (update <= c["critic_warmup_updates"])
        assert all(np.isfinite(v) for v in row.values() if isinstance(v, (int, float)))
    assert all(logs[-1][k] == value for k, value in counts.items())
    assert counts["warmup_decisions"] == c["critic_warmup_updates"] * batch
    assert counts["warmup_value_rows"] == 2 * counts["warmup_decisions"]
    for key in ("rounds", "matches"):
        assert counts["warmup_" + key] == logs[c["critic_warmup_updates"] - 1][key]
    assert warmup["counters"]["updates"] == c["critic_warmup_updates"]
    assert tensor_equal(warmup["actor"], initial["actor"])
    batches = ((batch + c["minibatch"] - 1) // c["minibatch"]) * c["epochs"]
    assert counts["optimizer_steps_value"] == p["updates"] * batches
    weights = current["actor"]["current"]["weights"]
    assert parameter_digest(weights) == current["actor"]["current"]["parameters_sha256"]
    tensors, steps = [], []
    for phase, name in ((0, "discard"), (1, "pick"), (3, "koi")):
        tensors += list(weights[phase].values())
        count = counts["optimizer_steps_" + name]
        assert 0 <= count <= counts["policy_updates"] * batches
        steps += [count] * len(weights[phase])
    tensors += list(current["value"].values())
    steps += [counts["optimizer_steps_value"]] * len(current["value"])
    groups, states = current["optimizer"]["param_groups"], current["optimizer"]["state"]
    assert (
        len(groups) == 2
        and groups[0]["lr"] == c["lr"]
        and groups[1]["lr"] == c["value_lr"]
    )
    ids = [idx for group in groups for idx in group["params"]]
    assert len(ids) == len(set(ids)) == len(tensors) and set(states) <= set(ids)
    for idx, tensor, expected in zip(ids, tensors, steps, strict=True):
        assert torch.isfinite(tensor).all()
        state = states.get(idx)
        assert (0 if state is None else int(state["step"])) == expected
        if state:
            for key in ("exp_avg", "exp_avg_sq"):
                assert (
                    state[key].shape == tensor.shape
                    and torch.isfinite(state[key]).all()
                )
    comparison = None
    if p["kind"] != "observer_selfplay_smoke":
        original, _ = read_checkpoint(spec["original_initial"])
        comparison = {
            key: tensor_equal(initial[key], original[key])
            for key in ("actor", "env", "torch_rng", "cuda_rng")
        }
        assert all(comparison.values())
    return {
        "status": "passed",
        "checkpoint_sha256": checkpoint_sha,
        "warmup_sha256": warmup_sha,
        "counters": counts,
        "initial_comparison_to_original_selfplay": comparison,
    }


def audit_warmup(base, p, indices=None):
    result = {}
    for index in range(1, len(p["parents"]) + 1) if indices is None else indices:
        left, left_sha = read_checkpoint(base / f"parent{index}_observer" / "warmup.pt")
        right, right_sha = read_checkpoint(base / f"parent{index}_signed" / "warmup.pt")
        for key in (
            "actor",
            "value",
            "optimizer",
            "env",
            "counters",
            "torch_rng",
            "cuda_rng",
        ):
            assert tensor_equal(left[key], right[key]), key
        result[f"parent{index}"] = {
            "status": "passed",
            "observer_sha256": left_sha,
            "signed_sha256": right_sha,
        }
    return result


def checked(base):
    p = read(base / "protocol.json")
    if read(base / "process.json")["protocol_sha256"] != sha(base / "protocol.json"):
        raise ValueError("Protocol changed")
    if provenance()["source_sha256"] != p["source"]["source_sha256"]:
        raise ValueError("Production changed")
    for path, digest in p["inputs"].items():
        if sha(path) != digest:
            raise ValueError(f"Study input changed: {path}")
    return p


def prior_ready(p, mode):
    if p["kind"] == "observer_selfplay_smoke":
        return True
    source = Path(p["source_study"])
    state = read(source / "process.json")
    if state["protocol_sha256"] != p["source_protocol_sha256"]:
        raise ValueError("Prior protocol changed")
    if state["status"] == "failed":
        raise RuntimeError("Prior study failed; inspect its retained evidence")
    if mode == "train":
        jobs = [j for j in state["jobs"] if j["mode"] == "train"]
        return (
            len(jobs) == len(p["source_training_cases"])
            and {j["case"] for j in jobs} == set(p["source_training_cases"])
            and all(j.get("returncode") == 0 for j in jobs)
        )
    if state["status"] != "complete":
        return False
    if sha(source / "analysis.json") != state["analysis_sha256"]:
        raise ValueError("Prior final analysis identity changed")
    return True


def candidate(base, case, p):
    return (
        base / case / "final.pt"
        if case in p["cases"]
        else Path(p["controls"][case]["path"])
    )


def evaluate(base, case, opponent):
    p = checked(base)
    if not prior_ready(p, "evaluate"):
        raise RuntimeError("Prior evaluation slots are still occupied")
    torch.set_num_threads(p["evaluation"]["threads"])
    path = candidate(base, case, p)
    if case in p["cases"]:
        assert (
            sha(path)
            == read(base / case / "training_complete.json")["checkpoint_sha256"]
        )
        a = ObserverAgent(path)
    else:
        a = make_agent(str(path), 0, True, "cpu")
    b = make_agent(opponent, 0, True, "cpu")
    folder = base / case
    folder.mkdir(parents=True, exist_ok=True)
    raw = folder / (opponent + ".jsonl")
    rows, start = [], time.perf_counter()
    with raw.open("x", encoding="utf-8", newline="\n") as stream:
        for seed in range(
            p["evaluation"]["first_seed"],
            p["evaluation"]["first_seed"] + p["evaluation"]["pairs"],
        ):
            left, right = duel(a, b, seed, "paper8"), duel(b, a, seed, "paper8")
            row = {
                "seed": seed,
                "seat0_score": left[0],
                "seat1_score": 1 - right[0],
                "seat0_point_difference": left[1],
                "seat1_point_difference": -right[1],
                "decisions": left[3] + right[3],
            }
            rows.append(row)
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            if len(rows) % 100 == 0:
                print(
                    case,
                    opponent,
                    len(rows),
                    round(time.perf_counter() - start, 1),
                    flush=True,
                )
    wins = [(r["seat0_score"] + r["seat1_score"]) / 2 for r in rows]
    points = [
        (r["seat0_point_difference"] + r["seat1_point_difference"]) / 2 for r in rows
    ]
    checked(base)
    write(
        folder / (opponent + ".json"),
        {
            "case": case,
            "candidate": str(path),
            "checkpoint_sha256": sha(path),
            "opponent": opponent,
            "pairs": len(rows),
            "matches": 2 * len(rows),
            "win_score": float(np.mean(wins)),
            "win_ci95": paired_interval(wins),
            "point_difference": float(np.mean(points)),
            "points_ci95": paired_interval(points),
            "decisions": sum(r["decisions"] for r in rows),
            "raw_sha256": sha(raw),
            "elapsed_seconds": time.perf_counter() - start,
        },
    )


def tasks(p):
    return [("train", case, None) for case in p["cases"]] + [
        ("evaluate", case, opponent)
        for case in list(p["controls"]) + list(p["cases"])
        for opponent in p["evaluation"]["opponents"]
    ]


def audit_schedule(p, process):
    jobs = process["jobs"]
    actual = [(j["mode"], j["case"], j["opponent"]) for j in jobs]
    assert len(actual) == len(tasks(p)) and set(actual) == set(tasks(p))
    previous = (
        []
        if p["kind"] == "observer_selfplay_smoke"
        else read(Path(p["source_study"]) / "process.json")["jobs"]
    )
    for job in jobs:
        assert job["returncode"] == 0 and job["started"] <= job["ended"]
        active = [
            j for j in previous + jobs if j["started"] <= job["started"] < j["ended"]
        ]
        assert sum(j["mode"] == "train" for j in active) <= p["resources"]["trainers"]
        assert (
            sum(j["mode"] == "evaluate" for j in active) <= p["resources"]["evaluators"]
        )
        if job["mode"] == "evaluate" and job["case"] in p["cases"]:
            trainer = next(
                j for j in jobs if j["mode"] == "train" and j["case"] == job["case"]
            )
            assert trainer["ended"] <= job["started"]


def analyze(base):
    p = checked(base)
    audit_schedule(p, read(base / "process.json"))
    training = {case: audit_training(base, case, p) for case in p["cases"]}
    reports, samples, contrasts = {}, {}, {}
    for case in list(p["controls"]) + list(p["cases"]):
        for opponent in p["evaluation"]["opponents"]:
            key = case + "/" + opponent
            report, wins, points = outcome(base, case, opponent, p)
            assert report["case"] == case and report["opponent"] == opponent
            assert (
                Path(report["candidate"]).resolve()
                == candidate(base, case, p).resolve()
            )
            reports[key], samples[key] = report, (wins, points)
    for index in range(1, len(p["parents"]) + 1):
        for opponent in p["evaluation"]["opponents"]:
            for a, b in (
                ("observer", "parent"),
                ("signed", "parent"),
                ("selfplay", "parent"),
                ("observer", "selfplay"),
                ("observer", "signed"),
                ("signed", "selfplay"),
            ):
                left = f"parent{index}_{a}"
                right = f"parent{index}" if b == "parent" else f"parent{index}_{b}"
                dw = (
                    samples[left + "/" + opponent][0]
                    - samples[right + "/" + opponent][0]
                )
                dp = (
                    samples[left + "/" + opponent][1]
                    - samples[right + "/" + opponent][1]
                )
                contrasts[left + "-" + right + "/" + opponent] = {
                    "win_change": float(dw.mean()),
                    "win_ci95": interval(dw),
                    "point_change": float(dp.mean()),
                    "point_ci95": interval(dp),
                }
    qualified = all(
        contrasts[f"parent{i}_observer-parent{i}{suffix}/{opponent}"]["win_ci95"][0] > 0
        for i in range(1, len(p["parents"]) + 1)
        for suffix in ("", "_signed")
        for opponent in p["evaluation"]["opponents"]
    )
    result = {
        "status": "passed",
        "kind": p["kind"],
        "protocol_sha256": sha(base / "protocol.json"),
        "training": training,
        "matched_warmup": audit_warmup(base, p),
        "outcomes": reports,
        "contrasts": contrasts,
        "replicated_observer_gain": qualified
        if p["kind"] == "observer_selfplay_development"
        else None,
        "scope": p["scope"],
    }
    write(base / "analysis.json", result)
    return result


def run(base, smoke=False):
    if base.exists():
        raise ValueError("Use a fresh study directory")
    p = make_plan(smoke)
    write(base / "protocol.json", p)
    process = {
        "status": "running",
        "protocol_sha256": sha(base / "protocol.json"),
        "jobs": [],
    }
    write(base / "process.json", process)
    pending, active, complete_training = tasks(p), [], set()
    try:
        while pending or active:
            for entry in active[:]:
                child, log, record = entry
                code = child.poll()
                if code is None:
                    continue
                log.close()
                record.update(returncode=code, ended=time.time())
                active.remove(entry)
                if code:
                    raise RuntimeError(
                        f"Worker failed: {record['mode']} {record['case']}, exit{code}"
                    )
                if record["mode"] == "train":
                    complete_training.add(record["case"])
                write(base / "process.json", process)
            for task in pending[:]:
                mode, case, opponent = task
                limit = p["resources"]["trainers" if mode == "train" else "evaluators"]
                if sum(
                    row[2]["mode"] == mode for row in active
                ) >= limit or not prior_ready(p, mode):
                    continue
                if (
                    mode == "evaluate"
                    and case in p["cases"]
                    and case not in complete_training
                ):
                    continue
                checked(base)
                folder = base / case
                folder.mkdir(parents=True, exist_ok=True)
                log = (
                    folder / (mode + ("_" + opponent if opponent else "") + ".log")
                ).open("x", encoding="utf-8")
                command = [
                    sys.executable,
                    "-m",
                    "scripts.run_observer_selfplay_study",
                    "--base",
                    str(base),
                    "--mode",
                    mode,
                    "--case",
                    case,
                ]
                if opponent:
                    command += ["--opponent", opponent]
                try:
                    child = subprocess.Popen(
                        command, stdout=log, stderr=subprocess.STDOUT
                    )
                except BaseException:
                    log.close()
                    raise
                record = {
                    "mode": mode,
                    "case": case,
                    "opponent": opponent,
                    "pid": child.pid,
                    "started": time.time(),
                    "command": command,
                }
                process["jobs"].append(record)
                active.append((child, log, record))
                pending.remove(task)
                write(base / "process.json", process)
            if pending or active:
                time.sleep(1)
        result = analyze(base)
        process.update(status="complete", analysis_sha256=sha(base / "analysis.json"))
        write(base / "process.json", process)
        print(
            json.dumps(
                {"status": "complete", "qualified": result["replicated_observer_gain"]}
            ),
            flush=True,
        )
    except BaseException as error:
        failures = stop_owned_workers(active)
        process.update(status="failed", error=repr(error), cleanup_failures=failures)
        write(base / "process.json", process)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base", type=Path, default=Path("runs/observer_selfplay_study")
    )
    parser.add_argument(
        "--mode", choices=("run", "train", "evaluate", "analyze"), default="run"
    )
    parser.add_argument("--case")
    parser.add_argument("--opponent")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.mode == "run":
        run(args.base, args.smoke)
    elif args.mode == "train":
        train(args.base, args.case)
    elif args.mode == "evaluate":
        evaluate(args.base, args.case, args.opponent)
    else:
        analyze(args.base)


if __name__ == "__main__":
    main()
