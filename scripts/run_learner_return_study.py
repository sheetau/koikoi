"""Fixed learner-decision versus signed-return development comparison."""

import argparse
import copy
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
from scripts.learner_response import LearnerAgent, LearnerConfig, LearnerTrainer
from scripts.published_response import ResponseAgent, ResponseConfig, ResponseTrainer
from scripts.run_arbitration_study import read, sha, write
from scripts.run_response_study import audit_training as original_training_audit
from scripts.run_response_study import checked as original_checked
from scripts.run_response_study import stop_owned_workers

SOURCE = Path("runs/response_study")
CAPABILITY = Path("runs/learner_response_capability_cuda/audit.json")


def make_plan(smoke=False, require_capability=False):
    prior = original_checked(SOURCE)
    parents = prior["parents"][:1] if smoke else prior["parents"]
    inputs = dict(prior["inputs"])
    paths = [
        Path(__file__),
        SOURCE / "protocol.json",
        Path("docs/LEARNER_RETURN_STUDY.rst"),
    ]
    paths += [
        Path("scripts") / name for name in ("learner_returns.py", "learner_response.py")
    ]
    cases, controls = {}, {}
    for index, parent in enumerate(parents, 1):
        old_case = f"parent{index}_response"
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
        optimizer_audit = Path("runs/response_optimizer_audits") / (old_case + ".json")
        audit = read(optimizer_audit)
        assert audit["status"] == "passed" and audit["case"] == old_case
        assert all(sha(path) == digest for path, digest in audit["inputs"].items())
        paths.append(optimizer_audit)
        controls[f"parent{index}"] = {
            "path": parent["path"],
            "format": "published_ppo_v1",
        }
        controls[old_case] = {
            "path": str(folder / "final.pt"),
            "format": "published_response_v1",
        }
        for arm in ("learner", "lambda1"):
            config = dict(prior["cases"][old_case]["config"])
            config.update(
                gae_lambda=0.95 if arm == "learner" else 1.0,
                returns_backend="torch"
                if arm == "learner"
                else config["returns_backend"],
            )
            if smoke:
                config.update(
                    device="cpu",
                    envs=4,
                    horizon=16,
                    minibatch=32,
                    epochs=1,
                    critic_warmup_updates=1,
                    seed=20297001,
                )
            cls = LearnerConfig if arm == "learner" else ResponseConfig
            cases[f"parent{index}_{arm}"] = {
                "arm": arm,
                "parent": parent,
                "config": asdict(cls(**config)),
                "original_initial": str(folder / "initial.pt"),
            }
    if require_capability and not smoke:
        capability = read(CAPABILITY)
        assert capability["checks_passed"] and capability["config"]["device"] == "cuda"
        assert capability["exact_restart_passed"]
        assert (
            capability["provenance"]["source_sha256"] == provenance()["source_sha256"]
        )
        from scripts.learner_response import source_identities

        assert capability["provenance"]["learner_sources"] == source_identities()
        assert sha(CAPABILITY.with_name("final.pt")) == capability["checkpoint_sha256"]
        assert (
            sha(CAPABILITY.with_name("restart_test.log"))
            == capability["restart_test_sha256"]
        )
        for path, digest in capability["verification_inputs"].items():
            assert sha(path) == digest
            paths.append(Path(path))
        payload, _ = read_checkpoint(CAPABILITY.with_name("final.pt"))
        assert payload["format"] == "learner_response_v1"
        assert payload["config"] == capability["config"]
        assert payload["counters"] == capability["counters"]
        assert (
            payload["counters"]["updates"] == 3
            and payload["counters"]["decisions"] == 49152
        )
        paths += [
            CAPABILITY,
            CAPABILITY.with_name("final.pt"),
            CAPABILITY.with_name("restart_test.log"),
        ]
    inputs.update({str(path): sha(path) for path in paths})
    return {
        "kind": "learner_return_smoke" if smoke else "learner_return_development",
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
            "first_seed": 20298000 if smoke else 24000000,
            "pairs": 2 if smoke else 5000,
            "threads": 2,
            "opponents": prior["evaluation"]["opponents"],
        },
        "resources": {"trainers": 1, "evaluators": 1 if smoke else 3},
        "criterion": "All eight paired win lower95 bounds positive: learner minus parent and learner minus signed-lambda0.95 response, both parents against both opponents. Lambda1 and points secondary. No ranking, extensions or promotion.",
        "scope": "Full intervention, including critic removal, loss normalization, clipping, tail exclusion and minibatch filtering. Development comparison, not independent final confirmation or a general unbiased-PPO theorem.",
    }


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
    if p["kind"] == "learner_return_smoke":
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


def train(base, case):
    p = checked(base)
    if not prior_ready(p, "train"):
        raise RuntimeError("Prior GPU training is not complete")
    spec = p["cases"][case]
    folder = base / case
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "initial.pt").exists():
        raise ValueError("Training already exists")
    cls, config = (
        (LearnerTrainer, LearnerConfig)
        if spec["arm"] == "learner"
        else (ResponseTrainer, ResponseConfig)
    )
    trainer = cls.from_parent(config(**spec["config"]), spec["parent"]["path"])
    trainer.save(folder / "initial.pt")
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
            if trainer.counters["updates"] % 20 == 0:
                print(json.dumps(row), flush=True)
                checked(base)
                trainer.save(folder / "latest.pt")
    trainer.save(folder / "final.pt")
    checked(base)
    write(
        folder / "training_complete.json",
        {
            "checkpoint_sha256": sha(folder / "final.pt"),
            "counters": trainer.counters,
            "opponent_work": trainer.opponent_work,
        },
    )
    write(folder / "training_audit.json", audit_training(base, case, p))


def audit_training(base, case, p):
    spec, folder = p["cases"][case], base / case
    current, digest = read_checkpoint(folder / "final.pt")
    initial, _ = read_checkpoint(folder / "initial.pt")
    parent, parent_sha = read_checkpoint(spec["parent"]["path"])
    receipt = read(folder / "training_complete.json")
    c, counts = spec["config"], current["counters"]
    assert digest == receipt["checkpoint_sha256"] and counts == receipt["counters"]
    assert parent_sha == spec["parent"]["sha256"]
    assert current["config"] == initial["config"] == c
    assert tensor_equal(initial["actor"], parent["actor"]) and tensor_equal(
        initial["value"], parent["value"]
    )
    assert (
        all(v == 0 for v in initial["counters"].values())
        and not initial["optimizer"]["state"]
    )
    assert current["provenance"]["source_sha256"] == p["source"]["source_sha256"]
    assert current["provenance"]["weight_initialization"]["sha256"] == parent_sha
    assert counts["updates"] == p["updates"]
    assert counts["decisions"] == p["updates"] * c["envs"] * c["horizon"]
    assert counts["matches"] == int(current["env"][:, EPISODE].sum())
    assert (
        counts["learner_decisions"] + counts["published_decisions"]
        == counts["decisions"]
    )
    assert counts["warmup_updates"] == c["critic_warmup_updates"]
    assert counts["policy_updates"] == p["updates"] - c["critic_warmup_updates"]
    assert tensor_equal(initial["opponents"], current["opponents"])
    assert current["opponent_work"] == receipt["opponent_work"]
    for name in ("decisions", "rounds", "matches"):
        assert (
            sum(row[name] for row in current["opponent_work"].values())
            == counts["published_" + name]
        )
    assert (
        counts["published_rounds"] == counts["rounds"]
        and counts["published_matches"] == counts["matches"]
    )
    logs = [
        json.loads(line)
        for line in (folder / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(logs) == p["updates"]
    for update, row in enumerate(logs, 1):
        assert (
            row["updates"] == update
            and row["decisions"] == update * c["envs"] * c["horizon"]
        )
        assert row["critic_only"] == (update <= c["critic_warmup_updates"])
        assert all(np.isfinite(v) for v in row.values() if isinstance(v, (int, float)))
    assert all(logs[-1][k] == v for k, v in counts.items())
    assert (
        counts["warmup_decisions"]
        == c["critic_warmup_updates"] * c["envs"] * c["horizon"]
    )
    for key in ("rounds", "matches"):
        assert counts["warmup_" + key] == (
            logs[c["critic_warmup_updates"] - 1][key]
            if c["critic_warmup_updates"]
            else 0
        )
    comparison = None
    if p["kind"] != "learner_return_smoke":
        original, _ = read_checkpoint(spec["original_initial"])
        for key in ("actor", "value", "env"):
            assert tensor_equal(initial[key], original[key])
        comparison = {
            "actor_value_environment_equal": True,
            "cpu_rng_equal": tensor_equal(initial["torch_rng"], original["torch_rng"]),
            "cuda_rng_equal": tensor_equal(initial["cuda_rng"], original["cuda_rng"]),
        }
    if spec["arm"] == "lambda1":
        adapted = copy.deepcopy(p)
        adapted["cases"][case]["arm"] = "response"
        result = original_training_audit(base, case, adapted)
    else:
        assert current["format"] == "learner_response_v1"
        for name, digest_source in current["provenance"]["learner_sources"].items():
            assert digest_source == p["inputs"][str(Path("scripts") / name)]
        for row in logs:
            assert (
                row["valid_learner_decisions"] + row["dropped_learner_tails"]
                == row["learner_decisions"]
            )
            assert 0 <= row["dropped_learner_tails"] <= row["updates"] * c["envs"]
        batches = (c["envs"] * c["horizon"] + c["minibatch"] - 1) // c["minibatch"]
        assert (
            counts["optimizer_steps_value"] + counts["skipped_empty_minibatches"]
            == p["updates"] * c["epochs"] * batches
        )
        weights = current["actor"]["current"]["weights"]
        assert (
            parameter_digest(weights)
            == current["actor"]["current"]["parameters_sha256"]
        )
        tensors, steps = [], []
        for phase, name in ((0, "discard"), (1, "pick"), (3, "koi")):
            tensors += list(weights[phase].values())
            count = counts["optimizer_steps_" + name]
            assert 0 <= count <= counts["policy_updates"] * c["epochs"] * batches
            steps += [count] * len(weights[phase])
        tensors += list(current["value"].values())
        steps += [counts["optimizer_steps_value"]] * len(current["value"])
        groups, states = (
            current["optimizer"]["param_groups"],
            current["optimizer"]["state"],
        )
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
        result = {
            "status": "passed",
            "checkpoint_sha256": digest,
            "counters": counts,
            "opponent_work": current["opponent_work"],
        }
    result["initial_comparison_to_original_response"] = comparison
    return result


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
        a = (
            LearnerAgent(path)
            if p["cases"][case]["arm"] == "learner"
            else ResponseAgent(path)
        )
    else:
        a = (
            ResponseAgent(path)
            if p["controls"][case]["format"] == "published_response_v1"
            else make_agent(str(path), 0, True, "cpu")
        )
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
        if p["kind"] == "learner_return_smoke"
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
                ("learner", "parent"),
                ("lambda1", "parent"),
                ("response", "parent"),
                ("learner", "response"),
                ("learner", "lambda1"),
                ("lambda1", "response"),
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
        contrasts[f"parent{i}_learner-parent{i}{suffix}/{opponent}"]["win_ci95"][0] > 0
        for i in range(1, len(p["parents"]) + 1)
        for suffix in ("", "_response")
        for opponent in p["evaluation"]["opponents"]
    )
    result = {
        "status": "passed",
        "kind": p["kind"],
        "protocol_sha256": sha(base / "protocol.json"),
        "training": training,
        "outcomes": reports,
        "contrasts": contrasts,
        "replicated_learner_gain": qualified
        if p["kind"] == "learner_return_development"
        else None,
        "scope": p["scope"],
    }
    write(base / "analysis.json", result)
    return result


def run(base, smoke=False):
    if base.exists():
        raise ValueError("Use a fresh study directory")
    p = make_plan(smoke, require_capability=not smoke)
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
                    "scripts.run_learner_return_study",
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
                {"status": "complete", "qualified": result["replicated_learner_gain"]}
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
    parser.add_argument("--base", type=Path, default=Path("runs/learner_return_study"))
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
