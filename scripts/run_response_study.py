"""Fixed two-parent response-versus-self-play development study."""

import argparse
import json
import os
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
from koikoi.published_train import PublishedConfig, PublishedTrainer
from koikoi.train import provenance
from koikoi.upstream import UPSTREAM_PATH, load_source
from koikoi.upstream_batch import parameter_digest
from scripts.analyze_arbitration_study import interval, outcome, tensor_equal
from scripts.published_response import ResponseAgent, ResponseConfig, ResponseTrainer
from scripts.run_arbitration_study import PARENTS, read, sha, write


def make_plan(smoke=False):
    load_source()
    parents = [{"path": path, "sha256": digest} for path, digest in PARENTS]
    if smoke:
        parents = parents[:1]
    inputs = [Path(__file__)] + [
        Path("scripts") / f"{name}.py"
        for name in (
            "published_response",
            "run_arbitration_study",
            "analyze_arbitration_study",
        )
    ]
    inputs += list(UPSTREAM_PATH.rglob("*.py"))
    inputs += [
        UPSTREAM_PATH / f"model_agent/{phase}_rl_{variant}.pt"
        for phase in ("discard", "pick", "koikoi")
        for variant in ("wp", "point")
    ]
    cases = {}
    for index, parent in enumerate(parents, 1):
        if sha(parent["path"]) != parent["sha256"]:
            raise ValueError("Parent changed")
        inputs.append(Path(parent["path"]))
        for arm in ("response", "selfplay"):
            config_class = ResponseConfig if arm == "response" else PublishedConfig
            options = {"seed": 19910001 + (index - 1) * 100000}
            if smoke:
                options.update(
                    envs=4, horizon=16, minibatch=32, epochs=1, critic_warmup_updates=1
                )
            cases[f"parent{index}_{arm}"] = {
                "arm": arm,
                "parent": parent,
                "config": asdict(config_class(**options)),
            }
    return {
        "kind": "response_smoke" if smoke else "response_development",
        "source": provenance(),
        "inputs": {str(path): sha(path) for path in inputs},
        "parents": parents,
        "cases": cases,
        "updates": 2 if smoke else 520,
        "evaluation": {
            "first_seed": 19873000 if smoke else 23000000,
            "pairs": 2 if smoke else 5000,
            "threads": 2,
            "opponents": ["rl-wp-assisted", "rl-point-assisted"],
        },
        "resources": {"trainers": 1, "evaluators": 3},
        "criterion": "Both parents: positive paired win-change lower95 bound versus unchanged parent and versus fresh self-play control, against each of WP and Point. Retain every endpoint; no intermediate ranking, sample extension or automatic promotion.",
        "scope": "Development study, not untouched final confirmation. Fixed mixture training is established methodology. Same simulator budget is not equal learner sample count; role critic and clipping also differ. Report those differences. Both parents share the same evaluation deals.",
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


def train(base, case):
    p = checked(base)
    spec = p["cases"][case]
    folder = base / case
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "initial.pt").exists():
        raise ValueError("Training already exists")
    if spec["arm"] == "response":
        trainer = ResponseTrainer.from_parent(
            ResponseConfig(**spec["config"]), spec["parent"]["path"]
        )
    else:
        parent, _ = read_checkpoint(spec["parent"]["path"])
        trainer = PublishedTrainer(PublishedConfig(**spec["config"]), parent["actor"])
        trainer.model.value.load_state_dict(parent["value"])
        trainer.provenance["weight_initialization"] = spec["parent"]
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
            "opponent_work": getattr(trainer, "opponent_work", {}),
            "elapsed_seconds": time.perf_counter() - start,
        },
    )
    write(folder / "training_audit.json", audit_training(base, case, p))


def audit_training(base, case, p):
    spec, folder = p["cases"][case], base / case
    current, digest = read_checkpoint(folder / "final.pt")
    initial, _ = read_checkpoint(folder / "initial.pt")
    parent, parent_digest = read_checkpoint(spec["parent"]["path"])
    receipt = read(folder / "training_complete.json")
    c, counts = spec["config"], current["counters"]
    assert digest == receipt["checkpoint_sha256"]
    assert parent_digest == spec["parent"]["sha256"]
    assert current["config"] == initial["config"] == c
    assert counts == receipt["counters"]
    assert tensor_equal(initial["actor"], parent["actor"])
    assert tensor_equal(initial["value"], parent["value"])
    assert all(value == 0 for value in initial["counters"].values())
    assert counts["updates"] == p["updates"]
    assert counts["decisions"] == p["updates"] * c["envs"] * c["horizon"]
    assert int(current["env"][:, EPISODE].sum()) == counts["matches"]
    assert counts["warmup_updates"] == c["critic_warmup_updates"]
    assert counts["policy_updates"] == p["updates"] - c["critic_warmup_updates"]
    assert current["provenance"]["source_sha256"] == p["source"]["source_sha256"]
    assert current["provenance"]["weight_initialization"]["sha256"] == parent_digest
    assert current["actor"]["initialization"] == parent["actor"]["initialization"]
    weights = current["actor"]["current"]["weights"]
    assert parameter_digest(weights) == current["actor"]["current"]["parameters_sha256"]
    logs = [
        json.loads(line)
        for line in (folder / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(logs) == p["updates"]
    for update, row in enumerate(logs, 1):
        assert row["updates"] == update
        assert row["decisions"] == update * c["envs"] * c["horizon"]
        assert row["critic_only"] == (update <= c["critic_warmup_updates"])
        assert all(
            np.isfinite(value)
            for value in row.values()
            if isinstance(value, (float, int))
        )
    assert all(logs[-1][key] == value for key, value in counts.items())
    assert (
        counts["warmup_decisions"]
        == c["critic_warmup_updates"] * c["envs"] * c["horizon"]
    )
    step = c["critic_warmup_updates"] - 1
    for name in ("rounds", "matches"):
        expected = 0 if step < 0 else logs[step][name]
        assert counts["warmup_" + name] == expected
    tensors = [t for phase in (0, 1, 3) for t in weights[phase].values()] + list(
        current["value"].values()
    )
    expected_steps = []
    for phase, name in ((0, "discard"), (1, "pick"), (3, "koi")):
        expected_steps += [counts["optimizer_steps_" + name]] * len(weights[phase])
    expected_steps += [counts["optimizer_steps_value"]] * len(current["value"])
    if spec["arm"] == "response":
        assert current["format"] == "published_response_v1"
        assert tensor_equal(initial["opponents"], current["opponents"])
        assert (
            current["provenance"]["response_source_sha256"]
            == p["inputs"][str(Path("scripts/published_response.py"))]
        )
        assert (
            counts["learner_decisions"] + counts["published_decisions"]
            == counts["decisions"]
        )
        assert current["opponent_work"] == receipt["opponent_work"]
        for name in ("decisions", "rounds", "matches"):
            assert (
                sum(row[name] for row in current["opponent_work"].values())
                == counts["published_" + name]
            )
        assert counts["published_rounds"] == counts["rounds"]
        assert counts["published_matches"] == counts["matches"]
        tensors += list(current["opponent_value"].values())
        expected_steps += [counts["optimizer_steps_opponent_value"]] * len(
            current["opponent_value"]
        )
    else:
        assert current["format"] == "published_ppo_v1"
    groups, states = current["optimizer"]["param_groups"], current["optimizer"]["state"]
    ids = [index for group in groups for index in group["params"]]
    assert len(ids) == len(tensors) == len(expected_steps) and len(set(ids)) == len(ids)
    assert set(states) <= set(ids)
    assert len(groups) == (3 if spec["arm"] == "response" else 2)
    assert groups[0]["lr"] == c["lr"]
    assert all(group["lr"] == c["value_lr"] for group in groups[1:])
    minibatches = (c["envs"] * c["horizon"] + c["minibatch"] - 1) // c["minibatch"]
    assert counts["optimizer_steps_value"] == p["updates"] * c["epochs"] * minibatches
    for index, tensor, expected in zip(ids, tensors, expected_steps):
        assert torch.isfinite(tensor).all()
        state = states.get(index)
        assert (0 if state is None else int(state["step"])) == expected
        if state:
            for key in ("exp_avg", "exp_avg_sq"):
                assert (
                    state[key].shape == tensor.shape
                    and torch.isfinite(state[key]).all()
                )
    return {
        "status": "passed",
        "checkpoint_sha256": digest,
        "counters": counts,
        "opponent_work": receipt["opponent_work"],
    }


def evaluate(base, case, opponent):
    p = checked(base)
    torch.set_num_threads(p["evaluation"]["threads"])
    if case in p["cases"]:
        path = base / case / "final.pt"
        if (
            sha(path)
            != read(base / case / "training_complete.json")["checkpoint_sha256"]
        ):
            raise ValueError("Endpoint changed")
        a = (
            ResponseAgent(path)
            if p["cases"][case]["arm"] == "response"
            else make_agent(str(path), 0, True, "cpu")
        )
    else:
        path = Path(p["parents"][int(case.removeprefix("parent")) - 1]["path"])
        a = make_agent(str(path), 0, True, "cpu")
    b = make_agent(opponent, 0, True, "cpu")
    folder = base / case
    folder.mkdir(parents=True, exist_ok=True)
    raw = folder / (opponent + ".jsonl")
    rows = []
    start = time.perf_counter()
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
            "matches": len(rows) * 2,
            "win_score": float(np.mean(wins)),
            "win_ci95": paired_interval(wins),
            "point_difference": float(np.mean(points)),
            "points_ci95": paired_interval(points),
            "decisions": sum(r["decisions"] for r in rows),
            "raw_sha256": sha(raw),
            "elapsed_seconds": time.perf_counter() - start,
        },
    )


def analyze(base):
    p = checked(base)
    audit_schedule(p, read(base / "process.json"))
    training = {case: audit_training(base, case, p) for case in p["cases"]}
    reports, samples = {}, {}
    for case in [f"parent{i}" for i in range(1, len(p["parents"]) + 1)] + list(
        p["cases"]
    ):
        for opponent in p["evaluation"]["opponents"]:
            key = case + "/" + opponent
            report, wins, points = outcome(base, case, opponent, p)
            assert report["case"] == case and report["opponent"] == opponent
            expected = (
                base / case / "final.pt"
                if case in p["cases"]
                else Path(p["parents"][int(case.removeprefix("parent")) - 1]["path"])
            )
            assert Path(report["candidate"]).resolve() == expected.resolve()
            reports[key], samples[key] = report, (wins, points)
    contrasts = {}
    for i in range(1, len(p["parents"]) + 1):
        for opponent in p["evaluation"]["opponents"]:
            for arm, control in (
                ("response", "parent"),
                ("selfplay", "parent"),
                ("response", "selfplay"),
            ):
                a = f"parent{i}_{arm}"
                b = f"parent{i}" if control == "parent" else f"parent{i}_{control}"
                dw = samples[a + "/" + opponent][0] - samples[b + "/" + opponent][0]
                dp = samples[a + "/" + opponent][1] - samples[b + "/" + opponent][1]
                contrasts[a + "-" + b + "/" + opponent] = {
                    "win_change": float(dw.mean()),
                    "win_ci95": interval(dw),
                    "point_change": float(dp.mean()),
                    "point_ci95": interval(dp),
                }
    qualification = all(
        contrasts[f"parent{i}_response-parent{i}{suffix}/{opponent}"]["win_ci95"][0] > 0
        for i in range(1, len(p["parents"]) + 1)
        for suffix in ("", "_selfplay")
        for opponent in p["evaluation"]["opponents"]
    )
    result = {
        "status": "passed",
        "kind": p["kind"],
        "protocol_sha256": sha(base / "protocol.json"),
        "training": training,
        "outcomes": reports,
        "contrasts": contrasts,
        "replicated_response_gain": qualification
        if p["kind"] == "response_development"
        else None,
        "scope": p["scope"],
    }
    write(base / "analysis.json", result)
    return result


def audit_schedule(p, process):
    jobs = process["jobs"]
    expected = {("train", case, None) for case in p["cases"]}
    for case in [f"parent{i}" for i in range(1, len(p["parents"]) + 1)] + list(
        p["cases"]
    ):
        expected.update(
            ("evaluate", case, opponent) for opponent in p["evaluation"]["opponents"]
        )
    actual = [(job["mode"], job["case"], job["opponent"]) for job in jobs]
    assert len(actual) == len(expected) and set(actual) == expected
    for job in jobs:
        assert job["returncode"] == 0 and job["started"] <= job["ended"]
        concurrent = [
            row for row in jobs if row["started"] <= job["started"] < row["ended"]
        ]
        assert (
            sum(row["mode"] == "train" for row in concurrent)
            <= p["resources"]["trainers"]
        )
        assert (
            sum(row["mode"] == "evaluate" for row in concurrent)
            <= p["resources"]["evaluators"]
        )
        if job["mode"] == "evaluate" and job["case"] in p["cases"]:
            trainer = next(
                row
                for row in jobs
                if row["mode"] == "train" and row["case"] == job["case"]
            )
            assert trainer["ended"] <= job["started"]


def stop_owned_workers(active):
    failures = []
    for child, log, record in active:
        try:
            if child.poll() is None:
                if os.name == "nt":
                    subprocess.run(
                        ["taskkill", "/PID", str(child.pid), "/T", "/F"],
                        capture_output=True,
                        check=False,
                    )
                else:
                    child.terminate()
                child.wait(timeout=30)
        except (OSError, subprocess.TimeoutExpired) as error:
            failures.append({"pid": child.pid, "error": repr(error)})
        finally:
            log.close()
            record.update(
                returncode=child.poll(), ended=time.time(), stopped_after_failure=True
            )
    return failures


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
    pending = [("train", case, None) for case in p["cases"]]
    pending += [
        ("evaluate", f"parent{i}", opponent)
        for i in range(1, len(p["parents"]) + 1)
        for opponent in p["evaluation"]["opponents"]
    ]
    pending += [
        ("evaluate", case, opponent)
        for case in p["cases"]
        for opponent in p["evaluation"]["opponents"]
    ]
    active = []
    complete_training = set()
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
                if sum(row[2]["mode"] == mode for row in active) >= limit:
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
                    "scripts.run_response_study",
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
            if active:
                time.sleep(1)
        result = analyze(base)
        process["status"] = "complete"
        process["analysis_sha256"] = sha(base / "analysis.json")
        write(base / "process.json", process)
        print(
            json.dumps(
                {"status": "complete", "qualified": result["replicated_response_gain"]}
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
    parser.add_argument("--base", type=Path, default=Path("runs/response_study"))
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
