"""Fixed public-versus-masked seat-context continuation comparison.

Isolated adaptation of the frozen observer-study orchestration; its source is
retained unchanged while that study runs.
"""

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
from scripts.observer_selfplay_agent import ObserverAgent
from scripts.run_arbitration_study import read, sha, write
from scripts.run_observer_selfplay_study import (
    audit_training as original_training_audit,
)
from scripts.run_observer_selfplay_study import checked as original_checked
from scripts.run_response_study import stop_owned_workers
from scripts.seat_observer_agent import SeatObserverAgent
from scripts.seat_observer_selfplay import SeatObserverConfig, SeatObserverTrainer

SOURCE = Path("runs/observer_selfplay_study")
CAPABILITY = Path("runs/seat_observer_capability_cuda/audit.json")
CPU_CAPABILITY = Path("runs/seat_observer_device_smoke/audit.json")


def make_plan(smoke=False):
    prior = original_checked(SOURCE)
    if not smoke and read(SOURCE / "process.json")["status"] != "complete":
        raise ValueError("Current observer study must complete before the seat study")
    cap_path = CPU_CAPABILITY if smoke else CAPABILITY
    cap = read(cap_path)
    assert cap["status"] == "passed"
    assert cap["kind"] == ("cpu_smoke" if smoke else "cuda_capability")
    assert all(sha(p) == h for p, h in cap["inputs"].items())
    assert cap["total_physical_work"]["decisions"] == (2048 if smoke else 753664)
    assert cap["total_physical_work"]["value_rows"] == (4096 if smoke else 1507328)
    assert cap["initial_signatures"]["public"] == cap["initial_signatures"]["masked"]
    for key in ("actor", "env", "counters", "torch_rng", "cuda_rng"):
        assert (
            cap["warmup_signatures"]["public"][key]
            == cap["warmup_signatures"]["masked"][key]
        )
    inputs = prior["inputs"] | cap["inputs"]
    paths = [
        Path(__file__),
        SOURCE / "protocol.json",
        cap_path,
        Path("docs/SEAT_OBSERVER_STUDY.rst"),
        Path("scripts/seat_observer_agent.py"),
        Path("tests/test_seat_observer_agent.py"),
        Path("tests/test_seat_observer_study.py"),
    ]
    for context, row in cap["conditions"].items():
        assert row["counters"]["updates"] == (3 if smoke else 22)
        assert row["counters"]["warmup_updates"] == (1 if smoke else 20)
        assert len(row["optimizer_steps"]) == 96
        assert all(
            v
            == (
                (6 if smoke else 2112)
                if k.startswith("value.")
                else (4 if smoke else 192)
            )
            for k, v in row["optimizer_steps"].items()
        )
        for stage in ("initial", "warmup", "resume", "final"):
            path = cap_path.parent / f"{context}_{stage}.pt"
            assert sha(path) == row[f"{stage}_sha256"]
            paths.append(path)
    if not smoke:
        final_audit = read(SOURCE / "independent_audit.json")
        assert final_audit["status"] == "passed"
        assert all(sha(p) == h for p, h in final_audit["inputs"].items())
        inputs.update(final_audit["inputs"])
        paths += [
            SOURCE / name
            for name in ("process.json", "analysis.json", "independent_audit.json")
        ]
    parents = prior["parents"][:1] if smoke else prior["parents"]
    cases, controls = {}, {}
    for index, parent in enumerate(parents, 1):
        old_case = f"parent{index}_observer"
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
        optimizer = Path("runs/observer_selfplay_optimizer_audits") / f"{old_case}.json"
        audit = read(optimizer)
        assert audit["status"] == "passed" and audit["case"] == old_case
        assert all(sha(p) == h for p, h in audit["inputs"].items())
        paths.append(optimizer)
        controls[f"parent{index}"] = {
            "path": parent["path"],
            "format": "published_ppo_v1",
        }
        controls[old_case] = {
            "path": str(folder / "final.pt"),
            "format": "observer_selfplay_v1",
        }
        for context in ("public", "masked"):
            config = dict(prior["cases"][old_case]["config"])
            config.update(seat_context=context, policy_returns="observer")
            if smoke:
                config.update(
                    device="cpu",
                    envs=4,
                    horizon=16,
                    minibatch=32,
                    epochs=1,
                    critic_warmup_updates=1,
                    seed=26096001,
                )
            cases[f"parent{index}_{context}"] = {
                "arm": context,
                "parent": parent,
                "config": asdict(SeatObserverConfig(**config)),
                "observer_initial": str(folder / "initial.pt"),
            }
    assert all(sha(p) == h for p, h in inputs.items())
    inputs.update({str(path): sha(path) for path in paths})
    return {
        "kind": "seat_observer_smoke" if smoke else "seat_observer_development",
        "source": provenance(),
        "inputs": inputs,
        "parents": parents,
        "cases": cases,
        "controls": controls,
        "updates": 2 if smoke else 520,
        "source_study": str(SOURCE),
        "source_protocol_sha256": sha(SOURCE / "protocol.json"),
        "evaluation": {
            "first_seed": 26097000 if smoke else 28000000,
            "pairs": 2 if smoke else 5000,
            "threads": 2,
            "opponents": prior["evaluation"]["opponents"],
        },
        "resources": {"trainers": 1, "evaluators": 1 if smoke else 3},
        "criterion": "All eight public-minus-parent/public-minus-masked win lower95 bounds positive; historical observer and points secondary. No selection,extension or promotion.",
        "scope": "Matched public-seat information contrast; prior561 observer control is a separate architecture comparison. Development evidence,not final confirmation.",
    }


def compare_initial(initial, original):
    comparison = {
        key: tensor_equal(initial[key], original[key])
        for key in ("actor", "env", "torch_rng", "cuda_rng")
    }
    comparison["padded_value"] = all(
        torch.equal(
            initial["value"][key][:, :561]
            if key == "0.weight"
            else initial["value"][key],
            value,
        )
        for key, value in original["value"].items()
    ) and not bool(initial["value"]["0.weight"][:, 561].any())
    assert all(comparison.values())
    return comparison


def train(base, case):
    p = checked(base)
    if not prior_ready(p, "train"):
        raise RuntimeError("Prior GPU training is not complete")
    spec, folder = p["cases"][case], base / case
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "initial.pt").exists():
        raise ValueError("Training already exists")
    trainer = SeatObserverTrainer.from_parent(
        SeatObserverConfig(**spec["config"]), spec["parent"]["path"]
    )
    trainer.save(folder / "initial.pt")
    if p["kind"] != "seat_observer_smoke":
        initial, _ = read_checkpoint(folder / "initial.pt")
        original, _ = read_checkpoint(spec["observer_initial"])
        compare_initial(initial, original)
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
                if spec["arm"] == "masked":
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
        == "seat_observer_selfplay_v1"
    )
    assert tensor_equal(initial["actor"], parent["actor"])
    for key, value in parent["value"].items():
        actual = initial["value"][key]
        assert torch.equal(actual[:, :560] if key == "0.weight" else actual, value)
    assert initial["value"]["0.weight"].shape[1] == 562
    assert not initial["value"]["0.weight"][:, 560:].any()
    assert (
        all(value == 0 for value in initial["counters"].values())
        and not initial["optimizer"]["state"]
    )
    for payload in (initial, warmup, current):
        assert payload["provenance"]["source_sha256"] == p["source"]["source_sha256"]
        assert payload["provenance"]["initialized_from"]["sha256"] == parent_sha
        assert (
            payload["provenance"]["seat_observer_source"]
            == p["inputs"][str(Path("scripts/seat_observer_selfplay.py"))]
        )
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
    if p["kind"] != "seat_observer_smoke":
        original, _ = read_checkpoint(spec["observer_initial"])
        comparison = compare_initial(initial, original)
    return {
        "status": "passed",
        "checkpoint_sha256": checkpoint_sha,
        "warmup_sha256": warmup_sha,
        "counters": counts,
        "initial_comparison_to_archived_observer": comparison,
    }


def audit_warmup(base, p, indices=None):
    result = {}
    for index in range(1, len(p["parents"]) + 1) if indices is None else indices:
        left, left_sha = read_checkpoint(base / f"parent{index}_public" / "warmup.pt")
        right, right_sha = read_checkpoint(base / f"parent{index}_masked" / "warmup.pt")
        for key in ("actor", "env", "counters", "torch_rng", "cuda_rng"):
            assert tensor_equal(left[key], right[key]), key
        initial = [
            read_checkpoint(base / f"parent{index}_{c}" / "initial.pt")[0]
            for c in ("public", "masked")
        ]
        for key in (
            "actor",
            "value",
            "optimizer",
            "env",
            "counters",
            "torch_rng",
            "cuda_rng",
        ):
            assert tensor_equal(initial[0][key], initial[1][key]), key
        result[f"parent{index}"] = {
            "status": "passed",
            "public_sha256": left_sha,
            "masked_sha256": right_sha,
            "scope": "Initial full state and warmup actor/environment/RNG equality; critic differences are the treatment",
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
    if p["kind"] == "seat_observer_smoke" and mode == "train":
        return True
    source = Path(p["source_study"])
    state = read(source / "process.json")
    if state["protocol_sha256"] != p["source_protocol_sha256"]:
        raise ValueError("Prior protocol changed")
    if state["status"] == "failed":
        raise RuntimeError("Prior study failed; inspect its retained evidence")
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
        a = SeatObserverAgent(path)
    elif p["controls"][case]["format"] == "observer_selfplay_v1":
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
    previous = read(Path(p["source_study"]) / "process.json")["jobs"]
    for job in jobs:
        assert job["returncode"] == 0 and job["started"] <= job["ended"]
        active_prior = [
            j for j in previous if j["started"] <= job["started"] < j["ended"]
        ]
        active_own = [j for j in jobs if j["started"] <= job["started"] < j["ended"]]
        assert (
            sum(j["mode"] == "train" for j in active_own) <= p["resources"]["trainers"]
        )
        gpu_own = sum(
            j["mode"] == "train" and p["cases"][j["case"]]["config"]["device"] == "cuda"
            for j in active_own
        )
        assert gpu_own + sum(j["mode"] == "train" for j in active_prior) <= 1
        assert (
            sum(j["mode"] == "evaluate" for j in active_own)
            <= p["resources"]["evaluators"]
        )
        assert sum(j["mode"] == "evaluate" for j in active_prior + active_own) <= 3
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
                ("public", "parent"),
                ("masked", "parent"),
                ("observer", "parent"),
                ("public", "observer"),
                ("public", "masked"),
                ("masked", "observer"),
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
        contrasts[f"parent{i}_public-parent{i}{suffix}/{opponent}"]["win_ci95"][0] > 0
        for i in range(1, len(p["parents"]) + 1)
        for suffix in ("", "_masked")
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
        "replicated_seat_gain": qualified
        if p["kind"] == "seat_observer_development"
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
                    "scripts.run_seat_observer_study",
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
                {"status": "complete", "qualified": result["replicated_seat_gain"]}
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
    parser.add_argument("--base", type=Path, default=Path("runs/seat_observer_study"))
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
