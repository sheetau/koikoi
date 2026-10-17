"""Fixed fresh-seed temperature/LR comparison, with separate public qualification.

One GPU training worker and at most two CPU evaluation workers. Only final
update520 models are evaluated. Selection rules are fixed before launch.
"""

import argparse
import os
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from koikoi.compare import compare_reports
from koikoi.published_train import PublishedConfig
from koikoi.train import provenance
from koikoi.upstream import UPSTREAM_PATH, UPSTREAM_REV, load_source
from scripts.run_pretrained_ppo_screen import INITIAL, SOURCE, TEACHER, read, sha, write

BASE = Path("runs/cold_ppo_screen")
SEEDS = (6710001, 6720001)
FIRST, PAIRS = 7200000, 5000
INITIAL_PATH = Path("runs/pretrained_ppo_screen/initial/initial.pt")
INITIAL_SHA = "290e301702428a1762ffec2004ec9e9fafbb35a7d68a79e849c8e47a1126893e"
HELPER_SHA = "827ecd17d5728b4ac2e6f1b8a60f1634277a7e133ed4ed930c693014c9e23f7c"


def preflight():
    current = provenance()
    assert current["source_sha256"] == SOURCE and torch.cuda.is_available()
    assert sha(INITIAL_PATH) == INITIAL_SHA
    assert sha("scripts/run_pretrained_ppo_screen.py") == HELPER_SHA
    load_source()
    for name, digest in TEACHER.items():
        assert sha(UPSTREAM_PATH / f"model_agent/{name}_rl_wp.pt") == digest
    dependencies = {
        "runs/pretrained_ppo_screen/independent_audit.json": "643432440c451836559ab398d45a095f21043c4f04b60e26ec470dfa09d8c982",
        "runs/pretrained_policy_diagnostic/audit.json": "21aac38fb9d5c45715fbcccf8e15eafbafa78a15e9d8c4583c9cca153721fdc5",
        "runs/pretrained_policy_diagnostic/tie_audit.json": "c3cba0c0b4e708f4e4c7852491b0c65ffa4d3449c52c3dd4337c7e022cfd9f89",
    }
    for path, digest in dependencies.items():
        assert sha(path) == digest and read(path)["status"] == "passed"
    scan = Path("runs/cold_ppo_seed_range_check.json")
    assert read(scan)["range"] == [FIRST, FIRST + PAIRS - 1]
    assert read(scan)["overlapping_report_ranges"] == []
    scores_path = Path("runs/pretrained_policy_diagnostic/scores.npz")
    assert (
        sha(scores_path)
        == "b34b56d132bb691eb1cc1dfe85d6f88e90b941548e4694a778ed0724ee2859c8"
    )
    with np.load(scores_path, allow_pickle=False) as raw:
        for device in ("cpu", "cuda"):
            scores = np.where(raw["mask"], raw[device + "_initial"], -np.inf)
            for temperature in (0.05, 0.01):
                np.testing.assert_array_equal(
                    (scores / np.float32(temperature)).argmax(1), scores.argmax(1)
                )
    return {
        "provenance": current,
        "script_sha256": sha(__file__),
        "helper_sha256": HELPER_SHA,
        "upstream_revision": UPSTREAM_REV,
        "comparator_hashes": TEACHER,
        "initial_checkpoint": str(INITIAL_PATH),
        "initial_checkpoint_sha256": INITIAL_SHA,
        "initial_parameters_sha256": INITIAL,
        "dependency_hashes": dependencies,
        "seed_range_check_sha256": sha(scan),
        "initial_temperature_argmax_checks": 13029,
        "cases": {
            f"seed{seed}_{arm}": asdict(
                PublishedConfig(seed=seed, score_temperature=t, lr=lr)
            )
            for seed in SEEDS
            for arm, t, lr in (("warm", 0.05, 3e-6), ("cold", 0.01, 6e-7))
        },
        "updates": 520,
        "warmup_updates": 20,
        "policy_updates": 500,
        "decisions_per_case": 8519680,
        "warmup_decisions_per_case": 327680,
        "evaluation": {
            "pairs": PAIRS,
            "first_seed": FIRST,
            "device": "cpu",
            "opponent": "rl-wp-assisted",
            "profile": "paper8",
            "endgame": True,
            "deterministic": True,
            "search": "none",
        },
        "parallelism": {"gpu": 1, "cpu": 2},
        "timing_note": "Evaluations may share CPU resources with training/evaluation workers; reported decision latencies are not isolated benchmarks. Each evaluator uses two CPU threads and independently seeded deterministic pairs.",
        "intervention": "ColdT0.01/actorLR6e-7 versus warmT0.05/actorLR3e-6; everything else fixed. LR rescaling is heuristic; Adam, clipping, entropy and state visitation need not match. This is a bundled learning intervention, not an isolated causal test of exploration.",
        "criteria": {
            "temperature_benefit": "BOTH new training seeds: cold-minus-warm paired win-score95% CI lower bound >0.",
            "public_qualification": "Separately for each arm, BOTH new training seeds must have absolute win-score95% CI lower bound >0.5 against strict CPU WP. Points are secondary. Two arms' lower-tail nominal tests are each2.5%; no claim that bootstrap coverage or independence is exact.",
            "selection": "Prefer cold if publicly qualified AND replicated paired benefit; otherwise choose warm if publicly qualified; otherwise cold if publicly qualified; otherwise none. Choosing cold on absolute qualification alone does not establish benefit versus warm. Canonical checkpoint is always seed6710001 of the selected arm, never the better-scoring seed. Independent artifact/raw-record audit must pass first. This is development qualification only; untouched final confirmation remains separate.",
        },
        "stop_rule": "Exactlyfour fresh training runs and5000 new duplicate pairs for initial plus each final endpoint. No old checkpoint continuation, added old-endpoint evaluation, intermediate model selection, hyperparameter changes or extension based on interim outcomes. Initial outcome compatibility must pass before final evaluations. Abort pipeline on any failed job/nonfinite metric/identity drift. All final seeds100M+ remain unused.",
    }


def initial_gate(report):
    assert report["pairs"] == PAIRS and len(report["records"]) == PAIRS
    assert (
        report["draw_adjusted_win_rate"] == 0.5 and report["mean_point_difference"] == 0
    )
    assert [r["seed"] for r in report["records"]] == list(range(FIRST, FIRST + PAIRS))
    for row in report["records"]:
        assert row["seat0_score"] + row["seat1_score"] == 1
        assert row["seat0_point_difference"] + row["seat1_point_difference"] == 0


def decision(endpoints, paired):
    qualified = {
        arm: all(endpoints[f"seed{seed}_{arm}"]["win_ci95"][0] > 0.5 for seed in SEEDS)
        for arm in ("warm", "cold")
    }
    benefit = all(paired[str(seed)]["score_difference_ci95"][0] > 0 for seed in SEEDS)
    selected = (
        "cold"
        if qualified["cold"] and benefit
        else "warm"
        if qualified["warm"]
        else "cold"
        if qualified["cold"]
        else None
    )
    return {
        "public_qualified": qualified,
        "cold_paired_benefit": benefit,
        "selected_arm": selected,
        "canonical_case": None if selected is None else f"seed{SEEDS[0]}_{selected}",
    }


def worker(job):
    p = read(BASE / "protocol.json")
    assert p["script_sha256"] == sha(__file__) and p["helper_sha256"] == sha(
        "scripts/run_pretrained_ppo_screen.py"
    )
    assert provenance()["source_sha256"] == SOURCE
    for name, digest in TEACHER.items():
        assert sha(UPSTREAM_PATH / f"model_agent/{name}_rl_wp.pt") == digest
    stage, case = job.split(":")
    folder = BASE / case
    if stage == "train":
        from koikoi.published_train import main

        assert read(folder / "config.json") == p["cases"][case]
        sys.argv = [
            "koikoi.published_train",
            "--config",
            str(folder / "config.json"),
            "--run",
            str(folder),
            "--updates",
            "520",
            "--checkpoint-every",
            "20",
        ]
    else:
        from koikoi.evaluate import main

        assert stage == "evaluate"
        if case == "initial":
            checkpoint = INITIAL_PATH
            assert sha(checkpoint) == INITIAL_SHA
        else:
            initial_gate(read(BASE / "initial/evaluation.json"))
            checkpoint = folder / "update_00000520.pt"
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            assert payload["config"] == p["cases"][case]
            assert payload["provenance"]["source_sha256"] == SOURCE
            assert payload["actor"]["initialization"]["parameters_sha256"] == INITIAL
            assert (
                payload["counters"]["updates"] == 520
                and payload["counters"]["policy_updates"] == 500
            )
            assert payload["counters"]["decisions"] == 8519680
        sys.argv = [
            "koikoi.evaluate",
            "--agent",
            str(checkpoint),
            "--opponent",
            "rl-wp-assisted",
            "--profile",
            "paper8",
            "--pairs",
            str(PAIRS),
            "--seed",
            str(FIRST),
            "--device",
            "cpu",
            "--endgame",
            "--deterministic",
            "--output",
            str(folder / "evaluation.json"),
        ]
    main()
    if job == "evaluate:initial":
        initial_gate(read(folder / "evaluation.json"))
    elif stage == "train":
        payload = torch.load(
            folder / "update_00000520.pt", map_location="cpu", weights_only=False
        )
        assert payload["counters"]["updates"] == 520


def job_plan(cases):
    jobs = [{"id": "evaluate:initial", "resource": "cpu", "requires": []}]
    jobs += [
        {"id": "train:" + case, "resource": "gpu", "requires": []} for case in cases
    ]
    jobs += [
        {
            "id": "evaluate:" + case,
            "resource": "cpu",
            "requires": ["evaluate:initial", "train:" + case],
        }
        for case in cases
    ]
    return jobs


def ready_jobs(jobs, records, limits):
    active = {
        resource: sum(
            r["status"] == "running" and r["resource"] == resource
            for r in records.values()
        )
        for resource in limits
    }
    ready = []
    for job in jobs:
        resource = job["resource"]
        if job["id"] in records or active[resource] >= limits[resource]:
            continue
        if all(
            records.get(dep, {}).get("status") == "complete" for dep in job["requires"]
        ):
            ready.append(job)
            active[resource] += 1
    return ready


def summarize(p):
    baseline = read(BASE / "initial/evaluation.json")
    initial_gate(baseline)
    reports, endpoints = {}, {}
    for case in p["cases"]:
        path = BASE / case / "evaluation.json"
        reports[case] = read(path)
        endpoints[case] = {
            "report_sha256": sha(path),
            "win_rate": reports[case]["draw_adjusted_win_rate"],
            "win_ci95": reports[case]["win_rate_ci95"],
            "versus_initial": compare_reports(reports[case], baseline),
        }
    paired = {
        str(seed): compare_reports(
            reports[f"seed{seed}_cold"], reports[f"seed{seed}_warm"]
        )
        for seed in SEEDS
    }
    write(
        BASE / "summary.json",
        {
            "status": "awaiting_independent_audit",
            "protocol_sha256": sha(BASE / "protocol.json"),
            "initial_report_sha256": sha(BASE / "initial/evaluation.json"),
            "endpoints": endpoints,
            "paired": paired,
            "provisional_decision": decision(endpoints, paired),
        },
    )


def supervise(p):
    jobs = job_plan(p["cases"])
    records, live = {}, {}
    manifest = {
        "runner_pid": os.getpid(),
        "started": time.time(),
        "status": "running",
        "jobs": records,
    }
    try:
        while len(records) < len(jobs) or live:
            for job in ready_jobs(jobs, records, p["parallelism"]):
                stage, case = job["id"].split(":")
                path = BASE / case / f"{stage}.log"
                stream = path.open("wb")
                process = subprocess.Popen(
                    [
                        sys.executable,
                        "-u",
                        "-m",
                        "scripts.run_cold_ppo_screen",
                        "--worker",
                        job["id"],
                    ],
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
                record = job | {
                    "pid": process.pid,
                    "started": time.time(),
                    "status": "running",
                    "log": str(path),
                }
                records[job["id"]] = record
                live[job["id"]] = (process, stream)
                write(BASE / "process.json", manifest)
                print(record, flush=True)
            for name, (process, stream) in list(live.items()):
                code = process.poll()
                if code is None:
                    continue
                stream.close()
                del live[name]
                records[name].update(
                    returncode=code,
                    finished=time.time(),
                    status="complete" if code == 0 else "failed",
                )
                write(BASE / "process.json", manifest)
                if code:
                    raise RuntimeError(
                        f"{name} failed; abort remaining jobs and retain artifacts"
                    )
            time.sleep(0.5)
        summarize(p)
        manifest.update(status="complete_awaiting_audit", finished=time.time())
    except BaseException:
        for name, (process, stream) in live.items():
            if process.poll() is None:
                if os.name == "nt":
                    subprocess.run(
                        ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                        capture_output=True,
                        check=False,
                    )
                else:
                    process.terminate()
            code = process.wait()
            stream.close()
            records[name].update(
                status="aborted", returncode=code, finished=time.time()
            )
        manifest.update(status="failed", finished=time.time())
        write(BASE / "process.json", manifest)
        raise
    write(BASE / "process.json", manifest)
    print(manifest["status"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--worker")
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
        return
    p = preflight()
    if args.preflight:
        write("runs/cold_ppo_screen_preflight.json", p)
        print({"passed": True, "cases": list(p["cases"]), "jobs": job_plan(p["cases"])})
        return
    BASE.mkdir(exist_ok=False)
    write(BASE / "protocol.json", p)
    (BASE / "initial").mkdir()
    for case, config in p["cases"].items():
        (BASE / case).mkdir()
        write(BASE / case / "config.json", config)
    supervise(p)


if __name__ == "__main__":
    main()
