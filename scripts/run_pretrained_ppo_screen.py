"""Fixed two-seed pretrained-PPO development screen; final endpoints only."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch

from koikoi.compare import compare_reports
from koikoi.published_train import PublishedConfig, PublishedTrainer
from koikoi.train import provenance
from koikoi.upstream import UPSTREAM_PATH, UPSTREAM_REV, load_source

BASE = Path("runs/pretrained_ppo_screen")
SOURCE = "4b4360ed783c1d12cb5258e255f4b85f2e30dfa2e111fd8a078cf3439ce612cb"
INITIAL = "7d684d9063b8c4bbe406c7b50f9b43ac90bb774f5917b13522d28be1a81925f4"
TEACHER = {
    "discard": "31197cbd4b4fc17da6dbd6a6ae946f25000968aa941757028db92d480d548fdd",
    "pick": "4064b54356f1cda3baedcd6b94187effb51a2fac381f07dcfdb1649d8bf01e64",
    "koikoi": "408f931abafd7127648f9b933fbb8be7cc137a15d0b23b182090a9a83a70e3e4",
}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    os.replace(temporary, path)


def preflight():
    current = provenance()
    assert current["source_sha256"] == SOURCE and torch.cuda.is_available()
    load_source()
    for name, digest in TEACHER.items():
        assert sha(UPSTREAM_PATH / f"model_agent/{name}_rl_wp.pt") == digest
    benchmark = Path("runs/published_ppo_benchmark/report.json")
    assert (
        sha(benchmark)
        == "5611d2f0d7d0fcc44ae727f2c0c45859b29db0342cfe029ade4e95667110be70"
    )
    assert read(benchmark)["status"] == "complete"
    return {
        "provenance": current,
        "script_sha256": sha(__file__),
        "upstream_revision": UPSTREAM_REV,
        "comparator_hashes": TEACHER,
        "initial_parameters_sha256": INITIAL,
        "benchmark_report_sha256": sha(benchmark),
        "cases": {
            name: asdict(PublishedConfig(seed=seed))
            for name, seed in (("seed6681001", 6681001), ("seed6691001", 6691001))
        },
        "updates": 520,
        "warmup_updates": 20,
        "policy_updates": 500,
        "decisions_per_case": 520 * 128 * 128,
        "warmup_decisions_per_case": 20 * 128 * 128,
        "evaluation": {
            "pairs": 1000,
            "first_seed": 1000000,
            "device": "cpu",
            "opponent": "rl-wp-assisted",
            "profile": "paper8",
            "endgame": True,
            "deterministic": True,
            "search": "none",
        },
        "advance_criterion": "BOTH training seeds must have paired win-score improvement over initial with CI lower bound >0 AND absolute win-score CI lower bound >0.5. Independently audit before a decision. Point differences are secondary. Shared development deals are not additional independent samples; results are exploratory, not untouched final confirmation.",
        "initial_gate": "Before training, initial checkpoint on all1000 evaluation pairs must have each paired score0.5/point difference0. This is an outcome compatibility gate, not proof of universal action equality.",
        "stop_rule": "Exactly20 critic-only plus500 policy updates per seed; original WP weights, T0.05, actorLR3e-6/valueLR1e-4 fixed. Evaluate only initial and update520; no interim adaptation/reranking, unchanged extension or final seeds. Abort on failure, nonfinite metrics, source/teacher drift or initial compatibility failure. Terminal match counts are measured, not fixed; matches may cross update/warmup boundaries.",
    }


def initial_gate(report):
    assert report["pairs"] == 1000 and len(report["records"]) == 1000
    assert report["draw_adjusted_win_rate"] == 0.5
    for row in report["records"]:
        assert row["seat0_score"] + row["seat1_score"] == 1
        assert row["seat0_point_difference"] + row["seat1_point_difference"] == 0


def worker(job):
    protocol = read(BASE / "protocol.json")
    assert protocol["script_sha256"] == sha(__file__)
    assert provenance()["source_sha256"] == SOURCE
    for name, digest in TEACHER.items():
        assert sha(UPSTREAM_PATH / f"model_agent/{name}_rl_wp.pt") == digest
    stage, case = job.split(":")
    folder = BASE / case
    if stage == "train":
        from koikoi.published_train import main

        assert read(folder / "config.json") == protocol["cases"][case]
        initial_gate(read(BASE / "initial/evaluation.json"))
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
            trainer = PublishedTrainer(
                PublishedConfig(**next(iter(protocol["cases"].values())))
            )
            assert (
                trainer.model.actor.checkpoint()["current"]["parameters_sha256"]
                == INITIAL
            )
            checkpoint = folder / "initial.pt"
            trainer.save(checkpoint)
            del trainer
        else:
            checkpoint = folder / "update_00000520.pt"
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            assert payload["config"] == protocol["cases"][case]
            assert payload["provenance"]["source_sha256"] == SOURCE
            assert payload["actor"]["initialization"]["parameters_sha256"] == INITIAL
            counts = payload["counters"]
            assert counts["updates"] == 520 and counts["policy_updates"] == 500
            assert counts["warmup_updates"] == 20
            assert counts["decisions"] == protocol["decisions_per_case"]
        sys.argv = [
            "koikoi.evaluate",
            "--agent",
            str(checkpoint),
            "--opponent",
            "rl-wp-assisted",
            "--profile",
            "paper8",
            "--pairs",
            "1000",
            "--seed",
            "1000000",
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
    if stage == "train":
        final = torch.load(
            folder / "update_00000520.pt", map_location="cpu", weights_only=False
        )
        assert final["counters"]["updates"] == 520


def summarize(protocol):
    initial_path = BASE / "initial/evaluation.json"
    baseline = read(initial_path)
    initial_gate(baseline)
    results = {}
    for case in protocol["cases"]:
        path = BASE / case / "evaluation.json"
        report = read(path)
        results[case] = {
            "report_sha256": sha(path),
            "versus_initial": compare_reports(report, baseline),
            "win_rate": report["draw_adjusted_win_rate"],
            "win_ci95": report["win_rate_ci95"],
        }
    write(
        BASE / "summary.json",
        {
            "status": "awaiting_independent_audit",
            "protocol_sha256": sha(BASE / "protocol.json"),
            "initial_report_sha256": sha(initial_path),
            "results": results,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--worker")
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
        return
    protocol = preflight()
    if args.preflight:
        write("runs/pretrained_ppo_screen_preflight.json", protocol)
        print(json.dumps({"passed": True, "cases": list(protocol["cases"])}))
        return
    BASE.mkdir(exist_ok=False)
    write(BASE / "protocol.json", protocol)
    (BASE / "initial").mkdir()
    for name, config in protocol["cases"].items():
        (BASE / name).mkdir()
        write(BASE / name / "config.json", config)
    manifest = {
        "runner_pid": os.getpid(),
        "status": "running",
        "started": time.time(),
        "jobs": [],
    }
    jobs = ["evaluate:initial"] + [
        f"{stage}:{name}"
        for stage in ("train", "evaluate")
        for name in protocol["cases"]
    ]
    for job in jobs:
        stage, case = job.split(":")
        log_path = BASE / case / f"{stage}.log"
        with log_path.open("wb") as log:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-u",
                    "-m",
                    "scripts.run_pretrained_ppo_screen",
                    "--worker",
                    job,
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            record = {
                "job": job,
                "pid": process.pid,
                "started": time.time(),
                "status": "running",
                "log": str(log_path),
            }
            manifest["jobs"].append(record)
            write(BASE / "process.json", manifest)
            print(json.dumps(record), flush=True)
            result = process.wait()
        record.update(
            returncode=result,
            finished=time.time(),
            status="complete" if result == 0 else "failed",
        )
        if result:
            manifest["status"] = "failed"
            write(BASE / "process.json", manifest)
            raise RuntimeError(
                f"{job} failed; inspect retained artifacts before any restart"
            )
        write(BASE / "process.json", manifest)
    summarize(protocol)
    manifest.update(status="complete_awaiting_audit", finished=time.time())
    write(BASE / "process.json", manifest)
    print(
        json.dumps({"status": manifest["status"], "directory": str(BASE)}), flush=True
    )


if __name__ == "__main__":
    main()
