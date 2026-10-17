"""Fixed two-parent published-WP adaptation screen; no interim selection.

--preflight checks inputs without launching training. A normal invocation creates
an exclusive run directory and launches four training then four evaluation jobs.
Worker PIDs and every completed job are recorded atomically for external polling.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import torch

from koikoi.compare import compare_reports
from koikoi.train import Config, provenance
from koikoi.upstream import UPSTREAM_PATH, load_source

BASE = Path("runs/published_wp50_screen")
SOURCE = "1c36895f2b5164c4c0a86862e2270364a74c2ff0ee4d950f4b0c4cd970575914"
PARENTS = {
    "seed1": {
        "checkpoint": "runs/ppo8_v3/update_00005000.pt",
        "sha256": "d104ae00824aca114e6cd414c74d02670b9a8de5a009ad4fad557005f66233e5",
        "baseline": "runs/ppo8_v3/eval_5000_wp.json",
        "training_seed": 6300001,
    },
    "seed10001": {
        "checkpoint": "runs/ppo8_seed10001_captured/update_00005000.pt",
        "sha256": "edc44213bc5c5c840297afc33ee64a57b91e9dee48c202535906fc916cde109d",
        "baseline": "runs/ppo8_seed10001_captured/eval_5000_wp.json",
        "training_seed": 6310001,
    },
}
TEACHER = {
    "discard": "31197cbd4b4fc17da6dbd6a6ae946f25000968aa941757028db92d480d548fdd",
    "pick": "4064b54356f1cda3baedcd6b94187effb51a2fac381f07dcfdb1649d8bf01e64",
    "koikoi": "408f931abafd7127648f9b933fbb8be7cc137a15d0b23b182090a9a83a70e3e4",
}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    os.replace(temporary, path)


def preflight():
    current = provenance()
    assert current["source_sha256"] == SOURCE
    assert torch.cuda.is_available()
    load_source()
    for name, digest in TEACHER.items():
        assert sha(UPSTREAM_PATH / f"model_agent/{name}_rl_wp.pt") == digest
    config_path = Path("configs/ppo8_published_wp50.json")
    config = Config(**read(config_path))
    cases = {}
    baselines = {}
    for label, parent in PARENTS.items():
        assert sha(parent["checkpoint"]) == parent["sha256"]
        baseline = read(parent["baseline"])
        assert baseline["pairs"] == 1000 and baseline["seed_start"] == 1000000
        assert baseline["opponent"] == "rl-wp-assisted"
        assert baseline["profile"] == "paper8" and baseline["protocol_version"] == 2
        assert (
            baseline["device"] == "cpu"
            and baseline["deterministic"]
            and baseline["endgame"]
        )
        assert baseline["search"] == "none"
        assert any(
            row["sha256"] == parent["sha256"]
            for row in baseline["checkpoints"].values()
        )
        assert sorted(row["seed"] for row in baseline["records"]) == list(
            range(1000000, 1001000)
        )
        baselines[label] = sha(parent["baseline"])
        for arm in ("wp50", "control"):
            case = replace(config, seed=parent["training_seed"])
            if arm == "control":
                case = replace(case, published_opponent="", opponent_fraction=0.0)
            cases[f"{label}_{arm}"] = {
                "parent": parent["checkpoint"],
                "parent_sha256": parent["sha256"],
                "config": asdict(case),
            }
    return {
        "provenance": current,
        "script_sha256": sha(__file__),
        "config_sha256": sha(config_path),
        "teacher_hashes": TEACHER,
        "parent_baseline_hashes": baselines,
        "cases": cases,
        "updates": 500,
        "simulator_decisions_per_case": 500 * 2048 * 128,
        "evaluation": {
            "pairs": 1000,
            "first_seed": 1000000,
            "device": "cpu",
            "opponent": "rl-wp-assisted",
            "endgame": True,
            "deterministic": True,
            "search": "none",
        },
        "advance_criterion": "For BOTH parents, WP50-control and WP50-parent paired win-score CI lower bounds >0; separately require BOTH absolute WP50 win-rate lower bounds >0.5 for a public-superiority development signal. No pooling parents as extra deals. Independently audit before any method decision; all results remain exploratory development.",
        "stop_rule": "Exactly500 updates per case and1000 evaluation pairs per endpoint, no interim selection or unchanged extension. Abort on job failure; final seeds untouched.",
    }


def worker(job):
    protocol = read(BASE / "protocol.json")
    assert protocol["script_sha256"] == sha(__file__)
    assert provenance()["source_sha256"] == SOURCE
    stage, case = job.split(":")
    record = protocol["cases"][case]
    folder = BASE / case
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    if stage == "train":
        from koikoi.train import main

        assert sha(record["parent"]) == record["parent_sha256"]
        sys.argv = [
            "koikoi.train",
            "--config",
            str(folder / "config.json"),
            "--run",
            str(folder),
            "--initialize",
            record["parent"],
            "--updates",
            "500",
            "--checkpoint-every",
            "100",
        ]
    else:
        from koikoi.evaluate import main

        assert stage == "evaluate"
        checkpoint = folder / "update_00000500.pt"
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        assert payload["counters"]["updates"] == 500
        assert (
            payload["counters"]["decisions"] == protocol["simulator_decisions_per_case"]
        )
        assert payload["provenance"]["source_sha256"] == SOURCE
        assert (
            payload["provenance"]["initialized_from"]["sha256"]
            == record["parent_sha256"]
        )
        assert payload["config"] == record["config"]
        assert all(torch.isfinite(value).all() for value in payload["model"].values())
        if record["config"]["published_opponent"]:
            counters = payload["counters"]
            assert (
                counters["learner_decisions"] + counters["published_decisions"]
                == counters["decisions"]
            )
            assert 0 < counters["published_matches"] <= counters["matches"]
            assert 0 < counters["published_rounds"] <= counters["rounds"]
            assert payload["provenance"]["frozen_opponent"]["model_hashes"] == TEACHER
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


def summarize(protocol):
    results = {}
    for label, parent in PARENTS.items():
        treatment_path = BASE / f"{label}_wp50/evaluation.json"
        control_path = BASE / f"{label}_control/evaluation.json"
        treatment, control = read(treatment_path), read(control_path)
        assert sha(parent["baseline"]) == protocol["parent_baseline_hashes"][label]
        results[label] = {
            "treatment_report_sha256": sha(treatment_path),
            "control_report_sha256": sha(control_path),
            "wp50_vs_control": compare_reports(treatment, control),
            "wp50_vs_parent": compare_reports(treatment, read(parent["baseline"])),
            "control_vs_parent": compare_reports(control, read(parent["baseline"])),
            "wp50_win_rate": treatment["draw_adjusted_win_rate"],
            "wp50_win_ci95": treatment["win_rate_ci95"],
        }
    write(
        BASE / "summary.json",
        {
            "status": "awaiting_independent_audit",
            "protocol_sha256": sha(BASE / "protocol.json"),
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
        write("runs/published_wp50_screen_preflight.json", protocol)
        print(json.dumps({"passed": True, "cases": list(protocol["cases"])}))
        return
    BASE.mkdir(exist_ok=False)
    write(BASE / "protocol.json", protocol)
    for case, record in protocol["cases"].items():
        folder = BASE / case
        folder.mkdir()
        write(folder / "config.json", record["config"])
    manifest = {
        "runner_pid": os.getpid(),
        "status": "running",
        "started": time.time(),
        "jobs": [],
    }
    for stage in ("train", "evaluate"):
        for case in protocol["cases"]:
            job = f"{stage}:{case}"
            log_path = BASE / case / f"{stage}.log"
            with log_path.open("wb") as log:
                process = subprocess.Popen(
                    [
                        sys.executable,
                        "-u",
                        "-m",
                        "scripts.run_published_screen",
                        "--worker",
                        job,
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
                record = {
                    "job": job,
                    "pid": process.pid,
                    "status": "running",
                    "started": time.time(),
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
                    f"{job} failed; retain artifacts and inspect its log"
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
