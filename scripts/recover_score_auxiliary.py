"""Preserve the failed study and run remaining fixed arms in separate processes.

No learner or native-backend changes. Each child makes at most 101 CUDA backend
instances, below the independently reproduced Windows DLL-registration limit.
"""

import argparse
import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

from scripts.check_score_auxiliary_training import equal, equal_ppo, load
from scripts.finish_point_initialization import run_job
from scripts.run_arbitration_study import read, sha
from scripts.train_score_auxiliary_study import checked, train_arm, write_new

SOURCE = Path("runs/score_auxiliary_study")
OUTPUT = Path("runs/score_auxiliary_recovery")
PROTOCOL_SHA = "9ba0bd7b1d187d59269af336537ceab890bf8884a969a9df073dcd626e686f46"
RETAINED = tuple(f"seed1_{kind}" for kind in ("none", "win", "score"))
RESTARTED = tuple(f"seed2_{kind}" for kind in ("none", "win", "score"))


def copy_completed(source, destination):
    """Copy immutable arm artifacts; relocate only the receipt's endpoint path."""
    receipt = read(source / "complete.json")
    if receipt["status"] != "complete":
        raise ValueError("Only completed arms can be retained")
    old_endpoint = Path(receipt["endpoint"])
    if old_endpoint.parent != source:
        raise ValueError("Endpoint lies outside its arm")
    files = {
        p.relative_to(source).as_posix(): sha(p)
        for p in source.rglob("*")
        if p.is_file()
    }
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns("complete.json"))
    for name, digest in files.items():
        if name != "complete.json" and sha(destination / name) != digest:
            raise ValueError("Retained artifact copy differs")
    relocated = receipt | {"endpoint": str(destination / old_endpoint.name)}
    write_new(destination / "complete.json", relocated)
    return {"original_files": files, "receipt": relocated}


def verify_repeated_prefix(source, destination):
    """Check every saved old prefix state and log row against the fixed rerun."""
    import json

    old = [
        json.loads(s)
        for s in (source / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    new = [
        json.loads(s)
        for s in (destination / "metrics.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert old and len(new) >= len(old)
    for a, b in zip(old, new[: len(old)], strict=True):
        # Wall time and git provenance can change; all numerical learner state cannot.
        equal(
            {k: v for k, v in a.items() if k != "seconds"},
            {k: v for k, v in b.items() if k != "seconds"},
        )
    stages = []
    for manifest in sorted(source.glob("*/manifest.json")):
        stage = manifest.parent.name
        for filename in ("ppo.pt", "auxiliary.pt"):
            compare = equal_ppo if filename == "ppo.pt" else equal
            compare(
                load(source / stage / filename), load(destination / stage / filename)
            )
        stages.append(stage)
    collections = sorted(source.glob("collection_*.pt"))
    for path in collections:
        equal(load(path), load(destination / path.name))
    return {
        "status": "passed",
        "updates": len(old),
        "stages": stages,
        "collections": len(collections),
        "logged_ppo_work": {
            k: old[-1]["ppo"][k] for k in ("decisions", "rounds", "matches")
        },
        "logged_auxiliary_work": old[-1]["auxiliary_counters"],
        "scope": "Exact saved tensors, RNG, optimizer and all logged numerical metrics; wall time and git provenance excluded. No neural replay of original unsaved update165.",
    }


def prepare():
    if OUTPUT.exists():
        raise ValueError("Preserve recovery directory")
    suite = ET.parse("runs/score_auxiliary_recovery_tests.xml").getroot().find("testsuite")
    assert int(suite.attrib["tests"]) == 6
    assert all(int(suite.attrib[k]) == 0 for k in ("errors", "failures", "skipped"))
    assert sha(SOURCE / "protocol.json") == PROTOCOL_SHA
    protocol = read(SOURCE / "protocol.json")
    checked(protocol["inputs"])
    assert set(protocol["arms"]) == set(RETAINED + RESTARTED)
    assert (SOURCE / "failed.json").exists() and not (SOURCE / "complete.json").exists()
    assert not any((SOURCE / name).exists() for name in RESTARTED[1:])
    probe = read("runs/score_auxiliary_dll_failure_probe.json")
    assert probe["error"]["winerror"] == 206
    assert (
        probe["registrations_before_failure"] == 332
        and probe["registration_after_cleanup"]
    )
    inputs = dict(protocol["inputs"])
    for path in SOURCE.rglob("*"):
        if path.is_file():
            inputs[str(path)] = sha(path)
    for path in (
        Path(__file__),
        Path("scripts/finish_point_initialization.py"),
        Path("runs/score_auxiliary_dll_failure_probe.json"),
        Path("runs/score_auxiliary_completion/failed.json"),
        Path("runs/score_auxiliary_completion/protocol.json"),
        Path("tests/test_recover_score_auxiliary.py"),
        Path("runs/score_auxiliary_recovery_tests.xml"),
    ):
        inputs[str(path)] = sha(path)
    OUTPUT.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(SOURCE / "protocol.json", OUTPUT / "protocol.json")
    retained = {name: copy_completed(SOURCE / name, OUTPUT / name) for name in RETAINED}
    checked(inputs)
    write_new(
        OUTPUT / "recovery_protocol.json",
        {
            "format": "score_auxiliary_process_recovery_v1",
            "inputs": inputs,
            "original_protocol_sha256": PROTOCOL_SHA,
            "retained": retained,
            "fresh_process_arms": list(RESTARTED),
            "scope": "Original failure and partial arm remain untouched. Copy completed seed1 arms byte-for-byte except endpoint paths in receipts. Rerun seed2_none from the original seed; verify its recorded prefix exactly. Run seed2_win and seed2_score once. Original learning, evaluation seeds and primary criterion unchanged. No endpoint selection or promotion.",
            "additional_work": {
                "ppo_completed_updates_before_failure": 165,
                "ppo_decisions_before_failure": 2703360,
                "logged_updates_before_failure": 164,
                "unlogged_update_rounds_matches": None,
                "auxiliary_complete_collections": 28,
                "auxiliary_matches": 3584,
            },
        },
    )
    print({"prepared": str(OUTPUT), "inputs": len(inputs)}, flush=True)


def run():
    recovery = read(OUTPUT / "recovery_protocol.json")
    checked(recovery["inputs"])
    write_new(
        OUTPUT / "started.json",
        {
            "pid": os.getpid(),
            "recovery_protocol_sha256": sha(OUTPUT / "recovery_protocol.json"),
        },
    )
    try:
        for name in RESTARTED:
            print({"starting": name}, flush=True)
            run_job(
                name, ["scripts.recover_score_auxiliary", "arm", "--arm", name], OUTPUT
            )
            checked(recovery["inputs"])
            if name == "seed2_none":
                prefix = verify_repeated_prefix(SOURCE / name, OUTPUT / name)
                write_new(OUTPUT / "repeated_prefix.json", prefix)
        results = {
            name: read(OUTPUT / name / "complete.json") for name in RETAINED + RESTARTED
        }
        totals = {
            group: {
                key: sum(r[group][key] for r in results.values())
                for key in ("decisions", "rounds", "matches")
            }
            for group in ("ppo_work", "auxiliary_work")
        }
        assert totals["ppo_work"]["decisions"] == 51118080
        assert totals["auxiliary_work"]["matches"] == 76800
        checked(recovery["inputs"])
        write_new(
            OUTPUT / "complete.json",
            {
                "status": "complete",
                "arms": results,
                "totals": totals,
                "evaluation": "not_started",
                "deployment_changed": False,
                "protocol_sha256": PROTOCOL_SHA,
                "recovery_protocol_sha256": sha(OUTPUT / "recovery_protocol.json"),
                "repeated_prefix_sha256": sha(OUTPUT / "repeated_prefix.json"),
            },
        )
    except BaseException as error:
        write_new(OUTPUT / "failed.json", {"error": repr(error)})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "arm"))
    parser.add_argument("--arm", choices=RESTARTED)
    args = parser.parse_args()
    if args.command == "arm":
        if args.arm is None:
            parser.error("--arm required")
        checked(read(OUTPUT / "recovery_protocol.json")["inputs"])
        protocol = read(OUTPUT / "protocol.json")
        train_arm(OUTPUT, args.arm, protocol["arms"][args.arm], protocol)
    else:
        {"prepare": prepare, "run": run}[args.command]()
