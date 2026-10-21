"""Retain every source acquired before the mechanical gate-scope correction."""

import shutil
from pathlib import Path

from scripts.frontier_incremental_design import cases
from scripts.run_arbitration_study import read, sha
from scripts.uniform_controller_study import collect_case as fresh_collect
from scripts.uniform_controller_study import source_files

FAILED = Path("runs/frontier_incremental_evaluation")
MANIFEST = Path("runs/frontier_incremental_failure_manifest.json")
MANIFEST_SHA = "bbc63063d172d9c77525db53745abae274044d5a13a99805dbdc0cc90ac10f19"


def manifest():
    if sha(MANIFEST) != MANIFEST_SHA:
        raise ValueError("Original failed-run source manifest changed")
    value = read(MANIFEST)
    indices = [row["case"]["index"] for row in value["sources"]]
    if indices != [0, 16, 17, 18, 19, 20, 32, 48, 64, 65, 66, 67, 68, 69, 70]:
        raise ValueError("Retained source set changed")
    ordered = cases()
    for row in value["sources"]:
        if row["case"] != ordered[row["case"]["index"]]:
            raise ValueError("Retained source case changed")
    return value


def bindings():
    value = manifest()
    paths = {str(MANIFEST): MANIFEST_SHA}
    for name, field in (
        ("protocol.json", "protocol_sha256"),
        ("queue.json", "queue_sha256"),
    ):
        path = FAILED / name
        if sha(path) != value[field]:
            raise ValueError("Failed-run evidence changed")
        paths[str(path)] = value[field]
    if read(FAILED / "queue.json")["status"] != "failed":
        raise ValueError("Predecessor must be closed")
    for row in value["sources"]:
        folder = Path(row["folder"])
        if {p.name for p in folder.iterdir()} != set(row["files"]):
            raise ValueError("Retained ledger membership changed")
        for name, digest in row["files"].items():
            path = folder / name
            if sha(path) != digest:
                raise ValueError("Retained source evidence changed")
            paths[str(path)] = digest
    return paths


def acquisition(case):
    return (
        "retained_v1"
        if any(r["case"] == case for r in manifest()["sources"])
        else "new_v2"
    )


def audit_acquisition(folder, case):
    matches = [r for r in manifest()["sources"] if r["case"] == case]
    if matches:
        row = matches[0]
        if {p.name for p in folder.iterdir()} != set(row["files"]) or any(
            sha(folder / name) != digest for name, digest in row["files"].items()
        ):
            raise ValueError("Retained source was replaced or modified")


def collect_case(folder, case, config, parent, opponent, predictors, native):
    matches = [r for r in manifest()["sources"] if r["case"] == case]
    if not matches:
        return fresh_collect(folder, case, config, parent, opponent, predictors, native)
    row = matches[0]
    origin = Path(row["folder"])
    folder.mkdir()
    for name, digest in row["files"].items():
        if sha(origin / name) != digest:
            raise ValueError("Retained source changed before copy")
        shutil.copyfile(origin / name, folder / name)
        if sha(folder / name) != digest:
            raise ValueError("Retained source copy differs")
    source = read(folder / "record.json")
    source_files(folder, case, source, config)
    return source
