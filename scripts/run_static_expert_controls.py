"""Two fixed no-learning expert hybrids, matched to paired-regression deals."""

import argparse
import concurrent.futures
import subprocess
import sys
import time
from pathlib import Path

from koikoi.train import provenance
from koikoi.upstream import UPSTREAM_PATH, UPSTREAM_REV, load_source
from scripts.run_arbitration_study import PARENTS, read, sha, write

PREVIOUS = Path("runs/paired_arbitration_study")


def prepare(base):
    if base.exists():
        raise ValueError("Use a fresh controls directory")
    previous = read(PREVIOUS / "protocol.json")
    load_source()
    cases = {}
    for index, (parent, digest) in enumerate(PARENTS, 1):
        if sha(parent) != digest:
            raise ValueError("Parent changed")
        for arm in ("point_cards", "point_koi"):
            cases[f"parent{index}_{arm}"] = {
                "parent": parent,
                "parent_sha256": digest,
                "cards": "rl-point-assisted" if arm == "point_cards" else parent,
                "koi": parent if arm == "point_cards" else "rl-point-assisted",
            }
    p = {
        "source": provenance(),
        "runner_sha256": sha(__file__),
        "upstream_revision": UPSTREAM_REV,
        "comparator_hashes": {
            str(UPSTREAM_PATH / f"model_agent/{phase}_rl_{variant}.pt"): sha(
                UPSTREAM_PATH / f"model_agent/{phase}_rl_{variant}.pt"
            )
            for phase in ("discard", "pick", "koikoi")
            for variant in ("wp", "point")
        },
        "previous_protocol_sha256": sha(PREVIOUS / "protocol.json"),
        "evaluation": previous["evaluation"],
        "cases": cases,
        "purpose": "No-learning controls for whether trained expert composition exceeds fixed card/koi role substitution. Both hybrids and both parents are fixed before paired-regression outcomes. All results are retained; no best-seed selection.",
        "scope": "Same22000000--22000999 development deals as paired regression, deliberately reused for paired comparisons, not new independent confirmation. Eight matchups,1000 duplicate pairs each. No training and no search. CLI evaluator timing overlaps workers and is not an isolated latency benchmark.",
        "execution": "Wait for paired-regression study completion, then at most three CPU2 evaluators. Abort if the preceding study fails or source/parent identities change.",
    }
    write(base / "protocol.json", p)
    write(
        base / "process.json",
        {"status": "waiting_for_paired_study", "started": time.time()},
    )
    return p


def checked(base):
    p = read(base / "protocol.json")
    if (
        p["runner_sha256"] != sha(__file__)
        or p["source"]["source_sha256"] != provenance()["source_sha256"]
    ):
        raise ValueError("Source changed")
    if p["previous_protocol_sha256"] != sha(PREVIOUS / "protocol.json"):
        raise ValueError("Preceding protocol changed")
    if any(sha(path) != digest for path, digest in p["comparator_hashes"].items()):
        raise ValueError("Published comparator weights changed")
    for spec in p["cases"].values():
        if sha(spec["parent"]) != spec["parent_sha256"]:
            raise ValueError("Parent checkpoint changed")
    return p


def evaluate(base, case, opponent):
    p = checked(base)
    spec, evaluation = p["cases"][case], p["evaluation"]
    target = base / case / (opponent + ".json")
    if target.exists():
        raise ValueError("Evaluation already exists")
    target.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "koikoi.evaluate",
        "--agent",
        spec["cards"],
        "--koi-agent",
        spec["koi"],
        "--opponent",
        opponent,
        "--profile",
        "paper8",
        "--pairs",
        str(evaluation["pairs"]),
        "--seed",
        str(evaluation["first_seed"]),
        "--device",
        "cpu",
        "--deterministic",
        "--endgame",
        "--output",
        str(target),
    ]
    with target.with_suffix(".log").open("x", encoding="utf-8") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    report = read(target)
    assert report["checkpoints"][spec["parent"]]["sha256"] == spec["parent_sha256"]
    assert report["matches"] == 2 * evaluation["pairs"]
    assert report["endgame"] and report["opponent_endgame"]


def run(base):
    p = prepare(base)
    try:
        while True:
            state = read(PREVIOUS / "process.json")["status"]
            if state == "complete":
                break
            if state == "failed":
                raise RuntimeError("Preceding study failed")
            time.sleep(10)
        checked(base)
        write(base / "process.json", {"status": "evaluating"})
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            jobs = [
                pool.submit(evaluate, base, case, opponent)
                for case in p["cases"]
                for opponent in p["evaluation"]["opponents"]
            ]
            for job in concurrent.futures.as_completed(jobs):
                job.result()
    except BaseException as error:
        write(base / "process.json", {"status": "failed", "error": repr(error)})
        raise
    write(base / "process.json", {"status": "complete", "finished": time.time()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base", type=Path, default=Path("runs/static_expert_controls")
    )
    run(parser.parse_args().base)


if __name__ == "__main__":
    main()
