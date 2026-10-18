"""Qualified, fixed-budget final confirmation; never train or promote a model."""

import argparse
import concurrent.futures
import json
import xml.etree.ElementTree as ET
from pathlib import Path

from koikoi.agents import EndgameAgent, NeuralAgent
from koikoi.env import BatchEnv
from koikoi.train import provenance
from koikoi.upstream import PublishedAgent
from scripts.public_memory_continuation_match import (
    ContinuationAgent,
    play_continuation_match,
)
from scripts.renewed_confirmation_analysis import POLICIES, analyse, cases, load_points
from scripts.renewed_confirmation_gate import check
from scripts.renewed_continuation_match import RenewedContinuationAgent
from scripts.revealed_opponent_models import StrictPublishedPrediction
from scripts.run_arbitration_study import PARENTS, read, sha, write
from scripts.run_final_round_training_study import checked_inputs, configure

STUDY = Path("runs/renewed_continuation_study")
FIXED = Path("runs/public_continuation_study")
PREDECLARATION = Path("runs/renewed_confirmation_pre_evaluation.json")
PREDECLARATION_SHA = "a9547cb8a9601b40e162988dca2bd1515e4b696bc4749601806d869dda89a934"
TEST_XML = Path("runs/renewed_confirmation_workflow_tests.xml")


def make_plan(study=STUDY):
    # Qualification is checked before reading/loading candidate model payloads,
    # creating the output directory, or constructing a confirmation deal.
    qualification = check(study)
    if not qualification["qualified"]:
        raise ValueError("Replicated primary development gain required")
    if sha(PREDECLARATION) != PREDECLARATION_SHA:
        raise ValueError("Changed prospective confirmation receipt")
    inputs = dict(read(PREDECLARATION)["inputs"])
    audited = read(study / "independent_audit.json")["inputs"]
    for path, digest in audited.items():
        if path in inputs and inputs[path] != digest:
            raise ValueError(f"Conflicting evidence binding: {path}")
        inputs[path] = digest
    tests = list(ET.parse(TEST_XML).iter("testcase"))
    if len(tests) < 8 or any(
        case.find(tag) is not None
        for case in tests
        for tag in ("skipped", "failure", "error")
    ):
        raise ValueError("Passed confirmation workflow checks required")
    parents = []
    for index, (path, digest) in enumerate(PARENTS, 1):
        models = {
            "parent": path,
            "fixed_public": str(FIXED / f"parent{index}_public/final.pt"),
            "renewed_public": str(study / f"parent{index}_public/final.pt"),
        }
        if sha(path) != digest or any(p not in inputs for p in models.values()):
            # Windows receipts can contain equivalent slash spellings.
            resolved = {Path(p).resolve(): h for p, h in inputs.items()}
            if sha(path) != digest or any(
                Path(p).resolve() not in resolved for p in models.values()
            ):
                raise ValueError("Models must be bound by completed development audit")
        parents.append({"index": index, "models": models})
    files = [
        PREDECLARATION,
        TEST_XML,
        study / "independent_audit.json",
        study / "summary_receipt.json",
        study / "summary.md",
        study / "process.json",
        Path(__file__),
        Path("scripts/audit_renewed_confirmation.py"),
        Path("scripts/summarize_renewed_confirmation.py"),
        Path("scripts/renewed_confirmation_analysis.py"),
        Path("tests/test_renewed_confirmation_workflow.py"),
        Path("tests/test_renewed_confirmation_analysis.py"),
        Path("runs/renewed_confirmation_analysis_tests_final.xml"),
    ]
    inputs.update({str(p): sha(p) for p in files})
    plan = {
        "kind": "renewed_confirmation_v1",
        "study": str(study),
        "qualification": qualification,
        "parents": parents,
        "policies": list(POLICIES),
        "pairs_per_opponent": 4096,
        "deal_start": 101100001,
        "distinct_deals": 8192,
        "matches": 98304,
        "bootstrap_seed": 101300001,
        "bootstrap_replicates": 10000,
        "workers": 4,
        "torch_threads_per_worker": 2,
        "core_source_sha256": provenance()["source_sha256"],
        "inputs": inputs,
        "automatic_promotion": False,
    }
    checked_inputs(plan)
    return plan


def evaluate_cell(root_name, parent, opponent_name):
    configure()
    root = Path(root_name)
    plan = read(root / "protocol.json")
    checked_inputs(plan)
    if parent not in plan["parents"] or opponent_name not in ("wp", "point"):
        raise ValueError("Unexpected evaluation group")
    models = {name: PublishedAgent(name) for name in ("wp", "point")}
    predictors = {name: StrictPublishedPrediction(a) for name, a in models.items()}
    opponent = EndgameAgent(models[opponent_name])
    actors = {
        "parent": NeuralAgent(parent["models"]["parent"], deterministic=True),
        "fixed_public": ContinuationAgent(parent["models"]["fixed_public"], predictors),
        "renewed_public": RenewedContinuationAgent(
            parent["models"]["renewed_public"], predictors
        ),
    }
    native = BatchEnv(1, "cpu", "paper8")
    path = root / f"eval_parent{parent['index']}_{opponent_name}.jsonl"
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        for policy, actor in actors.items():
            for case in cases(parent["index"], opponent_name):
                row = play_continuation_match(case, actor, opponent, native)
                stream.write(json.dumps({"policy": policy, **row}) + "\n")
                stream.flush()
    checked_inputs(plan)
    return str(path)


def run(output, study=STUDY):
    if output.exists():
        raise ValueError("Use a fresh confirmation directory")
    plan = make_plan(study)
    output.mkdir(parents=True, exist_ok=False)
    write(output / "protocol.json", plan)
    try:
        write(output / "process.json", {"status": "running", "stage": "evaluation"})
        with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
            futures = [
                pool.submit(evaluate_cell, str(output), parent, opponent)
                for parent in plan["parents"]
                for opponent in ("wp", "point")
            ]
            for future in concurrent.futures.as_completed(futures):
                print(json.dumps({"evaluation_complete": future.result()}), flush=True)
        checked_inputs(plan)
        results = analyse(load_points(output))
        paths = [output / "protocol.json"] + sorted(output.glob("eval_*.jsonl"))
        write(
            output / "analysis.json",
            {
                "status": "complete",
                **results,
                "inputs": {str(p): sha(p) for p in paths},
                "scope": "Fixed balanced WP/Point benchmark; independent audit pending; no automatic promotion or mechanism confirmation.",
            },
        )
        write(output / "process.json", {"status": "complete"})
    except BaseException as error:
        write(output / "process.json", {"status": "failed", "error": repr(error)})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--study", type=Path, default=STUDY)
    args = parser.parse_args()
    run(args.output, args.study)
