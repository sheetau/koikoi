"""Prospective matched chance-renewal study with fixed-bank controls."""

import argparse
import concurrent.futures
import json
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict
from pathlib import Path

import torch

from koikoi.agents import EndgameAgent, NeuralAgent
from koikoi.env import BatchEnv
from koikoi.train import provenance
from koikoi.upstream import PublishedAgent
from scripts.check_renewed_continuation import require_completed_study
from scripts.public_memory_continuation_match import (
    ContinuationAgent,
    play_continuation_match,
)
from scripts.renewed_continuation import RenewedContinuationConfig as ContinuationConfig
from scripts.renewed_continuation import (
    RenewedContinuationTrainer as ContinuationTrainer,
)
from scripts.renewed_continuation_analysis import (
    POLICIES,
    analyse,
    cases,
    decisions,
    load_points,
)
from scripts.renewed_continuation_match import RenewedContinuationAgent
from scripts.revealed_opponent_models import StrictPublishedPrediction
from scripts.run_arbitration_study import PARENTS, read, sha, write
from scripts.run_final_round_training_study import checked_inputs, configure

BASE = Path("runs/public_memory_continuation_bank")
FIXED = Path("runs/public_continuation_study")
CAPABILITY = Path("runs/renewed_continuation_capability")


def validate_allocations():
    allocations = [set(range(start, start + 66560)) for start in (94110001, 94210001)]
    allocations += [set(range(94990001, 94990001 + 384))]
    allocations += [set(range(start, start + 1024)) for start in (95110001, 95120001)]
    allocations += [
        set(range(start, start + 1024))
        for start in (93110001, 93120001, 91110001, 91120001)
    ]
    for i, values in enumerate(allocations):
        assert min(values) >= 0 and max(values) < 2**31
        assert all(not values & other for other in allocations[i + 1 :])
        assert not any(101000000 <= seed < 102000000 for seed in values)


def require_capability(root):
    if (
        not (root / "process.json").exists()
        or read(root / "process.json")["status"] != "complete"
    ):
        raise ValueError("Complete renewed-deal CUDA capability required")
    audit = read(root / "audit.json")
    if audit["status"] != "passed" or not audit["matched_initial_and_warmup_exact"]:
        raise ValueError("Passed paired capability required")
    assert (
        audit["logical_conditional_deal_constructions"],
        audit["replayed_conditional_deal_constructions"],
        audit["distinct_chance_keys_shared_between_arms"],
    ) == (768, 256, 384)
    for mode in ("public", "masked"):
        assert audit["arms"][mode]["exact_resume"]
        assert audit["arms"][mode]["counters"]["fresh_conditional_deals"] == 384
    for path, digest in audit["inputs"].items():
        if sha(path) != digest:
            raise ValueError(f"Changed capability input: {path}")
    return {**audit["inputs"], str(root / "audit.json"): sha(root / "audit.json")}


def make_plan():
    validate_allocations()
    inputs = require_completed_study(FIXED)
    assert (
        sha(FIXED / "protocol.json")
        == "95ce0a01a76c790f5dbdeb0c78028199e61219053167fc7309543cfa4c4815a1"
    )
    inputs.update(require_capability(CAPABILITY))
    design = Path("docs/RENEWED_CONTINUATION_STUDY.rst")
    design_receipt = Path("runs/renewed_continuation_design_pre_evaluation.json")
    assert (
        sha(design)
        == read(design_receipt)["design_sha256"]
        == "b6c0b7ada8039e26824db2c67eeed89f5051a7b7e478326343fb35ca1417c073"
    )
    files = [Path(__file__), design, design_receipt] + [
        Path(p)
        for p in (
            "scripts/audit_renewed_continuation_study.py",
            "scripts/renewed_continuation_analysis.py",
            "tests/test_renewed_continuation_analysis.py",
            "tests/test_renewed_continuation_protocol.py",
            "runs/renewed_continuation_analysis_tests.xml",
            "runs/renewed_continuation_protocol_tests.xml",
        )
    ]
    for filename in (
        "runs/renewed_continuation_analysis_tests.xml",
        "runs/renewed_continuation_protocol_tests.xml",
    ):
        cases_xml = list(ET.parse(filename).iter("testcase"))
        assert len(cases_xml) >= 4
        assert all(
            case.find(tag) is None
            for case in cases_xml
            for tag in ("skipped", "failure", "error")
        )
    parents = []
    for index, (path, digest) in enumerate(PARENTS, 1):
        assert sha(path) == digest
        files += [Path(path), BASE / f"parent{index}.pt"]
        for mode in ("public", "masked"):
            files += [
                FIXED / f"parent{index}_{mode}" / name
                for name in (
                    "initial.pt",
                    "final.pt",
                    "update_00000020.pt",
                    "update_00000100.pt",
                    "update_00000200.pt",
                    "update_00000300.pt",
                    "update_00000400.pt",
                    "update_00000500.pt",
                    "update_00000520.pt",
                )
            ]
        parents.append(
            {
                "index": index,
                "path": path,
                "sha256": digest,
                "seed": 92110001 + 10000 * (index - 1),
                "chance_seed": 94110001 + 100000 * (index - 1),
            }
        )
    inputs.update({str(p): sha(p) for p in files})
    return {
        "kind": "renewed_continuation_study_v1",
        "inputs": inputs,
        "core_source_sha256": provenance()["source_sha256"],
        "parents": parents,
        "fixed_root": str(FIXED),
        "updates": 520,
        "pairs_per_opponent": 1024,
        "bootstrap_seed": 95310001,
        "bootstrap_replicates": 10000,
        "selection": "Only update520;renewed-public minus parent/fixed-public positive mixture win bounds at both parents. Memory/interaction claims separate.",
    }


def train_arm(root, parent, mode, plan):
    from scripts.audit_renewed_continuation_study import audit_endpoint

    folder = root / f"parent{parent['index']}_{mode}"
    bank = torch.load(
        BASE / f"parent{parent['index']}.pt",
        weights_only=False,
        map_location="cpu",
    )
    data = tuple(bank[k] for k in ("states", "seats", "opponents", "memory"))
    config = ContinuationConfig(
        device="cuda",
        seed=parent["seed"],
        public_memory=mode == "public",
        chance_seed=parent["chance_seed"],
        chance_draws=66560,
    )
    trainer = ContinuationTrainer.from_parent(config, parent["path"], *data)
    trainer.save(folder / "initial.pt")
    write(folder / "config.json", asdict(config))
    start = time.perf_counter()
    with (folder / "metrics.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
        for update in range(1, 521):
            metrics = trainer.update()
            row = {
                "update": update,
                "seconds": time.perf_counter() - start,
                "counters": trainer.counters.copy(),
                "metrics": metrics,
            }
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            if update == 20 or update % 100 == 0 or update == 520:
                trainer.save(folder / f"update_{update:08d}.pt")
            if update % 20 == 0:
                print(
                    json.dumps({"parent": parent["index"], "mode": mode, **row}),
                    flush=True,
                )
    trainer.save(folder / "final.pt")
    del trainer
    checked_inputs(plan)
    write(folder / "training_audit.json", audit_endpoint(root, parent["index"], mode))


def evaluate_cell(root_name, parent, opponent_name):
    configure()
    root = Path(root_name)
    plan = read(root / "protocol.json")
    checked_inputs(plan)
    models = {n: PublishedAgent(n) for n in ("wp", "point")}
    predictors = {n: StrictPublishedPrediction(a) for n, a in models.items()}
    opponent = EndgameAgent(models[opponent_name])
    actors = {
        "parent": NeuralAgent(parent["path"], deterministic=True),
        **{
            f"fixed_{mode}": ContinuationAgent(
                FIXED / f"parent{parent['index']}_{mode}/final.pt", predictors
            )
            for mode in ("public", "masked")
        },
        **{
            f"renewed_{mode}": RenewedContinuationAgent(
                root / f"parent{parent['index']}_{mode}/final.pt", predictors
            )
            for mode in ("public", "masked")
        },
    }
    assert tuple(actors) == POLICIES
    native = BatchEnv(1, "cpu", "paper8")
    path = root / f"eval_parent{parent['index']}_{opponent_name}.jsonl"
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        for policy, actor in actors.items():
            for case in cases(parent["index"], opponent_name):
                row = play_continuation_match(case, actor, opponent, native)
                stream.write(json.dumps({"policy": policy, **row}) + "\n")
                stream.flush()
    return str(path)


def main():
    from scripts.audit_renewed_continuation_study import audit_matched_warmup

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a fresh study directory")
    configure()
    plan = make_plan()
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output / "protocol.json", plan)
    try:
        for parent in plan["parents"]:
            for mode in ("public", "masked"):
                write(
                    args.output / "process.json",
                    {
                        "status": "running",
                        "stage": "training",
                        "parent": parent["index"],
                        "mode": mode,
                    },
                )
                train_arm(args.output, parent, mode, plan)
            audit_matched_warmup(args.output, parent["index"])
        write(
            args.output / "process.json", {"status": "running", "stage": "evaluation"}
        )
        with concurrent.futures.ProcessPoolExecutor(max_workers=3) as pool:
            futures = [
                pool.submit(evaluate_cell, str(args.output), parent, opponent)
                for parent in plan["parents"]
                for opponent in ("wp", "point")
            ]
            for future in concurrent.futures.as_completed(futures):
                print(json.dumps({"evaluation_complete": future.result()}), flush=True)
        checked_inputs(plan)
        result = analyse(load_points(args.output))
        files = [
            p
            for p in args.output.rglob("*")
            if p.is_file() and p.name != "process.json"
        ]
        write(
            args.output / "analysis.json",
            {
                "status": "complete",
                "contrasts": result,
                "decisions": decisions(result),
                "inputs": {str(p): sha(p) for p in files},
                "scope": "Prospective fixed/renewed chance and public/masked comparison; independent audit pending; no promotion.",
            },
        )
        write(args.output / "process.json", {"status": "complete"})
    except BaseException as error:
        write(args.output / "process.json", {"status": "failed", "error": repr(error)})
        raise


if __name__ == "__main__":
    main()
