"""Frozen two-parent natural-tail adaptation and strict full-match comparison."""

import argparse
import concurrent.futures
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from koikoi.agents import EndgameAgent, NeuralAgent
from koikoi.env import BatchEnv
from koikoi.evaluate import paired_interval
from koikoi.train import provenance
from koikoi.upstream import UPSTREAM_PATH, PublishedAgent
from scripts.final_round_actual_match import play_final_round_match
from scripts.final_round_training import (
    TailAgent,
    TailConfig,
    TailTrainer,
    bank_identity,
    prefix_to_final_round,
)
from scripts.run_arbitration_study import PARENTS, read, sha, write


def configure():
    torch.set_num_threads(2)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def cases(parent, split):
    if split not in ("source", "evaluation"):
        raise ValueError("Unknown split")
    base, count = (88110001, 256) if split == "source" else (89110001, 1024)
    return [
        {
            "parent": parent,
            "opponent": name,
            "seat": seat,
            "replicate": rep,
            "seed": base + 10000 * opponent + rep,
        }
        for opponent, name in enumerate(("wp", "point"))
        for seat in (0, 1)
        for rep in range(count)
    ]


def checked_inputs(plan):
    for name, digest in plan["inputs"].items():
        if sha(name) != digest:
            raise ValueError(f"Study input changed: {name}")
    if provenance()["source_sha256"] != plan["core_source_sha256"]:
        raise ValueError("Core source changed")


def make_plan():
    capability = Path("runs/final_round_training_capability/audit.json")
    audit = read(capability)
    assert audit["status"] == "passed"
    for path, digest in audit["inputs"].items():
        assert sha(path) == digest
    files = [Path(__file__), capability] + [
        Path(p)
        for p in (
            "scripts/audit_final_round_training_study.py",
            "scripts/final_round_training.py",
            "scripts/final_round_actual_match.py",
            "scripts/fixed_microbatch_continuation.py",
            "scripts/run_arbitration_study.py",
            "docs/FINAL_ROUND_TRAINING.rst",
            "docs/FINAL_ROUND_TRAINING_STUDY.rst",
            "tests/test_final_round_training.py",
            "tests/test_final_round_training_protocol.py",
            "runs/final_round_training_tests.xml",
            "runs/final_round_training_protocol_tests.xml",
            "runs/final_round_search_completion.json",
        )
    ]
    files.extend(Path(p) for p in audit["inputs"])
    files.extend(
        p for p in Path("koikoi").rglob("*") if p.suffix in (".py", ".h", ".cpp", ".cu")
    )
    files.extend(
        UPSTREAM_PATH / f"model_agent/{phase}_rl_{name}.pt"
        for phase in ("discard", "pick", "koikoi")
        for name in ("wp", "point")
    )
    parents = []
    for i, (path, expected) in enumerate(PARENTS, 1):
        assert sha(path) == expected
        files.append(Path(path))
        parents.append(
            {
                "index": i,
                "path": path,
                "sha256": expected,
                "config": asdict(
                    TailConfig(device="cuda", seed=88210001 + 10000 * (i - 1))
                ),
            }
        )
    return {
        "kind": "fixed_prefix_terminal_tail_v1",
        "parents": parents,
        "updates": 520,
        "source_cases_per_parent": 1024,
        "evaluation_pairs_per_cell": 1024,
        "bootstrap_seed": 89310001,
        "bootstrap_replicates": 10000,
        "inputs": {str(p): sha(p) for p in files},
        "core_source_sha256": provenance()["source_sha256"],
        "selection": "Only update520; all four paired win lower bounds must be positive.",
    }


def collect_bank(root, parent):
    folder = root / f"parent{parent['index']}"
    folder.mkdir()
    actor = NeuralAgent(parent["path"], deterministic=True)
    opponents = {n: EndgameAgent(PublishedAgent(n)) for n in ("wp", "point")}
    native = BatchEnv(1, "cpu", "paper8")
    states, seats, labels = [], [], []
    with (folder / "source.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for index, case in enumerate(cases(parent["index"], "source")):
            state, row = prefix_to_final_round(
                case, actor, opponents[case["opponent"]], native
            )
            row["root_index"] = len(states) if state is not None else None
            if state is not None:
                states.append(state)
                seats.append(case["seat"])
                labels.append(("wp", "point").index(case["opponent"]))
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            if (index + 1) % 128 == 0:
                print(
                    json.dumps(
                        {
                            "parent": parent["index"],
                            "source_prefixes": index + 1,
                            "survivors": len(states),
                        }
                    ),
                    flush=True,
                )
    bank = (
        torch.from_numpy(np.stack(states)),
        torch.tensor(seats),
        torch.tensor(labels),
    )
    torch.save(bank, folder / "bank.pt")
    return bank


def audit_checkpoint(folder, parent, plan):
    payload = torch.load(folder / "final.pt", map_location="cpu", weights_only=False)
    initial = torch.load(folder / "initial.pt", map_location="cpu", weights_only=False)
    original = torch.load(parent["path"], map_location="cpu", weights_only=False)
    c = payload["counters"]
    assert c["updates"] == plan["updates"] == 520
    assert c["policy_updates"] == 500 and c["warmup_updates"] == 20
    assert c["conditional_tail_completions"] == c["conditional_round_endings"] == 66560
    assert c["matches"] == c["rounds"] == 0
    assert c["decisions"] == c["learner_decisions"] + c["opponent_decisions"]
    assert payload["config"] == parent["config"]
    assert payload["provenance"]["weight_initialization"]["sha256"] == parent["sha256"]
    bank = torch.load(folder / "bank.pt", map_location="cpu", weights_only=False)
    assert bank_identity(*bank) == payload["provenance"]["root_bank_sha256"]
    for actual in (payload["prefix_actor"], initial["prefix_actor"], initial["actor"]):
        for phase, weights in original["actor"]["current"]["weights"].items():
            for name, value in weights.items():
                assert torch.equal(actual["current"]["weights"][phase][name], value)
    for name in ("wp", "point"):
        for phase, weights in initial["opponents"][name]["current"]["weights"].items():
            for key, value in weights.items():
                assert torch.equal(
                    payload["opponents"][name]["current"]["weights"][phase][key], value
                )
    model = TailTrainer.restore(folder / "final.pt")
    optimizer_counts = {}
    for name, parameter in model.model.named_parameters():
        key = (
            "value"
            if name.startswith("value.")
            else {"0": "discard", "1": "pick", "3": "koi"}[name.split(".")[2]]
        )
        entry = model.optimizer.state[parameter]
        assert int(entry["step"]) == c["optimizer_steps_" + key]
        assert all(bool(torch.isfinite(value).all()) for value in entry.values())
        optimizer_counts[name] = int(entry["step"])
    del model
    write(
        folder / "training_audit.json",
        {
            "status": "passed",
            "counters": c,
            "optimizer_steps": optimizer_counts,
            "inputs": {
                str(p): sha(p)
                for p in (
                    folder / "bank.pt",
                    folder / "source.jsonl",
                    folder / "initial.pt",
                    folder / "final.pt",
                    folder / "metrics.jsonl",
                )
            },
        },
    )


def train_parent(root, parent, plan):
    bank = collect_bank(root, parent)
    folder = root / f"parent{parent['index']}"
    trainer = TailTrainer.from_parent(
        TailConfig(**parent["config"]), parent["path"], *bank
    )
    trainer.save(folder / "initial.pt")
    start = time.perf_counter()
    with (folder / "metrics.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for update in range(1, plan["updates"] + 1):
            metrics = trainer.update()
            row = {
                "update": update,
                "seconds": time.perf_counter() - start,
                "counters": trainer.counters.copy(),
                "metrics": metrics,
            }
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            if update % 100 == 0 or update == plan["updates"]:
                trainer.save(folder / f"update_{update:08d}.pt")
            if update % 20 == 0:
                print(json.dumps({"parent": parent["index"], **row}), flush=True)
    trainer.save(folder / "final.pt")
    del trainer
    checked_inputs(plan)
    audit_checkpoint(folder, parent, plan)


def evaluate_cell(root_name, parent, opponent_name):
    configure()
    root = Path(root_name)
    plan = read(root / "protocol.json")
    checked_inputs(plan)
    opponent = EndgameAgent(PublishedAgent(opponent_name))
    actors = {
        "parent": NeuralAgent(parent["path"], deterministic=True),
        "tail": TailAgent(root / f"parent{parent['index']}" / "final.pt"),
    }
    native = BatchEnv(1, "cpu", "paper8")
    path = root / f"eval_parent{parent['index']}_{opponent_name}.jsonl"
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        for policy, actor in actors.items():
            for case in cases(parent["index"], "evaluation"):
                if case["opponent"] != opponent_name:
                    continue
                _, row = play_final_round_match(case, actor, opponent, native)
                stream.write(json.dumps({"policy": policy, **row}) + "\n")
                stream.flush()
    return str(path)


def contrasts(root, plan):
    result = []
    for parent in (1, 2):
        for opponent in ("wp", "point"):
            points = np.zeros((2, 1024, 2), dtype=float)
            seen = set()
            path = root / f"eval_parent{parent}_{opponent}.jsonl"
            for line in path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                i = ("parent", "tail").index(row["policy"])
                j, k = row["replicate"], row["seat"]
                assert (i, j, k) not in seen
                seen.add((i, j, k))
                scores = row["terminal_points"]
                points[i, j, k] = scores[k] - scores[1 - k]
            assert len(seen) == 4096
            wins = (np.sign(points) + 1) / 2
            row = {"parent": parent, "opponent": opponent}
            for label, values in (("win", wins), ("point", points)):
                change = (values[1] - values[0]).mean(axis=1)
                row[label] = {
                    "parent_mean": float(values[0].mean()),
                    "tail_mean": float(values[1].mean()),
                    "change": float(change.mean()),
                    "interval95": paired_interval(
                        change, plan["bootstrap_seed"], plan["bootstrap_replicates"]
                    ),
                }
            result.append(row)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    configure()
    plan = make_plan()
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output / "protocol.json", plan)
    write(args.output / "process.json", {"status": "running", "stage": "training"})
    try:
        for parent in plan["parents"]:
            train_parent(args.output, parent, plan)
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
        result = contrasts(args.output, plan)
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
                "replicated_gain": all(r["win"]["interval95"][0] > 0 for r in result),
                "inputs": {str(p): sha(p) for p in files},
                "scope": "Prospective development comparison; independent audit pending; no promotion.",
            },
        )
        write(args.output / "process.json", {"status": "complete"})
    except BaseException as error:
        write(args.output / "process.json", {"status": "failed", "error": repr(error)})
        raise


if __name__ == "__main__":
    main()
