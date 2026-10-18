"""Fixed training-prefix diagnostic of bounded public-history hand inference."""

import argparse
import json
import math
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from koikoi.agents import EndgameAgent
from koikoi.layout import DISCARD, PHASE, PLAYER, ROUND, TERMINAL
from koikoi.reference import ReferenceEnv
from koikoi.upstream import UPSTREAM_PATH, PublishedAgent
from scripts.enumerate_history_support import context
from scripts.run_arbitration_study import read, sha, write
from scripts.sampled_history_support import marginal_summary, propose_history_support


def eligible(env, seat, identification, decision):
    if identification is None or env.state[PHASE] != DISCARD:
        return False
    if identification["model"] not in ("wp", "point"):
        raise ValueError("Invalid public identification")
    return bool(
        env.state[PLAYER] == seat
        and env.legal().sum() > 1
        and 0 <= identification["round"] < env.state[ROUND] < 8
        and 1 <= identification["available_after_decision"] <= decision
        and round(float(env.observe()[541]) * 8) in (3, 4)
    )


def select_root(row, identification):
    env = ReferenceEnv(row["seed"], "paper8")
    root = None
    work = dict.fromkeys(("decisions", "round_endings", "completed_matches"), 0)
    for decision in range(len(row["actions"]) + 1):
        if root is None and eligible(env, row["seat"], identification, decision):
            root = (env.state.copy(), decision)
        if decision == len(row["actions"]):
            break
        if env.state[PHASE] == TERMINAL:
            raise ValueError("Source continues beyond termination")
        result = env.step(row["actions"][decision])
        if int(result[0]) == 99:
            raise ValueError("Illegal source action")
        work["decisions"] += 1
        work["round_endings"] += int(result[1])
        work["completed_matches"] += int(result[2])
    np.testing.assert_array_equal(env.state, row["endpoint"])
    assert work == row["work"]
    return root, work


def beliefs(batch, identified):
    uniform = replace(
        batch, model_names=("uniform",), compatible=((True,),) * len(batch.hands)
    )
    column = batch.model_names.index(identified)
    known = replace(
        batch,
        model_names=(identified,),
        compatible=tuple((row[column],) for row in batch.compatible),
    )
    result = {}
    for name, proposal in (
        ("uniform", uniform),
        ("joint", batch),
        ("identified", known),
    ):
        try:
            result[name] = {"status": "estimated", **marginal_summary(proposal)}
        except ValueError as error:
            if str(error) != "Zero compatible proposal mass":
                raise
            result[name] = {"status": "zero_sampled_mass"}
    return result


def aggregate(records):
    roots = [r for r in records if r["status"] == "eligible"]
    result = {"attempts": len(records), "eligible_roots": len(roots), "beliefs": {}}
    for name in ("uniform", "joint", "identified"):
        values = [r["beliefs"][name] for r in roots]
        available = [v for v in values if v["status"] == "estimated"]
        result["beliefs"][name] = {
            "estimated": len(available),
            "zero_sampled_mass": len(values) - len(available),
            "mean_card_brier": float(np.mean([v["card_brier"] for v in available]))
            if available
            else None,
        }
    result["paired_descriptive_differences"] = {}
    for left, right in (
        ("joint", "uniform"),
        ("identified", "uniform"),
        ("identified", "joint"),
    ):
        differences = [
            r["beliefs"][left]["card_brier"] - r["beliefs"][right]["card_brier"]
            for r in roots
            if all(r["beliefs"][n]["status"] == "estimated" for n in (left, right))
        ]
        result["paired_descriptive_differences"][f"{left}_minus_{right}"] = {
            "roots": len(differences),
            "mean": float(np.mean(differences)) if differences else None,
        }
    return result


def run(output):
    if output.exists():
        raise ValueError("Use a fresh diagnostic directory")
    audit_path = Path("runs/public_memory_continuation_bank/audit.json")
    audit = read(audit_path)
    assert (
        sha(audit_path)
        == "f2e49203737b4b51e54ab39cd5f73fc669e47d24c68c10081226e224f2556120"
    )
    inputs = dict(audit["inputs"])
    inputs[str(audit_path)] = sha(audit_path)
    paths = [
        Path(__file__),
        Path("scripts/sampled_history_support.py"),
        Path("scripts/pruned_history_support.py"),
        Path("scripts/enumerate_history_support.py"),
        Path("scripts/history_conditioned_worlds.py"),
        Path("scripts/diagnose_history_support.py"),
        Path("docs/SAMPLED_HISTORY_SUPPORT.rst"),
        Path("tests/test_sampled_history_support.py"),
        Path("runs/sampled_history_support_tests.xml"),
        Path("tests/test_sampled_history_diagnostic.py"),
        Path("runs/sampled_history_diagnostic_tests_final.xml"),
    ]
    paths += [
        UPSTREAM_PATH / f"model_agent/{phase}_rl_{name}.pt"
        for name in ("wp", "point")
        for phase in ("discard", "pick", "koikoi")
    ]
    inputs.update({str(p): sha(p) for p in paths})
    for path, digest in inputs.items():
        assert sha(path) == digest, path
    identity_path = Path("runs/revealed_opponent_identification/records.jsonl")
    assert sha(identity_path) == inputs[str(identity_path)]
    identities = {
        (r["parent"], r["case_index"]): r
        for r in map(json.loads, identity_path.read_text(encoding="utf-8").splitlines())
    }
    attempts = []
    for parent in (1, 2):
        path = Path(f"runs/final_round_training_study/parent{parent}/source.jsonl")
        assert sha(path) == inputs[str(path)]
        for index, row in enumerate(
            map(json.loads, path.read_text(encoding="utf-8").splitlines())
        ):
            if row["replicate"] not in range(4):
                continue
            identity = identities[parent, index]
            assert all(
                identity[k] == row[k] for k in ("parent", "seat", "seed", "opponent")
            )
            attempts.append((parent, index, row, identity["first_identification"]))
    assert len(attempts) == 32
    assert (
        len({(p, r["opponent"], r["seat"], r["replicate"]) for p, _, r, _ in attempts})
        == 32
    )
    output.mkdir(parents=True, exist_ok=False)
    write(
        output / "protocol.json",
        {
            "inputs": inputs,
            "attempts": 32,
            "replicates": list(range(4)),
            "proposal_budget": 128,
            "proposal_seed": "96010001 + 10000 * parent + case_index",
            "torch_threads": 2,
            "scope": "Existing training prefixes; descriptive belief diagnostic only.",
        },
    )
    torch.set_num_threads(2)
    models = {name: EndgameAgent(PublishedAgent(name)) for name in ("wp", "point")}
    records, source_work, inference_work = [], {}, {}
    for parent, index, row, identity in attempts:
        selected, work = select_root(row, identity)
        for name, value in work.items():
            source_work[name] = source_work.get(name, 0) + value
        record = {
            k: row[k] for k in ("parent", "opponent", "seat", "seed", "replicate")
        }
        record.update(
            case_index=index, public_identification=identity, status="ineligible"
        )
        if selected is not None:
            state, decision = selected
            env = ReferenceEnv.__new__(ReferenceEnv)
            env.state = state
            unknown, count, _ = context(state)
            assert math.comb(len(unknown), count) > 128
            start = time.perf_counter()
            batch = propose_history_support(
                env,
                models,
                np.random.default_rng(96010001 + parent * 10000 + index),
                max_draws=128,
            )
            prediction = beliefs(batch, identity["model"])
            elapsed = time.perf_counter() - start
            # Ground truth is read only after every estimator has returned.
            truth = (state[:48] == 1 - row["seat"]).astype(float)
            assert int(truth.sum()) == count
            for estimate in prediction.values():
                if estimate["status"] == "estimated":
                    values = np.asarray(estimate["card_probabilities"])
                    estimate["card_brier"] = float(
                        np.square(values[unknown] - truth[unknown]).mean()
                    )
            for name, value in batch.counts.items():
                inference_work[name] = inference_work.get(name, 0) + value
            record.update(
                status="eligible",
                decision=decision,
                round=int(state[ROUND]),
                opponent_cards=count,
                unknown_cards=len(unknown),
                inference_seconds=elapsed,
                proposals=asdict(batch),
                beliefs=prediction,
            )
        records.append(record)
        with (output / "records.jsonl").open(
            "a", encoding="utf-8", newline="\n"
        ) as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps({"processed_attempts": len(records)}), flush=True)
    for path, digest in inputs.items():
        assert sha(path) == digest, path
    report = {
        "status": "complete",
        "inputs": inputs,
        "protocol_sha256": sha(output / "protocol.json"),
        "records_sha256": sha(output / "records.jsonl"),
        "summary": aggregate(records),
        "source_replay": source_work,
        "hypothetical_history_replay": inference_work,
        "new_evaluation_deals": 0,
        "future_rollouts": 0,
        "source_distinct_seeds": len({r["seed"] for _, _, r, _ in attempts}),
    }
    write(output / "report.json", report)
    print(json.dumps(report["summary"]), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
