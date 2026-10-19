"""Enumerate every eligible root in the existing sampled-history diagnostic."""

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from koikoi.agents import EndgameAgent
from koikoi.layout import PLAYER
from koikoi.reference import ReferenceEnv
from koikoi.upstream import PublishedAgent
from scripts.diagnose_sampled_history_support import beliefs
from scripts.enumerate_history_support import context
from scripts.history_conditioned_worlds import observation_key
from scripts.run_arbitration_study import read, sha, write
from scripts.sampled_history_support import (
    HistoryProposalBatch,
    propose_history_support,
)

BASE = Path("runs/sampled_history_diagnostic")
PRIOR = Path("runs/sampled_history_exact_check")


def select_records(records):
    if (
        len(records) != 32
        or len({(r["parent"], r["case_index"]) for r in records}) != 32
    ):
        raise ValueError("Expected the complete unique32-attempt diagnostic")
    selected = [r for r in records if r["status"] == "eligible"]
    if len(selected) != 25 or any(
        r["status"] not in ("eligible", "ineligible") for r in records
    ):
        raise ValueError("Expected25eligible and7ineligible attempts")
    return selected


def batch_from_record(record):
    data = dict(record)
    data["model_names"] = tuple(data["model_names"])
    data["hands"] = tuple(tuple(h) for h in data["hands"])
    data["compatible"] = tuple(tuple(c) for c in data["compatible"])
    batch = HistoryProposalBatch(**data)
    if (
        batch.mode != "enumerated"
        or batch.candidate_hands != batch.proposed_draws
        or len(batch.hands) + batch.invalid_draws != batch.candidate_hands
        or len(batch.hands) != len(set(batch.hands))
        or len(batch.compatible) != len(batch.hands)
        or batch.model_names != ("point", "wp")
    ):
        raise ValueError("Invalid exact support table")
    return batch


def insert_cache(cache, batch):
    if batch.root_key in cache:
        old = cache[batch.root_key]
        for key in (
            "model_names",
            "candidate_hands",
            "invalid_draws",
            "hands",
            "compatible",
        ):
            if getattr(old, key) != getattr(batch, key):
                raise ValueError("Conflicting exact tables for the same public root")
    else:
        cache[batch.root_key] = batch


def comparisons(estimates, sampled_beliefs, unknown, truth):
    result = {}
    for name, estimate in estimates.items():
        if estimate["status"] != "estimated":
            result[name] = {"status": estimate["status"]}
            continue
        exact = np.asarray(estimate["card_probabilities"])
        row = {
            "status": "estimated",
            "exact_card_brier": float(
                np.square(exact[unknown] - truth[unknown]).mean()
            ),
            "exact_supported_hands": estimate["distinct_supported_hands"],
            "sampled_status": sampled_beliefs[name]["status"],
        }
        if sampled_beliefs[name]["status"] == "estimated":
            sampled = np.asarray(sampled_beliefs[name]["card_probabilities"])
            row.update(
                sampled_card_brier=sampled_beliefs[name]["card_brier"],
                mean_absolute_sampling_error=float(
                    np.abs(exact[unknown] - sampled[unknown]).mean()
                ),
                maximum_absolute_sampling_error=float(
                    np.abs(exact[unknown] - sampled[unknown]).max()
                ),
                sampled_effective_proposals=sampled_beliefs[name][
                    "effective_proposals"
                ],
            )
        result[name] = row
    return result


def aggregate(results):
    groups = {"all": results}
    groups.update(
        {f"parent{p}": [r for r in results if r["parent"] == p] for p in (1, 2)}
    )
    groups.update(
        {
            name: [r for r in results if r["opponent"] == name]
            for name in ("wp", "point")
        }
    )
    output = {}
    for scope, rows in groups.items():
        metrics = {}
        for name in ("uniform", "joint", "identified"):
            values = [r["comparison"][name] for r in rows]
            exact = [
                v["exact_card_brier"] for v in values if v["status"] == "estimated"
            ]
            sampled = [
                v["sampled_card_brier"] for v in values if "sampled_card_brier" in v
            ]
            metrics[name] = {
                "exact_estimated": len(exact),
                "sampled_estimated": len(sampled),
                "exact_mean_card_brier": float(np.mean(exact)) if exact else None,
                "sampled_mean_card_brier": float(np.mean(sampled)) if sampled else None,
            }
        output[scope] = {"roots": len(rows), "beliefs": metrics}
    return output


def run(output):
    if output.exists():
        raise ValueError("Use a fresh diagnostic directory")
    audits = [BASE / "independent_replay.json", PRIOR / "arithmetic_verification.json"]
    assert [sha(p) for p in audits] == [
        "2c3b08ca3cfbf19a420576533c34a2bb7db8ce0937d8fb88bfc815580ec1eac4",
        "c9a9b51f4f337bfc9bf70396ca14604679f0fa920cf5593b2cafda7d205e6bf9",
    ]
    inputs = {}
    for path in audits:
        audit = read(path)
        assert audit["status"] == "passed"
        for name, digest in audit["inputs"].items():
            assert name not in inputs or inputs[name] == digest
            inputs[name] = digest
        inputs[str(path)] = sha(path)
    for path in [
        Path(__file__),
        Path("docs/SAMPLED_HISTORY_COMPLETE_CHECK.rst"),
        Path("tests/test_complete_sampled_history_exact.py"),
        Path("runs/complete_sampled_history_exact_tests.xml"),
    ]:
        inputs[str(path)] = sha(path)
    for path, digest in inputs.items():
        assert sha(path) == digest, path
    records = [
        json.loads(line)
        for line in (BASE / "records.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    selected = select_records(records)
    cache = {}
    for row in read(PRIOR / "report.json")["results"]:
        path = PRIOR / row["support_file"]
        assert sha(path) == row["support_sha256"]
        inputs[str(path)] = sha(path)
        insert_cache(cache, batch_from_record(read(path)))
    output.mkdir(parents=True, exist_ok=False)
    write(
        output / "protocol.json",
        {
            "inputs": inputs,
            "selection": [
                {
                    k: r[k]
                    for k in ("parent", "opponent", "case_index", "seed", "decision")
                }
                for r in selected
            ],
            "attempts": 32,
            "eligible_roots": 25,
            "ineligible_attempts": 7,
            "max_candidates": 6000,
            "torch_threads": 2,
            "prior_cached_public_roots": len(cache),
            "scope": "All previously inspected eligible development roots; inference diagnostic only.",
        },
    )
    torch.set_num_threads(2)
    models = {name: EndgameAgent(PublishedAgent(name)) for name in ("wp", "point")}
    results, work, source_steps, new_tables = [], {}, 0, 0
    write(output / "process.json", {"status": "running", "completed_roots": 0})
    try:
        for row in selected:
            path = Path(
                f"runs/final_round_training_study/parent{row['parent']}/source.jsonl"
            )
            source = json.loads(
                path.read_text(encoding="utf-8").splitlines()[row["case_index"]]
            )
            env = ReferenceEnv(source["seed"], "paper8")
            for action in source["actions"][: row["decision"]]:
                assert env.legal()[action]
                env.step(action)
                source_steps += 1
            assert env.state[PLAYER] == row["seat"]
            key = observation_key(env)
            assert key == row["proposals"]["root_key"]
            unknown, count, _ = context(env.state)
            candidates = math.comb(len(unknown), count)
            assert candidates <= 6000
            cached = key in cache
            start = time.perf_counter()
            if not cached:
                rng = np.random.default_rng(0)
                before = json.dumps(rng.bit_generator.state)
                batch = propose_history_support(env, models, rng, max_draws=6000)
                assert json.dumps(rng.bit_generator.state) == before
                insert_cache(cache, batch_from_record(asdict(batch)))
                new_tables += 1
                for name, value in batch.counts.items():
                    work[name] = work.get(name, 0) + value
            batch = cache[key]
            assert batch.candidate_hands == candidates
            table = dict(zip(batch.hands, batch.compatible))
            assert batch.model_names == tuple(row["proposals"]["model_names"])
            for hand, compatible in zip(
                row["proposals"]["hands"], row["proposals"]["compatible"]
            ):
                assert table[tuple(hand)] == tuple(compatible)
            estimates = beliefs(batch, row["public_identification"]["model"])
            # Assessment labels become visible only after public-only inference.
            truth = (env.state[:48] == 1 - row["seat"]).astype(float)
            assert int(truth.sum()) == count
            result = {
                k: row[k]
                for k in ("parent", "opponent", "case_index", "seed", "decision")
            }
            support_file = f"support_{key}.json"
            if not (output / support_file).exists():
                write(output / support_file, asdict(batch))
            result.update(
                public_root_key=key,
                reused_exact_table=cached,
                enumeration_seconds=time.perf_counter() - start,
                comparison=comparisons(estimates, row["beliefs"], unknown, truth),
                exact_beliefs=estimates,
                support_file=support_file,
                support_sha256=sha(output / support_file),
            )
            results.append(result)
            write(
                output / "process.json",
                {
                    "status": "running",
                    "completed_roots": len(results),
                    "new_exact_tables": new_tables,
                },
            )
            print(
                json.dumps(
                    {"completed_roots": len(results), "new_exact_tables": new_tables}
                ),
                flush=True,
            )
        for path, digest in inputs.items():
            assert sha(path) == digest, path
        write(
            output / "report.json",
            {
                "status": "complete",
                "inputs": inputs,
                "results": results,
                "aggregate": aggregate(results),
                "protocol_sha256": sha(output / "protocol.json"),
                "distinct_public_roots": len({r["public_root_key"] for r in results}),
                "shared_source_deals": len({r["seed"] for r in results}),
                "new_exact_tables": new_tables,
                "cached_root_uses": sum(r["reused_exact_table"] for r in results),
                "source_prefix_replay_decisions": source_steps,
                "new_hypothetical_history_replay": work,
                "new_evaluation_deals": 0,
                "future_rollouts": 0,
                "new_training_matches": 0,
            },
        )
        write(output / "process.json", {"status": "complete"})
    except BaseException as error:
        write(output / "process.json", {"status": "failed", "error": repr(error)})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
