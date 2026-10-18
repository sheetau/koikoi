"""Four fixed development roots: sampled estimates versus full hand enumeration."""

import argparse
import json
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
from scripts.run_arbitration_study import read, sha, write
from scripts.sampled_history_support import propose_history_support


def run(output):
    if output.exists():
        raise ValueError("Use a fresh output directory")
    base = Path("runs/sampled_history_diagnostic")
    audit_path = base / "independent_replay.json"
    assert (
        sha(audit_path)
        == "2c3b08ca3cfbf19a420576533c34a2bb7db8ce0937d8fb88bfc815580ec1eac4"
    )
    audit = read(audit_path)
    assert audit["status"] == "passed"
    inputs = dict(audit["inputs"])
    for path in (
        audit_path,
        Path(__file__),
        Path("docs/SAMPLED_HISTORY_EXACT_CHECK.rst"),
    ):
        inputs[str(path)] = sha(path)
    for path, digest in inputs.items():
        assert sha(path) == digest, path
    records = list(
        map(
            json.loads,
            (base / "records.jsonl").read_text(encoding="utf-8").splitlines(),
        )
    )
    selected = [
        min(
            (
                r
                for r in records
                if r["parent"] == parent
                and r["opponent"] == opponent
                and r["status"] == "eligible"
            ),
            key=lambda r: r["case_index"],
        )
        for parent in (1, 2)
        for opponent in ("wp", "point")
    ]
    assert [(r["parent"], r["case_index"]) for r in selected] == [
        (1, 0),
        (1, 512),
        (2, 0),
        (2, 512),
    ]
    output.mkdir(parents=True, exist_ok=False)
    write(
        output / "protocol.json",
        {
            "inputs": inputs,
            "selection": [
                {k: r[k] for k in ("parent", "opponent", "case_index")}
                for r in selected
            ],
            "max_candidates": 6000,
            "torch_threads": 2,
            "scope": "Four previously inspected development roots; exact posterior diagnostic.",
        },
    )
    torch.set_num_threads(2)
    models = {name: EndgameAgent(PublishedAgent(name)) for name in ("wp", "point")}
    results, source_steps, work = [], 0, {}
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
        unknown, count, _ = context(env.state)
        start = time.perf_counter()
        # Enumerated mode never consumes RNG; the seed has no statistical role.
        rng = np.random.default_rng(0)
        before = json.dumps(rng.bit_generator.state)
        batch = propose_history_support(env, models, rng, max_draws=6000)
        assert batch.mode == "enumerated" and batch.candidate_hands == 5985
        assert json.dumps(rng.bit_generator.state) == before
        estimates = beliefs(batch, row["public_identification"]["model"])
        seconds = time.perf_counter() - start
        table = dict(zip(batch.hands, batch.compatible))
        assert batch.model_names == tuple(row["proposals"]["model_names"])
        for hand, compatible in zip(
            row["proposals"]["hands"], row["proposals"]["compatible"]
        ):
            assert table[tuple(hand)] == tuple(compatible)
        # Hidden truth is accessed only after exact predictions have returned.
        truth = (env.state[:48] == 1 - row["seat"]).astype(float)
        assert int(truth.sum()) == count
        comparison = {}
        for name, estimate in estimates.items():
            assert estimate["status"] == "estimated"
            exact = np.asarray(estimate["card_probabilities"])
            sampled = np.asarray(row["beliefs"][name]["card_probabilities"])
            comparison[name] = {
                "exact_card_brier": float(
                    np.square(exact[unknown] - truth[unknown]).mean()
                ),
                "sampled_card_brier": row["beliefs"][name]["card_brier"],
                "mean_absolute_sampling_error": float(
                    np.abs(exact[unknown] - sampled[unknown]).mean()
                ),
                "maximum_absolute_sampling_error": float(
                    np.abs(exact[unknown] - sampled[unknown]).max()
                ),
                "exact_supported_hands": estimate["distinct_supported_hands"],
                "sampled_effective_proposals": row["beliefs"][name][
                    "effective_proposals"
                ],
            }
        for name, value in batch.counts.items():
            work[name] = work.get(name, 0) + value
        result = {
            k: row[k] for k in ("parent", "opponent", "case_index", "seed", "decision")
        }
        result.update(
            comparison=comparison, exact_beliefs=estimates, enumeration_seconds=seconds
        )
        result["support_file"] = f"parent{row['parent']}_{row['opponent']}_support.json"
        write(output / result["support_file"], asdict(batch))
        result["support_sha256"] = sha(output / result["support_file"])
        results.append(result)
        print(json.dumps({"enumerated_roots": len(results)}), flush=True)
    for path, digest in inputs.items():
        assert sha(path) == digest, path
    write(
        output / "report.json",
        {
            "status": "complete",
            "inputs": inputs,
            "results": results,
            "protocol_sha256": sha(output / "protocol.json"),
            "source_prefix_replay_decisions": source_steps,
            "hypothetical_history_replay": work,
            "new_evaluation_deals": 0,
            "future_rollouts": 0,
            "new_training_matches": 0,
        },
    )
    print(
        json.dumps(
            [{k: r[k] for k in ("parent", "opponent", "comparison")} for r in results]
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
