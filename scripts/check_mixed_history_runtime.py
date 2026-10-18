"""Actual published-policy likelihoods versus the four saved exact supports."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from koikoi.agents import EndgameAgent
from koikoi.reference import ReferenceEnv
from koikoi.search import sample_hidden
from koikoi.upstream import PublishedAgent
from scripts.diagnose_history_support import reference, replay
from scripts.enumerate_history_support import canonical_world, context
from scripts.mixed_history_sampler import draw_hand_chain, sample_public_hand_chain
from scripts.run_arbitration_study import read, sha, write


def run(output):
    if output.exists():
        raise ValueError("Use a fresh output path")
    bench_path = Path("runs/mixed_history_sampler_benchmark/report.json")
    bench = read(bench_path)
    assert bench["status"] == "complete"
    inputs = dict(bench["inputs"])
    for path in (bench_path, Path(__file__)):
        inputs[str(path)] = sha(path)
    for path, digest in inputs.items():
        assert sha(path) == digest, path
    torch.set_num_threads(2)
    models = {name: EndgameAgent(PublishedAgent(name)) for name in ("wp", "point")}
    base = Path("runs/sampled_history_exact_check")
    records, source_steps, producer_work, full_work = [], 0, {}, {}
    for row in read(base / "report.json")["results"]:
        source_path = Path(
            f"runs/final_round_training_study/parent{row['parent']}/source.jsonl"
        )
        source = json.loads(
            source_path.read_text(encoding="utf-8").splitlines()[row["case_index"]]
        )
        env = ReferenceEnv(source["seed"], "paper8")
        for action in source["actions"][: row["decision"]]:
            assert env.legal()[action]
            env.step(action)
            source_steps += 1
        batch = read(base / row["support_file"])
        model_name = next(
            iter(row["exact_beliefs"]["identified"]["model_probabilities"])
        )
        model = models[model_name]
        column = batch["model_names"].index(model_name)
        support = {
            tuple(h)
            for h, flags in zip(batch["hands"], batch["compatible"])
            if flags[column]
        }
        seed = 1701 + 1000000 * row["parent"] + 1000 * row["case_index"]
        actual = sample_public_hand_chain(
            env, {model_name: model}, np.random.default_rng(seed)
        )
        unknown, size, _ = context(env.state)
        oracle = draw_hand_chain(
            unknown, size, support.__contains__, np.random.default_rng(seed)
        )
        assert actual["chain"] == oracle
        alternate = reference(
            sample_hidden(env, np.random.default_rng(seed + 90000000))
        )
        substituted = sample_public_hand_chain(
            alternate, {model_name: model}, np.random.default_rng(seed)
        )
        assert actual == substituted
        for result in (actual, substituted):
            for name, value in result["history_work"].items():
                producer_work[name] = producer_work.get(name, 0) + value
        for hand, compatible in actual["chain"].checked:
            world = canonical_world(env.state, hand)
            if world is None:
                assert not compatible
                continue
            full = replay(world, model)
            assert full["compatible"] == compatible
            for name, value in full["counts"].items():
                full_work[name] = full_work.get(name, 0) + value
        records.append(
            {
                "parent": row["parent"],
                "opponent": row["opponent"],
                "chain": asdict(actual["chain"]),
                "history_work": actual["history_work"],
            }
        )
        print(json.dumps({"verified_cases": len(records)}), flush=True)
    for path, digest in inputs.items():
        assert sha(path) == digest, path
    write(
        output,
        {
            "status": "passed",
            "inputs": inputs,
            "records": records,
            "source_prefix_replay_decisions": source_steps,
            "actual_and_hidden_substituted_history_work": producer_work,
            "unpruned_likelihood_recheck": full_work,
            "proposal_attempts": 4 * 128 * 3,
            "new_evaluation_deals": 0,
            "future_rollouts": 0,
            "scope": "One fixed RNG replicate per development case; actual wrapper, cached oracle and hidden-world substitute agree. Full replay verifies every queried valid hand, sharing canonical construction and published inference adapter.",
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
