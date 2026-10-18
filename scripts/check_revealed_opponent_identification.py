"""Identify fixed WP/Point candidates from retained public training histories.

Run only after the ongoing tail study finishes its strict policy evaluation.
No evaluation records, private oracle inputs or opponent label reach predictors.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from koikoi.agents import solved_koi_action
from koikoi.layout import PLAYER, ROUND
from koikoi.reference import ReferenceEnv
from koikoi.upstream import UPSTREAM_PATH, PublishedAgent, reconstruct
from scripts.revealed_opponent_inputs import public_snapshot
from scripts.revealed_opponent_models import RevealedModelSet, StrictPublishedPrediction
from scripts.run_arbitration_study import read, sha, write


def verify_oracle(event, states):
    """Audit only: predictors have already run using public reconstruction.

    These simulator states are never supplied to the candidate-set tracker.
    Verify every input hash against the strict original builder, not merely
    against another invocation of the new decoder.
    """
    for check in event["checks"]:
        ref = ReferenceEnv.__new__(ReferenceEnv)
        ref.state = states[check["decision"]]
        strict = reconstruct(ref).feature_tensor.numpy()
        padded = np.zeros((300, 50), dtype=np.float32)
        padded[:, : strict.shape[1]] = strict
        assert hashlib.sha256(padded.tobytes()).hexdigest() == check["features_sha256"]
        mask = ref.legal()
        solved = solved_koi_action(ref)
        if solved is not None:
            mask[:] = False
            mask[solved] = True
        assert (
            hashlib.sha256(mask.tobytes()).hexdigest() == check["assisted_mask_sha256"]
        )
        assert all(mask[a] for a in check["predictions"].values())


def run(base, output):
    if read(base / "process.json")["status"] != "complete":
        raise ValueError("Wait for the strict tail-study evaluation to finish")
    if read(base / "analysis.json")["status"] != "complete":
        raise ValueError("Tail study is incomplete")
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    files = [Path(__file__)] + [
        Path(p)
        for p in (
            "scripts/revealed_opponent_models.py",
            "scripts/revealed_opponent_inputs.py",
            "scripts/check_seat_feature_reconstruction.py",
            "scripts/check_public_hand_revelation.py",
            "tests/test_revealed_opponent_models.py",
            "runs/revealed_opponent_model_tests.xml",
            "runs/revealed_opponent_input_capability.json",
            "runs/public_hand_revelation.json",
            "docs/REVEALED_OPPONENT_IDENTIFICATION.rst",
            "koikoi/upstream.py",
            "koikoi/reference.py",
            "koikoi/layout.py",
            "koikoi/agents.py",
        )
    ]
    files += [base / name for name in ("protocol.json", "training_completion.json")]
    files += [base / f"parent{i}" / "source.jsonl" for i in (1, 2)]
    files += [UPSTREAM_PATH / "koikoigame.py", UPSTREAM_PATH / "koikoinet2L.py"]
    files += [
        UPSTREAM_PATH / f"model_agent/{phase}_rl_{variant}.pt"
        for phase in ("discard", "pick", "koikoi")
        for variant in ("wp", "point")
    ]
    inputs = {str(p): sha(p) for p in files}
    completion = read(base / "training_completion.json")
    opportunities = read("runs/public_hand_revelation.json")
    for path in (base / f"parent{i}" / "source.jsonl" for i in (1, 2)):
        assert (
            completion["inputs"][str(path)]
            == opportunities["inputs"][str(path)]
            == sha(path)
        )
    expected_events = {
        (r["parent"], r["case_index"]): r["revelations"]
        for r in opportunities["records"]
    }
    models = {
        name: StrictPublishedPrediction(PublishedAgent(name))
        for name in ("wp", "point")
    }
    groups = {}
    work = dict.fromkeys(("decisions", "round_endings", "completed_matches"), 0)
    predictions = dict.fromkeys(models, 0)
    oracle_checks = 0
    start = time.perf_counter()
    write(
        output / "protocol.json",
        {
            "inputs": inputs,
            "source_attempts": 2048,
            "candidate_models": list(models),
            "selection": "All retained training prefixes; no evaluation records.",
        },
    )
    with (output / "records.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for parent in (1, 2):
            rows = [
                json.loads(line)
                for line in (base / f"parent{parent}" / "source.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            assert len(rows) == 1024
            for index, row in enumerate(rows):
                tracker = RevealedModelSet(
                    f"parent{parent}-case{index}", row["seat"], models
                )
                env = ReferenceEnv(row["seed"], "paper8")
                oracle_states, old_round = {}, -1
                available_learner_decisions = 0
                for decision, action in enumerate(row["actions"]):
                    current_round = int(env.state[ROUND])
                    if current_round != old_round:
                        oracle_states = {}
                        old_round = current_round
                    opponent_turn = int(env.state[PLAYER]) != row["seat"]
                    if opponent_turn:
                        snapshot = public_snapshot(
                            env.observe(),
                            int(env.state[PLAYER]),
                            decision,
                            match_key=tracker.match_key,
                        )
                        oracle_states[decision] = env.state.copy()
                    elif tracker.status == "identified":
                        available_learner_decisions += 1
                    result = env.step(action)
                    work["decisions"] += 1
                    work["round_endings"] += int(result[1])
                    work["completed_matches"] += int(result[2])
                    if opponent_turn:
                        event = tracker.observe(snapshot, int(action))
                        if event is not None:
                            verify_oracle(event, oracle_states)
                            oracle_checks += len(event["checks"])
                np.testing.assert_array_equal(env.state, row["endpoint"])
                assert [
                    {
                        k: e[k]
                        for k in (
                            "round",
                            "available_after_decision",
                            "initial_opponent_hand",
                        )
                    }
                    for e in tracker.events
                ] == expected_events[parent, index]
                label = f"parent{parent}_{row['opponent']}"
                group = groups.setdefault(
                    label,
                    {
                        "source_prefixes": 0,
                        "identified_correct": 0,
                        "identified_wrong": 0,
                        "ambiguous": 0,
                        "inconsistent": 0,
                        "true_model_excluded": 0,
                        "first_identification_round_counts": [0] * 7,
                        "final_round_survivors": 0,
                        "correctly_identified_final_round_survivors": 0,
                        "learner_decisions_with_singleton_candidate": 0,
                    },
                )
                group["source_prefixes"] += 1
                correct = tracker.candidates == {row["opponent"]}
                if tracker.status == "identified":
                    group["identified_correct" if correct else "identified_wrong"] += 1
                else:
                    group[tracker.status] += 1
                group["true_model_excluded"] += (
                    row["opponent"] not in tracker.candidates
                )
                group["final_round_survivors"] += row["reached_final_round"]
                group["correctly_identified_final_round_survivors"] += (
                    correct and row["reached_final_round"]
                )
                group["learner_decisions_with_singleton_candidate"] += (
                    available_learner_decisions
                )
                if tracker.first_identification is not None:
                    group["first_identification_round_counts"][
                        tracker.first_identification["round"]
                    ] += 1
                for name, count in tracker.prediction_calls.items():
                    predictions[name] += count
                stream.write(
                    json.dumps(
                        {
                            "parent": parent,
                            "case_index": index,
                            "opponent": row["opponent"],
                            "seat": row["seat"],
                            "seed": row["seed"],
                            "status": tracker.status,
                            "candidates": sorted(tracker.candidates),
                            "first_identification": tracker.first_identification,
                            "events": tracker.events,
                            "prediction_calls": tracker.prediction_calls,
                            "learner_decisions_with_singleton_candidate": available_learner_decisions,
                        }
                    )
                    + "\n"
                )
                stream.flush()
                if (index + 1) % 128 == 0:
                    print(
                        json.dumps(
                            {
                                "parent": parent,
                                "source_prefixes": index + 1,
                                "prediction_calls": predictions,
                                "seconds": time.perf_counter() - start,
                            }
                        ),
                        flush=True,
                    )
    assert work == completion["source_work"]
    for p, digest in inputs.items():
        assert sha(p) == digest, p
    write(
        output / "audit.json",
        {
            "status": "complete",
            "inputs": inputs,
            "groups": groups,
            "records_sha256": sha(output / "records.jsonl"),
            "source_attempts": 2048,
            "distinct_deals": 512,
            "replayed_source_work": work,
            "strict_oracle_input_checks": oracle_checks,
            "prediction_calls": predictions,
            "seconds": time.perf_counter() - start,
            "new_independent_matches": 0,
            "scope": "Finite fixed WP/Point identification on retained training histories; no response policy or strength evaluation.",
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.base, args.output)


if __name__ == "__main__":
    main()
