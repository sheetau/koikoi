"""Training-history opportunities for using public identification before round8.

Read-only replay; no neural inference, new policy, training or held-out data.
Eligibility receives public counters and a timestamped public identification.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from koikoi.layout import DISCARD, PHASE, PLAYER, ROUND, TURN
from koikoi.reference import ReferenceEnv
from scripts.run_arbitration_study import read, sha, write


def eligible_boundary(round_index, turn, phase, decision, identification):
    """First *subsequent* fresh round, after the identifying action is observed."""
    if identification is None:
        return False
    if (
        identification["model"] not in ("wp", "point")
        or identification["round"] not in range(7)
        or identification["available_after_decision"] < 1
    ):
        raise ValueError("Invalid timestamped public identification")
    return bool(
        decision >= identification["available_after_decision"]
        and round_index > identification["round"]
        and round_index in range(1, 8)
        and turn == 0
        and phase == DISCARD
    )


def run(base, diagnostic, output):
    if output.exists():
        raise ValueError("Use a fresh diagnostic output")
    audit_path = diagnostic / "independent_audit.json"
    audit = read(audit_path)
    assert audit["status"] == "passed"
    inputs = dict(audit["inputs"])
    paths = [
        audit_path,
        Path(__file__),
        Path("koikoi/reference.py"),
        Path("koikoi/layout.py"),
        Path("tests/test_public_memory_activation.py"),
        Path("runs/public_memory_activation_tests.xml"),
    ]
    inputs.update({str(p): sha(p) for p in paths})
    for path, digest in inputs.items():
        assert sha(path) == digest, path
    identities = {
        (r["parent"], r["case_index"]): r
        for r in [
            json.loads(s)
            for s in (diagnostic / "records.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
    }
    groups, records = {}, []
    work = dict.fromkeys(("decisions", "round_endings", "completed_matches"), 0)
    for parent in (1, 2):
        source = base / f"parent{parent}/source.jsonl"
        assert sha(source) == inputs[str(source)]
        rows = [json.loads(s) for s in source.read_text(encoding="utf-8").splitlines()]
        for index, row in enumerate(rows):
            identity = identities[parent, index]
            first = identity["first_identification"]
            env = ReferenceEnv(row["seed"], "paper8")
            activation = None
            later_learner_decisions = 0
            for decision in range(len(row["actions"]) + 1):
                # The final prefix state may itself be the first eligible boundary.
                if activation is None and eligible_boundary(
                    int(env.state[ROUND]),
                    int(env.state[TURN]),
                    int(env.state[PHASE]),
                    decision,
                    first,
                ):
                    activation = {
                        "decision": decision,
                        "round": int(env.state[ROUND]),
                        "candidate": first["model"],
                    }
                if decision == len(row["actions"]):
                    break
                if activation is not None and env.state[PLAYER] == row["seat"]:
                    later_learner_decisions += 1
                result = env.step(row["actions"][decision])
                work["decisions"] += 1
                work["round_endings"] += int(result[1])
                work["completed_matches"] += int(result[2])
            np.testing.assert_array_equal(env.state, row["endpoint"])
            group = groups.setdefault(
                f"parent{parent}_{row['opponent']}",
                {
                    "prefixes": 0,
                    "identified": 0,
                    "eligible_boundaries": 0,
                    "identified_without_later_boundary": 0,
                    "activation_round_counts": [0] * 7,
                    "recorded_learner_decisions_after_activation": 0,
                    "activated_prefixes_reaching_final_round": 0,
                },
            )
            group["prefixes"] += 1
            group["identified"] += first is not None
            group["eligible_boundaries"] += activation is not None
            group["identified_without_later_boundary"] += (
                first is not None and activation is None
            )
            group["recorded_learner_decisions_after_activation"] += (
                later_learner_decisions
            )
            if activation is not None:
                group["activation_round_counts"][activation["round"] - 1] += 1
                group["activated_prefixes_reaching_final_round"] += row[
                    "reached_final_round"
                ]
            records.append(
                {
                    "parent": parent,
                    "case_index": index,
                    "activation": activation,
                    "recorded_learner_decisions_after_activation": later_learner_decisions,
                }
            )
    assert work == audit["audit_replay_work"]
    for path, digest in inputs.items():
        assert sha(path) == digest, path
    write(
        output,
        {
            "status": "complete",
            "inputs": inputs,
            "groups": groups,
            "records": records,
            "replay_work": work,
            "new_neural_inferences": 0,
            "new_independent_deals": 0,
            "scope": "Public-timing opportunity in training prefixes only; no candidate continuation or strength evidence.",
        },
    )
    print(json.dumps({"groups": groups, "replay_work": work, "sha256": sha(output)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base", type=Path, default=Path("runs/final_round_training_study")
    )
    parser.add_argument(
        "--diagnostic", type=Path, default=Path("runs/revealed_opponent_identification")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.base, args.diagnostic, args.output)
