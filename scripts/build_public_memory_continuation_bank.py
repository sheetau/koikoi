"""Materialize natural continuation roots after audited public identification.

Packed roots are simulator restart data, never policy inputs. Public memory is
joined from the independent identification audit, not from opponent routing
labels. This replays existing training histories; it does not train or evaluate.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from koikoi.layout import PHASE, ROUND, TERMINAL, TURN
from koikoi.reference import ReferenceEnv
from scripts.check_public_memory_activation import eligible_boundary
from scripts.run_arbitration_study import read, sha, write

ACTIVATION_SHA = "b9086bbf6fbb82e8f1eef888bff715d868435e88ab16da82c7cdbb4241b2f625"


def extract_root(row, identification):
    """Copy the first eligible public boundary, then verify the retained suffix.

    Later actions are replayed only for source integrity and work accounting.
    They cannot modify the copied root or the selected activation event.
    """
    env = ReferenceEnv(row["seed"], "paper8")
    root = activation = None
    work = dict.fromkeys(("decisions", "round_endings", "completed_matches"), 0)
    for decision in range(len(row["actions"]) + 1):
        if activation is None and eligible_boundary(
            int(env.state[ROUND]),
            int(env.state[TURN]),
            int(env.state[PHASE]),
            decision,
            identification,
        ):
            root = env.state.copy()
            activation = {
                "decision": decision,
                "round": int(env.state[ROUND]),
                "candidate": identification["model"],
            }
        if decision == len(row["actions"]):
            break
        if env.state[PHASE] == TERMINAL:
            raise ValueError("Source contains actions after match termination")
        result = env.step(row["actions"][decision])
        if int(result[0]) == 99:
            raise ValueError("Illegal source action")
        work["decisions"] += 1
        work["round_endings"] += int(result[1])
        work["completed_matches"] += int(result[2])
    np.testing.assert_array_equal(env.state, row["endpoint"])
    if work != row["work"]:
        raise ValueError("Source work does not match replay")
    return root, activation, work


def build(output, activation_path, diagnostic, base):
    if output.exists():
        raise ValueError("Use a fresh bank directory")
    if sha(activation_path) != ACTIVATION_SHA:
        raise ValueError("Expected the completed training-only activation diagnostic")
    activation = read(activation_path)
    inputs = dict(activation["inputs"])
    inputs[str(activation_path)] = sha(activation_path)
    for path in (
        Path(__file__),
        Path("tests/test_public_memory_continuation_bank.py"),
        Path("runs/public_memory_continuation_bank_tests.xml"),
    ):
        inputs[str(path)] = sha(path)
    for path, digest in inputs.items():
        if sha(path) != digest:
            raise ValueError(f"Input changed: {path}")
    records_path = diagnostic / "records.jsonl"
    if sha(records_path) != inputs[str(records_path)]:
        raise ValueError("Identification records changed")
    identities = {
        (r["parent"], r["case_index"]): r
        for r in map(json.loads, records_path.read_text(encoding="utf-8").splitlines())
    }
    expected = {(r["parent"], r["case_index"]): r for r in activation["records"]}
    banks, ledger, groups = {}, [], {}
    work = dict.fromkeys(("decisions", "round_endings", "completed_matches"), 0)
    for parent in (1, 2):
        source = base / f"parent{parent}/source.jsonl"
        if sha(source) != inputs[str(source)]:
            raise ValueError("Training source changed")
        states, seats, labels, codes = [], [], [], []
        for index, row in enumerate(
            map(json.loads, source.read_text(encoding="utf-8").splitlines())
        ):
            identity = identities[parent, index]
            if any(
                identity[key] != row[key]
                for key in ("parent", "seat", "seed", "opponent")
            ):
                raise ValueError("Public history join mismatch")
            root, event, replay = extract_root(row, identity["first_identification"])
            if event != expected[parent, index]["activation"]:
                raise ValueError("Activation differs from completed diagnostic")
            for key, value in replay.items():
                work[key] += value
            entry = {
                "parent": parent,
                "case_index": index,
                "seed": row["seed"],
                "seat": row["seat"],
                "activation": event,
                "root_index": None,
            }
            group = groups.setdefault(
                f"parent{parent}_{row['opponent']}", {"attempts": 0, "roots": 0}
            )
            group["attempts"] += 1
            if root is not None:
                entry["root_index"] = len(states)
                states.append(root)
                seats.append(row["seat"])
                # Routing metadata selects a frozen training opponent only.
                labels.append({"wp": 0, "point": 1}[row["opponent"]])
                # Actor conditioning comes exclusively from public elimination.
                codes.append({"wp": 0, "point": 1}[event["candidate"]])
                group["roots"] += 1
            ledger.append(entry)
        banks[parent] = {
            "format": "public_memory_continuation_bank_v1",
            "states": torch.from_numpy(np.stack(states)),
            "seats": torch.tensor(seats, dtype=torch.int64),
            "opponents": torch.tensor(labels, dtype=torch.int64),
            "memory": torch.tensor(codes, dtype=torch.int64),
            "inputs": inputs,
        }
    if len(ledger) != len(expected) or work != activation["replay_work"]:
        raise ValueError("Incomplete source coverage")
    for key, group in groups.items():
        if group["roots"] != activation["groups"][key]["eligible_boundaries"]:
            raise ValueError("Root count mismatch")
    for path, digest in inputs.items():
        if sha(path) != digest:
            raise ValueError(f"Input changed during replay: {path}")
    output.mkdir(parents=True, exist_ok=False)
    outputs = {}
    for parent, bank in banks.items():
        path = output / f"parent{parent}.pt"
        torch.save(bank, path)
        outputs[str(path)] = sha(path)
    report = {
        "status": "complete",
        "inputs": inputs,
        "outputs": outputs,
        "groups": groups,
        "records": ledger,
        "replay_work": work,
        "new_neural_inferences": 0,
        "new_independent_deals": 0,
        "scope": "Existing training histories only; no continuation learning or strength result.",
    }
    write(output / "audit.json", report)
    print(
        json.dumps(
            {
                "groups": groups,
                "replay_work": work,
                "sha256": sha(output / "audit.json"),
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--activation", type=Path, default=Path("runs/public_memory_activation.json")
    )
    parser.add_argument(
        "--diagnostic", type=Path, default=Path("runs/revealed_opponent_identification")
    )
    parser.add_argument(
        "--base", type=Path, default=Path("runs/final_round_training_study")
    )
    args = parser.parse_args()
    build(args.output, args.activation, args.diagnostic, args.base)
