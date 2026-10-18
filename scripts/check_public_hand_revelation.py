"""Count retrospective hand revelation from public discards in training prefixes.

No model inference, outcome-based selection or held-out evaluation input is used.
The detector receives only public round/phase/player/action events.
"""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from koikoi.layout import DISCARD, PHASE, PLAYER, ROUND
from koikoi.reference import ReferenceEnv


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def revealed_hands(events, viewer):
    if viewer not in (0, 1):
        raise ValueError("Invalid viewer")
    hands = defaultdict(set)
    result = []
    for decision, (round_index, phase, player, action) in enumerate(events):
        if phase != DISCARD or player == viewer:
            continue
        if action not in range(48) or action in hands[round_index]:
            raise ValueError("Invalid or duplicate public hand discard")
        hands[round_index].add(action)
        if len(hands[round_index]) > 8:
            raise ValueError("More than eight hand discards in a round")
        if len(hands[round_index]) == 8:
            result.append(
                {
                    "round": round_index,
                    "available_after_decision": decision + 1,
                    "initial_opponent_hand": sorted(hands[round_index]),
                }
            )
    return result


def run(base, output):
    if output.exists():
        raise ValueError("Use a fresh diagnostic output path")
    paths = [
        Path(__file__),
        Path("koikoi/reference.py"),
        Path("koikoi/layout.py"),
        Path("tests/test_public_hand_revelation.py"),
        Path("runs/public_hand_revelation_tests.xml"),
        base / "protocol.json",
        base / "training_completion.json",
    ]
    paths += [base / f"parent{i}" / "source.jsonl" for i in (1, 2)]
    inputs = {str(p): sha(p) for p in paths}
    complete = json.loads(
        (base / "training_completion.json").read_text(encoding="utf-8")
    )
    for p in paths[-2:]:
        assert complete["inputs"][str(p)] == inputs[str(p)]
    groups = {}
    records = []
    replay_work = defaultdict(int)
    for parent in (1, 2):
        rows = [
            json.loads(s)
            for s in (base / f"parent{parent}" / "source.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        assert len(rows) == 1024
        for case_index, row in enumerate(rows):
            env = ReferenceEnv(row["seed"], "paper8")
            events = []
            for action in row["actions"]:
                events.append(
                    tuple(int(env.state[k]) for k in (ROUND, PHASE, PLAYER)) + (action,)
                )
                result = env.step(action)
                replay_work["decisions"] += 1
                replay_work["round_endings"] += int(result[1])
                replay_work["completed_matches"] += int(result[2])
            np.testing.assert_array_equal(env.state, row["endpoint"])
            reveals = revealed_hands(events, row["seat"])
            label = f"parent{parent}_{row['opponent']}"
            group = groups.setdefault(
                label,
                {
                    "source_prefixes": 0,
                    "prefixes_with_revelation": 0,
                    "revealed_rounds": 0,
                    "first_revelation_round_counts": [0] * 7,
                    "revealed_round_counts": [0] * 7,
                },
            )
            group["source_prefixes"] += 1
            group["prefixes_with_revelation"] += bool(reveals)
            group["revealed_rounds"] += len(reveals)
            if reveals:
                group["first_revelation_round_counts"][reveals[0]["round"]] += 1
            for event in reveals:
                group["revealed_round_counts"][event["round"]] += 1
            records.append(
                {
                    "parent": parent,
                    "case_index": case_index,
                    "opponent": row["opponent"],
                    "seat": row["seat"],
                    "seed": row["seed"],
                    "revelations": reveals,
                }
            )
    assert dict(replay_work) == complete["source_work"]
    for p, digest in inputs.items():
        assert sha(p) == digest
    report = {
        "status": "complete",
        "inputs": inputs,
        "groups": groups,
        "records": records,
        "source_attempts": 2048,
        "distinct_deals": 512,
        "replayed_source_work": dict(replay_work),
        "new_neural_inferences": 0,
        "new_independent_matches": 0,
        "scope": "Public hand-revelation opportunities only; neither opponent identification accuracy nor playing strength.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(
        json.dumps({"groups": groups, "replayed_source_work": dict(replay_work)}),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.base, args.output)


if __name__ == "__main__":
    main()
