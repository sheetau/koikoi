"""Independently replay public reveal timing and saved model eliminations."""

import argparse
import json
from pathlib import Path

import numpy as np

from koikoi.layout import PHASE, PLAYER, ROUND
from koikoi.reference import ReferenceEnv
from scripts.run_arbitration_study import read, sha, write


def elimination_ledger(events):
    candidates = {"wp", "point"}
    calls = dict.fromkeys(candidates, 0)
    first = None
    updates = {}
    previous_time = -1
    for event in events:
        when = event["available_after_decision"]
        assert when > previous_time
        previous_time = when
        assert event["before"] == sorted(candidates)
        last_decision = -1
        for check in event["checks"]:
            assert last_decision < check["decision"] < when
            last_decision = check["decision"]
            assert set(check["predictions"]) == candidates
            for name in candidates:
                calls[name] += 1
            candidates = {
                name
                for name in candidates
                if check["predictions"][name] == check["observed"]
            }
        assert event["after"] == sorted(candidates)
        updates[when] = candidates.copy()
        if len(candidates) == 1 and first is None:
            first = {
                "round": event["round"],
                "available_after_decision": when,
                "model": next(iter(candidates)),
            }
    return candidates, first, calls, updates


def audit(base, diagnostic):
    output = diagnostic / "independent_audit.json"
    if output.exists():
        raise ValueError("Use a fresh audit path")
    produced = read(diagnostic / "audit.json")
    assert produced["status"] == "complete"
    inputs = dict(produced["inputs"])
    inputs[str(diagnostic / "audit.json")] = sha(diagnostic / "audit.json")
    inputs[str(diagnostic / "records.jsonl")] = produced["records_sha256"]
    inputs[str(Path(__file__))] = sha(__file__)
    for path, digest in inputs.items():
        assert sha(path) == digest, path
    source = {}
    for parent in (1, 2):
        path = base / f"parent{parent}" / "source.jsonl"
        assert sha(path) == inputs[str(path)]
        for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
            source[parent, index] = json.loads(line)
    records = [
        json.loads(line)
        for line in (diagnostic / "records.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [(r["parent"], r["case_index"]) for r in records] == list(source)
    assert len(records) == 2048
    groups = {}
    work = dict.fromkeys(("decisions", "round_endings", "completed_matches"), 0)
    total_calls = {"wp": 0, "point": 0}
    total_checks = 0
    for record in records:
        row = source[record["parent"], record["case_index"]]
        for key in ("parent", "opponent", "seat", "seed"):
            assert row[key] == record[key]
        candidates, first, calls, updates = elimination_ledger(record["events"])
        assert calls == record["prediction_calls"]
        assert first == record["first_identification"]
        assert sorted(candidates) == record["candidates"]
        status = (
            "inconsistent"
            if not candidates
            else "identified"
            if len(candidates) == 1
            else "ambiguous"
        )
        assert status == record["status"]
        expected_match = f"parent{record['parent']}-case{record['case_index']}"
        event_by_time = {e["available_after_decision"]: e for e in record["events"]}
        assert len(event_by_time) == len(record["events"])
        env = ReferenceEnv(row["seed"], "paper8")
        hands, snapshots = {}, {}
        seen_events = set()
        active_candidates = {"wp", "point"}
        learner_opportunities = 0
        for decision, action in enumerate(row["actions"]):
            if decision in updates:
                active_candidates = updates[decision]
            round_index = int(env.state[ROUND])
            actor = int(env.state[PLAYER])
            phase = int(env.state[PHASE])
            if actor == row["seat"]:
                learner_opportunities += len(active_candidates) == 1
            else:
                snapshots.setdefault(round_index, []).append((decision, phase, action))
            newly_revealed = False
            if actor != row["seat"] and phase == 0:
                hand = hands.setdefault(round_index, [])
                assert action not in hand
                hand.append(action)
                newly_revealed = len(hand) == 8
            result = env.step(action)
            work["decisions"] += 1
            work["round_endings"] += int(result[1])
            work["completed_matches"] += int(result[2])
            if newly_revealed:
                event = event_by_time[decision + 1]
                seen_events.add(decision + 1)
                assert event["match_key"] == expected_match
                assert event["actor_seat"] == 1 - row["seat"]
                assert event["round"] == round_index
                assert event["initial_opponent_hand"] == sorted(hands[round_index])
                assert [
                    (c["decision"], c["phase"], c["observed"]) for c in event["checks"]
                ] == snapshots[round_index]
            elif decision + 1 in event_by_time:
                raise AssertionError(
                    "Model inference occurred before full public hand revelation"
                )
        np.testing.assert_array_equal(env.state, row["endpoint"])
        assert seen_events == set(event_by_time)
        assert (
            learner_opportunities
            == record["learner_decisions_with_singleton_candidate"]
        )
        label = f"parent{row['parent']}_{row['opponent']}"
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
        correct = candidates == {row["opponent"]}
        group[
            ("identified_correct" if correct else "identified_wrong")
            if status == "identified"
            else status
        ] += 1
        group["true_model_excluded"] += row["opponent"] not in candidates
        group["final_round_survivors"] += row["reached_final_round"]
        group["correctly_identified_final_round_survivors"] += (
            correct and row["reached_final_round"]
        )
        group["learner_decisions_with_singleton_candidate"] += learner_opportunities
        if first is not None:
            group["first_identification_round_counts"][first["round"]] += 1
        for name in total_calls:
            total_calls[name] += calls[name]
        total_checks += sum(len(e["checks"]) for e in record["events"])
    assert groups == produced["groups"]
    assert total_calls == produced["prediction_calls"]
    assert total_checks == produced["strict_oracle_input_checks"]
    assert work == produced["replayed_source_work"]
    write(
        output,
        {
            "status": "passed",
            "inputs": inputs,
            "groups": groups,
            "prediction_calls_verified": total_calls,
            "input_check_count_verified": total_checks,
            "audit_replay_work": work,
            "new_neural_inferences": 0,
            "new_independent_matches": 0,
            "scope": "Independent reveal timing, candidate elimination, source replay and arithmetic; no neural prediction rerun.",
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--diagnostic", type=Path, required=True)
    args = parser.parse_args()
    audit(args.base, args.diagnostic)


if __name__ == "__main__":
    main()
