"""Replay a closed cohort to measure public gate coverage, without evaluating gain."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from koikoi.layout import KOI, PHASE, PICK_DRAW, PLAYER, ROUND, TERMINAL, TURN
from scripts.uniform_deal_source import DeckSample, LoggedDealSource


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def replay_gates(record, caps=(5, 6, 7, 8)):
    """Use only recorded public identification and public turn/phase for gates."""
    position = 0

    def draw():
        nonlocal position
        entry = record["deals"][position]
        if entry["round"] != position:
            raise ValueError("Out-of-order recorded deal")
        position += 1
        return DeckSample(tuple(entry["deck"]), entry["attempts"], entry["bounded_draws"])

    source = LoggedDealSource(record["seed"], record["profile"], draw)
    eligible_events = [
        e for e in record["identification_events"]
        if e["round"] < 7 and len(e["after"]) == 1
    ]
    available = min((e["available_after_decision"] for e in eligible_events), default=None)
    roots = {cap: None for cap in caps}
    old_root_verified = False
    for decision, action in enumerate(record["actions"]):
        state = source.env.state
        if state[PHASE] == TERMINAL:
            raise ValueError("Actions after termination")
        if (
            available is not None and available <= decision
            and state[ROUND] == 7 and state[PLAYER] == record["seat"]
        ):
            draws = 16 - int(state[TURN]) - int(state[PHASE] in (PICK_DRAW, KOI))
            for cap in caps:
                if roots[cap] is None and draws <= cap:
                    roots[cap] = {
                        "decision": decision, "draws": draws,
                        "phase": int(state[PHASE]),
                        "legal_actions": int(np.count_nonzero(source.env.legal())),
                    }
                    if roots[cap]["legal_actions"] < 1:
                        raise ValueError("An eligible live root must have a legal action")
                    if cap == 5:
                        expected = record["selected_root"]
                        if expected is None or expected["decision"] != decision:
                            raise ValueError("Five-draw root differs")
                        np.testing.assert_array_equal(state, expected["state"])
                        old_root_verified = True
        source.step(action)
    if source.env.state[PHASE] != TERMINAL or position != len(record["deals"]):
        raise ValueError("Incomplete source replay")
    np.testing.assert_array_equal(source.env.state, record["final"])
    if source.work != record["work"]:
        raise ValueError("Replay work differs")
    if old_root_verified != (record["selected_root"] is not None):
        raise ValueError("Missing recorded five-draw root")
    if len(record["deals"]) < 8:
        reason = "ended_before_final_round"
    elif available is None:
        reason = "no_prior_round_identity"
    elif roots[5] is None:
        reason = "final_round_ended_before_five_draw_gate"
    else:
        reason = "five_draw_root"
    return {"reason": reason, "roots": roots, "work": source.work}


def diagnose(base, evidence, destination):
    if destination.exists():
        raise FileExistsError(destination)
    ledger = read(evidence)
    if ledger["status"] != "passed" or read(base / "queue.json")["status"] != "complete":
        raise ValueError("Complete checked cohort required")
    for path, digest in ledger["inputs"].items():
        if sha(path) != digest:
            raise ValueError(f"Changed checked input: {path}")
    protocol = read(base / "protocol.json")
    counts = {p: Counter() for p in ("0:wp", "0:point", "1:wp", "1:point")}
    cases, bindings, work = [], {}, Counter()
    for chunk in range(512):
        folder = base / f"chunk_{chunk:04}"
        complete, records = folder / "complete.json", folder / "records.json"
        if sha(complete) != ledger["summary"]["bindings"][str(complete)]:
            raise ValueError("Complete receipt changed")
        if sha(records) != read(complete)["records_sha256"]:
            raise ValueError("Source records changed")
        bindings[str(records)] = sha(records)
        for row in read(records):
            result = replay_gates(row["source"])
            count = counts[row["case"]["population"]]
            count["sources"] += 1
            count[result["reason"]] += 1
            work.update(result["work"])
            for cap, root in result["roots"].items():
                if root is not None:
                    count[f"root_at_{cap}"] += 1
                    count[f"choice_at_{cap}"] += int(root["legal_actions"] > 1)
                    if result["roots"][5] is None:
                        count[f"added_root_at_{cap}"] += 1
                        count[f"added_choice_at_{cap}"] += int(root["legal_actions"] > 1)
            cases.append({"case": row["case"], "reason": result["reason"], "roots": result["roots"]})
        if (chunk + 1) % 64 == 0:
            print(json.dumps({"replayed_chunks": chunk + 1}), flush=True)
    if [c["case"] for c in cases] != protocol["cases"]:
        raise ValueError("Cohort case order differs")
    if dict(work) != ledger["summary"]["work"]["sources"]:
        raise ValueError("Full replay work differs")
    result = {
        "status": "passed", "scope": "Closed-cohort gate coverage diagnosis only; no strength estimates.",
        "script_sha256": sha(__file__), "evidence_sha256": sha(evidence),
        "records": bindings, "counts": {p: dict(c) for p, c in counts.items()},
        "replay_work": dict(work), "fresh_chance_draws": 0, "neural_calls": 0, "cases": cases,
    }
    destination.write_bytes((json.dumps(result, indent=2) + "\n").encode("utf-8"))
    print(json.dumps({"counts": result["counts"], "replay_work": dict(work)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/uniform_gate_diagnosis.json"))
    args = parser.parse_args()
    diagnose(Path("runs/uniform_controller_evaluation"),
             Path("docs/paper/generated/uniform_results_evidence.json"), args.output)
