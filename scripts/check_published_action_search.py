"""Fixed saved-root all-card-action diagnostic; no match-strength inference."""

import argparse
import json
import math
import statistics
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from koikoi.layout import PHASE, PLAYER
from koikoi.reference import ReferenceEnv
from koikoi.train import provenance
from scripts.published_action_search import PublishedActionSearch
from scripts.run_arbitration_study import PARENTS, sha, write


def selected_roots(path):
    """First two unforced roots per card phase/seat, before any search outcomes."""
    states = np.load(path, allow_pickle=False)
    counts = {(phase, seat): 0 for phase in range(3) for seat in range(2)}
    selected = []
    for index, state in enumerate(states):
        key = (int(state[PHASE]), int(state[PLAYER]))
        env = ReferenceEnv(1, "paper8")
        env.state[:] = state
        if key in counts and counts[key] < 2 and env.legal().sum() > 1:
            selected.append((index, env))
            counts[key] += 1
    if set(counts.values()) != {2}:
        raise ValueError(f"Incomplete fixed root strata: {counts}")
    return selected


def audit_rows(rows, confidence):
    """Recompute recorded choices/statistics from terminal points using stdlib."""
    totals = {"decisions": 0, "round_endings": 0, "completed_branches": 0}
    confirmations = dict.fromkeys(totals, 0)
    proposed = changed = 0
    for row in rows:
        record = row["decision"]
        all_records = record["alternatives"] + (
            [record["confirmation"]] if record["confirmation"] is not None else []
        )
        for comparison in all_records:
            points = comparison["paired_points"]
            differences = [
                ((b > 0) - (b < 0) - (a > 0) + (a < 0)) / 2 for a, b in points
            ]
            mean = statistics.mean(differences)
            lower = mean - confidence * statistics.stdev(differences) / math.sqrt(
                len(points)
            )
            assert math.isclose(mean, comparison["win_change"], abs_tol=1e-12)
            assert math.isclose(lower, comparison["lower_margin"], abs_tol=1e-12)
            assert math.isclose(
                statistics.mean(b - a for a, b in points),
                comparison["point_change"],
                abs_tol=1e-12,
            )
            assert comparison["work"]["completed_branches"] == 2 * len(points)
            for key, value in comparison["work"].items():
                totals[key] += value
                if comparison is record["confirmation"]:
                    confirmations[key] += value
        selected = max(record["alternatives"], key=lambda item: item["win_change"])
        assert selected == record["proposal"]
        proposal_passes = selected["lower_margin"] > 0
        assert (record["confirmation"] is not None) == proposal_passes
        accepted = proposal_passes and record["confirmation"]["lower_margin"] > 0
        assert record["changed"] == accepted
        assert row["action"] == (
            selected["alternative"] if accepted else record["base"]
        )
        proposed += int(proposal_passes)
        changed += int(accepted)
    return {
        "work": totals,
        "confirmation_work": confirmations,
        "proposed_roots": proposed,
        "changed_roots": changed,
    }


def run(folder):
    if folder.exists():
        raise ValueError("Use a fresh diagnostic output directory")
    if not torch.cuda.is_available():
        raise ValueError("CUDA required for this fixed diagnostic")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    source = provenance()
    prior_path = Path("runs/observer_seat_information_v2/audit.json")
    assert sha(prior_path) == "ac30ff58a6b921716ee2b8f5b59c93de5d7e3a818f5c4626e7fc626191bd1684"
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    files = [
        Path(__file__),
        Path("scripts/published_action_search.py"),
        Path("scripts/published_koi_search.py"),
        Path("scripts/collect_reusable_paired_targets.py"),
        Path("scripts/paired_rollout_targets.py"),
        Path("tests/test_published_action_search.py"),
        Path("docs/PUBLISHED_ACTION_SEARCH.rst"),
        Path("runs/observer_seat_information_v2/audit.json"),
        Path("runs/observer_seat_information_v2/independent_audit.json"),
    ]
    roots = {}
    for index, (checkpoint, expected) in enumerate(PARENTS, 1):
        assert sha(checkpoint) == expected
        path = Path(f"runs/observer_seat_information_v2/parent{index}_states.npy")
        assert sha(path) == prior["parents"][f"parent{index}"]["states_sha256"]
        files.extend([Path(checkpoint), path])
        roots[index] = selected_roots(path)
    inputs = {str(path): sha(path) for path in files}
    folder.mkdir(parents=True)
    protocol = {
        "inputs": inputs,
        "provenance": source,
        "roots": {str(k): [i for i, _ in v] for k, v in roots.items()},
        "particles": 32,
        "confirmation_particles": 256,
        "confidence": 2.5,
        "seed": "29130001 + 1000 * parent_index + saved_root_index",
        "scope": "Saved-root capability diagnostic, not match-strength evidence; no root collection, training, selection, or final-test use.",
    }
    write(folder / "protocol.json", protocol)
    cases = {}
    for index, (checkpoint, _) in enumerate(PARENTS, 1):
        search = PublishedActionSearch(
            checkpoint, particles=32, confirmation_particles=256
        )
        rows = []
        for root_index, env in roots[index]:
            before = env.state.copy()
            search.reset_seed(29130001 + 1000 * index + root_index)
            torch.cuda.synchronize()
            start = time.perf_counter()
            action = search.act(env)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - start
            np.testing.assert_array_equal(before, env.state)
            assert env.legal()[action]
            row = {
                "root": root_index,
                "phase": int(env.state[PHASE]),
                "player": int(env.state[PLAYER]),
                "action": action,
                "seconds": seconds,
                "decision": search.last_decision,
            }
            rows.append(row)
            with (folder / f"parent{index}.jsonl").open(
                "a", encoding="utf-8", newline="\n"
            ) as stream:
                stream.write(json.dumps(row) + "\n")
            print(
                json.dumps({"parent": index, "completed_roots": len(rows)}), flush=True
            )
        audit = audit_rows(rows, search.confidence)
        assert audit["work"] == asdict(search.work)
        assert audit["confirmation_work"] == asdict(search.confirmation_work)
        assert audit["proposed_roots"] == search.proposed_decisions
        assert audit["changed_roots"] == search.changed_decisions
        cases[f"parent{index}"] = {
            **audit,
            "roots": len(rows),
            "mean_seconds": statistics.mean(row["seconds"] for row in rows),
            "max_seconds": max(row["seconds"] for row in rows),
            "rows_sha256": sha(folder / f"parent{index}.jsonl"),
        }
        del search
    assert all(sha(path) == expected for path, expected in inputs.items())
    assert provenance()["source_sha256"] == source["source_sha256"]
    result = {
        "status": "passed",
        "protocol_sha256": sha(folder / "protocol.json"),
        "inputs": inputs,
        "cases": cases,
        "new_source_work": {"decisions": 0, "round_endings": 0, "completed_matches": 0},
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "scope": protocol["scope"],
    }
    write(folder / "audit.json", result)
    print(
        json.dumps(
            {
                "output": str(folder / "audit.json"),
                "sha256": sha(folder / "audit.json"),
                "cases": cases,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
