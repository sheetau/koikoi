"""Independently audit natural final-round sources, interventions and estimates."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from koikoi.agents import EndgameAgent, NeuralAgent
from koikoi.layout import PHASE, PLAYER, POINTS, ROUND, TERMINAL
from koikoi.reference import ReferenceEnv
from koikoi.search import sample_hidden_batch
from koikoi.upstream import PublishedAgent
from scripts.audit_round_search_comparison import close, terminal_comparison
from scripts.diagnose_history_support import reference
from scripts.run_arbitration_study import PARENTS, sha, write


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def audit(folder):
    output = folder / "independent_audit.json"
    if output.exists():
        raise ValueError("Preserve previous audit")
    torch.set_num_threads(2)
    report, protocol, process = [
        read(folder / p) for p in ("audit.json", "protocol.json", "process.json")
    ]
    assert report["status"] == "passed" and process["status"] == "complete"
    assert sha(folder / "audit.json") == process["audit_sha256"]
    inputs = report["inputs"].copy()
    for name, digest in report["artifacts"].items():
        assert sha(folder / name) == digest
        inputs[str(folder / name)] = digest
    for path in (
        Path(__file__),
        Path("scripts/audit_round_search_comparison.py"),
        folder / "audit.json",
    ):
        inputs[str(path)] = sha(path)
    assert all(sha(p) == h for p, h in inputs.items())
    expected_cases = [
        {
            "parent": p,
            "opponent": name,
            "seat": seat,
            "replicate": rep,
            "seed": 87110001 + 100 * oi + rep,
        }
        for p in (1, 2)
        for oi, name in enumerate(("wp", "point"))
        for seat in (0, 1)
        for rep in range(16)
    ]
    assert protocol["cases"] == expected_cases
    assert [
        protocol[k]
        for k in (
            "source_matches",
            "source_unique_deals",
            "candidate_matches",
            "proposal_particles",
            "confirmation_particles",
            "reference_particles",
            "reference_chunk_particles",
            "confidence",
            "microbatch",
            "bootstrap_samples",
            "bootstrap_seed",
        )
    ] == [128, 32, 384, 1024, 2048, 4096, 512, 2.5, 64, 10000, 87410001]
    assert protocol["phases"] == [0, 1, 2]
    sources = read(folder / "source.json")
    metadata = read(folder / "root_metadata.json")
    roots = np.load(folder / "roots.npy", allow_pickle=False)
    rows = [
        json.loads(s)
        for s in (folder / "rows.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert (
        len(sources) == 128
        and len(metadata) == len(roots) == len(rows) == report["root_count"]
    )
    parents = [NeuralAgent(p, deterministic=True) for p, _ in PARENTS]
    opponents = {name: EndgameAgent(PublishedAgent(name)) for name in ("wp", "point")}
    source_work = dict.fromkeys(("decisions", "round_endings", "completed_matches"), 0)

    def replay(record, case, intervention=None):
        assert all(record[k] == v for k, v in case.items())
        env = ReferenceEnv(case["seed"], "paper8")
        found, eligible_indices, applied = {}, [], []
        work = dict.fromkeys(source_work, 0)
        for i, action in enumerate(record["actions"]):
            ours = int(env.state[PLAYER]) == case["seat"]
            phase = int(env.state[PHASE])
            legal = np.flatnonzero(env.legal())
            eligible = ours and env.state[ROUND] == 7 and phase < 3 and len(legal) > 1
            if eligible:
                eligible_indices.append(i)
                if phase not in found:
                    found[phase] = (i, env.state.copy())
            if intervention is not None and i == intervention[0]:
                assert eligible
                np.testing.assert_array_equal(env.state, intervention[1])
                assert action == intervention[2] and action in legal
                applied.append(i)
            else:
                actor = (
                    parents[case["parent"] - 1] if ours else opponents[case["opponent"]]
                )
                assert actor.act(env) == action
            result = env.step(action)
            work["decisions"] += 1
            work["round_endings"] += int(result[1])
            work["completed_matches"] += int(result[2])
        assert work == record["work"] and work["completed_matches"] == 1
        assert env.state[PHASE] == TERMINAL
        assert record["terminal_round"] == env.state[ROUND]
        assert record["terminal_points"] == env.state[POINTS : POINTS + 2].tolist()
        assert record["eligible_decision_indices"] == eligible_indices
        assert record["intervention_indices"] == applied
        assert applied == ([] if intervention is None else [intervention[0]])
        assert record["selected_action"] == (
            None if intervention is None else intervention[2]
        )
        return found, work

    next_root = 0
    for ci, (source, case) in enumerate(zip(sources, expected_cases)):
        found, work = replay(source, case)
        assert set(source["root_indices"]) == {"0", "1", "2"}
        for phase in range(3):
            ri = source["root_indices"][str(phase)]
            if phase not in found:
                assert ri is None
                continue
            assert ri == next_root
            index, state = found[phase]
            assert metadata[ri] == {
                "root_index": ri,
                "case_index": ci,
                "phase": phase,
                "decision_index": index,
            }
            np.testing.assert_array_equal(roots[ri], state)
            next_root += 1
        for k, v in work.items():
            source_work[k] += v
    assert next_root == len(roots) and source_work == report["source_work"]
    totals = {
        name: {
            "work": dict.fromkeys(e["work"], 0),
            "confirmation_work": dict.fromkeys(e["confirmation_work"], 0),
            "proposed": 0,
            "changed": 0,
        }
        for name, e in report["totals"].items()
    }
    summaries, by_root = [], {}
    for row in rows:
        ri, ci, phase, parent = (
            row[k] for k in ("root_index", "case_index", "phase", "parent")
        )
        assert ri not in by_root and row["parent"] == sources[ci]["parent"]
        assert all(row[k] == v for k, v in metadata[ri].items())
        by_root[ri] = row
        root = reference(roots[ri])
        base = sources[ci]["actions"][row["decision_index"]]
        alternatives = sorted(set(map(int, np.flatnonzero(root.legal()))) - {base})
        comparison = row["comparison"]
        path = folder / f"search_{ri}.json"
        assert read(path) == comparison
        inputs[str(path)] = sha(path)
        decision = comparison["decision"]
        assert decision["base"] == base
        assert comparison["context"] == {
            "status": "final_round_uniform",
            "model_names": ["point", "wp"],
            "model_prior": [0.5, 0.5],
            "history_likelihood_calls": 0,
        }
        assert [r["alternative"] for r in decision["alternatives"]] == alternatives
        selected = max(decision["alternatives"], key=lambda r: r["win_change"])
        assert (
            decision["proposal"] == selected
            and decision["alternative"] == selected["alternative"]
        )
        proposed = selected["lower_margin"] > 0
        assert (decision["confirmation"] is not None) == proposed
        accepted = proposed and decision["confirmation"]["lower_margin"] > 0
        assert decision["changed"] == accepted
        assert comparison["action"] == (selected["alternative"] if accepted else base)
        total = totals[f"parent{parent}_search"]
        total["proposed"] += int(proposed)
        total["changed"] += int(accepted)

        def verify_sampling(sampling, count, rng, root=root):
            worlds = sample_hidden_batch(root, rng, count)
            labels = rng.integers(0, 2, count, dtype=np.int32)
            assert (
                hashlib.sha256(worlds.tobytes()).hexdigest()
                == sampling["worlds_sha256"]
            )
            assert labels.tolist() == sampling["model_indices"]

        def verify_records(records, count, entry, confirmation=False):
            baseline = None
            for record in records:
                terminal_comparison(record, count)
                assert record["work"]["round_endings"] == 2 * count
                points = np.asarray(record["paired_points"])[:, 0]
                if baseline is not None:
                    np.testing.assert_array_equal(points, baseline)
                baseline = points
                for k, v in record["work"].items():
                    entry["work"][k] += v
                    if confirmation:
                        entry["confirmation_work"][k] += v

        seed = 87210001 + 3 * ci + phase
        verify_sampling(
            comparison["sampling"]["proposal"], 1024, np.random.default_rng(seed)
        )
        verify_records(decision["alternatives"], 1024, total)
        if proposed:
            confirm = decision["confirmation"]
            assert confirm["alternative"] == selected["alternative"]
            verify_sampling(
                comparison["sampling"]["confirmation"],
                2048,
                np.random.default_rng(np.random.SeedSequence([seed, 1])),
            )
            verify_records([confirm], 2048, total, True)
        else:
            assert "confirmation" not in comparison["sampling"]
        ref = row["reference"]
        assert ref["alternatives"] == alternatives and len(ref["chunks"]) == 8
        rng = np.random.default_rng(87310001 + 3 * ci + phase)
        changes = {a: [] for a in alternatives}
        for chunk_index, chunk in enumerate(ref["chunks"]):
            path = folder / f"reference_{ri}_{chunk_index}.json"
            assert read(path) == chunk
            inputs[str(path)] = sha(path)
            assert [r["alternative"] for r in chunk["comparisons"]] == alternatives
            verify_sampling(chunk["sampling"], 512, rng)
            verify_records(
                chunk["comparisons"], 512, totals[f"parent{parent}_reference"]
            )
            for record in chunk["comparisons"]:
                p = np.asarray(record["paired_points"])
                changes[record["alternative"]].extend(
                    ((np.sign(p[:, 1]) - np.sign(p[:, 0])) / 2).tolist()
                )
        assert [r["alternative"] for r in ref["summary"]] == alternatives
        references = {}
        for record in ref["summary"]:
            values = np.asarray(changes[record["alternative"]])
            assert len(values) == 4096
            mean = float(values.sum() / 4096)
            se = float(np.sqrt(np.square(values - mean).sum() / 4095 / 4096))
            close(record["win_change"], mean)
            close(record["se"], se)
            close(record["lower_margin"], mean - 2.5 * se)
            references[record["alternative"]] = {
                "mean": mean,
                "se": se,
                "lower": mean - 2.5 * se,
            }
        summaries.append(
            {
                **metadata[ri],
                "parent": parent,
                "base": base,
                "proposed": proposed,
                "action": comparison["action"],
                "changed": accepted,
                "act_seconds": comparison["seconds"],
                "reference_seconds": ref["seconds"],
                "references": references,
                "proposal_errors": {
                    r["alternative"]: r["win_change"]
                    - references[r["alternative"]]["mean"]
                    for r in decision["alternatives"]
                },
            }
        )
    assert set(by_root) == set(range(len(roots))) and totals == report["totals"]
    pilot = read(folder / "actual_pilot.json")
    assert len(pilot) == 384
    seen, outcomes = set(), {}
    pilot_work = dict.fromkeys(source_work, 0)
    for record in pilot:
        ci, phase = record["case_index"], record["candidate_phase"]
        assert phase in (0, 1, 2) and (ci, phase) not in seen
        seen.add((ci, phase))
        ri = sources[ci]["root_indices"][str(phase)]
        intervention = (
            None
            if ri is None
            else (
                metadata[ri]["decision_index"],
                roots[ri],
                by_root[ri]["comparison"]["action"],
            )
        )
        _, work = replay(record, expected_cases[ci], intervention)
        for k, v in work.items():
            pilot_work[k] += v
        p0, p1 = record["terminal_points"]
        outcomes[(ci, phase)] = (p0 - p1) * (1 - 2 * record["seat"])
    assert seen == {(ci, phase) for ci in range(128) for phase in range(3)}
    assert pilot_work == report["actual_pilot_work"]
    indices = np.random.default_rng(87410001).integers(0, 16, size=(10000, 16))
    contrasts = []
    for parent in (1, 2):
        for opponent in ("wp", "point"):
            for phase in range(3):
                points = np.zeros((16, 2, 2))
                for ci, case in enumerate(expected_cases):
                    if case["parent"] == parent and case["opponent"] == opponent:
                        p0, p1 = sources[ci]["terminal_points"]
                        points[case["replicate"], case["seat"]] = [
                            (p0 - p1) * (1 - 2 * case["seat"]),
                            outcomes[(ci, phase)],
                        ]
                entry = {
                    "parent": parent,
                    "opponent": opponent,
                    "phase": phase,
                    "unique_deals": 16,
                }
                for metric, values in (
                    ("points", points),
                    ("win", (np.sign(points) + 1) / 2),
                ):
                    paired = (values[:, :, 1] - values[:, :, 0]).mean(1)
                    samples = paired[indices].mean(1)
                    entry[metric] = {
                        "mean": float(paired.mean()),
                        "interval95": np.quantile(samples, [0.025, 0.975]).tolist(),
                        "paired_changes": paired.tolist(),
                    }
                contrasts.append(entry)
    assert all(sha(p) == h for p, h in inputs.items())
    write(
        output,
        {
            "status": "passed",
            "inputs": inputs,
            "source_work_replayed": source_work,
            "actual_work_replayed": pilot_work,
            "summaries": summaries,
            "contrasts": contrasts,
            "scope": "All128 source and384 candidate matches, first-per-phase natural roots, uniform sampling, paired terminal statistics and actual contrasts verified. Neural future trajectories are not rerun;32 distinct deals do not establish replicated population strength.",
        },
    )
    print(
        json.dumps(
            {
                "status": "passed",
                "roots": len(roots),
                "audit_sha256": sha(output),
                "contrasts": contrasts,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    audit(parser.parse_args().folder)
