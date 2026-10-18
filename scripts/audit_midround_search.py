"""Independently replay source selection, sampling and midround statistics."""

import argparse
import hashlib
import json
import math
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

from koikoi.agents import EndgameAgent, NeuralAgent
from koikoi.env import BatchEnv
from koikoi.layout import (
    HISTORY,
    KOI,
    PHASE,
    PLAYER,
    POINTS,
    ROUND,
    STOP,
    TERMINAL,
    TURN,
)
from koikoi.reference import ReferenceEnv
from koikoi.search import sample_hidden_batch
from koikoi.upstream import PublishedAgent
from scripts.audit_round_search_comparison import (
    close,
    round_comparison,
    terminal_comparison,
)
from scripts.diagnose_history_support import reference
from scripts.enumerate_history_support import canonical_world, context
from scripts.history_conditioned_worlds import (
    HistorySupport,
    observation_key,
    sample_joint_worlds,
)
from scripts.run_arbitration_study import PARENTS, sha, write


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_histories(root, hands):
    """Native transitions validate even suffixes omitted by likelihood pruning."""
    targets = np.stack([canonical_world(root.state, hand) for hand in hands])
    initial = []
    for target in targets:
        env = reference(target)
        env.deal(deck=target[48:96].copy())
        initial.append(env.state)
    native = BatchEnv(len(hands), "cpu", "paper8", seed=0)
    native.state.copy_(torch.from_numpy(np.stack(initial)))
    steps, opponent_steps = 0, 0
    while not torch.all(
        (native.state[:, TURN] == root.state[TURN])
        & (native.state[:, PHASE] == root.state[PHASE])
    ):
        assert steps < 64
        phase, turn = int(native.state[0, PHASE]), int(native.state[0, TURN])
        assert torch.all(native.state[:, PHASE] == phase)
        assert torch.all(native.state[:, TURN] == turn)
        encoded = int(root.state[HISTORY + turn * 5 + (0, 1, 3, 4)[phase]])
        assert encoded > 0
        action = STOP + encoded - 1 if phase == KOI else encoded - 1
        assert native.observe()[1][:, action].all()
        opponent_steps += int(native.state[0, PLAYER] != root.state[PLAYER])
        result = native.step(torch.full((len(hands),), action, dtype=torch.int64))
        assert not result[:, 1:3].any() and not (result[:, 0] == 99).any()
        steps += 1
    np.testing.assert_array_equal(native.state.numpy(), targets)
    return {
        "native_validation_decisions": steps * len(hands),
        "full_replay_equivalent_decisions": 2 * steps * len(hands),
        "full_replay_equivalent_model_calls": 2 * opponent_steps * len(hands),
        "new_model_calls": 0,
    }


def audit(folder):
    output = folder / "independent_audit.json"
    if output.exists():
        raise ValueError("Preserve previous audit")
    torch.set_num_threads(2)
    protocol, report, process = [
        read(folder / f) for f in ("protocol.json", "audit.json", "process.json")
    ]
    assert report["status"] == "passed" and process["status"] == "complete"
    assert process["audit_sha256"] == sha(folder / "audit.json")
    assert report["protocol_sha256"] == sha(folder / "protocol.json")
    inputs = report["inputs"].copy()
    for name, key in (
        ("source.json", "source_sha256"),
        ("roots.npy", "roots_sha256"),
        ("rows.jsonl", "rows_sha256"),
    ):
        assert sha(folder / name) == report[key]
        inputs[str(folder / name)] = report[key]
    for path in (
        Path(__file__),
        Path("scripts/audit_round_search_comparison.py"),
        folder / "protocol.json",
        folder / "audit.json",
    ):
        inputs[str(path)] = sha(path)
    assert all(sha(p) == h for p, h in inputs.items())
    sources = read(folder / "source.json")
    roots = np.load(folder / "roots.npy", allow_pickle=False)
    rows = [
        json.loads(s)
        for s in (folder / "rows.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(sources) == 16 and len(rows) == len(roots) == report["root_count"]
    expected_cases = [
        {
            "parent": p,
            "opponent": name,
            "seat": seat,
            "replicate": replicate,
            "seed": 83110001 + 100 * opponent + replicate,
        }
        for p in (1, 2)
        for opponent, name in enumerate(("wp", "point"))
        for seat in (0, 1)
        for replicate in (0, 1)
    ]
    assert protocol["cases"] == expected_cases
    assert [
        protocol[k]
        for k in (
            "proposal_particles",
            "confirmation_particles",
            "reference_particles",
            "confidence",
        )
    ] == [1024, 2048, 4096, 2.5]
    parents = [NeuralAgent(p, deterministic=True) for p, _ in PARENTS]
    opponents = {name: EndgameAgent(PublishedAgent(name)) for name in ("wp", "point")}
    source_work = dict.fromkeys(("decisions", "round_endings", "completed_matches"), 0)
    root_by_case, base_by_case = {}, {}
    for case_index, (source, case) in enumerate(zip(sources, expected_cases)):
        assert all(source[k] == v for k, v in case.items())
        env = ReferenceEnv(case["seed"], "paper8")
        work = dict.fromkeys(source_work, 0)
        found = False
        for action in source["actions"]:
            ours = int(env.state[PLAYER]) == case["seat"]
            actor = parents[case["parent"] - 1] if ours else opponents[case["opponent"]]
            assert actor.act(env) == action
            if (
                ours
                and env.state[ROUND] == 0
                and env.state[TURN] in (8, 9)
                and env.state[PHASE] == 0
            ):
                assert not found and source["root_index"] == len(root_by_case)
                np.testing.assert_array_equal(env.state, roots[source["root_index"]])
                root_by_case[case_index] = env.state.copy()
                base_by_case[case_index] = action
                found = True
            result = env.step(action)
            work["decisions"] += 1
            work["round_endings"] += int(result[1])
            work["completed_matches"] += int(result[2])
        assert found or source["root_index"] is None
        assert work == source["work"] and env.state[PHASE] == TERMINAL
        assert env.state[POINTS : POINTS + 2].tolist() == source["terminal_points"]
        for k, v in work.items():
            source_work[k] += v
    assert source_work == report["source_work"]
    study = Path(protocol["tables"])
    analysis = read(study / "analysis.json")
    assert all(sha(p) == h for p, h in analysis["inputs"].items())
    assert all(sha(study / p) == h for p, h in analysis["artifacts"].items())
    with np.load(study / "bootstrap_values.npz", allow_pickle=False) as data:
        bootstrap = data["values"]
    tables = []
    for p in (1, 2):
        pair = []
        for t in ("A", "B"):
            with np.load(
                study / f"parent{p}_{t}" / "kernels.npz", allow_pickle=False
            ) as data:
                pair.append(data["values"])
        tables.append(pair)
    tables = np.asarray(tables)
    totals = {
        name: {
            "work": dict.fromkeys(e["work"], 0),
            "confirmation_work": dict.fromkeys(e["confirmation_work"], 0),
            "proposed": 0,
            "changed": 0,
            "history_work": {},
        }
        for name, e in report["totals"].items()
    }
    seen, summaries = set(), []
    for row in rows:
        ci, parent = row["case_index"], row["parent"]
        assert ci in root_by_case and ci not in seen and parent == sources[ci]["parent"]
        seen.add(ci)
        root = reference(root_by_case[ci])
        seat, base = int(root.state[PLAYER]), base_by_case[ci]
        alternatives = sorted(set(np.flatnonzero(root.legal())) - {base})
        assert len(alternatives) == 3
        path = folder / f"support_{ci}.json"
        assert sha(path) == row["support_sha256"]
        inputs[str(path)] = sha(path)
        raw = read(path)
        for k in ("hands", "compatible"):
            raw[k] = tuple(tuple(r) for r in raw[k])
        raw["model_names"] = tuple(raw["model_names"])
        support = HistorySupport(**raw)
        assert support.root_key == observation_key(root) and support.model_names == (
            "point",
            "wp",
        )
        unknown, n, _ = context(root.state)
        assert support.candidate_hands == math.comb(len(unknown), n) in (4845, 816)
        valid = [
            h
            for h in combinations(unknown, n)
            if canonical_world(root.state, h) is not None
        ]
        assert tuple(valid) == support.hands and support.valid_hands == len(valid)
        assert support.valid_hands + support.invalid_hands == support.candidate_hands
        history_validation = validate_histories(root, valid)
        matrix = np.asarray(support.compatible)
        assert matrix.dtype == bool and matrix.shape == (len(valid), 2) and matrix.any()
        truth = tuple(np.flatnonzero(root.state[:48] == 1 - seat))
        assert matrix[
            valid.index(truth), support.model_names.index(sources[ci]["opponent"])
        ]
        counts = support.counts
        assert counts["pruned_hands"] + counts["completed_hands"] == len(valid)
        assert counts["rejected_models"] == np.logical_not(matrix).sum()
        assert (
            counts["model_calls"]
            == counts["guarded_model_calls"] + counts["unguarded_model_calls"]
        )
        assert (
            counts["replay_decisions"]
            == counts["candidate_decisions"] + counts["opponent_decisions"]
        )
        for k, v in counts.items():
            entry = totals[f"parent{parent}_history"]["history_work"]
            entry[k] = entry.get(k, 0) + v

        def verify_worlds(sampling, count, rng, mode, root=root, support=support):
            if mode == "history":
                worlds, labels = sample_joint_worlds(root, support, rng, count)
            else:
                worlds = sample_hidden_batch(root, rng, count)
                labels = rng.integers(0, 2, count, dtype=np.int32)
            assert (
                hashlib.sha256(worlds.tobytes()).hexdigest()
                == sampling["worlds_sha256"]
            )
            assert labels.tolist() == sampling["model_indices"]

        selected, estimates = {}, {}
        assert set(row["comparisons"]) == {"uniform", "history"}
        for mode, comparison in row["comparisons"].items():
            decision, ctx = comparison["decision"], comparison["context"]
            assert decision["base"] == base and ctx["status"] == "searched"
            assert (
                ctx["candidate_hands"] == support.candidate_hands
                and ctx["mode"] == mode
            )
            if mode == "history":
                assert (
                    ctx["replay_work"] == counts
                    and ctx["supported_model_counts"] == matrix.sum(0).tolist()
                )
                close(ctx["model_posterior"], matrix.sum(0) / matrix.sum())
            assert [r["alternative"] for r in decision["alternatives"]] == alternatives
            best = max(decision["alternatives"], key=lambda r: r["win_change"])
            assert (
                best == decision["proposal"]
                and best["alternative"] == decision["alternative"]
            )
            selected[mode] = best["alternative"]
            estimates[mode] = {
                r["alternative"]: r["win_change"] for r in decision["alternatives"]
            }
            proposed = best["lower_margin"] > 0
            assert (decision["confirmation"] is not None) == proposed
            accepted = proposed and decision["confirmation"]["lower_margin"] > 0
            assert decision["changed"] == accepted and comparison["action"] == (
                selected[mode] if accepted else base
            )
            entry = totals[f"parent{parent}_{mode}"]
            entry["proposed"] += int(proposed)
            entry["changed"] += int(accepted)
            for stage, count, records, rng in (
                (
                    "proposal",
                    1024,
                    decision["alternatives"],
                    np.random.default_rng(83210001 + ci),
                ),
                (
                    "confirmation",
                    2048,
                    [decision["confirmation"]] if proposed else [],
                    np.random.default_rng(np.random.SeedSequence([83210001 + ci, 1])),
                ),
            ):
                if not records:
                    assert stage not in comparison["sampling"]
                    continue
                verify_worlds(comparison["sampling"][stage], count, rng, mode)
                baseline = None
                for record in records:
                    assert record["table"] == ("A" if stage == "proposal" else "B")
                    if stage == "confirmation":
                        assert record["alternative"] == selected[mode]
                    round_comparison(
                        record,
                        comparison["sampling"][stage],
                        count,
                        seat,
                        tables[parent - 1],
                        bootstrap[parent - 1],
                    )
                    fields = np.asarray(record["paired_boundary_fields"])
                    assert (fields[:, :, 0][fields[:, :, 3] != TERMINAL] == 1).all()
                    if baseline is not None:
                        np.testing.assert_array_equal(fields[:, 0], baseline)
                    baseline = fields[:, 0]
                    for k, v in record["work"].items():
                        entry["work"][k] += v
                        if stage == "confirmation":
                            entry["confirmation_work"][k] += v
        ref = row["reference"]
        assert ref["alternatives"] == alternatives and len(ref["chunks"]) == 8
        changes, rng = (
            {a: [] for a in alternatives},
            np.random.default_rng(83310001 + ci),
        )
        for chunk in ref["chunks"]:
            verify_worlds(chunk["sampling"], 512, rng, "history")
            assert [r["alternative"] for r in chunk["comparisons"]] == alternatives
            baseline = None
            for record in chunk["comparisons"]:
                changes[record["alternative"]].extend(terminal_comparison(record, 512))
                current = np.asarray(record["paired_points"])[:, 0]
                if baseline is not None:
                    np.testing.assert_array_equal(current, baseline)
                baseline = current
                for k, v in record["work"].items():
                    totals[f"parent{parent}_reference"]["work"][k] += v
        assert [r["alternative"] for r in ref["summary"]] == alternatives
        references = {}
        for record in ref["summary"]:
            d = np.asarray(changes[record["alternative"]])
            assert len(d) == 4096
            mean, se = float(d.mean()), float(d.std(ddof=1) / 64)
            close(record["win_change"], mean)
            close(record["se"], se)
            close(record["lower_margin"], mean - 2.5 * se)
            references[int(record["alternative"])] = {
                "win_change": mean,
                "se": se,
                "lower": mean - 2.5 * se,
            }
        summaries.append(
            {
                "case_index": ci,
                "parent": parent,
                "base": int(base),
                "selected": selected,
                "accepted": {m: c["action"] for m, c in row["comparisons"].items()},
                "references": references,
                "model_counts": matrix.sum(0).tolist(),
                "proposal_errors": {
                    m: {
                        int(a): float(estimates[m][a] - references[a]["win_change"])
                        for a in alternatives
                    }
                    for m in selected
                },
                "seconds": {m: c["seconds"] for m, c in row["comparisons"].items()},
                "history_validation": history_validation,
            }
        )
    assert seen == set(root_by_case) and totals == report["totals"]
    assert all(sha(p) == h for p, h in inputs.items())
    write(
        output,
        {
            "status": "passed",
            "inputs": inputs,
            "source_work_replayed": source_work,
            "new_independent_source_deals": 0,
            "totals": totals,
            "summaries": summaries,
            "scope": "All source actions, root selection, every canonical hand's full native history, actual-model membership, sampled worlds and raw statistics verified. Other support likelihood memberships and future neural trajectories are not rerun; no full-match strength result.",
        },
    )
    print(
        json.dumps(
            {"status": "passed", "audit_sha256": sha(output), "roots": len(rows)}
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    audit(parser.parse_args().folder)
