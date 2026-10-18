"""Replay opening sources and independently check all selected-action statistics."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from koikoi.agents import EndgameAgent, NeuralAgent
from koikoi.layout import PHASE, PLAYER, POINTS, TERMINAL
from koikoi.reference import ReferenceEnv
from koikoi.upstream import PublishedAgent
from scripts.audit_round_search_comparison import (
    close,
    round_comparison,
    terminal_comparison,
)
from scripts.run_arbitration_study import PARENTS, sha, write


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def audit(folder):
    output = folder / "independent_audit.json"
    if output.exists():
        raise ValueError("Preserve previous independent audit")
    torch.set_num_threads(2)
    protocol, report, process = (
        read(folder / name) for name in ("protocol.json", "audit.json", "process.json")
    )
    assert report["status"] == "passed" and process["status"] == "complete"
    assert process["audit_sha256"] == sha(folder / "audit.json")
    assert report["protocol_sha256"] == sha(folder / "protocol.json")
    inputs = report["inputs"].copy()
    inputs.update(
        {
            str(p): sha(p)
            for p in (
                Path(__file__),
                Path("scripts/audit_round_search_comparison.py"),
                folder / "protocol.json",
                folder / "audit.json",
            )
        }
    )
    for name, key in (
        ("source.json", "source_sha256"),
        ("roots.npy", "roots_sha256"),
        ("rows.jsonl", "rows_sha256"),
    ):
        assert sha(folder / name) == report[key]
        inputs[str(folder / name)] = report[key]
    assert all(sha(p) == h for p, h in inputs.items())
    sources = read(folder / "source.json")
    roots = np.load(folder / "roots.npy", allow_pickle=False)
    rows = [
        json.loads(line)
        for line in (folder / "rows.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(sources) == 16 and len(roots) == len(rows) == 8
    parents = [NeuralAgent(p, deterministic=True) for p, _ in PARENTS]
    opponents = {name: EndgameAgent(PublishedAgent(name)) for name in ("wp", "point")}
    source_work = dict.fromkeys(("decisions", "round_endings", "completed_matches"), 0)
    root_by_case, base_by_case = {}, {}
    for case_index, (source, case) in enumerate(zip(sources, protocol["cases"])):
        assert all(source[k] == v for k, v in case.items())
        env = ReferenceEnv(case["seed"], "paper8")
        if env.state[PLAYER] == case["seat"]:
            assert source["root_index"] == len(root_by_case)
            np.testing.assert_array_equal(env.state, roots[source["root_index"]])
            root_by_case[case_index] = env.state.copy()
            base_by_case[case_index] = parents[case["parent"] - 1].act(env)
        else:
            assert source["root_index"] is None
        work = dict.fromkeys(source_work, 0)
        for action in source["actions"]:
            ours = int(env.state[PLAYER]) == case["seat"]
            assert (
                parents[case["parent"] - 1] if ours else opponents[case["opponent"]]
            ).act(env) == action
            result = env.step(action)
            work["decisions"] += 1
            work["round_endings"] += int(result[1])
            work["completed_matches"] += int(result[2])
        assert work == source["work"] and env.state[PHASE] == TERMINAL
        assert env.state[POINTS : POINTS + 2].tolist() == source["terminal_points"]
        for name, value in work.items():
            source_work[name] += value
    assert source_work == report["source_work"] and len(root_by_case) == 8
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
            "work": dict.fromkeys(entry["work"], 0),
            "confirmation_work": dict.fromkeys(entry["confirmation_work"], 0),
            "proposed": 0,
            "changed": 0,
            "history_work": {},
        }
        for name, entry in report["totals"].items()
    }
    summaries, seen = [], set()
    for row in rows:
        case_index, parent = row["case_index"], row["parent"]
        assert case_index in root_by_case and case_index not in seen
        seen.add(case_index)
        assert parent == sources[case_index]["parent"]
        state, base = root_by_case[case_index], base_by_case[case_index]
        seat = int(state[PLAYER])
        alternatives = sorted(set(np.flatnonzero(state[:48] == seat)) - {base})
        assert len(alternatives) == 7
        selected, estimates = {}, {}
        for mode, comparison in row["comparisons"].items():
            assert mode in ("terminal", "round")
            decision = comparison["decision"]
            assert (
                decision["base"] == base
                and comparison["context"]["status"] == "opening_prior"
            )
            assert comparison["context"]["history_likelihood_calls"] == 0
            assert comparison["context"]["model_prior"] == [0.5, 0.5]
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
            for stage, count, records in (
                ("proposal", 256, decision["alternatives"]),
                ("confirmation", 1024, [decision["confirmation"]] if proposed else []),
            ):
                prior_baseline = None
                for record in records:
                    if stage == "confirmation":
                        assert record["alternative"] == selected[mode]
                    if mode == "terminal":
                        terminal_comparison(record, count)
                        baseline = np.asarray(record["paired_points"])[:, 0]
                    else:
                        assert record["table"] == ("A" if stage == "proposal" else "B")
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
                        baseline = fields[:, 0]
                    if prior_baseline is not None:
                        np.testing.assert_array_equal(baseline, prior_baseline)
                    prior_baseline = baseline
                    for name, value in record["work"].items():
                        entry["work"][name] += value
                        if stage == "confirmation":
                            entry["confirmation_work"][name] += value
        assert set(selected) == {"terminal", "round"}
        for stage in ("proposal", "confirmation"):
            left, right = (
                row["comparisons"][m]["sampling"] for m in ("terminal", "round")
            )
            if stage in left and stage in right:
                for key in ("worlds_sha256", "model_indices"):
                    assert left[stage][key] == right[stage][key]
        union = sorted(set(selected.values()))
        ref = row["reference"]
        assert ref["alternatives"] == union and len(ref["chunks"]) == 8
        changes = {a: [] for a in union}
        for chunk in ref["chunks"]:
            assert [r["alternative"] for r in chunk["comparisons"]] == union
            baseline = None
            for record in chunk["comparisons"]:
                changes[record["alternative"]].extend(terminal_comparison(record, 512))
                current = np.asarray(record["paired_points"])[:, 0]
                if baseline is not None:
                    np.testing.assert_array_equal(current, baseline)
                baseline = current
                for name, value in record["work"].items():
                    totals[f"parent{parent}_reference"]["work"][name] += value
        assert [r["alternative"] for r in ref["summary"]] == union
        references = {}
        for record in ref["summary"]:
            d = np.asarray(changes[record["alternative"]])
            assert len(d) == 4096
            mean, se = float(d.mean()), float(d.std(ddof=1) / 64)
            close(record["win_change"], mean)
            close(record["se"], se)
            close(record["lower_margin"], mean - 2.5 * se)
            references[record["alternative"]] = {
                "win_change": mean,
                "se": se,
                "lower": mean - 2.5 * se,
            }
        summaries.append(
            {
                "case_index": case_index,
                "parent": parent,
                "base": int(base),
                "selected": selected,
                "accepted": {
                    mode: row["comparisons"][mode]["action"] for mode in selected
                },
                "selected_references": references,
                "union_proposal_errors": {
                    mode: {
                        int(a): float(estimates[mode][a] - references[a]["win_change"])
                        for a in union
                    }
                    for mode in selected
                },
                "seconds": {
                    mode: row["comparisons"][mode]["seconds"] for mode in selected
                },
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
            "scope": "Every source action/opening selection and all proposal/confirmation/selected-reference arithmetic reconstructed. Future neural trajectories not rerun; no unselected-action error or full-match strength inference.",
        },
    )
    print(
        json.dumps(
            {"status": "passed", "audit_sha256": sha(output), "summaries": summaries}
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    audit(parser.parse_args().folder)
