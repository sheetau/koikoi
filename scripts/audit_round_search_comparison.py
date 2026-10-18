"""Replay source choices and reconstruct every saved round/terminal comparison."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from koikoi.agents import EndgameAgent, NeuralAgent, solved_koi_action
from koikoi.layout import KOI, PHASE, PLAYER, POINTS, TERMINAL
from koikoi.reference import ReferenceEnv
from koikoi.upstream import PublishedAgent
from scripts.run_arbitration_study import PARENTS, sha, write


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def close(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-14)


def terminal_comparison(record, count):
    points = np.asarray(record["paired_points"])
    assert (
        points.shape == (count, 2)
        and points.dtype.kind in "iu"
        and (points % 2 == 0).all()
    )
    changes = (
        (points[:, 1] > 0).astype(int)
        - (points[:, 1] < 0).astype(int)
        - (points[:, 0] > 0).astype(int)
        + (points[:, 0] < 0).astype(int)
    ) / 2
    mean = changes.sum() / count
    se = np.sqrt(((changes - mean) ** 2).sum() / (count - 1) / count)
    close(record["win_change"], mean)
    close(record["lower_margin"], mean - 2.5 * se)
    close(record["point_change"], (points[:, 1] - points[:, 0]).sum() / count)
    work = record["work"]
    assert work["completed_branches"] == 2 * count
    assert 2 * count <= work["round_endings"] <= 16 * count
    assert work["round_endings"] <= work["decisions"] <= 64 * work["round_endings"]
    return changes


def round_comparison(record, sampling, count, seat, values, bootstrap):
    table = ("A", "B").index(record["table"])
    assert sampling["table"] == record["table"]
    fields = np.asarray(record["paired_boundary_fields"])
    assert fields.shape == (count, 2, 4) and fields.dtype.kind in "iu"
    assert record["boundary_field_order"] == ["round", "points0", "dealer", "phase"]
    labels = sampling["model_indices"]
    assert len(labels) == count and set(labels) <= {0, 1}
    v, b = np.empty((count, 2)), np.empty((256, count, 2))
    for i in range(count):
        for a in range(2):
            r, q, d, phase = map(int, fields[i, a])
            assert 0 <= r <= 7 and d in (0, 1)
            if phase == TERMINAL:
                v[i, a] = ((q > 30) - (q < 30)) * (1 - 2 * seat)
                b[:, i, a] = v[i, a]
            else:
                assert phase == 0 and 1 <= r <= 7 and 1 <= q <= 59
                key = (r, seat, labels[i], q - 1, d)
                v[i, a] = values[table][key]
                b[:, i, a] = bootstrap[table][(slice(None), *key)]
    close(record["paired_boundary_values"], v)
    changes = (v[:, 1] - v[:, 0]) / 2
    changes_boot = (b[:, :, 1] - b[:, :, 0]) / 2
    weights = np.random.default_rng(sampling["bootstrap_seed"]).multinomial(
        count, np.full(count, 1 / count), size=256
    )
    means = np.array([np.dot(changes_boot[i], weights[i]) / count for i in range(256)])
    close(record["joint_bootstrap_means"], means)
    close(record["win_change"], changes.mean())
    close(record["lower_margin"], changes.mean() - 2.5 * means.std(ddof=1))
    close(record["joint_se"], means.std(ddof=1))
    close(record["particle_se"], changes.std(ddof=1) / np.sqrt(count))
    close(record["table_se"], changes_boot.mean(1).std(ddof=1))
    assert record["work"]["round_endings"] == 2 * count
    assert record["work"]["terminal_match_endings"] == int(
        (fields[:, :, 3] == TERMINAL).sum()
    )
    assert 2 * count <= record["work"]["decisions"] <= 128 * count
    return changes


def audit(folder):
    output = folder / "independent_audit.json"
    if output.exists():
        raise ValueError("Preserve the previous independent audit")
    torch.set_num_threads(2)
    protocol, report, process = (
        read(folder / name) for name in ("protocol.json", "audit.json", "process.json")
    )
    assert report["status"] == "passed" and process["status"] == "complete"
    assert process["audit_sha256"] == sha(folder / "audit.json")
    assert report["protocol_sha256"] == sha(folder / "protocol.json")
    inputs = report["inputs"].copy()
    assert all(sha(p) == h for p, h in inputs.items())
    for name, key in (
        ("source.json", "source_sha256"),
        ("roots.npy", "roots_sha256"),
        ("rows.jsonl", "rows_sha256"),
    ):
        assert sha(folder / name) == report[key]
    for path in (
        folder / "protocol.json",
        folder / "audit.json",
        folder / "source.json",
        folder / "roots.npy",
        folder / "rows.jsonl",
        Path(__file__),
    ):
        inputs[str(path)] = sha(path)
    sources = read(folder / "source.json")
    roots = np.load(folder / "roots.npy", allow_pickle=False)
    rows = [
        json.loads(line)
        for line in (folder / "rows.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(sources) == 16 and len(rows) == len(roots) == report["root_count"]
    parents = [NeuralAgent(p, deterministic=True) for p, _ in PARENTS]
    opponents = {name: EndgameAgent(PublishedAgent(name)) for name in ("wp", "point")}
    source_work = dict.fromkeys(("decisions", "round_endings", "completed_matches"), 0)
    root_by_case = {}
    for case_index, (source, case) in enumerate(zip(sources, protocol["cases"])):
        assert all(source[k] == v for k, v in case.items())
        env = ReferenceEnv(case["seed"], "paper8")
        found, work = None, dict.fromkeys(source_work, 0)
        for action in source["actions"]:
            ours = int(env.state[PLAYER]) == case["seat"]
            obs = env.observe()
            unknown = int(np.count_nonzero(obs[:528].reshape(48, 11)[:, 4]))
            hand_count = round(float(obs[541]) * 8)
            if (
                found is None
                and ours
                and env.state[PHASE] == KOI
                and env.legal().sum() == 2
                and solved_koi_action(env) is None
                and math.comb(unknown, hand_count) <= 100
            ):
                found = env.state.copy()
            assert (
                parents[case["parent"] - 1] if ours else opponents[case["opponent"]]
            ).act(env) == action
            result = env.step(action)
            work["decisions"] += 1
            work["round_endings"] += int(result[1])
            work["completed_matches"] += int(result[2])
        assert work == source["work"] and env.state[PHASE] == TERMINAL
        assert env.state[POINTS : POINTS + 2].tolist() == source["terminal_points"]
        if found is None:
            assert source["root_index"] is None
        else:
            assert source["root_index"] == len(root_by_case)
            np.testing.assert_array_equal(found, roots[source["root_index"]])
            root_by_case[case_index] = found
        for k in work:
            source_work[k] += work[k]
    assert source_work == report["source_work"]
    study = Path(protocol["tables"])
    analysis = read(study / "analysis.json")
    assert all(sha(p) == h for p, h in analysis["inputs"].items())
    assert all(sha(study / p) == h for p, h in analysis["artifacts"].items())
    with np.load(study / "bootstrap_values.npz", allow_pickle=False) as data:
        boot = data["values"]
    values = []
    for p in (1, 2):
        pair = []
        for t in ("A", "B"):
            with np.load(
                study / f"parent{p}_{t}" / "kernels.npz", allow_pickle=False
            ) as data:
                pair.append(data["values"])
        values.append(pair)
    values = np.asarray(values)
    totals = {
        name: {
            "work": dict.fromkeys(entry["work"], 0),
            "confirmation_work": dict.fromkeys(entry["confirmation_work"], 0),
            "proposed": 0,
            "changed": 0,
        }
        for name, entry in report["totals"].items()
    }
    summaries = []
    seen = set()
    for row in rows:
        case_index, p = row["case_index"], row["parent"]
        assert case_index not in seen and case_index in root_by_case
        seen.add(case_index)
        assert p == sources[case_index]["parent"]
        seat = int(root_by_case[case_index][PLAYER])
        root = ReferenceEnv(1, "paper8")
        root.state[:] = root_by_case[case_index]
        parent_action = parents[p - 1].act(root)
        candidate = {}
        for mode, compared in row["comparisons"].items():
            assert mode in ("terminal", "round")
            entry = totals[f"parent{p}_{mode}"]
            decision = compared["decision"]
            if decision is None:
                assert compared["context"]["status"] == "empty_support"
                continue
            assert compared["context"]["status"] == "searched"
            base, alternative = decision["base"], decision["alternative"]
            assert base == parent_action
            assert {base, alternative} == {48, 49} and len(
                decision["alternatives"]
            ) == 1
            assert decision["proposal"] == decision["alternatives"][0]
            proposed = decision["proposal"]["lower_margin"] > 0
            assert (decision["confirmation"] is not None) == proposed
            accepted = proposed and decision["confirmation"]["lower_margin"] > 0
            assert decision["changed"] == accepted and compared["action"] == (
                alternative if accepted else base
            )
            entry["proposed"] += int(proposed)
            entry["changed"] += int(accepted)
            for stage, count in (("proposal", 512), ("confirmation", 1024)):
                record = decision[stage]
                if record is None:
                    continue
                assert record["alternative"] == alternative
                if mode == "terminal":
                    terminal_comparison(record, count)
                else:
                    assert record["table"] == ("A" if stage == "proposal" else "B")
                    round_comparison(
                        record,
                        compared["sampling"][stage],
                        count,
                        seat,
                        values[p - 1],
                        boot[p - 1],
                    )
                for name, value in record["work"].items():
                    entry["work"][name] += value
                    if stage == "confirmation":
                        entry["confirmation_work"][name] += value
            candidate[mode] = {
                "proposal": decision["proposal"]["win_change"],
                "changed": bool(accepted),
                "seconds": compared["seconds"],
            }
        ref = row["reference"]
        if candidate:
            assert (
                set(candidate) == {"round", "terminal"}
                and ref is not None
                and len(ref["chunks"]) == 8
            )
            for stage in ("proposal", "confirmation"):
                terminal_sampling = row["comparisons"]["terminal"]["sampling"]
                round_sampling = row["comparisons"]["round"]["sampling"]
                if stage in terminal_sampling and stage in round_sampling:
                    for key in ("worlds_sha256", "model_indices"):
                        assert (
                            terminal_sampling[stage][key] == round_sampling[stage][key]
                        )
            changes = []
            for chunk in ref["chunks"]:
                changes.extend(terminal_comparison(chunk["comparison"], 512))
                for name, value in chunk["comparison"]["work"].items():
                    totals[f"parent{p}_reference"]["work"][name] += value
            changes = np.asarray(changes)
            mean, se = changes.mean(), changes.std(ddof=1) / 64
            close(ref["win_change"], mean)
            close(ref["se"], se)
            close(ref["lower_margin"], mean - 2.5 * se)
            for value in candidate.values():
                value["proposal_minus_reference"] = float(value["proposal"] - mean)
            summaries.append(
                {
                    "case_index": case_index,
                    "parent": p,
                    "reference_change": float(mean),
                    "reference_se": float(se),
                    "reference_lower": float(mean - 2.5 * se),
                    **candidate,
                }
            )
        else:
            assert ref is None
    assert seen == set(root_by_case)
    for name, computed in totals.items():
        for key in computed:
            assert computed[key] == report["totals"][name][key]
    assert all(sha(p) == h for p, h in inputs.items())
    result = {
        "status": "passed",
        "inputs": inputs,
        "source_work_replayed": source_work,
        "new_independent_source_deals": 0,
        "summaries": summaries,
        "work": totals,
        "offline_table_work_reused": analysis["table_work"],
        "scope": "All source policy actions/root selections and raw terminal/round/uncertainty statistics independently reconstructed. Future neural trajectories and every support hypothesis not rerun; no full-match strength claim.",
    }
    write(output, result)
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
