"""Recheck raw capability records and independently replay source decisions."""

import argparse
import json
import math
import statistics
from pathlib import Path

import numpy as np
import torch

from koikoi.agents import EndgameAgent, NeuralAgent, solved_koi_action
from koikoi.layout import PHASE, PLAYER, POINTS, TERMINAL
from koikoi.reference import ReferenceEnv
from koikoi.upstream import PublishedAgent
from scripts.diagnose_history_support import reference
from scripts.run_arbitration_study import PARENTS, sha, write


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def audit(folder):
    output = folder / "independent_audit.json"
    if output.exists():
        raise ValueError("Preserve the previous audit")
    torch.set_num_threads(2)
    protocol, original = read(folder / "protocol.json"), read(folder / "audit.json")
    assert original["status"] == "passed"
    assert original["protocol_sha256"] == sha(folder / "protocol.json")
    inputs = protocol["inputs"].copy()
    for name, key in (
        ("source.json", "source_sha256"),
        ("roots.npy", "roots_sha256"),
        ("rows.jsonl", "rows_sha256"),
    ):
        assert sha(folder / name) == original[key]
        inputs[str(folder / name)] = sha(folder / name)
    for path in (folder / "protocol.json", folder / "audit.json", Path(__file__)):
        inputs[str(path)] = sha(path)
    assert all(sha(p) == h for p, h in inputs.items())
    roots = np.load(folder / "roots.npy", allow_pickle=False)
    source = read(folder / "source.json")
    rows = [
        json.loads(s)
        for s in (folder / "rows.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    parents = [NeuralAgent(p, deterministic=True) for p, _ in PARENTS]
    opponents = {name: EndgameAgent(PublishedAgent(name)) for name in ("wp", "point")}
    source_work = {"decisions": 0, "round_endings": 0, "completed_matches": 0}
    root_index = 0
    assert len(source) == 8
    for case, expected_case in zip(source, protocol["cases"]):
        assert all(case[k] == v for k, v in expected_case.items())
        env = ReferenceEnv(case["seed"], "paper8")
        selected = None
        counts = dict.fromkeys(source_work, 0)
        for action in case["actions"]:
            assert env.state[PHASE] != TERMINAL
            ours = int(env.state[PLAYER]) == case["seat"]
            obs = env.observe()
            unknown = int(np.count_nonzero(obs[:528].reshape(48, 11)[:, 4]))
            hand_count = round(float(obs[541]) * 8)
            if (
                selected is None
                and ours
                and env.legal().sum() > 1
                and solved_koi_action(env) is None
                and math.comb(unknown, hand_count) <= 100
            ):
                selected = env.state.copy()
            actor = parents[case["parent"] - 1] if ours else opponents[case["opponent"]]
            assert actor.act(env) == action
            result = env.step(action)
            counts["decisions"] += 1
            counts["round_endings"] += int(result[1])
            counts["completed_matches"] += int(result[2])
        assert env.state[PHASE] == TERMINAL and counts == case["work"]
        assert env.state[POINTS : POINTS + 2].tolist() == case["terminal_points"]
        if selected is None:
            assert case["root_index"] is None
        else:
            assert case["root_index"] == root_index
            np.testing.assert_array_equal(selected, roots[root_index])
            root_index += 1
        for key in source_work:
            source_work[key] += counts[key]
    assert root_index == len(roots) == original["roots"]
    assert source_work == original["source_work"]
    totals, seen = {}, set()
    for row in rows:
        identity = (row["case_index"], row["mode"])
        assert identity not in seen
        seen.add(identity)
        case = source[row["case_index"]]
        assert row["parent"] == case["parent"]
        root = reference(roots[case["root_index"]])
        assert root.legal()[row["action"]]
        key = f"parent{case['parent']}_{row['mode']}"
        total = totals.setdefault(
            key,
            {
                "work": {"decisions": 0, "round_endings": 0, "completed_branches": 0},
                "proposed_roots": 0,
                "changed_roots": 0,
                "roots": 0,
            },
        )
        total["roots"] += 1
        ctx, record = row["context"], row["decision"]
        assert ctx["status"] == "searched" and record is not None
        if row["mode"] == "history":
            n = ctx["supported_model_counts"]
            np.testing.assert_allclose(
                ctx["model_posterior"], np.array(n) / sum(n), rtol=0, atol=1e-15
            )
        assert record["base"] == parents[case["parent"] - 1].act(root)
        assert [r["alternative"] for r in record["alternatives"]] == [
            int(a) for a in np.flatnonzero(root.legal()) if a != record["base"]
        ]
        for value in record["alternatives"] + (
            [record["confirmation"]] if record["confirmation"] is not None else []
        ):
            points = value["paired_points"]
            differences = [
                ((b > 0) - (b < 0) - (a > 0) + (a < 0)) / 2 for a, b in points
            ]
            mean = statistics.mean(differences)
            lower = mean - 2.5 * statistics.stdev(differences) / math.sqrt(len(points))
            assert math.isclose(mean, value["win_change"], abs_tol=1e-12)
            assert math.isclose(lower, value["lower_margin"], abs_tol=1e-12)
            assert math.isclose(
                statistics.mean(b - a for a, b in points),
                value["point_change"],
                abs_tol=1e-12,
            )
            assert value["work"]["completed_branches"] == 2 * len(points)
            for k, v in value["work"].items():
                total["work"][k] += v
        selected = max(record["alternatives"], key=lambda r: r["win_change"])
        assert selected == record["proposal"]
        proposed = selected["lower_margin"] > 0
        assert proposed == (record["confirmation"] is not None)
        changed = proposed and record["confirmation"]["lower_margin"] > 0
        assert changed == record["changed"]
        assert row["action"] == (selected["alternative"] if changed else record["base"])
        total["proposed_roots"] += int(proposed)
        total["changed_roots"] += int(changed)
    expected = {
        (i, mode)
        for i, case in enumerate(source)
        if case["root_index"] is not None
        for mode in ("uniform", "history")
    }
    assert seen == expected
    for key, values in totals.items():
        for name, value in values.items():
            assert original["cases"][key][name] == value
    assert all(sha(p) == h for p, h in inputs.items())
    result = {
        "status": "passed",
        "inputs": inputs,
        "recomputed": totals,
        "verification_replay_work": source_work,
        "scope": "Independent source-policy/reference replay, root eligibility and raw terminal-point arithmetic. Does not rerun future branches, every historical hypothesis, timing or training. Verification reuses the eight source matches and adds no independent deals.",
    }
    write(output, result)
    print(
        json.dumps({"output": str(output), "sha256": sha(output), "recomputed": totals})
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    audit(parser.parse_args().base)
