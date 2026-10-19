"""Independent arithmetic and recorded-action replay for the belief diagnostic."""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from koikoi.layout import PHASE, POINTS, TERMINAL
from koikoi.reference import ReferenceEnv
from koikoi.train import provenance
from scripts.run_arbitration_study import read, sha, write

KINDS = ("uniform", "identified_uniform", "history", "identified_history")


def check_estimate(row, particles):
    points = np.asarray(row["paired_points"], dtype=np.int64)
    assert points.shape == (particles, 2)
    differences = [(int(np.sign(b)) - int(np.sign(a))) / 2 for a, b in points]
    mean = sum(differences) / particles
    variance = sum((d - mean) ** 2 for d in differences) / (particles - 1)
    lower = mean - 2.5 * math.sqrt(variance / particles)
    for key, expected in (
        ("win_change", mean),
        ("lower_margin", lower),
        ("point_change", sum(int(b - a) for a, b in points) / particles),
    ):
        assert abs(row[key] - expected) < 1e-12, key
    assert row["work"]["completed_branches"] == particles * 2
    assert row["work"]["decisions"] >= particles * 2
    assert row["work"]["round_endings"] >= particles * 2
    return mean, lower


def check_choice(row, legal, base):
    decision = row["decision"]
    assert decision["base"] == base
    alternatives = decision["alternatives"]
    assert [r["alternative"] for r in alternatives] == [a for a in legal if a != base]
    for estimate in alternatives:
        check_estimate(estimate, 128)
        assert [p[0] for p in estimate["paired_points"]] == [
            p[0] for p in alternatives[0]["paired_points"]
        ]
    selected = max(alternatives, key=lambda r: r["win_change"])
    assert decision["proposal"] == selected
    assert decision["alternative"] == selected["alternative"]
    proposed = selected["lower_margin"] > 0
    confirmation = decision["confirmation"]
    assert (confirmation is not None) == proposed
    if proposed:
        check_estimate(confirmation, 512)
        assert confirmation["alternative"] == selected["alternative"]
    changed = proposed and confirmation["lower_margin"] > 0
    assert decision["changed"] is changed
    assert row["action"] == (selected["alternative"] if changed else base)


def audit(output):
    receipt = output / "audit.json"
    if receipt.exists():
        raise ValueError("Preserve the existing audit")
    report, protocol = read(output / "report.json"), read(output / "protocol.json")
    assert report["status"] == "complete"
    assert report["protocol_sha256"] == sha(output / "protocol.json")
    assert report["inputs"] == protocol["inputs"]
    assert report["choices_sha256"] == sha(output / "choices.json")
    assert report["outcomes_sha256"] == sha(output / "outcomes.json")
    inputs = dict(report["inputs"])
    # The pre-run inputs include the source study protocol, whose aggregate
    # hash covers every package Python/native source, including .cpp files.
    source_protocol = Path("runs/final_round_training_study/protocol.json")
    assert str(source_protocol) in inputs
    assert provenance()["source_sha256"] == read(source_protocol)["core_source_sha256"]
    supplement_path = output / "native_binding.json"
    supplement = read(supplement_path)
    for path, digest in supplement["inputs"].items():
        assert path not in inputs or inputs[path] == digest
        inputs[path] = digest
    for path in [
        Path(__file__),
        supplement_path,
        Path("tests/test_audit_prepared_belief_actions.py"),
        Path("runs/audit_prepared_belief_actions_tests.xml"),
        output / "report.json",
        output / "protocol.json",
        output / "choices.json",
        output / "outcomes.json",
    ]:
        inputs[str(path)] = sha(path)
    for path, digest in inputs.items():
        assert sha(path) == digest, path
    choices, outcomes = read(output / "choices.json"), read(output / "outcomes.json")
    originals = [
        json.loads(line)
        for line in Path("runs/sampled_history_diagnostic/records.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    originals = [r for r in originals if r["status"] == "eligible"]
    assert len(choices) == len(outcomes) == len(originals) == 25
    aggregate = {
        scope: {
            kind: {"roots": 0, "changed": 0, "win_delta_sum": 0.0, "point_delta_sum": 0}
            for kind in KINDS
        }
        for scope in ("all", "parent1", "parent2", "wp", "point")
    }
    actual_work = {
        key: 0 for key in ("decisions", "round_endings", "conditional_completions")
    }
    search_work = {
        str(p): {key: 0 for key in ("decisions", "round_endings", "completed_branches")}
        for p in (1, 2)
    }
    confirm_work = {str(p): dict.fromkeys(search_work[str(p)], 0) for p in (1, 2)}
    prefix_steps = 0
    for choice, outcome, original in zip(choices, outcomes, originals):
        for key in ("parent", "case_index", "opponent", "seed", "decision", "seat"):
            assert choice[key] == original[key]
        assert (
            outcome["parent"] == choice["parent"]
            and outcome["case_index"] == choice["case_index"]
        )
        source = json.loads(
            Path(
                f"runs/final_round_training_study/parent{choice['parent']}/source.jsonl"
            )
            .read_text(encoding="utf-8")
            .splitlines()[choice["case_index"]]
        )
        root = ReferenceEnv(source["seed"], "paper8")
        for action in source["actions"][: choice["decision"]]:
            assert root.legal()[action]
            root.step(action)
            prefix_steps += 1
        # Original source used the same unchanged strict CPU parent.
        assert choice["base"] == source["actions"][choice["decision"]]
        legal = np.flatnonzero(root.legal()).tolist()
        assert tuple(choice["conditions"]) == KINDS
        for kind, condition in choice["conditions"].items():
            check_choice(condition, legal, choice["base"])
            context = condition["context"]
            assert context["public_identification"] == original["public_identification"]
            assert context["root_key"] == original["proposals"]["root_key"]
            assert (
                context["kind"] == kind and context["new_historical_neural_calls"] == 0
            )
            for phase, count in (("proposal", 128), ("confirmation", 512)):
                if phase in condition["sampling"]:
                    labels = condition["sampling"][phase]["model_indices"]
                    assert len(labels) == count and set(labels) <= {0, 1}
                    if kind.startswith("identified"):
                        assert set(labels) == {
                            ("point", "wp").index(
                                original["public_identification"]["model"]
                            )
                        }
            estimates = condition["decision"]["alternatives"]
            confirmation = condition["decision"]["confirmation"]
            for estimate in estimates + ([confirmation] if confirmation else []):
                for key, value in estimate["work"].items():
                    search_work[str(choice["parent"])][key] += value
                    if estimate is confirmation:
                        confirm_work[str(choice["parent"])][key] += value
        expected_actions = {str(choice["base"])} | {
            str(c["action"]) for c in choice["conditions"].values()
        }
        assert set(outcome["completions"]) == expected_actions
        for action, completion in outcome["completions"].items():
            env = ReferenceEnv.__new__(ReferenceEnv)
            env.state = root.state.copy()
            assert completion["actions"][0] == int(action)
            rounds = 0
            for act in completion["actions"]:
                assert env.state[PHASE] != TERMINAL and env.legal()[act]
                result = env.step(act)
                rounds += int(result[1])
            assert env.state[PHASE] == TERMINAL
            assert env.state.tolist() == completion["endpoint"]
            points = int(env.state[POINTS] - env.state[POINTS + 1]) * (
                1 - 2 * choice["seat"]
            )
            assert completion["points"] == points
            assert completion["win"] == (int(np.sign(points)) + 1) / 2
            assert completion["work"] == {
                "decisions": len(completion["actions"]),
                "round_endings": rounds,
                "conditional_completions": 1,
            }
            for key, value in completion["work"].items():
                actual_work[key] += value
        baseline = outcome["completions"][str(choice["base"])]
        for kind, condition in choice["conditions"].items():
            actual = outcome["completions"][str(condition["action"])]
            for scope in ("all", f"parent{choice['parent']}", choice["opponent"]):
                group = aggregate[scope][kind]
                group["roots"] += 1
                group["changed"] += int(condition["action"] != choice["base"])
                group["win_delta_sum"] += actual["win"] - baseline["win"]
                group["point_delta_sum"] += actual["points"] - baseline["points"]
    assert report["source_prefix_replay_decisions"] == prefix_steps * 2
    assert report["actual_completion_work"] == actual_work
    assert report["search_work"] == search_work
    assert report["confirmation_work"] == confirm_work
    assert (
        report["reused_completion_lookups"]
        == 125 - actual_work["conditional_completions"]
    )
    assert (
        report["new_historical_neural_calls"]
        == report["new_evaluation_deals"]
        == report["full_evaluation_matches"]
        == 0
    )
    write(
        receipt,
        {
            "status": "passed",
            "inputs": inputs,
            "aggregate": aggregate,
            "additional_source_replay_decisions": prefix_steps,
            "additional_completion_replay": actual_work,
            "additional_neural_calls": 0,
            "scope": "Scalar search arithmetic and recorded actual-completion actions replayed; no independent neural rerun or new deals.",
        },
    )
    print(json.dumps({"status": "passed", "aggregate": aggregate["all"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    audit(parser.parse_args().output)
