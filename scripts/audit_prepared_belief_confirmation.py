"""Verify the fresh-confirmation follow-up without rerunning neural policies."""

import argparse
import json
from pathlib import Path

import numpy as np

from koikoi.layout import PHASE, POINTS, TERMINAL
from koikoi.reference import ReferenceEnv
from scripts.audit_prepared_belief_actions import KINDS, check_estimate
from scripts.run_arbitration_study import read, sha, write

SOURCE = Path("runs/prepared_belief_actions")


def verify_condition(condition, previous, base):
    old = previous["decision"]
    proposed = old["proposal"]["win_change"] > 0
    assert condition["confirmed"] is proposed
    confirmation = condition["confirmation"]
    if proposed:
        assert confirmation is not None
        check_estimate(confirmation, 512)
        assert confirmation["alternative"] == old["alternative"]
        labels = condition["sampling"]["model_indices"]
        assert len(labels) == 512 and set(labels) <= {0, 1}
        assert (
            condition["sampling"]["worlds_sha256"]
            != previous["sampling"]["proposal"]["worlds_sha256"]
        )
        changed = confirmation["lower_margin"] > 0
    else:
        assert confirmation is None and condition["sampling"] is None
        changed = False
    assert condition["action"] == (old["alternative"] if changed else base)


def audit(output):
    receipt = output / "audit.json"
    if receipt.exists():
        raise ValueError("Preserve the existing audit")
    report, protocol = read(output / "report.json"), read(output / "protocol.json")
    assert report["status"] == "complete" and report["inputs"] == protocol["inputs"]
    for name in ("protocol", "choices", "outcomes"):
        assert report[name + "_sha256"] == sha(output / (name + ".json"))
    inputs = dict(report["inputs"])
    for path in (
        Path(__file__),
        Path("tests/test_audit_prepared_belief_confirmation.py"),
        Path("runs/audit_prepared_belief_confirmation_tests.xml"),
        *(
            output / (name + ".json")
            for name in ("protocol", "report", "choices", "outcomes")
        ),
    ):
        inputs[str(path)] = sha(path)
    for path, digest in inputs.items():
        assert sha(path) == digest, path
    choices, outcomes = read(output / "choices.json"), read(output / "outcomes.json")
    previous, previous_outcomes = (
        read(SOURCE / "choices.json"),
        read(SOURCE / "outcomes.json"),
    )
    assert (
        len(choices) == len(outcomes) == len(previous) == len(previous_outcomes) == 25
    )
    work = {
        str(p): dict.fromkeys(("decisions", "round_endings", "completed_branches"), 0)
        for p in (1, 2)
    }
    actual_work = dict.fromkeys(
        ("decisions", "round_endings", "conditional_completions"), 0
    )
    aggregate = {
        scope: {
            kind: {
                "roots": 0,
                "confirmed": 0,
                "changed": 0,
                "win_delta_sum": 0.0,
                "point_delta_sum": 0,
            }
            for kind in KINDS
        }
        for scope in ("all", "parent1", "parent2", "wp", "point")
    }
    prefix_steps = reused = 0
    for choice, outcome, old, prior in zip(
        choices, outcomes, previous, previous_outcomes
    ):
        for key in (
            "parent",
            "case_index",
            "opponent",
            "seed",
            "decision",
            "seat",
            "base",
        ):
            assert choice[key] == old[key]
        for row in (outcome, prior):
            assert (row["parent"], row["case_index"]) == (
                choice["parent"],
                choice["case_index"],
            )
        assert tuple(choice["conditions"]) == KINDS
        for kind, condition in choice["conditions"].items():
            verify_condition(condition, old["conditions"][kind], choice["base"])
            if condition["confirmed"]:
                if kind.startswith("identified"):
                    label = ("point", "wp").index(
                        old["conditions"][kind]["context"]["public_identification"][
                            "model"
                        ]
                    )
                    assert set(condition["sampling"]["model_indices"]) == {label}
                for key, value in condition["confirmation"]["work"].items():
                    work[str(choice["parent"])][key] += value
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
        expected = {str(choice["base"])} | {
            str(c["action"]) for c in choice["conditions"].values()
        }
        assert set(outcome["completions"]) == expected
        assert outcome["new_actions"] == sorted(
            int(a) for a in expected if a not in prior["completions"]
        )
        for action, completion in outcome["completions"].items():
            if action in prior["completions"]:
                assert completion == prior["completions"][action]
                reused += 1
                continue
            env = ReferenceEnv.__new__(ReferenceEnv)
            env.state = root.state.copy()
            assert completion["actions"][0] == int(action)
            rounds = 0
            for act in completion["actions"]:
                assert env.state[PHASE] != TERMINAL and env.legal()[act]
                result = env.step(act)
                rounds += int(result[1])
            assert (
                env.state[PHASE] == TERMINAL
                and env.state.tolist() == completion["endpoint"]
            )
            points = int(env.state[POINTS] - env.state[POINTS + 1]) * (
                1 - 2 * choice["seat"]
            )
            assert (
                completion["points"] == points
                and completion["win"] == (int(np.sign(points)) + 1) / 2
            )
            assert completion["work"] == {
                "decisions": len(completion["actions"]),
                "round_endings": rounds,
                "conditional_completions": 1,
            }
            for key, value in completion["work"].items():
                actual_work[key] += value
        baseline = prior["completions"][str(choice["base"])]
        for kind, condition in choice["conditions"].items():
            actual = outcome["completions"][str(condition["action"])]
            for scope in ("all", f"parent{choice['parent']}", choice["opponent"]):
                group = aggregate[scope][kind]
                group["roots"] += 1
                group["confirmed"] += int(condition["confirmed"])
                group["changed"] += int(condition["action"] != choice["base"])
                group["win_delta_sum"] += actual["win"] - baseline["win"]
                group["point_delta_sum"] += actual["points"] - baseline["points"]
    assert report["source_prefix_replay_decisions"] == prefix_steps * 2
    assert (
        report["search_work"] == work
        and report["new_actual_completion_work"] == actual_work
    )
    assert report["reused_completions"] == reused
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
            "additional_new_completion_replay": actual_work,
            "reused_completions": reused,
            "additional_neural_calls": 0,
            "scope": "Independent scalar checks and new recorded completion replay; no neural rerun or new deals.",
        },
    )
    print(json.dumps({"status": "passed", "aggregate": aggregate["all"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    audit(parser.parse_args().output)
