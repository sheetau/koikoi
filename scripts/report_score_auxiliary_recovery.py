"""Run the frozen mechanism/export code against the recovered training artifacts."""

import argparse
import json
from pathlib import Path

import torch

from scripts.finish_score_auxiliary_recovery import EVALUATION, QUEUE, TRAINING
from scripts.run_arbitration_study import read, sha
from scripts.train_score_auxiliary_study import checked, write_new

RECOVERY_SHA = "b38aa097e1ce9a4f2989eab445025d7f4388fbf97d7dd0df8722860bddbe5294"
QUEUE_SHA = "81de97227a5a280e31ecbc797bcaeb2e40f8fe189e51443f58191ac2213fc5c7"
MECHANISM = Path("runs/score_auxiliary_mechanism")
RESULTS = Path("runs/score_auxiliary_results")


def recovery_inputs():
    path = TRAINING / "recovery_protocol.json"
    if sha(path) != RECOVERY_SHA:
        raise ValueError("Recovery identity differs")
    recovery = read(path)
    checked(recovery["inputs"])
    prefix = read(TRAINING / "repeated_prefix.json")
    if (prefix["status"], prefix["updates"], prefix["collections"]) != (
        "passed",
        164,
        28,
    ):
        raise ValueError("Require exact repeated-prefix verification")
    complete = read(TRAINING / "complete.json")
    if complete["repeated_prefix_sha256"] != sha(TRAINING / "repeated_prefix.json"):
        raise ValueError("Repeated-prefix receipt differs")
    return {
        str(p): sha(p)
        for p in (
            path,
            TRAINING / "repeated_prefix.json",
            TRAINING / "complete.json",
            Path(__file__),
        )
    }


def mechanism(command):
    inputs = recovery_inputs()
    from scripts import audit_score_auxiliary_mechanism as auditor
    from scripts import score_auxiliary_mechanism as producer

    producer.STUDY = auditor.STUDY = TRAINING
    torch.set_num_threads(2)
    if command == "prepare":
        producer.prepare(MECHANISM)
        path = MECHANISM / "protocol.json"
        plan = read(path)
        plan["inputs"].update(inputs)
        checked(plan["inputs"])
        path.write_bytes((json.dumps(plan, indent=2) + "\n").encode("utf-8"))
        print({"recovery_bound_protocol_sha256": sha(path)}, flush=True)
    else:
        plan = read(MECHANISM / "protocol.json")
        if any(plan["inputs"].get(p) != digest for p, digest in inputs.items()):
            raise ValueError("Mechanism is not bound to this recovery adapter")
        {"run": producer.run, "audit": auditor.audit}[command](MECHANISM)


def export(audit_sha):
    from scripts import export_score_auxiliary_results as exporter

    if RESULTS.exists():
        raise ValueError("Preserve existing export")
    bindings = recovery_inputs()
    exporter.TRAINING, exporter.QUEUE_SHA = TRAINING, QUEUE_SHA
    summary, evaluation, training, inputs = exporter.load_completed(
        EVALUATION, QUEUE, audit_sha
    )
    inputs.update(bindings)
    checked(inputs)
    text = exporter.render(summary, evaluation, training)
    text += (
        "\nRecovery work outside the retained comparison\n"
        "---------------------------------------------\n\n"
        "The original process failed before auxiliary collection29, after seed2_none\n"
        "PPO update165. The failed prefix executed2,703,360PPO decisions. Its164logged\n"
        "updates record2,686,976decisions/174,865round endings/22,736matches. Round and\n"
        "match counts for unlogged update165 are unavailable. The28complete auxiliary\n"
        "collections contain421,608decisions/27,488rounds/3,584matches. These failed-run\n"
        "costs are additional to the retained six-arm work above. Repeating the same\n"
        "prefix adds no independent evidence. Saved states, RNG, optimizers, numerical\n"
        "logs and collections match exactly; wall time and git provenance may differ.\n"
        "All original artifacts and failure receipts remain preserved.\n"
    )
    RESULTS.mkdir(parents=True, exist_ok=False)
    (RESULTS / "results.rst").write_bytes(text.encode("utf-8"))
    write_new(
        RESULTS / "evidence.json",
        {
            "status": "complete",
            "summary": summary,
            "training": training,
            "evaluation_and_replay_work": evaluation,
            "recovery_overhead": read(TRAINING / "recovery_protocol.json")[
                "additional_work"
            ],
            "repeated_prefix": read(TRAINING / "repeated_prefix.json"),
            "inputs": inputs,
            "outputs": {"results.rst": sha(RESULTS / "results.rst")},
            "new_matches": 0,
            "new_neural_calls": 0,
            "scope": "All original fixed audited contrasts and descriptive rows, with recovery overhead reported separately. No new analysis or promotion.",
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "audit", "export"))
    parser.add_argument("--audit-sha256")
    args = parser.parse_args()
    if args.command == "export":
        if args.audit_sha256 is None:
            parser.error("--audit-sha256 required")
        export(args.audit_sha256)
    else:
        mechanism(args.command)
