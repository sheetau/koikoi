"""Audit one checkpoint's learning ancestry without counting resumed steps twice."""

import argparse
import hashlib
import io
import json
from pathlib import Path

import torch


def lineage_budget(payload, digest):
    stage = {
        "sha256": digest,
        "profile": payload["config"]["profile"],
        "counters": payload["counters"],
        "provenance": payload["provenance"],
    }
    stages, totals, seen = [], {}, set()
    while stage is not None:
        if stage["sha256"] in seen:
            raise ValueError("Repeated initialization ancestor")
        seen.add(stage["sha256"])
        counters = stage["counters"].copy()
        counters.setdefault("learner_decisions", counters["decisions"])
        if any(
            type(counters[key]) is not int or counters[key] < 0
            for key in (
                "matches",
                "rounds",
                "decisions",
                "updates",
                "learner_decisions",
            )
        ):
            raise ValueError("Learning counters must be nonnegative integers")
        if counters["learner_decisions"] > counters["decisions"]:
            raise ValueError("Learner decisions cannot exceed simulator decisions")
        profile = stage["profile"]
        bucket = totals.setdefault(
            profile,
            {
                key: 0
                for key in (
                    "matches",
                    "rounds",
                    "decisions",
                    "updates",
                    "learner_decisions",
                )
            },
        )
        for key in bucket:
            bucket[key] += counters[key]
        stages.append(
            {
                "role": "current" if not stages else "pretraining",
                "sha256": stage["sha256"],
                "profile": profile,
                "counters": {key: counters[key] for key in bucket},
                "source_sha256": stage["provenance"].get("source_sha256"),
                "revision": stage["provenance"].get("revision"),
            }
        )
        # parent_checkpoint describes a resume: those cumulative counts are
        # already included. Only weight-only initialization adds a new stage.
        stage = stage["provenance"].get("initialized_from")
    return {"by_profile": totals, "stages": stages}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    content = args.checkpoint.read_bytes()
    payload = torch.load(io.BytesIO(content), map_location="cpu", weights_only=False)
    if payload["format"] != 1:
        raise ValueError("Unsupported checkpoint format")
    result = lineage_budget(payload, hashlib.sha256(content).hexdigest())
    result["checkpoint"] = str(args.checkpoint)
    serialized = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")


if __name__ == "__main__":
    main()
