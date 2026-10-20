"""Measure certificate coverage on saved policies; perform no new simulations."""

import argparse
import json
from pathlib import Path

import torch

from scripts.export_belief_tail import read
from scripts.export_full_source_results import table
from scripts.public_tail_bounds import (
    certified_outcome,
    final_margin_bounds,
    public_tail,
    public_tail_from_key,
)
from scripts.train_branch_policy_gradient import sha, write


def coverage(keys, viewer):
    keys = set(keys)
    contexts = {}
    counts = {"keys": len(keys), "win": 0, "loss": 0, "unresolved": 0}
    for key in keys:
        context = public_tail_from_key(key)
        if context not in contexts:
            contexts[context] = certified_outcome(context, viewer)
        certificate = contexts[context]
        counts[{1: "win", -1: "loss", None: "unresolved"}[certificate]] += 1
    counts["distinct_public_contexts"] = len(contexts)
    return counts


def run(output):
    if output.exists():
        raise ValueError("Use a fresh output directory")
    base = Path("runs/earlier_belief_tail_capability")
    recovery = Path("runs/ceiling_tail_recovery")
    receipt_path = Path("runs/ceiling_tail_recovery_export.json")
    receipt = read(receipt_path)
    assert receipt["status"] == "complete"
    assert receipt["report_sha256"] == sha("docs/CEILING_RECOVERY_RESULTS.rst")
    assert receipt["exporter_sha256"] == sha("scripts/export_ceiling_recovery.py")
    bindings = {
        str(path): sha(path)
        for path in (
            Path(__file__),
            Path("scripts/public_tail_bounds.py"),
            Path("koikoi/reference.py"),
            Path("koikoi/layout.py"),
            Path("scripts/belief_tail.py"),
            Path("docs/PUBLIC_TAIL_BOUNDS.rst"),
            receipt_path,
            base / "sources.json",
        )
    }
    sources = read(base / "sources.json")
    rows, root_rows, completed_indices = [], [], []
    for directory, expected in (
        (base, receipt["original_complete_sha256"]),
        (recovery, receipt["recovery_complete_sha256"]),
    ):
        complete_path = directory / "complete.json"
        assert sha(complete_path) == expected
        bindings[str(complete_path)] = expected
        complete = read(complete_path)
        plans_path = directory / "plans_complete.json"
        assert sha(plans_path) == complete["plans_sha256"]
        bindings[str(plans_path)] = complete["plans_sha256"]
        plans = read(plans_path)
        assert plans["sources_sha256"] == bindings[str(base / "sources.json")]
        for record in plans["records"]:
            if record["status"] != "complete":
                continue
            index = record["source_index"]
            assert index not in completed_indices
            completed_indices.append(index)
            artifact_path = directory / record["evidence_file"]
            expected = plans["artifacts"][record["evidence_file"]]
            assert sha(artifact_path) == expected
            bindings[str(artifact_path)] = expected
            artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
            viewer = sources[index]["seat"]
            for name, plan in artifact["plans"].items():
                keys = {key for policy in plan.root_policies.values() for key in policy}
                rows.append({"source": index, "kind": name, **coverage(keys, viewer)})
            opponent_keys = [key[1] for key in artifact["opponent_action_cache"]]
            rows.append(
                {"source": index, "kind": "opponent", **coverage(opponent_keys, viewer)}
            )
            print(f"source {index}: saved keys checked", flush=True)
    assert len(completed_indices) == 28
    for index, source in enumerate(sources):
        if source["selected_root"] is None:
            continue
        context = public_tail(source["selected_root"]["state"])
        viewer = source["seat"]
        root_rows.append(
            {
                "source": index,
                "completed": index in completed_indices,
                "margin_bounds": final_margin_bounds(context, viewer),
                "certificate": certified_outcome(context, viewer),
            }
        )
    assert len(root_rows) == 33
    totals = {
        kind: {
            key: sum(row[key] for row in rows if row["kind"] == kind)
            for key in ("keys", "win", "loss", "unresolved", "distinct_public_contexts")
        }
        for kind in ("fixed", "optimized", "opponent")
    }
    text = (
        "Public final-round outcome bound coverage\n"
        "========================================\n\n"
        "This saved-data analysis covers all 28 completed source roots and all\n"
        "33 available source root states. It performs no new transitions, model\n"
        "calls, training updates, matches or independent strength measurements.\n"
        "Keys are counted once within each root/method. Totals sum those counts;\n"
        "they are not globally distinct states or probability-weighted coverage.\n"
        "Fixed/optimized maps include every computed root action. Opponent caches\n"
        "include evaluated alternatives. No cache of partial refused trees was\n"
        "saved, so their interior coverage cannot be inferred here.\n\n"
    )
    text += table(
        "Saved keys with certified outcomes, from the learner's perspective",
        ["Kind", "Keys", "Win", "Loss", "Unresolved", "Public contexts"],
        [[kind, *(totals[kind][k] for k in totals[kind])] for kind in totals],
    )
    text += table(
        "Every available root",
        ["Source", "Completed", "Final margin interval", "Certificate"],
        [
            [r["source"], r["completed"], r["margin_bounds"], r["certificate"]]
            for r in root_rows
        ],
    )
    text += (
        "A certificate proves a fixed final win/loss under every legal continuation\n"
        "of the local rules, independently of the opponent policy. Its absence\n"
        "does not imply the outcome is changeable. Saved key coverage is not a\n"
        "measured speedup or an estimate of physical transitions saved. An exact\n"
        "planner still needs explicit certificate leaves, executable fallback\n"
        "semantics, independent verification and a paired cost comparison.\n"
    )
    assert all(sha(path) == digest for path, digest in bindings.items())
    output.mkdir(parents=True)
    report = output / "report.rst"
    report.write_bytes(text.encode("utf-8"))
    write(
        output / "complete.json",
        {
            "status": "complete",
            "bindings": bindings,
            "report_sha256": sha(report),
            "rows": rows,
            "root_rows": root_rows,
            "totals": totals,
            "new_simulator_decisions": 0,
            "new_policy_calls": 0,
            "new_source_matches": 0,
        },
    )
    print(json.dumps(totals, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="runs/public_tail_bound_coverage")
    run(Path(parser.parse_args().output))
