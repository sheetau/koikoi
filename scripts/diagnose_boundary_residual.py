"""Fixed paired accuracy/cost diagnostic for residual-corrected action values."""

import argparse
import hashlib
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch

from koikoi.agents import NeuralAgent
from koikoi.published_actor import PublishedActor
from scripts.boundary_residual_values import (
    BoundaryResidualWorkspace,
    corrected_estimate,
)
from scripts.fixed_microbatch_continuation import FixedMicrobatchContinuation
from scripts.history_conditioned_worlds import sample_joint_worlds
from scripts.prepared_belief_search import make_support
from scripts.round_value_search import BoundaryTables
from scripts.run_arbitration_study import PARENTS, read, sha, write
from scripts.run_prepared_belief_actions import load_roots

TABLES = Path("runs/round_value_accuracy")
KINDS = ("uniform", "identified_history")


def statistics(high, cheap, reference):
    x, y, c, ref = (
        np.asarray(v, dtype=float)
        for v in (high["high"], high["proxy"], cheap["proxy"], reference)
    )
    assert (len(x), len(y), len(c), len(ref)) == (512, 512, 2048, 2048)
    estimate = corrected_estimate(c, x, y)
    corrected_blocks = c.reshape(16, 128).mean(1) + (x - y).reshape(16, 32).mean(1)
    ordinary_blocks = x.reshape(16, 32).mean(1)
    reference_mean = float(ref.mean())
    estimate.update(
        reference_mean=reference_mean,
        reference_standard_error=float(ref.std(ddof=1) / np.sqrt(len(ref))),
        corrected_block_means=corrected_blocks.tolist(),
        ordinary_block_means=ordinary_blocks.tolist(),
        corrected_block_mse=float(np.square(corrected_blocks - reference_mean).mean()),
        ordinary_block_mse=float(np.square(ordinary_blocks - reference_mean).mean()),
        corrected_block_variance=float(corrected_blocks.var(ddof=1)),
        ordinary_block_variance=float(ordinary_blocks.var(ddof=1)),
    )
    return estimate


def run(output):
    if output.exists():
        raise ValueError("Use a fresh output directory")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    audit_path = Path("runs/prepared_belief_actions/audit.json")
    audit = read(audit_path)
    assert audit["status"] == "passed"
    inputs = dict(audit["inputs"])
    analysis = read(TABLES / "analysis.json")
    assert analysis["status"] == "passed"
    for path, digest in analysis["inputs"].items():
        assert path not in inputs or inputs[path] == digest
        inputs[path] = digest
    test_path = Path("runs/boundary_residual_values_tests.xml")
    suite = ET.parse(test_path).getroot().find("testsuite")
    assert int(suite.attrib["tests"]) == 4
    assert all(int(suite.attrib[k]) == 0 for k in ("failures", "errors", "skipped"))
    paths = [
        Path(__file__),
        audit_path,
        TABLES / "analysis.json",
        test_path,
        Path("scripts/boundary_residual_values.py"),
        Path("scripts/round_value_search.py"),
        Path("scripts/round_transition_values.py"),
        Path("tests/test_boundary_residual_values.py"),
        Path("docs/BOUNDARY_RESIDUAL_CHECK.rst"),
    ]
    paths += [TABLES / name for name in analysis["artifacts"]]
    for path in paths:
        digest = sha(path)
        assert str(path) not in inputs or inputs[str(path)] == digest
        inputs[str(path)] = digest
    for path, digest in inputs.items():
        assert sha(path) == digest, path
    output.mkdir(parents=True, exist_ok=False)
    write(
        output / "protocol.json",
        {
            "inputs": inputs,
            "conditions": KINDS,
            "high": 512,
            "cheap": 2048,
            "reference": 2048,
            "blocks": 16,
            "alpha": 1,
            "attempts": 32,
            "eligible_roots": 25,
            "seeds": [104100001, 105100001, 106100001],
            "seed_offset": "10000*parent+case_index",
            "alternative": "First legal action other than strict parent, ascending order",
            "scope": "Estimator development diagnostic, no strength or model-selection criterion",
        },
    )
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    parents = {
        p: NeuralAgent(path, device="cpu", deterministic=True)
        for p, (path, _) in enumerate(PARENTS, 1)
    }
    actors = {
        p: FixedMicrobatchContinuation(
            PublishedActor(a.model.actor.checkpoint(), "cuda"), "cuda", 64
        )
        for p, a in parents.items()
    }
    tables = {p: BoundaryTables.from_accuracy_study(TABLES, p) for p in parents}
    workspaces = {n: BoundaryResidualWorkspace(n) for n in (512, 2048)}
    rows, prefix_steps = [], 0
    try:
        for original, root, batch in load_roots():
            p, seat = original["parent"], original["seat"]
            prefix_steps += original["decision"]
            base = int(parents[p].act(root))
            alternative = next(
                int(a) for a in np.flatnonzero(root.legal()) if a != base
            )
            offset = 10000 * p + original["case_index"]
            for kind in KINDS:
                support = make_support(
                    batch, kind, original["public_identification"]["model"]
                )
                records = {}
                for category, n, seed in (
                    ("high", 512, 104100001),
                    ("cheap", 2048, 105100001),
                    ("reference", 2048, 106100001),
                ):
                    worlds, labels = sample_joint_worlds(
                        root, support, np.random.default_rng(seed + offset), n
                    )
                    start = time.perf_counter()
                    parts = []
                    chunk = 2048 if category == "cheap" else 512
                    for i in range(0, n, chunk):
                        parts.append(
                            workspaces[chunk].run(
                                torch.from_numpy(worlds[i : i + chunk]).cuda(),
                                base,
                                alternative,
                                actors[p],
                                seat,
                                labels[i : i + chunk],
                                tables[p],
                                terminal=category != "cheap",
                            )
                        )
                    records[category] = {
                        "parts": parts,
                        "labels": labels.tolist(),
                        "seconds": time.perf_counter() - start,
                        "worlds_sha256": hashlib.sha256(worlds.tobytes()).hexdigest(),
                    }
                high, cheap = records["high"]["parts"][0], records["cheap"]["parts"][0]
                ref = [
                    x for part in records["reference"]["parts"] for x in part["high"]
                ]
                row = {
                    k: original[k]
                    for k in (
                        "parent",
                        "case_index",
                        "opponent",
                        "seed",
                        "seat",
                        "decision",
                    )
                }
                row.update(
                    kind=kind,
                    base=base,
                    alternative=alternative,
                    public_identification=original["public_identification"],
                    public_root_key=batch.root_key,
                    samples=records,
                    statistics=statistics(high, cheap, ref),
                )
                name = f"cell_{p}_{original['case_index']}_{kind}.json"
                write(output / name, row)
                rows.append({"file": name, "sha256": sha(output / name)})
                write(
                    output / "process.json",
                    {"status": "running", "completed_cells": len(rows)},
                )
                print(len(rows), flush=True)
        for path, digest in inputs.items():
            assert sha(path) == digest, path
        totals = {
            category: {
                key: 0 for key in ("decisions", "round_endings", "terminal_endings")
            }
            for category in ("high", "cheap", "reference")
        }
        for entry in rows:
            row = read(output / entry["file"])
            for category, category_totals in totals.items():
                for part in row["samples"][category]["parts"]:
                    for key, value in part["work"].items():
                        category_totals[key] += value
        write(
            output / "report.json",
            {
                "status": "complete",
                "inputs": inputs,
                "protocol_sha256": sha(output / "protocol.json"),
                "rows": rows,
                "work": totals,
                "source_prefix_replay_decisions": prefix_steps,
                "new_historical_neural_calls": 0,
                "actual_completions": 0,
                "new_evaluation_deals": 0,
                "training_updates": 0,
            },
        )
        write(output / "process.json", {"status": "complete"})
    except BaseException as error:
        write(output / "process.json", {"status": "failed", "error": repr(error)})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
