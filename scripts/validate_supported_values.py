"""Fixed nine-root comparison of uniform and exactly supported hidden worlds."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from koikoi.agents import NeuralAgent
from koikoi.search import sample_hidden_batch
from koikoi.upstream import PublishedAgent
from scripts.diagnose_history_support import ROOTS, read, reference, sha, write
from scripts.enumerate_history_support import metadata as enumeration_metadata
from scripts.rollout_opponent import rollout
from scripts.sample_history_support import sample_supported_worlds

MODELS = [
    ("seed1", "runs/ppo8_v3/update_00005000.pt"),
    ("seed10001", "runs/ppo8_seed10001_captured/update_00005000.pt"),
]
ENUMERATION = Path("runs/history_support_enumeration.json")
AUDIT = Path("runs/history_support_enumeration_audit.json")
PILOT = Path("runs/supported_values_pilot.json")
OUTPUT = Path("runs/supported_values_diagnostic.json")
PROCESS = Path("runs/supported_values_process.json")
ARCHIVE_SHA = "baf2c400b572cf96556f9ffdc9bdfdd9e617a3ffdba47b353db818419f93cfd2"


class CountedTeacher(PublishedAgent):
    def __init__(self):
        super().__init__("wp")
        self.total_calls = 0

    def act(self, env):
        self.total_calls += 1
        return super().act(env)


def metadata():
    assert (
        sha(ENUMERATION)
        == "57f3c88691b183aa3dfcdf85b0ef761fe7a08b90c2472d80b0e8a03017b6c052"
    )
    assert (
        sha(AUDIT) == "ff79e702f4871f47a9136d17acde346652a830b4275b6ed72ae770d51926d7e4"
    )
    assert read(AUDIT)["passed"]
    enum_meta = enumeration_metadata()
    assert read(ENUMERATION)["metadata"] == enum_meta
    models = {label: sha(path) for label, path in MODELS}
    for previous in read("runs/rollout_opponent_diagnostic.json")["results"]:
        assert models[previous["identity"]["model"]] == previous["checkpoint_sha256"]
    return {
        "script_sha256": sha(__file__),
        "rollout_sha256": sha("scripts/rollout_opponent.py"),
        "sampler_sha256": sha("scripts/sample_history_support.py"),
        "enumeration_sha256": sha(ENUMERATION),
        "audit_sha256": sha(AUDIT),
        "enumeration_metadata": enum_meta,
        "checkpoints": models,
    }


def statistics(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "se": float(values.std(ddof=1) / np.sqrt(len(values))),
    }


@torch.no_grad()
def main(pilot):
    torch.set_num_threads(2)
    meta = metadata()
    output = PILOT if pilot else OUTPUT
    assert not output.exists()
    if not pilot:
        previous_pilot = read(PILOT)
        assert (
            previous_pilot["status"] == "complete"
            and previous_pilot["metadata"] == meta
        )
        assert len(previous_pilot["results"]) == 4
        assert all(r["archived_exact"] for r in previous_pilot["results"])
    else:
        assert sha("runs/diagnose_rollout_opponent.py") == ARCHIVE_SHA
        from runs.diagnose_rollout_opponent import rollout as archived_rollout
    enumeration = read(ENUMERATION)["results"]
    if pilot:
        enumeration = [
            next(r for r in enumeration if r["identity"]["model"] == label)
            for label, _ in MODELS
        ]
    roots = read(ROOTS)["roots"]
    particles = 2 if pilot else 1024
    manifest = {
        "pid": os.getpid(),
        "status": "running",
        "pilot": pilot,
        "particles": particles,
        "metadata": meta,
    }
    if not pilot:
        manifest["pilot_sha256"] = sha(PILOT)
    write(PROCESS, manifest)
    report = {
        **manifest,
        "results": [],
        "distribution_differences": [],
        "note": "Selected correlated roots; no match-strength inference. Paired actions/continuations within each distribution, independent streams across distributions. No new training, final seeds, reranking or deployment-policy change.",
    }
    for model_index, (label, checkpoint) in enumerate(MODELS):
        neural = NeuralAgent(checkpoint, "cuda", deterministic=True)
        teacher = CountedTeacher()
        for entry in [e for e in enumeration if e["identity"]["model"] == label]:
            index = entry["root_index"]
            cached = roots[index]
            ident = cached["identity"]
            assert ident == entry["identity"]
            root = reference(cached["state"])
            assert (
                hashlib.sha256(root.state.tobytes()).hexdigest()
                == cached["state_sha256"]
            )
            hands = [
                h["hand"]
                for h in entry["hands"]
                if h["valid"] and h["replay"]["compatible"]
            ]
            assert len(hands) == entry["compatible_count"] and hands
            root_results = []
            for distribution, name in enumerate(("uniform", "supported")):
                started = time.perf_counter()
                entropy = [
                    94003 if pilot else 94009,
                    model_index,
                    ident["seed"],
                    ident["seat"],
                    ident["candidate_decision"],
                    distribution,
                ]
                rng = np.random.default_rng(np.random.SeedSequence(entropy))
                if distribution == 0:
                    worlds = sample_hidden_batch(root, rng, particles)
                    hand_indices = None
                else:
                    worlds, indices = sample_supported_worlds(
                        root.state, hands, rng, particles
                    )
                    hand_indices = indices.tolist()
                before_calls = teacher.total_calls
                values, counts = rollout(neural, teacher, cached, worlds, verify=True)
                total_calls = teacher.total_calls - before_calls
                archived_exact, archive_counts = False, None
                if pilot:
                    before_archive = teacher.total_calls
                    old_values, archive_counts = archived_rollout(
                        neural, teacher, cached, worlds, verify=True
                    )
                    np.testing.assert_array_equal(values, old_values)
                    assert counts == archive_counts
                    archived_exact = True
                    archive_counts = {
                        **archive_counts,
                        "total_teacher_calls": teacher.total_calls - before_archive,
                    }
                counts["total_teacher_calls"] = total_calls
                counts["verification_teacher_calls"] = (
                    total_calls - counts["published_calls"]
                )
                assert counts["verification_teacher_calls"] >= 0
                self_gain = values[1].astype(np.float64) - values[0]
                published_gain = values[3].astype(np.float64) - values[2]
                row = {
                    "root_index": index,
                    "identity": ident,
                    "distribution": name,
                    "entropy": entropy,
                    "worlds_sha256": hashlib.sha256(worlds.tobytes()).hexdigest(),
                    "hand_indices": hand_indices,
                    "values": values.tolist(),
                    "counts": counts,
                    "self_gain": statistics(self_gain),
                    "published_gain": statistics(published_gain),
                    "gain_shift": statistics(published_gain - self_gain),
                    "archived_exact": archived_exact,
                    "archive_counts": archive_counts,
                    "seconds": time.perf_counter() - started,
                }
                report["results"].append(row)
                root_results.append(row)
                print(
                    json.dumps(
                        {
                            "completed": len(report["results"]),
                            "root_index": index,
                            "distribution": name,
                            "seconds": row["seconds"],
                            "counts": counts,
                        }
                    ),
                    flush=True,
                )
                write("runs/supported_values_partial.json", report)
            uniform, supported = root_results
            differences = {}
            for key in ("self_gain", "published_gain", "gain_shift"):
                differences[key] = {
                    "mean": supported[key]["mean"] - uniform[key]["mean"],
                    "se": float(np.hypot(supported[key]["se"], uniform[key]["se"])),
                }
            report["distribution_differences"].append(
                {"root_index": index, **differences}
            )
    assert metadata() == meta
    assert len(report["results"]) == (4 if pilot else 18)
    report["status"] = "complete"
    write(output, report)
    manifest.update(status="complete", output=str(output), output_sha256=sha(output))
    write(PROCESS, manifest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["pilot", "run"])
    main(parser.parse_args().mode == "pilot")
