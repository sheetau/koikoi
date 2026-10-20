"""Frozen six-arm score-auxiliary learning; no strength evaluation or promotion."""

import argparse
import json
import math
import os
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict
from pathlib import Path

import torch

from koikoi.published_train import PublishedConfig, PublishedTrainer
from scripts.audit_pretrained_ppo_screen import validate_accounting
from scripts.check_score_auxiliary_training import equal, equal_ppo, load
from scripts.run_arbitration_study import PARENTS
from scripts.score_auxiliary_trainer import AuxiliaryConfig, ScoreAuxiliaryTrainer, sha

CAPABILITY = Path("runs/score_auxiliary_training_capability/report.json")
CAPABILITY_SHA = "49e24dc6f33247b69b8922d71d2716fcdaf676220b75c2630785ea829b833f9e"


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, indent=2) + "\n")


def checked(inputs):
    for path, digest in inputs.items():
        if sha(path) != digest:
            raise ValueError(f"Changed input: {path}")


def design():
    arms, ranges = {}, {}
    for index, seed in enumerate((106110001, 106210001), 1):
        ranges[f"ppo{index}"] = [seed, seed + 127]
        for kind in ("none", "win", "score"):
            ppo = PublishedConfig(device="cuda", seed=seed, initial_variant="point")
            aux = AuxiliaryConfig(
                kind=kind,
                interval=5,
                deal_seed=107010001 + index * 100000,
                action_seed=108010001 + index * 100000,
                reservoir_seed=108210001 + index * 100000,
                order_seed=108410001 + index * 100000,
            )
            arms[f"seed{index}_{kind}"] = {"ppo": asdict(ppo), "auxiliary": asdict(aux)}
        for name, first, size in (
            ("deal", aux.deal_seed, 12800),
            ("action", aux.action_seed, 100),
            ("reservoir", aux.reservoir_seed, 100),
            ("order", aux.order_seed, 100),
        ):
            ranges[f"auxiliary{index}_{name}"] = [first, first + size - 1]
    ranges.update(
        wp_evaluation=[109110001, 109114096],
        point_evaluation=[109210001, 109214096],
        bootstrap=[109310001, 109310001],
    )
    return {
        "format": "score_auxiliary_main_v1",
        "arms": arms,
        "seed_ranges": ranges,
        "updates": 520,
        "warmup": 20,
        "ppo_decisions_per_arm": 8519680,
        "total_ppo_decisions": 51118080,
        "auxiliary_matches_per_arm": 12800,
        "total_auxiliary_matches": 76800,
        "evaluation": {
            "pairs_per_opponent": 4096,
            "wp_first_seed": 109110001,
            "point_first_seed": 109210001,
            "bootstrap_seed": 109310001,
            "bootstrap_replicates": 10000,
            "full_matches": 131072,
            "primary": "Six simultaneous95% mixture-win lower bounds: score minus none, win and retained parent at both seeds; all strictly positive",
            "candidate": "seed1_score",
            "secondary_coordinates": 120,
        },
        "scope": "Fixed endpoints. No evaluation, promotion, restart or outcome-based extension in this runner.",
    }


def scan_seed_ranges(directory, ranges):
    overlaps, unreadable, found = [], [], {}
    count = 0

    def walk(value, trail=(), seed_context=False):
        if isinstance(value, dict):
            for key, item in value.items():
                yield from walk(
                    item, (*trail, str(key)), seed_context or "seed" in str(key).lower()
                )
        elif isinstance(value, list):
            for index, item in enumerate(value):
                yield from walk(item, (*trail, str(index)), seed_context)
        elif type(value) is int and seed_context:
            yield trail, value

    for path in sorted(Path(directory).rglob("*.json")):
        count += 1
        try:
            content = path.read_bytes()
            encoding = (
                "utf-32"
                if content.startswith((b"\xff\xfe\x00\x00", b"\x00\x00\xfe\xff"))
                else "utf-16"
                if content.startswith((b"\xff\xfe", b"\xfe\xff"))
                else "utf-8-sig"
            )
            values = list(walk(json.loads(content.decode(encoding))))
        except (OSError, UnicodeError, ValueError) as error:
            unreadable.append({"path": str(path), "error": str(error)})
            continue
        if values:
            found[str(path)] = sha(path)
        for trail, value in values:
            for name, (lo, hi) in ranges.items():
                if lo <= value <= hi:
                    overlaps.append(
                        {
                            "path": str(path),
                            "field": list(trail),
                            "value": value,
                            "range": name,
                        }
                    )
    return {
        "ranges": ranges,
        "json_files_scanned": count,
        "seed_bearing_files": found,
        "overlaps": overlaps,
        "unreadable": unreadable,
        "scope": "Recorded scalar integers under seed-named JSON fields/descendants; not unrecorded work or inferred arithmetic schedules.",
    }


def prepare(root):
    root = Path(root)
    if root.exists():
        raise ValueError("Preserve existing study directory")
    if sha(CAPABILITY) != CAPABILITY_SHA:
        raise ValueError("Wrong production capability report")
    capability = json.loads(CAPABILITY.read_text(encoding="utf-8"))
    if not capability["passed"] or capability["strength_evidence"]:
        raise ValueError("Invalid capability status")
    checked(capability["files"])
    inputs = dict(capability["files"])
    for name, count in (
        ("score_auxiliary_collection_fixed_tests.xml", 3),
        ("score_auxiliary_trainer_tests.xml", 6),
        ("score_auxiliary_study_runner_tests.xml", 5),
        ("score_auxiliary_seed_scan_tests.xml", 1),
    ):
        path = Path("runs") / name
        suite = ET.parse(path).getroot().find("testsuite")
        if int(suite.attrib["tests"]) != count or any(
            int(suite.attrib[k]) for k in ("errors", "failures", "skipped")
        ):
            raise ValueError("Required capability tests did not pass")
        inputs[str(path)] = sha(path)
    for path in (
        CAPABILITY,
        Path(__file__),
        Path("docs/SCORE_AUXILIARY_MAIN.rst"),
        Path("scripts/score_auxiliary.py"),
        Path("scripts/collect_score_auxiliary.py"),
        Path("scripts/score_auxiliary_trainer.py"),
        Path("scripts/check_score_auxiliary_training.py"),
        Path("scripts/audit_pretrained_ppo_screen.py"),
        Path("scripts/run_arbitration_study.py"),
        Path("tests/test_train_score_auxiliary_study.py"),
        Path("runs/score_auxiliary_study_runner_tests.xml"),
        Path("runs/published_batch_validation.npz"),
    ):
        inputs[str(path)] = sha(path)
    for path in Path("koikoi").rglob("*"):
        if path.suffix in (".py", ".h", ".cpp", ".cu"):
            inputs[str(path)] = sha(path)
    for path in Path("external/KoiKoi-AI/model_agent").glob("*_rl_*.pt"):
        inputs[str(path)] = sha(path)
    for name in ("koikoigame.py", "koikoinet2L.py", "koikoilearn.py"):
        path = Path("external/KoiKoi-AI") / name
        inputs[str(path)] = sha(path)
    for path, digest in PARENTS:
        if sha(path) != digest:
            raise ValueError("Changed retained comparator")
        inputs[path] = digest
    protocol = design()
    scan = scan_seed_ranges("runs", protocol["seed_ranges"])
    if scan["overlaps"] or scan["unreadable"]:
        write_new(Path("runs/score_auxiliary_seed_scan_rejected.json"), scan)
        raise ValueError("Seed scan has overlaps or unreadable files")
    root.mkdir(parents=True, exist_ok=False)
    write_new(root / "seed_scan.json", scan)
    inputs[str(root / "seed_scan.json")] = sha(root / "seed_scan.json")
    protocol["inputs"] = inputs
    checked(inputs)
    write_new(root / "protocol.json", protocol)
    print(
        json.dumps(
            {
                "prepared": str(root),
                "inputs": len(inputs),
                "protocol_sha256": sha(root / "protocol.json"),
            }
        ),
        flush=True,
    )


def validate_auxiliary(trainer, cycle_rows):
    c, counts = trainer.config, trainer.counters
    assert (
        counts["cycles"]
        == trainer.ppo.counters["policy_updates"] // c.interval
        == len(cycle_rows)
    )
    assert counts["matches"] == counts["cycles"] * c.matches
    for key in ("decisions", "rounds", "matches"):
        assert counts[key] == sum(row["collection_work"][key] for row in cycle_rows)
    assert counts["retained_rows"] == sum(row["retained_rows"] for row in cycle_rows)
    assert counts["example_presentations"] == sum(
        row["example_presentations"] for row in cycle_rows
    )
    steps = sum(
        math.ceil(row["retained_rows"] / c.minibatch) * c.epochs for row in cycle_rows
    )
    assert counts["optimizer_steps"] == (0 if c.kind == "none" else steps)


def train_arm(root, name, specification, protocol):
    checked(protocol["inputs"])
    folder = root / name
    folder.mkdir(exist_ok=False)
    trainer = ScoreAuxiliaryTrainer(
        PublishedTrainer(PublishedConfig(**specification["ppo"])),
        AuxiliaryConfig(**specification["auxiliary"]),
    )
    trainer.save(folder / "initial")
    control = root / (name.split("_")[0] + "_none")
    if name.endswith(("_win", "_score")):
        equal_ppo(load(folder / "initial/ppo.pt"), load(control / "initial/ppo.pt"))
    records, cycle_rows = [], []
    started = time.perf_counter()
    with (folder / "metrics.jsonl").open("x", encoding="utf-8", newline="\n") as log:
        for update in range(1, protocol["updates"] + 1):
            before = trainer.counters["cycles"]
            metrics = trainer.update()
            record = metrics | trainer.ppo.counters
            if not all(math.isfinite(value) for value in record.values()):
                raise FloatingPointError("Nonfinite main learning metrics")
            records.append(record)
            if trainer.counters["cycles"] != before:
                cycle_rows.append(trainer.last_metrics)
                with (folder / f"collection_{before + 1:04d}.pt").open("xb") as stream:
                    torch.save(trainer.last_collection, stream)
                if before == 0 and name.endswith(("_win", "_score")):
                    equal(trainer.last_collection, load(control / "collection_0001.pt"))
            row = {
                "ppo": record,
                "auxiliary_counters": trainer.counters.copy(),
                "auxiliary_phase": trainer.last_metrics
                if trainer.counters["cycles"] != before
                else None,
                "seconds": time.perf_counter() - started,
            }
            log.write(json.dumps(row) + "\n")
            log.flush()
            policy_updates = trainer.ppo.counters["policy_updates"]
            if (
                update == 20
                or policy_updates == 5
                or (policy_updates and policy_updates % 25 == 0)
            ):
                checkpoint = folder / f"update_{update:08d}"
                trainer.save(checkpoint)
                payload = load(checkpoint / "ppo.pt")
                validate_accounting(payload, records, update)
                validate_auxiliary(trainer, cycle_rows)
                if update == 20 and name.endswith(("_win", "_score")):
                    equal_ppo(payload, load(control / "update_00000020/ppo.pt"))
            if update % 20 == 0:
                print(
                    json.dumps(
                        {
                            "arm": name,
                            "update": update,
                            "auxiliary_cycles": trainer.counters["cycles"],
                        }
                    ),
                    flush=True,
                )
    assert trainer.ppo.counters["decisions"] == protocol["ppo_decisions_per_arm"]
    assert trainer.counters["matches"] == protocol["auxiliary_matches_per_arm"]
    checked(protocol["inputs"])
    result = {
        "status": "complete",
        "ppo_work": trainer.ppo.counters,
        "auxiliary_work": trainer.counters,
        "endpoint": str(checkpoint),
        "manifest_sha256": sha(checkpoint / "manifest.json"),
        "metrics_sha256": sha(folder / "metrics.jsonl"),
        "protocol_sha256": sha(root / "protocol.json"),
    }
    write_new(folder / "complete.json", result)
    return result


def run(root):
    root = Path(root)
    protocol = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
    checked(protocol["inputs"])
    write_new(
        root / "started.json",
        {"pid": os.getpid(), "protocol_sha256": sha(root / "protocol.json")},
    )
    try:
        results = {
            name: train_arm(root, name, config, protocol)
            for name, config in protocol["arms"].items()
        }
        totals = {
            group: {
                key: sum(result[group][key] for result in results.values())
                for key in ("decisions", "rounds", "matches")
            }
            for group in ("ppo_work", "auxiliary_work")
        }
        assert totals["ppo_work"]["decisions"] == protocol["total_ppo_decisions"]
        assert (
            totals["auxiliary_work"]["matches"] == protocol["total_auxiliary_matches"]
        )
        write_new(
            root / "complete.json",
            {
                "status": "complete",
                "arms": results,
                "totals": totals,
                "evaluation": "not_started",
                "deployment_changed": False,
                "protocol_sha256": sha(root / "protocol.json"),
            },
        )
    except BaseException as error:
        write_new(root / "failed.json", {"error": repr(error)})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    {"prepare": prepare, "run": run}[args.command](args.output)
