"""Fixed eight-fixture feasibility screen of earlier entry, using closed sources."""

import argparse
import copy
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from koikoi.env import BatchEnv
from scripts.logged_public_source import audit, collect
from scripts.prospective_incumbent import compressed_save
from scripts.run_arbitration_study import PARENTS, read, sha, write
from scripts.uniform_controller_assessment import assess_pair
from scripts.uniform_controller_study import actors, audit_plan, check, plan_source
from scripts.uniform_deal_source import DeckSample, LoggedDealSource

ORIGINAL = Path("runs/uniform_controller_evaluation")
DIAGNOSIS = Path("runs/uniform_gate_diagnosis_v2.json")


def freeze(base):
    if base.exists():
        raise FileExistsError(base)
    original = check(ORIGINAL)
    diagnosis = read(DIAGNOSIS)
    if diagnosis["status"] != "passed":
        raise ValueError("Passed gate replay required")
    selected = {}
    for entry in diagnosis["cases"]:
        if entry["roots"]["5"] is not None or entry["roots"]["6"] is None:
            continue
        case = entry["case"]
        chunk = case["index"] // 16
        path = ORIGINAL / f"chunk_{chunk:04}" / "records.json"
        if sha(path) != diagnosis["records"][str(path)]:
            raise ValueError("Diagnosed source changed")
        source = read(path)[case["index"] % 16]["source"]
        key = (case["population"], source["seat"])
        if key not in selected:
            selected[key] = {"case": case, "seat": source["seat"],
                             "six_draw_root": entry["roots"]["6"],
                             "records": str(path), "records_sha256": sha(path)}
        if len(selected) == 8:
            break
    if len(selected) != 8:
        raise ValueError("Expected all four populations and both seats")
    config = copy.deepcopy(original["design"])
    config["max_draws"] = 6
    protocol = {
        "scope": "Development feasibility only; mechanically first added six-draw root per population/seat.",
        "original_protocol_sha256": sha(ORIGINAL / "protocol.json"),
        "diagnosis_sha256": sha(DIAGNOSIS), "script_sha256": sha(__file__),
        "design": config, "opponent_identities": original["opponent_identities"],
        "fixtures": list(selected.values()),
        "criterion": "Descriptive completion and exact-gain feasibility; no population inference.",
    }
    base.mkdir(parents=True)
    write(base / "protocol.json", protocol)
    print(json.dumps({"fixtures": [e["case"]["index"] for e in protocol["fixtures"]]}))


def checked(base):
    original = check(ORIGINAL)
    p = read(base / "protocol.json")
    expected = copy.deepcopy(original["design"])
    expected["max_draws"] = 6
    if (p["script_sha256"] != sha(__file__) or p["design"] != expected
            or p["diagnosis_sha256"] != sha(DIAGNOSIS)
            or p["original_protocol_sha256"] != sha(ORIGINAL / "protocol.json")
            or p["opponent_identities"] != original["opponent_identities"]):
        raise ValueError("Frozen screen input changed")
    return p


def worker(base, index):
    p = checked(base)
    fixture = p["fixtures"][index]
    if sha(fixture["records"]) != fixture["records_sha256"]:
        raise ValueError("Fixture changed")
    old = read(fixture["records"])[fixture["case"]["index"] % 16]["source"]
    if old["selected_root"] is not None:
        raise ValueError("Expected a previously missing root")
    folder = base / f"fixture_{index:02}"
    folder.mkdir()
    parents, opponents, predictors = actors()
    deals = iter(old["deals"])

    def draw():
        d = next(deals)
        return DeckSample(tuple(d["deck"]), d["attempts"], d["bounded_draws"])

    parent, opponent = parents[old["parent"]], opponents[old["opponent"]]
    source = collect(LoggedDealSource(old["seed"], old["profile"], draw),
                     old["source_key"], old["seat"], parent, opponent, predictors,
                     BatchEnv(1, debug=True), max_draws=6)
    source.update(parent=old["parent"], opponent=old["opponent"])
    for key in ("actions", "final", "deals", "work", "native_work", "identification_events"):
        if source[key] != old[key]:
            raise ValueError(f"Source reproduction differs: {key}")
    if source["selected_root"]["decision"] != fixture["six_draw_root"]["decision"]:
        raise ValueError("Earlier root differs from gate diagnosis")
    write(folder / "source.json", source)
    print(json.dumps({"fixture": index, "stage": "source_verified"}), flush=True)
    row, evidence = plan_source(source, parents, opponents, p)
    write(folder / "selection.json", row)
    compressed_save(folder / "program.pkl.gz", evidence)
    print(json.dumps({"fixture": index, "stage": "planned", "selection": row.get("selection")}), flush=True)
    source_audit = audit(source, predictors, parent, opponent)
    selection, support_counts = audit_plan(row, evidence, source, opponents, p)
    assessment = None if selection is None else assess_pair(
        source, selection, parent, opponent, PARENTS[old["parent"]][1]
    )
    np.testing.assert_array_equal(source["final"], old["final"])
    write(folder / "result.json", {
        "status": "passed", "fixture": fixture, "protocol_sha256": sha(base / "protocol.json"),
        "selection": row, "source_audit": source_audit, "support_audit_counts": support_counts,
        "assessment": assessment, "fresh_chance_draws": 0,
        "artifacts": {f.name: sha(f) for f in folder.iterdir() if f.is_file()},
    })
    print(json.dumps({"fixture": index, "stage": "complete"}), flush=True)


def run(base):
    p = checked(base)
    queue = base / "queue.json"
    if queue.exists():
        raise FileExistsError(queue)
    write(queue, {"status": "running", "fixtures": len(p["fixtures"]), "workers": 2})

    def launch(index):
        with (base / f"worker_{index:02}.log").open("w", encoding="utf-8") as log:
            subprocess.run([sys.executable, "-m", "scripts.screen_six_draw_gate", "--base", str(base),
                            "--worker", str(index)], stdout=log, stderr=subprocess.STDOUT, check=True)
        return index

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(launch, i) for i in range(len(p["fixtures"]))]
            for future in as_completed(futures):
                print(json.dumps({"complete_fixture": future.result()}), flush=True)
        results = [read(base / f"fixture_{i:02}" / "result.json") for i in range(8)]
        write(base / "summary.json", {"status": "passed", "scope": p["scope"],
              "selections": [r["selection"].get("selection") for r in results],
              "results": {str(base / f"fixture_{i:02}" / "result.json"):
                          sha(base / f"fixture_{i:02}" / "result.json") for i in range(8)}})
        write(queue, {"status": "complete", "fixtures": 8})
    except Exception as error:
        write(queue, {"status": "failed", "error": repr(error)})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path("runs/six_draw_gate_screen"))
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--worker", type=int)
    args = parser.parse_args()
    if args.freeze:
        freeze(args.base)
    elif args.worker is not None:
        worker(args.base, args.worker)
    elif args.run:
        run(args.base)
    else:
        parser.error("Choose --freeze, --run or --worker")
