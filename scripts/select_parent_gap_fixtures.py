"""Select the first closed five-draw parent refusal per population and seat."""

import argparse
from pathlib import Path

from scripts.run_arbitration_study import read, sha, write
from scripts.uniform_controller_study import check

ORIGINAL = Path("runs/uniform_controller_evaluation")
DIAGNOSIS = Path("runs/uniform_controller_opportunities.json")


def select(output):
    if output.exists():
        raise FileExistsError(output)
    protocol = check(ORIGINAL)
    diagnosis = read(DIAGNOSIS)
    if (
        diagnosis["status"] != "passed"
        or read(ORIGINAL / "queue.json")["status"] != "complete"
    ):
        raise ValueError("Completed frozen original cohort required")
    selected, inspected = {}, 0
    records_bindings = {}
    for chunk in range(512):
        folder = ORIGINAL / f"chunk_{chunk:04}"
        path = folder / "records.json"
        if sha(path) != diagnosis["records"][str(path)]:
            raise ValueError("Diagnosed original record changed")
        records_bindings[str(path)] = sha(path)
        for record in read(path):
            case = record["case"]
            if case != protocol["cases"][inspected]:
                raise ValueError("Original source order changed")
            inspected += 1
            if (
                record["status"] != "evaluated"
                or record["selection"]["reason"] != "parent_unvalued"
            ):
                continue
            source = record["source"]
            group = (case["population"], source["seat"])
            if group in selected:
                continue
            index = case["index"]
            source_path = folder / f"source_{index:05}" / "record.json"
            selection_path = folder / f"selection_{index:05}.json"
            program_path = folder / f"program_{index:05}.pkl.gz"
            if (
                read(source_path) != source
                or sha(selection_path) != record["selection_sha256"]
            ):
                raise ValueError("Saved source or selection differs")
            selection = read(selection_path)
            if (
                selection["selection"]["reason"] != "parent_unvalued"
                or selection["selection"]["parent_q"] is not None
                or source["max_draws"] != 5
                or source["selected_root"] is None
                or set(selection["phases"]) != {"fixed"}
            ):
                raise ValueError("Expected a five-draw parent-refusal fixture")
            selected[group] = {
                "case": case,
                "seat": source["seat"],
                "source": str(source_path),
                "selection": str(selection_path),
                "program": str(program_path),
                "original_fixed_work": selection["phases"]["fixed"]["work"],
                "inputs": {
                    str(p): sha(p) for p in (source_path, selection_path, program_path)
                },
            }
            if len(selected) == 8:
                break
        if len(selected) == 8:
            break
    if len(selected) != 8:
        raise ValueError("Missing a population/seat group")
    manifest = {
        "scope": "Development only; first parent-unvalued five-draw source in each population/seat.",
        "script_sha256": sha(__file__),
        "protocol_sha256": sha(ORIGINAL / "protocol.json"),
        "diagnosis_sha256": sha(DIAGNOSIS),
        "prefix_sources_inspected": inspected,
        "records": records_bindings,
        "fixtures": list(selected.values()),
    }
    write(output, manifest)
    return manifest


def checked(path):
    check(ORIGINAL)
    manifest = read(path)
    if (
        manifest["script_sha256"] != sha(__file__)
        or manifest["protocol_sha256"] != sha(ORIGINAL / "protocol.json")
        or manifest["diagnosis_sha256"] != sha(DIAGNOSIS)
        or len(manifest["fixtures"]) != 8
    ):
        raise ValueError("Frozen fixture selection changed")
    for name, digest in manifest["records"].items():
        if sha(name) != digest:
            raise ValueError("Selection-prefix record changed")
    for fixture in manifest["fixtures"]:
        for name, digest in fixture["inputs"].items():
            if sha(name) != digest:
                raise ValueError("Selected source/evidence changed")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("runs/five_draw_parent_gap_fixtures.json")
    )
    args = parser.parse_args()
    result = select(args.output)
    print([f["case"]["index"] for f in result["fixtures"]])
