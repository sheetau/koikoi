"""Plan/build a local evidence archive; verify bytes and final raw statistics.

No upload, extraction, model deserialization, inference, or training is performed.
The verifier needs Python and NumPy only, independent of original absolute paths.
"""

import argparse
import hashlib
import json
import math
import subprocess
import zipfile
from pathlib import Path, PurePosixPath, PureWindowsPath

ROOT = Path(__file__).resolve().parents[1]
FINAL = "runs/final_published_test"
LATENCY = "runs/final_latency"
FINAL_SHA = "4a69cff95079e2bd4cfcda84241f4096149c05cfe408bd11cbfde154eaa060e4"
DESTINATION = ROOT / "dist/koikoi_research_artifacts.zip"


def safe_name(name):
    p = PurePosixPath(name)
    if (
        not name
        or p.is_absolute()
        or ".." in p.parts
        or "\\" in name
        or ":" in name
        or p.as_posix() != name
        or name == "."
    ):
        raise ValueError(f"Noncanonical archive member: {name}")
    return name


def normalized_path(value, original_root):
    p = PureWindowsPath(value)
    if p.is_absolute():
        value = p.relative_to(PureWindowsPath(original_root)).as_posix()
    else:
        value = value.replace("\\", "/")
    return safe_name(value)


def digest_stream(stream):
    h = hashlib.sha256()
    size = 0
    while chunk := stream.read(1024 * 1024):
        h.update(chunk)
        size += len(chunk)
    return h.hexdigest(), size


def sha(path):
    with Path(path).open("rb") as stream:
        return digest_stream(stream)[0]


def read(path):
    return json.loads(Path(path).read_bytes())


def inventory():
    protocol = read(ROOT / FINAL / "protocol.json")
    assert sha(ROOT / FINAL / "protocol.json") == FINAL_SHA
    cold = read(ROOT / "runs/cold_ppo_screen/protocol.json")
    names = set(
        subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines()
    )
    for mapping in (
        protocol["bindings"],
        protocol["seed_scan"]["reports"],
        cold["dependency_hashes"],
    ):
        names.update(normalized_path(path, str(ROOT)) for path in mapping)
    names.add(normalized_path(cold["initial_checkpoint"], str(ROOT)))
    for case in cold["cases"]:
        for filename in (
            "config.json",
            "metrics.jsonl",
            "update_00000020.pt",
            "update_00000520.pt",
        ):
            names.add(f"runs/cold_ppo_screen/{case}/{filename}")
    names.add("runs/cold_ppo_seed_range_check.json")
    for filename in (
        "protocol.json",
        "process.json",
        "summary.json",
        "independent_audit.json",
    ):
        names.add(f"{FINAL}/{filename}")
    names.update(f"{FINAL}/{job}/evaluation.json" for job in protocol["jobs"])
    for filename in (
        "protocol.json",
        "process.json",
        "report.json",
        "raw.npz",
        "independent_audit.json",
    ):
        names.add(f"{LATENCY}/{filename}")
    names.update(
        (
            "runs/published_batch_validation.npz",
            "runs/published_batch_independent_audit.json",
            "runs/ppo12_scale_seed30001_captured/final_100m.pt",
            "runs/ppo12_scale_seed30001_captured/budget_100m.json",
            "runs/ppo12_scale_seed30001_captured/audit_100m.json",
            "runs/ppo12_scale_seed30001_captured/audit_100m_independent.json",
            "external/KoiKoi-AI/LICENSE",
            "external/KoiKoi-AI/README.md",
            "output/pdf/manuscript.pdf",
            "runs/manuscript_final_qa.json",
        )
    )
    rows = {}
    for name in sorted(names):
        safe_name(name)
        path = ROOT / name
        if path.is_symlink() or not path.resolve().is_relative_to(ROOT.resolve()):
            raise ValueError(f"Artifact escapes workspace or is a symlink: {name}")
        rows[name] = path.stat().st_size if path.is_file() else None
    return rows


def readiness():
    for folder in (FINAL, LATENCY):
        p = read(ROOT / folder / "process.json")
        if p.get("status") != "complete_awaiting_audit":
            raise RuntimeError(f"Evidence still running or failed: {folder}")
        audit = read(ROOT / folder / "independent_audit.json")
        if audit.get("status") != "passed":
            raise RuntimeError(f"Complete independent audit required: {folder}")
        assert audit["protocol_sha256"] == sha(ROOT / folder / "protocol.json")
        report = "summary.json" if folder == FINAL else "report.json"
        key = "summary_sha256" if folder == FINAL else "report_sha256"
        assert audit[key] == sha(ROOT / folder / report)
    evidence = read(ROOT / "docs/paper/generated/evidence.json")
    if evidence["status"] != "audited_final_results_available":
        raise RuntimeError("Refresh the manuscript evidence after final auditing")
    assert evidence["evidence"]["final"] == read(
        ROOT / FINAL / "independent_audit.json"
    )
    qa = read(ROOT / "runs/manuscript_final_qa.json")
    if qa.get("status") != "passed" or qa.get("pending_final_results") is not False:
        raise RuntimeError("Final manuscript PDF review is still incomplete")
    for key, path in (
        ("pdf_sha256", "output/pdf/manuscript.pdf"),
        ("manuscript_sha256", "docs/paper/manuscript.tex"),
        ("evidence_sha256", "docs/paper/generated/evidence.json"),
    ):
        assert qa[key] == sha(ROOT / path)
    if subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    ).strip():
        raise RuntimeError("Commit the intended archive source snapshot first")


def build(destination=DESTINATION):
    readiness()  # Gate before creating either archive or temporary directory.
    files = inventory()
    missing = [name for name, size in files.items() if size is None]
    if missing:
        raise FileNotFoundError(f"Required artifacts missing: {missing}")
    destination = Path(destination)
    partial = destination.with_suffix(".partial.zip")
    if destination.exists() or partial.exists():
        raise FileExistsError("Retain the existing archive; choose a new destination")
    manifest = {
        "format": "koikoi_research_evidence_v1",
        "original_root": str(ROOT),
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "final_protocol_sha256": FINAL_SHA,
        "source_sha256": read(ROOT / FINAL / "protocol.json")["provenance"][
            "source_sha256"
        ],
        "files": {},
        "scope": "Local evidence archive. No external publication. Original protocols retained byte-for-byte; absolute binding paths are mapped beneath original_root for archive verification. Historical experiments are preserved as evidence, not claimed fully rerunnable from this selected archive. No model or training replay is performed by verification.",
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        partial, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True
    ) as archive:
        for name, expected_size in files.items():
            path = ROOT / name
            h, size = hashlib.sha256(), 0
            with (
                path.open("rb") as source,
                archive.open(name, "w", force_zip64=True) as target,
            ):
                while chunk := source.read(1024 * 1024):
                    h.update(chunk)
                    size += len(chunk)
                    target.write(chunk)
            assert size == expected_size, f"Artifact changed while packaging: {name}"
            manifest["files"][name] = {"sha256": h.hexdigest(), "bytes": size}
        archive.writestr(
            "ARTIFACT_MANIFEST.json", json.dumps(manifest, indent=2) + "\n"
        )
    readiness()
    assert (
        subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
        == manifest["revision"]
    )
    result = verify(partial)
    partial.rename(destination)
    result.update(archive=str(destination), archive_sha256=sha(destination))
    receipt = ROOT / "runs/research_artifact_bundle.json"
    receipt.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(result))


def verify_members(archive):
    names = archive.namelist()
    if len({name.casefold() for name in names}) != len(names):
        raise ValueError("Duplicate or case-colliding archive members")
    for name in names:
        safe_name(name)
    manifest = json.loads(archive.read("ARTIFACT_MANIFEST.json"))
    assert manifest["format"] == "koikoi_research_evidence_v1"
    assert set(names) == set(manifest["files"]) | {"ARTIFACT_MANIFEST.json"}
    for name, expected in manifest["files"].items():
        with archive.open(name) as source:
            digest, size = digest_stream(source)
        if digest != expected["sha256"] or size != expected["bytes"]:
            raise ValueError(f"Archive payload mismatch: {name}")
    return manifest


def paired_statistics(report, count=10000, first=100000000):
    import numpy as np

    rows = sorted(report["records"], key=lambda row: row["seed"])
    assert len(rows) == report["pairs"] == count and report["matches"] == 2 * count
    assert [row["seed"] for row in rows] == list(range(first, first + count))
    scores, points = [], []
    for row in rows:
        for seat in (0, 1):
            point, score = row[f"seat{seat}_point_difference"], row[f"seat{seat}_score"]
            assert math.isfinite(point) and point == int(point) and int(point) % 2 == 0
            assert score == (1 if point > 0 else 0 if point < 0 else 0.5)
        scores.append((row["seat0_score"] + row["seat1_score"]) / 2)
        points.append(
            (row["seat0_point_difference"] + row["seat1_point_difference"]) / 2
        )
    scores, points = np.array(scores), np.array(points)
    rng = np.random.default_rng(99)
    sampled = [[], []]
    for _ in range(5000):
        indices = rng.integers(0, count, count)
        sampled[0].append(float(scores[indices].sum() / count))
        sampled[1].append(float(points[indices].sum() / count))

    def bounds(values):
        ordered = sorted(values)
        result = []
        for q in (0.025, 0.975):
            x = (len(ordered) - 1) * q
            i = math.floor(x)
            result.append(ordered[i] + (x - i) * (ordered[i + 1] - ordered[i]))
        return result

    return {
        "score_mean": float(scores.sum() / count),
        "score_ci95": bounds(sampled[0]),
        "point_mean": float(points.sum() / count),
        "point_ci95": bounds(sampled[1]),
    }


def verify(path):
    import numpy as np

    with zipfile.ZipFile(path) as archive:
        manifest = verify_members(archive)
        get = lambda name: json.loads(archive.read(name))
        files = manifest["files"]
        assert (
            files[f"{FINAL}/protocol.json"]["sha256"]
            == manifest["final_protocol_sha256"]
            == FINAL_SHA
        )
        p = get(f"{FINAL}/protocol.json")
        audit = get(f"{FINAL}/independent_audit.json")
        assert audit["status"] == "passed" and audit["protocol_sha256"] == FINAL_SHA
        assert files[f"{FINAL}/summary.json"]["sha256"] == audit["summary_sha256"]
        assert (
            files["scripts/run_final_published_test.py"]["sha256"] == p["runner_sha256"]
        )
        assert (
            files["scripts/audit_final_published_test.py"]["sha256"]
            == audit["auditor_sha256"]
            == p["auditor_sha256"]
        )
        for mapping in (p["bindings"], p["seed_scan"]["reports"]):
            for original, digest in mapping.items():
                assert (
                    files[normalized_path(original, manifest["original_root"])][
                        "sha256"
                    ]
                    == digest
                )
        cold = get("runs/cold_ppo_screen/protocol.json")
        cold_audit = get("runs/cold_ppo_screen/independent_audit.json")
        for name, check in cold_audit["checkpoints"].items():
            if name == "initial":
                path = normalized_path(
                    cold["initial_checkpoint"], manifest["original_root"]
                )
            else:
                case, update = name.split(":")
                path = f"runs/cold_ppo_screen/{case}/update_{int(update):08d}.pt"
            assert files[path]["sha256"] == check["sha256"]
        latency = get(f"{LATENCY}/independent_audit.json")
        assert latency["status"] == "passed"
        assert files[f"{LATENCY}/protocol.json"]["sha256"] == latency["protocol_sha256"]
        assert files[f"{LATENCY}/report.json"]["sha256"] == latency["report_sha256"]
        assert (
            files[f"{LATENCY}/raw.npz"]["sha256"]
            == get(f"{LATENCY}/report.json")["raw_sha256"]
        )
        assert (
            files["scripts/audit_final_latency.py"]["sha256"]
            == latency["auditor_sha256"]
        )
        scale = get("runs/ppo12_scale_seed30001_captured/audit_100m_independent.json")
        assert (
            files["runs/ppo12_scale_seed30001_captured/final_100m.pt"]["sha256"]
            == scale["checkpoint_sha256"]
        )
        source = hashlib.sha256()
        for name in sorted(files):
            if name.startswith("koikoi/") and PurePosixPath(name).suffix in (
                ".py",
                ".h",
                ".cpp",
                ".cu",
            ):
                source.update(name.encode())
                source.update(archive.read(name))
        assert (
            source.hexdigest()
            == manifest["source_sha256"]
            == p["provenance"]["source_sha256"]
        )
        results = {}
        for job, settings in p["jobs"].items():
            name = f"{FINAL}/{job}/evaluation.json"
            report = get(name)
            assert files[name]["sha256"] == audit["endpoints"][job]["report_sha256"]
            assert (
                report["split"] == "test" and report["opponent"] == settings["opponent"]
            )
            assert normalized_path(
                report["agent"], manifest["original_root"]
            ) == normalized_path(
                p["models"][settings["case"]]["checkpoint"], manifest["original_root"]
            )
            stats = paired_statistics(report)
            for key, value in stats.items():
                np.testing.assert_allclose(
                    value, audit["endpoints"][job][key], atol=1e-12, rtol=0
                )
            results[job] = stats
        for label, variant in (
            ("primary_wp_replicated", "wp"),
            ("secondary_point_replicated", "point"),
        ):
            bounds = [
                results[job]["score_ci95"][0]
                for job, settings in p["jobs"].items()
                if settings["variant"] == variant
            ]
            assert len(bounds) == 2 and audit["conclusions"][label] == (
                min(bounds) > 0.5
            )
        assert files["external/KoiKoi-AI/LICENSE"] and files["THIRD_PARTY_NOTICES.md"]
    return {
        "status": "passed",
        "scope": "archive byte integrity, portable frozen path bindings, production source identity and independently recalculated final paired statistics/conclusions; no training, trajectory, inference or timing replay",
        "files": len(files),
        "uncompressed_bytes": sum(row["bytes"] for row in files.values()),
        "final_conclusions": audit["conclusions"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--plan", action="store_true")
    modes.add_argument("--build", action="store_true")
    modes.add_argument("--verify", type=Path)
    parser.add_argument("--output", type=Path, default=DESTINATION)
    args = parser.parse_args()
    if args.plan:
        files = inventory()
        print(
            json.dumps(
                {
                    "files": len(files),
                    "existing_bytes": sum(size or 0 for size in files.values()),
                    "missing": [name for name, size in files.items() if size is None],
                    "scope": "read-only inventory; no archive created",
                },
                indent=2,
            )
        )
    elif args.build:
        build(args.output)
    else:
        print(json.dumps(verify(args.verify)))


if __name__ == "__main__":
    main()
