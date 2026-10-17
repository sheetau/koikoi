"""Portable evidence verification without a real model load or archive release."""

import hashlib
import json
import zipfile

import pytest

from scripts import package_research_artifacts as package


@pytest.mark.parametrize(
    "name",
    [
        "../escape",
        "/absolute",
        "C:/drive",
        "folder\\file",
        "a//b",
        "a/./b",
        "a/../b",
        ".",
        "",
    ],
)
def test_archive_names_must_be_canonical_relative_files(name):
    with pytest.raises(ValueError):
        package.safe_name(name)


def test_original_windows_binding_is_portable_and_cannot_escape_its_root():
    root = r"C:\workspace\koikoi"
    assert (
        package.normalized_path(root + r"\scripts\audit.py", root) == "scripts/audit.py"
    )
    assert (
        package.normalized_path(r"runs\final\report.json", root)
        == "runs/final/report.json"
    )
    with pytest.raises(ValueError):
        package.normalized_path(r"C:\elsewhere\private.txt", root)


@pytest.mark.parametrize(
    "root,ordered_bytes",
    [
        (
            r"C:\workspace",
            (
                b"koikoi/a.pyalpha\nkoikoi/native/core.hnested\n"
                b"koikoi/native.pyflat\nkoikoi/Z.pyzeta\n"
            ),
        ),
        (
            "/workspace",
            (
                b"koikoi/Z.pyzeta\nkoikoi/a.pyalpha\n"
                b"koikoi/native/core.hnested\nkoikoi/native.pyflat\n"
            ),
        ),
    ],
)
def test_source_identity_uses_original_path_component_and_case_order(
    tmp_path, root, ordered_bytes
):
    # native/core.h precedes native.py in pathlib order; raw strings reverse it.
    # Windows also folds case, whereas POSIX places uppercase Z before a.
    payload = {
        "koikoi/native.py": b"flat\n",
        "koikoi/Z.py": b"zeta\n",
        "koikoi/native/core.h": b"nested\n",
        "koikoi/a.py": b"alpha\n",
        "koikoi/notes.md": b"excluded documentation",
        "scripts/helper.py": b"excluded tooling",
    }
    path = tmp_path / "source.zip"
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in payload.items():
            archive.writestr(name, content)
    with zipfile.ZipFile(path) as archive:
        assert (
            package.source_digest(archive, payload, root)
            == hashlib.sha256(ordered_bytes).hexdigest()
        )


@pytest.mark.parametrize("fault", [None, "bytes", "digest", "extra", "case_collision"])
def test_archive_integrity_checks_exact_members_bytes_and_hashes(tmp_path, fault):
    content = b"archived evidence"
    metadata = {"sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}
    if fault == "bytes":
        metadata["bytes"] += 1
    elif fault == "digest":
        metadata["sha256"] = "0" * 64
    manifest = {
        "format": "koikoi_research_evidence_v1",
        "files": {"data.json": metadata},
    }
    path = tmp_path / "evidence.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("data.json", content)
        archive.writestr("ARTIFACT_MANIFEST.json", json.dumps(manifest))
        if fault == "extra":
            archive.writestr("extra", b"unexpected")
        elif fault == "case_collision":
            archive.writestr("DATA.JSON", content)
    with zipfile.ZipFile(path) as archive:
        if fault is None:
            assert package.verify_members(archive) == manifest
        else:
            with pytest.raises((AssertionError, ValueError)):
                package.verify_members(archive)


def records():
    return {
        "pairs": 2,
        "matches": 4,
        "records": [
            {
                "seed": 100000000 + i,
                "seat0_score": 1 - i,
                "seat1_score": 1 - i,
                "seat0_point_difference": 2 - 4 * i,
                "seat1_point_difference": 2 - 4 * i,
            }
            for i in range(2)
        ],
    }


def test_portable_statistics_have_known_two_pair_mean_and_bootstrap_endpoints():
    assert package.paired_statistics(records(), count=2) == {
        "score_mean": 0.5,
        "score_ci95": [0, 1],
        "point_mean": 0,
        "point_ci95": [-2, 2],
    }


@pytest.mark.parametrize("fault", ["sign", "duplicate", "wrong_count"])
def test_portable_statistics_reject_inconsistent_raw_records(fault):
    report = records()
    if fault == "sign":
        report["records"][0]["seat0_point_difference"] = -2
    elif fault == "duplicate":
        report["records"][1]["seed"] = 100000000
    else:
        report["matches"] = 3
    with pytest.raises(AssertionError):
        package.paired_statistics(report, count=2)


def test_live_study_blocks_archive_creation_before_inventory(tmp_path, monkeypatch):
    monkeypatch.setattr(package, "ROOT", tmp_path)
    final = tmp_path / package.FINAL
    final.mkdir(parents=True)
    (final / "process.json").write_text('{"status":"running"}', encoding="utf-8")
    monkeypatch.setattr(
        package,
        "inventory",
        lambda: pytest.fail("Incomplete study reached archive inventory"),
    )
    with pytest.raises(RuntimeError, match="still running"):
        package.build(tmp_path / "out/evidence.zip")
    assert not (tmp_path / "out").exists()
