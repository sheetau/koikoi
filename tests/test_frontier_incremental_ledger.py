"""Exercise complete source denominators and immutable incremental-study ledgers."""

import pytest

from koikoi.agents import EndgameAgent
from scripts import frontier_incremental_study as study
from scripts.export_frontier_incremental import export
from scripts.frontier_incremental_design import cases, specification
from scripts.run_arbitration_study import read, write
from tests.test_logged_public_source import Minimum, predict
from tests.test_uniform_controller_study import deck_tape


@pytest.fixture
def complete_zero_chunk(tmp_path, monkeypatch):
    actor = EndgameAgent(Minimum())
    protocol = {
        "design": {
            **specification(),
            "chunk_size": 4,
            "source_matches": 4,
            "sources_per_population": 1,
        },
        "cases": cases()[:4],
    }
    write(tmp_path / "protocol.json", protocol)
    monkeypatch.setattr(study, "check", lambda base: protocol)
    monkeypatch.setattr(
        study,
        "actors",
        lambda: (
            [actor, actor],
            {"wp": actor, "point": actor},
            {"wp": predict, "point": predict},
        ),
    )
    collect = study.collect_case

    def fixed_collect(folder, case, config, parent, opponent, predictors, native):
        return collect(
            folder,
            case,
            config,
            parent,
            opponent,
            predictors,
            native,
            seat_draw=lambda n: case["index"] % 2,
            deck_draw=deck_tape(),
        )

    monkeypatch.setattr(study, "collect_case", fixed_collect)
    study.worker(tmp_path, 0)
    study.audit_chunk(tmp_path, 0)
    return tmp_path, protocol


def test_all_zero_sources_count_and_cannot_be_resampled_or_reaudited(
    complete_zero_chunk,
):
    base, protocol = complete_zero_chunk
    rows = study.chunk_records(base, 0, protocol)
    assert len(rows) == 4 and all(r["status"] == "missing_root" for r in rows)
    with pytest.raises(FileExistsError):
        study.worker(base, 0)
    with pytest.raises(FileExistsError):
        study.audit_chunk(base, 0)
    write(base / "queue.json", {"status": "complete"})
    result = study.summarize_study(base)
    assert result["sources"] == result["work"]["sources"]["completions"] == 4
    assert result["work"]["audit_source_replay"] == result["work"]["sources"]
    assert result["status_counts"] == {"missing_root": 4}
    assert not result["inference"]["criterion_met"]
    assert all(
        v["sources"] == 1 and v["mean"] == "0"
        for v in result["inference"]["populations"].values()
    )
    with pytest.raises(ValueError):
        study.summarize_study(base)
    exported = export(base)
    assert exported["status"] == "passed" and exported["sources"] == 4
    with pytest.raises(FileExistsError):
        export(base)


def test_export_recomputes_inference_from_complete_audited_rows(complete_zero_chunk):
    base, _ = complete_zero_chunk
    write(base / "queue.json", {"status": "complete"})
    result = study.summarize_study(base)
    result["inference"]["criterion_met"] = True
    write(base / "summary.json", result)
    with pytest.raises(ValueError, match="reduction differs"):
        export(base)
    assert not (base / "export.json").exists()


@pytest.mark.parametrize("change", ["artifact", "missing_audit", "incomplete_queue"])
def test_changed_or_incomplete_evidence_cannot_produce_inference(
    complete_zero_chunk, change
):
    base, _ = complete_zero_chunk
    write(base / "queue.json", {"status": "complete"})
    if change == "artifact":
        path = base / "chunk_0000" / "source_00000" / "seat.json"
        value = read(path)
        value["seat"] = 1 - value["seat"]
        write(path, value)
    elif change == "missing_audit":
        # Rename inside this test's temporary workspace; retain the evidence.
        (base / "chunk_0000" / "audit.json").rename(
            base / "chunk_0000" / "saved_audit.json"
        )
    else:
        write(base / "queue.json", {"status": "running"})
    with pytest.raises((ValueError, FileNotFoundError)):
        study.summarize_study(base)
    assert not (base / "summary.json").exists()
