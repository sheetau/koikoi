from copy import deepcopy
from dataclasses import replace
from fractions import Fraction

import pytest

from koikoi.agents import EndgameAgent
from koikoi.env import BatchEnv
from koikoi.reference import ReferenceEnv
from scripts import uniform_controller_study as study
from scripts.run_arbitration_study import read, write
from scripts.uniform_controller_design import POPULATIONS, specification
from scripts.uniform_deal_source import DeckSample
from tests.test_logged_public_source import Minimum, predict
from tests.test_symbolic_tail_assessment import prepared


def deck_tape():
    decks = iter(tuple(map(int, ReferenceEnv(71 + i).state[48:96])) for i in range(8))
    return lambda: DeckSample(next(decks), 1, 47)


@pytest.mark.parametrize("seat", [0, 1])
def test_durable_chance_ledger_collection_and_policy_audit(tmp_path, seat):
    folder, config = tmp_path / "source", specification()
    case = {"index": 0, "population": "0:wp", "replicate": 0}
    actor = EndgameAgent(Minimum())
    predictors = {"wp": predict, "point": predict}

    def draw():
        assert read(folder / "seat.json")["seat"] == seat
        return tape()

    tape = deck_tape()
    source = study.collect_case(
        folder,
        case,
        config,
        actor,
        actor,
        predictors,
        BatchEnv(1, debug=True),
        seat_draw=lambda n: seat,
        deck_draw=draw,
    )
    assert source["selected_root"] is None
    study.source_files(folder, case, source, config)
    checked = study.audit_source(source, predictors, actor, actor)
    assert checked["replay_work"] == source["work"]
    assert checked["fresh_external_chance_draws"] == 0
    with pytest.raises(FileExistsError):
        study.collect_case(folder, case, config, actor, actor, predictors, None)
    write(folder / "deal_08.json", {})
    with pytest.raises(ValueError, match="Extra"):
        study.source_files(folder, case, source, config)


def test_partial_sampling_failure_is_preserved_and_cannot_resample(tmp_path):
    folder = tmp_path / "partial"
    case = {"index": 3, "population": "1:point", "replicate": 0}

    def fail():
        raise RuntimeError("fixture entropy failure")

    with pytest.raises(RuntimeError, match="entropy"):
        study.collect_case(
            folder,
            case,
            specification(),
            None,
            None,
            {},
            None,
            seat_draw=lambda n: 1,
            deck_draw=fail,
        )
    assert read(folder / "seat.json") == {"case": case, "seat": 1, "bounded_draws": 1}
    assert not (folder / "record.json").exists()
    with pytest.raises(FileExistsError):
        study.collect_case(folder, case, specification(), None, None, {}, None)


@pytest.mark.parametrize("swap", [False, True])
def test_source_planning_independent_true_parent_oracle_and_rebuilt_posterior(swap):
    actor, _, source, _, _, _ = prepared(swap)
    source.update(parent=0, opponent="fixture")
    source["selected_root"]["public_identification"] = {"model": "fixture"}
    config = {**specification(), "phase_decisions": 1258}
    protocol = {"design": config, "opponent_identities": {"fixture": "fixture"}}
    row, evidence = study.plan_source(source, [actor], {"fixture": actor}, protocol)
    assert row["selection"]["gain"] == "53/550"
    assert row["oracle"]["parent"]["value"] == "204/275"
    assert row["oracle"]["selected"]["value"] == "461/550"
    selected, counts = study.audit_plan(
        row, evidence, source, {"fixture": actor}, protocol
    )
    assert counts == row["history_counts"]
    assert selected.gain == Fraction(53, 550)
    result = study.assess_pair(source, selected, actor, actor, study.PARENTS[0][1])
    assert result["actual_work"]["completions"] == 2
    altered = deepcopy(evidence)
    altered["worlds"].weights[0] += 1
    with pytest.raises(AssertionError):
        study.audit_plan(row, altered, source, {"fixture": actor}, protocol)
    altered = deepcopy(evidence)
    altered["selection"] = replace(selected, selected_q=Fraction(1))
    with pytest.raises(ValueError, match="selection"):
        study.audit_plan(row, altered, source, {"fixture": actor}, protocol)
    altered_row = deepcopy(row)
    altered_row["oracle"]["parent"]["value"] = "1"
    with pytest.raises(ValueError, match="receipt"):
        study.audit_plan(altered_row, evidence, source, {"fixture": actor}, protocol)
    altered_row = deepcopy(row)
    altered_row["phases"]["fixed"]["work"]["decisions"] += 1
    with pytest.raises(ValueError, match="work"):
        study.audit_plan(altered_row, evidence, source, {"fixture": actor}, protocol)


def test_missing_roots_and_public_support_refusal_are_explicit():
    assert study.plan_source(
        {"selected_root": None}, [], {}, {"design": specification()}
    ) == ({"status": "missing_root"}, None)
    actor, _, source, _, _, _ = prepared(cap=1)
    source.update(parent=0, opponent="fixture")
    source["selected_root"]["public_identification"] = {"model": "fixture"}
    protocol = {"design": {**specification(), "max_hands": 1}}
    row, evidence = study.plan_source(source, [actor], {"fixture": actor}, protocol)
    assert row["status"] == "support_refused"
    selected, counts = study.audit_plan(
        row, evidence, source, {"fixture": actor}, protocol
    )
    assert selected is None and counts == {}


def test_full_fixed_tape_worker_audit_and_complete_only_statistical_export(
    tmp_path, monkeypatch
):
    # Four tiny fixture populations, not the production design or strength samples.
    config = {
        **specification(),
        "source_matches": 4,
        "sources_per_population": 1,
        "chunk_size": 4,
    }
    cases = [
        {"index": i, "population": p, "replicate": 0} for i, p in enumerate(POPULATIONS)
    ]
    protocol = {"design": config, "cases": cases}
    write(tmp_path / "protocol.json", protocol)
    monkeypatch.setattr(study, "check", lambda _: protocol)

    def actors():
        actor = EndgameAgent(Minimum())
        return (
            [actor, actor],
            {"wp": actor, "point": actor},
            {"wp": predict, "point": predict},
        )

    monkeypatch.setattr(study, "actors", actors)
    original_collect = study.collect_case

    def collect(folder, case, *args):
        return original_collect(
            folder,
            case,
            *args,
            seat_draw=lambda n: case["index"] % 2,
            deck_draw=deck_tape(),
        )

    monkeypatch.setattr(study, "collect_case", collect)
    study.worker(tmp_path, 0)
    assert len(study.chunk_records(tmp_path, 0, protocol)) == 4
    study.audit_chunk(tmp_path, 0)
    audit_path = tmp_path / "chunk_0000/audit.json"
    audited = read(audit_path)
    assert audited["work"]["source_replay"]["completions"] == 4
    assert audited["fresh_external_chance_draws"] == 0
    assert all(x["gain"] == "0" for x in audited["gains"])
    with pytest.raises(ValueError, match="already exists"):
        study.audit_chunk(tmp_path, 0)
    with pytest.raises(FileExistsError):
        study.worker(tmp_path, 0)
    write(tmp_path / "queue.json", {"status": "running"})
    with pytest.raises(ValueError, match="Complete queue"):
        study.summarize_study(tmp_path)
    write(tmp_path / "queue.json", {"status": "complete"})
    altered = deepcopy(audited)
    altered["gains"].pop()
    write(audit_path, altered)
    with pytest.raises(ValueError, match="denominator"):
        study.summarize_study(tmp_path)
    write(audit_path, audited)
    result = study.summarize_study(tmp_path)
    assert not result["criterion_met"]
    summary = read(tmp_path / "summary.json")
    assert summary["sources"] == 4
    assert summary["status_counts"] == {"missing_root": 4}
    assert summary["work"]["sources"]["completions"] == 4
    assert summary["work"]["audit_source_replay"]["completions"] == 4
    assert set(summary["actual_mean_difference_descriptive"].values()) == {"0"}
    study.export_report(tmp_path)
    exported = read(tmp_path / "export.json")
    assert exported["status"] == "passed" and not exported["criterion_met"]
    assert exported["source_matches"] == 4
    report = (tmp_path / "results.rst").read_text(encoding="utf-8")
    assert "Joint prespecified criterion: FAILED" in report
    assert "descriptive only" in report
    with pytest.raises(ValueError, match="Export already"):
        study.export_report(tmp_path)
    records_path = tmp_path / "chunk_0000/records.json"
    records = read(records_path)
    write(records_path, records[::-1])
    with pytest.raises(ValueError, match="changed chunk"):
        study.chunk_records(tmp_path, 0, protocol)


def test_frozen_design_and_dependency_mutations_refuse_to_run(tmp_path, monkeypatch):
    dependency = tmp_path / "dependency.txt"
    dependency.write_text("fixture", encoding="utf-8")
    monkeypatch.setattr(
        study, "dependencies", lambda: {str(dependency): study.sha(dependency)}
    )
    runtime = {"source_sha256": "fixture", "torch": "fixture", "numpy": "fixture"}
    monkeypatch.setattr(study, "provenance", lambda: runtime)
    base = tmp_path / "study"
    protocol = study.freeze(base)
    assert study.check(base) == protocol
    with pytest.raises(ValueError, match="fresh"):
        study.freeze(base)
    changed = deepcopy(protocol)
    changed["design"]["source_matches"] -= 1
    write(base / "protocol.json", changed)
    with pytest.raises(ValueError, match="fixed design"):
        study.check(base)
    write(base / "protocol.json", protocol)
    dependency.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="dependency"):
        study.check(base)
