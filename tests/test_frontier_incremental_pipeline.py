"""Fixed historical tapes validate the pipeline; none are fresh strength samples."""

from copy import deepcopy
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest

from koikoi.env import BatchEnv
from scripts import frontier_incremental_assessment as assessment
from scripts.check_frontier_hybrid_integration import compare_patch
from scripts.frontier_incremental_design import specification
from scripts.frontier_incremental_study import oracle_work
from scripts.logged_public_source import audit as audit_source
from scripts.prospective_incumbent import compressed_save
from scripts.run_arbitration_study import PARENTS, read
from scripts.uniform_controller_study import actors, collect_case, load, source_files
from scripts.uniform_deal_source import DeckSample


@pytest.fixture(scope="module")
def pinned():
    protocol = read("runs/selected_frontier_integration/protocol.json")
    original = read("runs/uniform_controller_evaluation/protocol.json")
    return (
        protocol,
        {
            "design": specification(),
            "opponent_identities": original["opponent_identities"],
        },
        actors(),
    )


@pytest.mark.parametrize("index", [0, 3, 9])
def test_logged_source_selected_planning_actual_execution_and_independent_increment(
    pinned, tmp_path, index
):
    fixtures, protocol, (parents, opponents, predictors) = pinned
    fixture = fixtures["fixtures"][index]
    old = read(fixture["source"])
    parent, opponent = parents[old["parent"]], opponents[old["opponent"]]
    tape = iter(old["deals"])

    def draw():
        deal = next(tape)
        return DeckSample(**{k: deal[k] for k in ("deck", "attempts", "bounded_draws")})

    source = collect_case(
        tmp_path / "source",
        fixture["case"],
        protocol["design"],
        parent,
        opponent,
        predictors,
        BatchEnv(1, debug=True),
        seat_draw=lambda n: old["seat"],
        deck_draw=draw,
    )
    assert source == old
    source_files(tmp_path / "source", fixture["case"], source, protocol["design"])
    checked = audit_source(source, predictors, parent, opponent)
    assert checked["fresh_external_chance_draws"] == 0
    row, evidence = assessment.plan_source(source, parents, opponents, protocol)
    # Freeze the complete program before independent value/actual evaluation.
    compressed_save(tmp_path / "program.pkl.gz", evidence)
    evidence = load(tmp_path / "program.pkl.gz")
    expected = load(
        Path("runs/selected_frontier_integration")
        / f"fixture_{index:02}"
        / "program.pkl.gz"
    )
    hybrid = evidence["hybrid"]
    assert (hybrid.fixed, hybrid.optimized, hybrid.original) == (
        expected.fixed,
        expected.optimized,
        expected.original,
    )
    if expected.patch is not None:
        compare_patch(hybrid.patch, expected.patch)
    else:
        assert hybrid.patch is None
    verified, counts = assessment.audit_plan(
        row, evidence, source, parent, opponents, protocol
    )
    assert counts == row["history_counts"]
    actual = assessment.assess_pair(
        source, hybrid, parent, opponent, PARENTS[source["parent"]][1]
    )
    assert actual["actual_work"] == actual["replay_work"]
    if index == 3:
        assert Fraction(verified["exact_increment"]) == Fraction(1, 5265)
        assert oracle_work(verified["oracle"])["local_physical"]["decisions"] > 0
    else:
        assert verified["exact_increment"] == "0"
    if index == 9:
        assert hybrid.original.gain > 0
        assert verified == {
            "exact_increment": "0",
            "reason": "identical_programs",
            "oracle": None,
        }
        assert actual["outcomes"]["old"] == actual["outcomes"]["hybrid"]


@pytest.mark.parametrize("change", ["work", "binding", "order", "candidate"])
def test_structural_audit_rejects_changed_program_or_accounting(pinned, change):
    fixtures, protocol, _ = pinned
    hybrid = deepcopy(
        load("runs/selected_frontier_integration/fixture_03/program.pkl.gz")
    )
    worlds = load(fixtures["fixtures"][3]["program"])["worlds"]
    binding = hybrid.fixed.binding
    if change == "work":
        hybrid.decisions += 1
    elif change == "binding":
        hybrid.fixed = replace(hybrid.fixed, binding=replace(binding, digest="changed"))
    elif change == "order":
        hybrid.patch.comparisons = tuple(reversed(hybrid.patch.comparisons))
    else:
        local = next(
            r
            for _, r in hybrid.patch.comparisons
            if r["selection"].candidate is not None
        )
        local["candidate_certificate"] = None
    with pytest.raises(ValueError):
        assessment.validate_program(
            hybrid,
            worlds,
            binding.parent_identity,
            dict(binding.opponent_identities),
            protocol["design"],
        )


def test_missing_root_is_zero_without_planning():
    row, evidence = assessment.plan_source({"selected_root": None}, None, None, None)
    assert (
        row == {"status": "missing_root", "increment_lower": "0"} and evidence is None
    )
    assert all(not w for w in oracle_work(None).values())


def test_actual_execution_rejects_a_different_public_root(pinned):
    fixtures, _, (parents, opponents, _) = pinned
    source = deepcopy(read(fixtures["fixtures"][9]["source"]))
    source["seat"] = 1 - source["seat"]
    hybrid = load("runs/selected_frontier_integration/fixture_09/program.pkl.gz")
    with pytest.raises(ValueError, match="Actual root differs"):
        assessment.assess_pair(
            source,
            hybrid,
            parents[source["parent"]],
            opponents[source["opponent"]],
            PARENTS[source["parent"]][1],
        )
