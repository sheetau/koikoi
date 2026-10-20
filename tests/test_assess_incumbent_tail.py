import torch

from koikoi.agents import EndgameAgent, NeuralAgent
from koikoi.upstream import PublishedAgent
from scripts.assess_incumbent_tail import assess
from scripts.export_belief_tail import read
from scripts.run_arbitration_study import PARENTS
from scripts.train_branch_policy_gradient import sha


def test_assessment_helper_reproduces_frozen_adopted_source_zero():
    torch.set_num_threads(2)
    base = "runs/incumbent_saved_execution"
    complete = read(f"{base}/complete.json")
    assert complete["selections_sha256"] == sha(f"{base}/selections_complete.json")
    assert complete["assessments_sha256"] == sha(f"{base}/assessments.json")
    assert read(f"{base}/selections_complete.json")["artifact_sha256"] == sha(
        f"{base}/selections.pt"
    )
    selection = torch.load(
        f"{base}/selections.pt", map_location="cpu", weights_only=False
    )[0]
    source = read("runs/earlier_belief_tail_capability/sources.json")[0]
    parent_path, parent_id = PARENTS[source["parent"]]
    assert sha(parent_path) == parent_id
    parent = NeuralAgent(parent_path, "cpu", deterministic=True)
    opponent = EndgameAgent(PublishedAgent(source["opponent"]))
    outcomes, physical, replay, calls = assess(
        source, selection, parent, opponent, parent_id
    )
    assert outcomes == read(f"{base}/assessments.json")["0"]
    assert physical == replay
    assert physical["completions"] == 2
    assert sum(calls.values()) == physical["decisions"]
    assert selection.gain > 0 and selection.mode == "fixed"
