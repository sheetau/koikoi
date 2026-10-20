import copy
import json

import pytest
import torch

from koikoi.published_train import PublishedConfig, PublishedTrainer
from koikoi.upstream import UPSTREAM_PATH
from scripts.collect_score_auxiliary import IncompleteScoreCollection
from scripts.score_auxiliary_trainer import AuxiliaryConfig, ScoreAuxiliaryTrainer

pytestmark = [
    pytest.mark.upstream,
    pytest.mark.skipif(not UPSTREAM_PATH.exists(), reason="Upstream not fetched"),
]


def config(device="cpu", **extra):
    return PublishedConfig(
        **(
            {
                "device": device,
                "initial_variant": "point",
                "envs": 2,
                "horizon": 16,
                "minibatch": 32,
                "epochs": 1,
                "critic_width": 32,
                "critic_warmup_updates": 1,
                "seed": 105050001,
            }
            | extra
        )
    )


def auxiliary(kind, **extra):
    return AuxiliaryConfig(
        **(
            {
                "kind": kind,
                "interval": 1,
                "matches": 2,
                "epochs": 2,
                "minibatch": 16,
                "deal_seed": 105060001,
                "action_seed": 105070001,
                "reservoir_seed": 105080001,
                "order_seed": 105090001,
            }
            | extra
        )
    )


def equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            equal(a, b)
    else:
        assert left == right


def snapshot(ppo):
    return copy.deepcopy(
        {
            "model": ppo.model.state_dict(),
            "optimizer": ppo.optimizer.state_dict(),
            "env": ppo.env.state,
            "counters": ppo.counters,
            "rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all(),
        }
    )


def test_no_auxiliary_control_exactly_preserves_original_ppo(tmp_path, record_property):
    original = PublishedTrainer(config())
    initial = tmp_path / "initial.pt"
    original.save(initial)
    expected = [original.update() for _ in range(3)]
    baseline = snapshot(original)
    control = ScoreAuxiliaryTrainer(
        PublishedTrainer.restore(initial), auxiliary("none")
    )
    for item in expected:
        actual = control.update()
        assert {k: v for k, v in actual.items() if k != "auxiliary_cycles"} == item
    equal(snapshot(control.ppo), baseline)
    assert control.counters["cycles"] == 2
    assert control.counters["matches"] == 4
    assert control.counters["optimizer_steps"] == 0
    assert not control.optimizer.state
    record_property(
        "ppo_executed_work",
        json.dumps(
            {k: 2 * original.counters[k] for k in ("decisions", "rounds", "matches")}
        ),
    )
    record_property("auxiliary_executed_work", json.dumps(control.counters))


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)]
)
@pytest.mark.parametrize("kind", ["win", "score"])
def test_full_auxiliary_cycle_and_exact_resume(device, kind, tmp_path, record_property):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    trainer = ScoreAuxiliaryTrainer(PublishedTrainer(config(device)), auxiliary(kind))
    before = copy.deepcopy(trainer.auxiliary.heads.state_dict())
    trainer.update()
    assert trainer.counters["cycles"] == 0
    trainer.update()
    assert trainer.counters["cycles"] == 1
    assert trainer.counters["optimizer_steps"] >= 2
    assert any(
        not torch.equal(before[k], v)
        for k, v in trainer.auxiliary.heads.state_dict().items()
    )
    # The base PPO optimizer never owns the auxiliary heads or auxiliary Adam state.
    head_ids = {id(p) for p in trainer.auxiliary.heads.parameters()}
    assert not head_ids.intersection(
        id(p) for group in trainer.ppo.optimizer.param_groups for p in group["params"]
    )
    first = copy.deepcopy(trainer.counters)
    prefix_ppo = trainer.ppo.counters.copy()
    checkpoint = tmp_path / "cycle1"
    trainer.save(checkpoint)
    expected_metrics = trainer.update()
    expected = snapshot(trainer.ppo)
    auxiliary_state = copy.deepcopy(trainer.auxiliary.heads.state_dict())
    optimizer_state = copy.deepcopy(trainer.optimizer.state_dict())
    totals = copy.deepcopy(trainer.counters)
    last = copy.deepcopy(trainer.last_collection)
    restored = ScoreAuxiliaryTrainer.restore(checkpoint)
    assert restored.update() == expected_metrics
    equal(snapshot(restored.ppo), expected)
    equal(restored.auxiliary.heads.state_dict(), auxiliary_state)
    equal(restored.optimizer.state_dict(), optimizer_state)
    equal(restored.counters, totals)
    equal(restored.last_collection, last)
    equal(restored.last_metrics, trainer.last_metrics)
    for phase, name in ((0, "discard"), (1, "pick"), (3, "koi")):
        count = totals["optimizer_steps_" + name]
        parameters = list(
            trainer.auxiliary.actor.models[str(phase)].parameters()
        ) + list(trainer.auxiliary.heads[str(phase)].parameters())
        assert all(int(trainer.optimizer.state[p]["step"]) == count for p in parameters)
    with pytest.raises(FileExistsError):
        restored.save(checkpoint)
    manifest_path = checkpoint / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["auxiliary.pt"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="digest"):
        ScoreAuxiliaryTrainer.restore(checkpoint)
    record_property(
        "ppo_executed_work",
        json.dumps(
            {
                k: 2 * trainer.ppo.counters[k] - prefix_ppo[k]
                for k in ("decisions", "rounds", "matches")
            }
        ),
    )
    record_property(
        "auxiliary_executed_work",
        json.dumps({k: 2 * totals[k] - first[k] for k in totals}),
    )


def test_failed_collection_cannot_save_or_continue_a_partial_cycle(
    tmp_path, record_property
):
    trainer = ScoreAuxiliaryTrainer(
        PublishedTrainer(config(critic_warmup_updates=0)),
        auxiliary("score", max_steps=1),
    )
    with pytest.raises(IncompleteScoreCollection) as error:
        trainer.update()
    assert error.value.evidence["work"] == {"decisions": 2, "rounds": 0, "matches": 0}
    with pytest.raises(RuntimeError, match="Incomplete"):
        trainer.save(tmp_path / "invalid")
    with pytest.raises(RuntimeError, match="Incomplete"):
        trainer.update()
    assert not (tmp_path / "invalid").exists()
    record_property(
        "ppo_executed_work",
        json.dumps(
            {k: trainer.ppo.counters[k] for k in ("decisions", "rounds", "matches")}
        ),
    )
    record_property("auxiliary_executed_work", json.dumps(error.value.evidence["work"]))
