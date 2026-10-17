import numpy as np
import pytest
import torch

from koikoi.arbitration import ExpertBank
from koikoi.env import BatchEnv
from koikoi.layout import PHASE
from scripts.benchmark_final_latency import cached
from scripts.cache_paired_expert_features import expert_features, replay
from scripts.feature_score_selector import (
    FeatureScoreDistributionAgent,
    FeatureScoreSelector,
    observe_features,
    pack_features,
)
from scripts.run_arbitration_study import PARENTS
from scripts.train_paired_arbitration import collect, setup


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)]
)
def test_exact_root_replay_and_reject_misalignment(tmp_path, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    dataset = tmp_path / "targets.pt"
    data = collect(
        PARENTS[0][0], dataset, seed=18961001, roots=32, batch=16, device=device
    )
    output = tmp_path / "features.pt"
    result = replay(dataset, output)
    assert result["embeddings"].shape == (32, 2, 2, 256)
    assert result["scores"].shape == (32, 2, 2)
    assert torch.isfinite(result["embeddings"]).all()
    assert torch.isfinite(result["scores"]).all()
    assert result["public_dealer"].unique().tolist() == [0, 1]
    assert result["new_source_replay_work"] == data["source_work"]
    assert result["new_branch_decisions"] == 0
    assert result["reused_branch_work"] == data["branch_work"]
    assert "outcomes" not in result and "states" not in result
    loaded = torch.load(output, weights_only=False)
    torch.testing.assert_close(
        loaded["embeddings"], result["embeddings"], rtol=0, atol=0
    )
    with pytest.raises(ValueError, match="fresh"):
        replay(dataset, output)
    data["public_observations"][0, 0] += 0.25
    corrupted = tmp_path / "corrupted.pt"
    torch.save(data, corrupted)
    with pytest.raises(ValueError, match="public inputs differ"):
        replay(corrupted, tmp_path / "rejected.pt")
    assert not (tmp_path / "rejected.pt").exists()
    if device == "cpu":
        from koikoi.reference import ReferenceEnv
        from scripts.train_score_distribution import fit

        initial_bodies, order_seeds = set(), set()
        for kind in ("categorical", "mean", "win", "paired_win", "paired_mean"):
            target = tmp_path / f"{kind}.pt"
            trained = fit(dataset, target, kind=kind, epochs=2, feature_cache=output)
            initial_bodies.add(trained["training_config"]["initial_shared_body_sha256"])
            order_seeds.add(trained["training_config"]["minibatch_order_seed"])
            assert trained["feature_cache"]["source_replay_work"] == data["source_work"]
            assert trained["counters"]["new_simulation_decisions"] == 0
            assert all(
                np.isfinite(row["validation_loss"]) for row in trained["history"]
            )
            agent = FeatureScoreDistributionAgent(target)
            game = ReferenceEnv(18963001, "paper8")
            assert game.legal()[agent.act(game)]
            from scripts.diagnose_score_targets import diagnose
            from scripts.paired_rollout_targets import match_split

            report = diagnose(dataset, target, tmp_path / f"{kind}_diagnostic.json")
            valid = match_split(data["match_groups"])
            assert report["sections"]["all"]["roots"] == int(valid.sum())
            original = torch.load(dataset, weights_only=False)
            inputs = pack_features(
                original["public_observations"][valid],
                result["embeddings"][valid],
                result["scores"][valid],
            )
            proposals = inputs[:, 560:562].long()
            mask = torch.zeros(len(inputs), 50, dtype=torch.bool).scatter_(
                1, proposals, True
            )
            chosen = agent.model(inputs, mask)[0].argmax(-1) == proposals[:, 1]
            outcomes = data["outcomes"][valid]
            gain = (
                torch.where(chosen, outcomes[:, 1], outcomes[:, 0]) - outcomes[:, 0]
            ) / 2
            assert report["sections"]["all"]["single_action_comparisons"]["parent"][
                "win_score_gain"
            ] == pytest.approx(float(gain.mean()))
        assert len(initial_bodies) == len(order_seeds) == 1
        bad_cache = torch.load(output, weights_only=False)
        bad_cache["dataset_sha256"] = "0" * 64
        wrong_path = tmp_path / "wrong_cache.pt"
        torch.save(bad_cache, wrong_path)
        with pytest.raises(ValueError, match="does not match"):
            fit(dataset, tmp_path / "rejected_fit.pt", feature_cache=wrong_path)


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)]
)
def test_features_match_original_heads_all_phases(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    setup(18962001)
    data = torch.load(PARENTS[0][0], map_location="cpu", weights_only=False)
    from koikoi.published_actor import PublishedActor

    bank = ExpertBank(
        [data["actor"], PublishedActor.from_published("point", "cpu").checkpoint()],
        device,
    )
    corpus = cached()["states"]
    ids = np.concatenate([np.flatnonzero(corpus[:, PHASE] == p)[:8] for p in range(4)])
    states = torch.from_numpy(corpus[ids]).to(device)
    env = BatchEnv(len(states), device, "paper8")
    env.state.copy_(states)
    public, _ = bank.observe(env)
    features, legal = env.published_observe()
    proposals = public[:, 560:].long()
    encoded, scores = expert_features(
        bank, features, states[:, PHASE], legal, proposals
    )
    packed, mask = observe_features(bank, env)
    torch.testing.assert_close(
        packed, pack_features(public, encoded, scores), rtol=0, atol=0
    )
    assert mask.gather(1, proposals).all()
    with torch.no_grad():
        for expert, actor in enumerate(bank.experts):
            original = actor(features, states[:, PHASE], legal).gather(1, proposals)
            torch.testing.assert_close(scores[:, :, expert], original, rtol=0, atol=0)
        for phase in (0, 1, 2, 3):
            active = states[:, PHASE] == phase
            model_phase = 1 if phase == 2 else phase
            for expert, actor in enumerate(bank.experts):
                head = actor.models[str(model_phase)].out
                # Independent reconstruction from exported latent vectors.
                actual = torch.nn.functional.linear(
                    encoded[active, :, expert], head.weight[:, :, 0], head.bias
                ).squeeze(-1)
                torch.testing.assert_close(
                    actual, scores[active, :, expert], rtol=1e-5, atol=2e-6
                )
    from koikoi.reference import ReferenceEnv
    from koikoi.search import sample_hidden_batch

    game = ReferenceEnv(18964001, "paper8")
    variant = sample_hidden_batch(game, np.random.default_rng(11), 1)[0]
    hidden = BatchEnv(2, device, "paper8")
    hidden.state.copy_(torch.tensor(np.stack((game.state, variant)), device=device))
    identical, masks = observe_features(bank, hidden)
    torch.testing.assert_close(identical[0], identical[1], rtol=0, atol=0)
    assert torch.equal(masks[0], masks[1])


@pytest.mark.parametrize("kind", ["categorical", "mean", "win"])
def test_feature_scorer_is_proposal_order_invariant(kind):
    torch.manual_seed(18965001)
    public = torch.zeros(3, 562)
    public[:, 560:] = torch.tensor([3, 9])
    embeddings = torch.randn(3, 2, 2, 256)
    scores = torch.randn(3, 2, 2)
    public[1, 561] = public[1, 560]
    embeddings[1, 1] = embeddings[1, 0]
    scores[1, 1] = scores[1, 0]
    packed = pack_features(public, embeddings, scores)
    mask = torch.zeros(3, 50, dtype=torch.bool).scatter_(
        1, public[:, 560:].long(), True
    )
    model = FeatureScoreSelector(kind, width=16)
    torch.nn.init.normal_(model.head.weight)
    before = model(packed, mask)[0]
    swapped = public.clone()
    swapped[:, 560:] = public[:, 560:].flip(-1)
    after = model(pack_features(swapped, embeddings.flip(1), scores.flip(1)), mask)[0]
    torch.testing.assert_close(before, after, rtol=0, atol=0)
    assert model.values(packed)[1, 0] == model.values(packed)[1, 1]
    assert mask.gather(1, before.argmax(-1)[:, None]).all()


def test_score_decision_summary_scaling():
    from scripts.diagnose_score_targets import decision_summary

    values = decision_summary(
        np.array([True, False, True, False]),
        np.array([[-1, 1], [1, -1], [0, 1], [-1, 0]]),
        np.array([[-2, 4], [6, -8], [0, 2], [-4, 0]]),
        np.array([[1, 1], [1, 1], [1, 2], [1, 3]]),
    )
    assert values["parent"]["win_score_gain"] == 0.375
    assert values["point"]["win_score_gain"] == 0.125
    assert values["parent"]["point_margin_gain"] == 2
    assert values["point"]["point_margin_gain"] == 2.5
