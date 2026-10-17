import numpy as np
import pytest
import torch

from koikoi.arbitration import ExpertBank
from koikoi.env import BatchEnv
from koikoi.layout import PHASE
from scripts.benchmark_final_latency import cached
from scripts.cache_paired_expert_features import expert_features, replay
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
    # Changing unrelated hidden simulation state cannot reach this pure API:
    # it accepts only the public encoder input, phase, mask and proposal IDs.
