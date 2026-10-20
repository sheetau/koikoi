import numpy as np
import pytest
import torch

from koikoi.env import BatchEnv
from koikoi.layout import PHASE, PLAYER, POINTS, ROUND, TERMINAL
from koikoi.reference import ReferenceEnv
from scripts.branch_policy_gradient import (
    BranchWorkspace,
    collect_roots,
    expected_return_loss,
    inverse_cdf,
    probabilities,
)


class PublicFixtureActor(torch.nn.Module):
    def forward(self, features, phases, legal):
        # Deliberately uses visible features, with nonconstant legal probabilities.
        scores = features[:, :3].sum(1) * 0.01
        return scores.masked_fill(~legal, -torch.inf)


def replay(initial, actions, final):
    decisions = rounds = completions = 0
    for lane, state in enumerate(initial.cpu().numpy()):
        game = ReferenceEnv()
        game.state = state.copy()
        for action in actions[:, lane].tolist():
            if game.state[PHASE] == TERMINAL:
                continue
            old_round = int(game.state[ROUND])
            assert game.legal()[action]
            game.step(action)
            decisions += 1
            rounds += int(game.state[ROUND]) - old_round
            # Terminal rounds finish without advancing to another round.
            rounds += int(game.state[PHASE] == TERMINAL)
            completions += int(game.state[PHASE] == TERMINAL)
        assert np.array_equal(game.state, final[lane].cpu().numpy())
    return {
        "decisions": decisions,
        "round_endings": rounds,
        "completions": completions,
    }


def test_variable_length_exact_gradient_and_detached_outcomes():
    # A two-stage game: stop yields 2; continue reaches a second decision,
    # whose outcomes are -1 and 4. The same policy parameter governs both.
    theta = torch.tensor(0.31, dtype=torch.float64, requires_grad=True)
    p = torch.stack([theta, -theta]).softmax(0)
    last = torch.tensor([-1.0, 4.0], dtype=torch.float64)
    objective = p[0] * 2 + p[1] * (p * last).sum()
    exact = torch.autograd.grad(objective, theta, retain_graph=True)[0]
    first = torch.stack([theta.new_tensor(2), (p * last).sum()])
    estimate = theta * 0
    # Enumerate all source paths and all equally likely reservoir roots.
    for path_prob, values in [
        (p[0].detach(), [first]),
        (p[1].detach() * p[0].detach(), [first, last]),
        (p[1].detach() * p[1].detach(), [first, last]),
    ]:
        k = len(values)
        for q in values:
            estimate -= (
                path_prob
                / k
                * expected_return_loss(p[None], q[None], theta.new_tensor([k]))
            )
    estimated = torch.autograd.grad(estimate, theta)[0]
    torch.testing.assert_close(exact, estimated, rtol=1e-12, atol=1e-12)

    logits = torch.tensor([[0.3, -0.2]], requires_grad=True)
    q = torch.tensor([[1.0, -1.0]], requires_grad=True)
    count = torch.tensor([3.0], requires_grad=True)
    expected_return_loss(logits.softmax(-1), q, count).backward()
    assert q.grad is None and count.grad is None
    assert logits.grad[0, 0] < 0 < logits.grad[0, 1]


def test_categorical_boundaries_and_validation():
    p = torch.tensor([[0.0, 0.25, 0.75, 0.0]]).repeat(4, 1)
    u = torch.tensor([0.0, 0.249, 0.25, 0.999999])
    assert inverse_cdf(p, u).tolist() == [1, 1, 2, 2]
    with pytest.raises(ValueError):
        inverse_cdf(p, torch.ones(4))
    with pytest.raises(ValueError):
        inverse_cdf(torch.zeros_like(p), u)


def test_conditional_variance_matches_scalar_computation():
    from scripts.diagnose_branch_policy_gradient import statistics

    p = np.array([[0.2, 0.8, 0.0], [0.1, 0.3, 0.6]], dtype=np.float64)
    q = np.array(
        [[[1, -1, 0], [0, 1, -1]], [[1, 1, 0], [-1, 0, 1]], [[-1, 1, 0], [1, -1, 0]]],
        dtype=np.float64,
    )
    counts = np.array([2, 7])
    actual = statistics(torch.tensor(p), torch.tensor(counts), torch.tensor(q))
    expected = []
    for row in range(2):
        coordinates = []
        for action in range(3):
            samples = []
            for rep in range(3):
                baseline = sum(p[row, a] * q[rep, row, a] for a in range(3))
                samples.append(
                    counts[row] * p[row, action] * (q[rep, row, action] - baseline)
                )
            mean = sum(samples) / 3
            coordinates.append(sum((x - mean) ** 2 for x in samples) / 2)
        expected.append(sum(coordinates))
    np.testing.assert_allclose(
        actual["conditional_logit_variance_per_root"].numpy(),
        expected,
        rtol=1e-14,
        atol=1e-14,
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_source_branches_reference_replay_and_reuse(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.set_num_threads(2)
    actor = PublicFixtureActor().to(device)
    env = BatchEnv(3, device, seed=110010001, debug=True)
    rng = lambda seed: torch.Generator(device=device).manual_seed(seed)
    data = collect_roots(env, actor, rng(13), rng(17))
    assert replay(data["initial"], data["actions"], data["final"]) == data["work"]
    assert (data["eligible_decisions"] > 0).all()
    # Reconstruct selected roots without using the reservoir implementation.
    for lane in range(3):
        game = ReferenceEnv()
        game.state = data["initial"][lane].cpu().numpy().copy()
        for action in data["actions"][: int(data["selected_step"][lane]), lane]:
            game.step(int(action))
        assert np.array_equal(game.state, data["states"][lane].cpu().numpy())
    workspace = BranchWorkspace(24, device)
    backend = workspace.env.backend
    original = data["states"].clone()
    for shared in (True, False):
        result = workspace.run(original, actor, rng(19), shared_actions=shared)
        roots = result["root_indices"]
        assert (
            replay(original[roots], result["actions"], result["final"])
            == result["work"]
        )
        expected = (result["final"][:, POINTS] - result["final"][:, POINTS + 1]) * (
            1 - 2 * original[roots, PLAYER]
        )
        assert torch.equal(expected, result["points"])
        assert torch.equal(
            result["q"][roots, result["first_actions"]], expected.sign().float()
        )
        assert torch.equal(original, data["states"])
        assert workspace.env.backend is backend
    # A different root count reuses the same native backend and overwrites it.
    small = workspace.run(original[:1], actor, rng(19))
    assert small["work"]["completions"] == int(small["legal"].sum())
    assert workspace.env.backend is backend
    with pytest.raises(ValueError, match="capacity"):
        workspace.run(original.repeat(25, 1), actor, rng(19))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_published_actor_backward_and_private_world_invariance(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    from koikoi.published_actor import PublishedActor
    from koikoi.search import sample_hidden_batch

    torch.set_num_threads(2)
    actor = PublishedActor.from_published("wp", device)
    game = ReferenceEnv(110020001)
    hidden = sample_hidden_batch(game, np.random.default_rng(29), 1)[0]
    states = torch.tensor(np.stack([game.state, hidden]), device=device)
    workspace = BranchWorkspace(16, device)
    features, phases, legal = workspace.root_inputs(states)
    torch.testing.assert_close(features[0], features[1], rtol=0, atol=0)
    probs = probabilities(actor, features, phases, legal)
    torch.testing.assert_close(probs[0], probs[1], rtol=0, atol=0)
    q = torch.linspace(-1, 1, 50, device=device).repeat(2, 1)
    loss = expected_return_loss(probs, q, torch.tensor([3, 5], device=device))
    loss.backward()
    gradients = [p.grad for p in actor.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert sum(float(g.square().sum()) for g in gradients) > 0
