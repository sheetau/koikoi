"""Replay every diagnostic trajectory on CPU and recompute its statistics."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from koikoi.env import BatchEnv
from koikoi.layout import PHASE, PLAYER, POINTS, TERMINAL
from koikoi.reference import ReferenceEnv
from koikoi.train import provenance


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def audit(base):
    base = Path(base)
    output = base / "audit.json"
    if output.exists():
        raise ValueError("Audit already exists")
    plan, summary = read(base / "protocol.json"), read(base / "summary.json")
    assert summary["status"] == "complete"
    assert summary["protocol_sha256"] == sha(base / "protocol.json")
    assert plan["source"]["source_sha256"] == provenance()["source_sha256"]
    bindings = {
        str(base / "protocol.json"): sha(base / "protocol.json"),
        str(base / "summary.json"): sha(base / "summary.json"),
        __file__: sha(__file__),
        **plan["bindings"],
        **{str(base / name): value for name, value in summary["artifacts"].items()},
        **{p["path"]: p["sha256"] for p in plan["parents"]},
    }
    for name, digest in bindings.items():
        assert sha(name) == digest, name
    env = BatchEnv(512, "cpu", debug=True)
    storage = env.state
    replay_work = dict.fromkeys(("decisions", "round_endings", "completions"), 0)

    def replay(initial, data, source=False):
        env.count = len(initial)
        env.state = storage[: len(initial)]
        env.state.copy_(initial)
        work = dict.fromkeys(replay_work, 0)
        eligible_counts = torch.zeros(len(initial), dtype=torch.long)
        for step, actions in enumerate(data["actions"].long()):
            active = env.state[:, PHASE] != TERMINAL
            _, legal = env.observe()
            solved = env.solved_actions()
            legal &= (solved[:, None] < 0) | (torch.arange(50)[None] == solved[:, None])
            assert legal[active].gather(1, actions[active, None]).all()
            if source:
                chosen = data["selected_step"] == step
                assert torch.equal(env.state[chosen], data["states"][chosen])
                eligible_counts += active & (legal.sum(-1) > 1)
            result = env.step(actions, freeze_done=True)
            work["decisions"] += int(active.sum())
            work["round_endings"] += int(result[:, 1].sum())
            work["completions"] += int(result[:, 2].sum())
        assert torch.equal(env.state, data["final"])
        assert work == data["work"]
        if source:
            assert torch.equal(eligible_counts, data["eligible_decisions"])
        for key in work:
            replay_work[key] += work[key]

    for index, row in enumerate(summary["parents"]):
        directory = base / f"parent{index + 1}"
        source, policy = (
            load(directory / "source.pt"),
            load(directory / "root_policy.pt"),
        )
        for lane in range(64):
            expected = ReferenceEnv(plan["source_seeds"][index] + lane).state
            assert np.array_equal(expected, source["initial"][lane].numpy())
        replay(source["initial"], source, source=True)
        assert row["source_work"] == source["work"]
        assert row["eligible_decisions"] == int(source["eligible_decisions"].sum())
        assert row["zero_weight_roots"] == int(
            (source["eligible_decisions"] == 0).sum()
        )
        p = policy["probabilities"].double().numpy()
        np.testing.assert_allclose(p.sum(-1), 1, rtol=1e-6, atol=1e-7)
        assert np.isfinite(p).all() and (p >= 0).all()
        assert not p[~policy["legal"].numpy()].any()
        k = source["eligible_decisions"].numpy()
        assert policy["phases"].tolist() == source["states"][:, PHASE].tolist()
        np.testing.assert_array_equal(
            np.bincount(policy["phases"].numpy(), minlength=4), row["phase_counts"]
        )
        variances = {}
        for mode in plan["modes"]:
            outcomes, work = [], dict.fromkeys(replay_work, 0)
            for replicate in range(plan["replicates_per_mode"]):
                branch = load(directory / f"{mode}_{replicate}.pt")
                roots, actions = torch.where(policy["legal"])
                assert torch.equal(roots, branch["root_indices"])
                assert torch.equal(actions, branch["first_actions"])
                assert torch.equal(actions, branch["actions"][0].long())
                replay(source["states"][roots], branch)
                points = (
                    branch["final"][:, POINTS] - branch["final"][:, POINTS + 1]
                ) * (1 - 2 * source["states"][roots, PLAYER])
                assert torch.equal(points, branch["points"])
                expected_q = torch.zeros_like(branch["q"])
                expected_q[roots, actions] = points.sign().float()
                assert torch.equal(expected_q, branch["q"])
                outcomes.append(expected_q.numpy())
                for key in work:
                    work[key] += branch["work"][key]
            assert work == row["modes"][mode]["branch_work"]
            q = np.stack(outcomes).astype(np.float64)
            gradient = p * (q - np.sum(p * q, axis=-1, keepdims=True))
            g = gradient * k[None, :, None]
            variance = np.var(g, axis=0, ddof=1).sum(-1)
            unweighted = np.var(gradient, axis=0, ddof=1).sum(-1)
            saved = load(directory / f"{mode}_statistics.pt")
            np.testing.assert_allclose(
                variance,
                saved["conditional_logit_variance_per_root"].numpy(),
                atol=1e-10,
                rtol=1e-12,
            )
            np.testing.assert_allclose(
                unweighted,
                saved["unweighted_conditional_logit_variance_per_root"].numpy(),
                atol=1e-12,
                rtol=1e-12,
            )
            np.testing.assert_allclose(
                variance.mean(),
                row["modes"][mode]["mean_conditional_logit_variance"],
                atol=1e-10,
                rtol=1e-12,
            )
            np.testing.assert_allclose(
                np.square(g.mean(0)).sum(-1).mean(),
                row["modes"][mode]["mean_gradient_squared_norm"],
                atol=1e-10,
                rtol=1e-12,
            )
            differing = np.max(np.where(p > 0, q, -2), axis=-1) != np.min(
                np.where(p > 0, q, 2), axis=-1
            )
            assert (
                int(differing.any(0).sum())
                == row["modes"][mode]["roots_with_any_outcome_difference"]
            )
            assert (
                float(differing.mean())
                == row["modes"][mode]["replicate_root_outcome_difference_fraction"]
            )
            variances[mode] = float(variance.mean())
        ratio = variances["shared_actions"] / variances["independent_actions"]
        np.testing.assert_allclose(
            ratio,
            row["conditional_variance_ratio_shared_over_independent"],
            atol=1e-12,
            rtol=1e-12,
        )
        print(f"Verified parent {index + 1}: variance ratio {ratio:.6f}", flush=True)
    result = {
        "status": "passed",
        "bindings": bindings,
        "replayed_work": replay_work,
        "scope": "All recorded actions, initial deals, roots, outcomes and variance statistics; no neural inference replay or new independent matches",
    }
    output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps({"status": "passed", "replayed_work": replay_work}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="runs/branch_policy_gradient_diagnostic")
    audit(parser.parse_args().input)
