"""Saved-data analysis of branch-root signal and cumulative policy movement.

Optimizer steps reconstruct the already completed shared-action arms; they do
not create new training endpoints or new game trajectories.
"""

import json
from pathlib import Path

import numpy as np
import torch

from koikoi.published_actor import PublishedActor
from scripts.branch_policy_gradient import expected_return_loss, probabilities
from scripts.train_branch_policy_gradient import Config, Trainer, sha, write


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def assert_equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a.cpu(), b.cpu())
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_equal(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for left, right in zip(a, b):
            assert_equal(left, right)
    else:
        assert a == b


def summarize(rows):
    p = rows["probabilities"].astype(np.float64)
    q = rows["q"].astype(np.float64)
    k = rows["counts"].astype(np.float64)
    g = k[:, None] * p * (q - np.sum(p * q, axis=1, keepdims=True))
    norm2 = np.square(g).sum(1)
    sensitivity = np.maximum(0, 1 - np.square(p).sum(1))
    priority = 0.05 + np.sqrt(sensitivity)
    # A hypothetical reweighting of this finite root pool only. No claim about
    # future within-match root sampling or full network-gradient variance.
    second_moment_ratio = priority.mean() * np.mean(norm2 / priority) / norm2.mean()
    different = np.where(rows["legal"], q, -2).max(1) != np.where(
        rows["legal"], q, 2
    ).min(1)
    confident = p.max(1) >= 0.99
    return {
        "roots": len(p),
        "constant_action_outcome_fraction": float((~different).mean()),
        "max_probability_at_least_099_fraction": float(confident.mean()),
        "confident_roots_logit_gradient_energy_fraction": float(
            norm2[confident].sum() / norm2.sum()
        ),
        "priority_pool_second_moment_ratio": float(second_moment_ratio),
        "probability_max_quantiles": np.quantile(
            p.max(1), [0, 0.25, 0.5, 0.75, 0.9, 0.99, 1]
        ).tolist(),
        "entropy_quantiles": np.quantile(
            -(p * np.log(np.maximum(p, 1e-38))).sum(1),
            [0, 0.25, 0.5, 0.75, 0.9, 0.99, 1],
        ).tolist(),
        "phase_counts": np.bincount(rows["phases"], minlength=4).tolist(),
        "priority_formula": "0.05 + sqrt(max(0, 1-sum_a pi(a)^2))",
    }


def run():
    base = Path("runs/branch_policy_gradient_training")
    output = Path("runs/branch_update_coverage")
    if output.exists():
        raise ValueError("Use this saved-data diagnostic only once")
    plan, complete = read(base / "protocol.json"), read(base / "complete.json")
    audit = read(base / "audit.json")
    if audit["status"] != "passed" or complete["protocol_sha256"] != sha(
        base / "protocol.json"
    ):
        raise ValueError("Expected the completed audited training study")
    for path, digest in plan["bindings"].items():
        if sha(path) != digest:
            raise ValueError("Original numerical sources changed")
    diagnostic = Path("runs/branch_policy_gradient_diagnostic")
    diagnostic_summary = read(diagnostic / "summary.json")
    for parent_index in range(2):
        for filename in ("source.pt", "root_policy.pt"):
            relative = str(Path(f"parent{parent_index + 1}") / filename)
            if sha(diagnostic / relative) != diagnostic_summary["artifacts"][relative]:
                raise ValueError("Original diagnostic bank changed")
    checkpoint_bindings = {
        str(path): sha(path)
        for index in range(2)
        for path in (base / f"parent{index + 1}_shared_actions").glob("update_*.pt")
    }
    output.mkdir()
    write(
        output / "protocol.json",
        {
            "training_audit_sha256": sha(base / "audit.json"),
            "training_complete_sha256": sha(base / "complete.json"),
            "script_sha256": sha(__file__),
            "intermediate_checkpoint_bindings": checkpoint_bindings,
            "diagnostic_summary_sha256": sha(diagnostic / "summary.json"),
            "scope": "Post-hoc; both shared arms, all128updates; all six endpoints on the fixed64-root bank of their corresponding parent.",
            "pool_priority_fixed_before_analysis": "0.05 + sqrt(max(0, 1-sum(pi**2)))",
            "new_simulator_decisions": 0,
            "new_training_updates": 0,
        },
    )
    results = []
    for parent_index in range(2):
        name = f"parent{parent_index + 1}_shared_actions"
        case = plan["cases"][name]
        trainer = Trainer(
            plan["parents"][parent_index]["path"], Config(**case["config"])
        )
        directory = base / name
        parts = {key: [] for key in ("probabilities", "q", "counts", "phases", "legal")}
        checked = 0
        for step in range(1, plan["updates_per_arm"] + 1):
            path = directory / f"evidence_{step:04}.pt"
            if sha(path) != complete["cases"][name]["evidence"][path.name]:
                raise ValueError("Saved training evidence changed")
            data = load(path)
            states = data["source"]["states"].cuda()
            counts = data["source"]["eligible_decisions"].cuda()
            q = data["branch"]["q"].cuda()
            features, phases, legal = trainer.workspace.root_inputs(states)
            p = probabilities(trainer.actor, features, phases, legal)
            for key, value in (
                ("probabilities", p),
                ("q", q),
                ("counts", counts),
                ("phases", phases),
                ("legal", legal),
            ):
                parts[key].append(value.detach().cpu().numpy())
            trainer.optimizer.zero_grad(set_to_none=True)
            expected_return_loss(p, q, counts).backward()
            torch.nn.utils.clip_grad_norm_(
                trainer.actor.parameters(),
                trainer.config.clip_norm,
                error_if_nonfinite=True,
            )
            trainer.optimizer.step()
            if step % plan["save_interval"] == 0:
                expected = load(directory / f"update_{step:04}.pt")
                assert_equal(trainer.actor.checkpoint(), expected["actor"])
                assert_equal(trainer.optimizer.state_dict(), expected["optimizer"])
                checked += 1
        rows = {key: np.concatenate(values) for key, values in parts.items()}
        target = output / f"parent{parent_index + 1}_roots.npz"
        np.savez_compressed(target, **rows)
        result = summarize(rows)
        result["parent"] = parent_index + 1
        result["exact_actor_optimizer_checkpoint_matches"] = checked
        result["replayed_optimizer_steps"] = plan["updates_per_arm"]
        result["root_data_sha256"] = sha(target)

        bank = (
            Path("runs/branch_policy_gradient_diagnostic") / f"parent{parent_index + 1}"
        )
        bank_states = load(bank / "source.pt")["states"].cuda()
        features, phases, legal = trainer.workspace.root_inputs(bank_states)
        original = PublishedActor(
            load(plan["parents"][parent_index]["path"])["actor"], "cuda"
        )
        with torch.no_grad():
            before = probabilities(original, features, phases, legal)
            assert torch.equal(
                before.cpu(), load(bank / "root_policy.pt")["probabilities"]
            )
            movement = {}
            for mode in ("shared_actions", "independent_actions", "sampled_return"):
                endpoint_name = f"parent{parent_index + 1}_{mode}"
                endpoint = base / endpoint_name / "update_0128.pt"
                assert (
                    sha(endpoint) == complete["cases"][endpoint_name]["endpoint_sha256"]
                )
                actor = PublishedActor(load(endpoint)["actor"], "cuda")
                after = probabilities(actor, features, phases, legal)
                kl = (
                    before
                    * (before.clamp_min(1e-38).log() - after.clamp_min(1e-38).log())
                ).sum(-1)
                movement[mode] = {
                    "roots": len(before),
                    "greedy_disagreements": int(
                        (before.argmax(-1) != after.argmax(-1)).sum()
                    ),
                    "mean_kl_from_parent": float(kl.mean()),
                    "mean_total_variation": float(
                        (before - after).abs().sum(-1).mean() / 2
                    ),
                    "phase_disagreements": {
                        str(phase): int(
                            (
                                (before.argmax(-1) != after.argmax(-1))
                                & (phases == phase)
                            ).sum()
                        )
                        for phase in range(4)
                    },
                }
                del actor
        result["fixed_bank_movement"] = movement
        results.append(result)
        print(json.dumps(result), flush=True)
        del trainer, original
    write(
        output / "summary.json",
        {
            "status": "complete",
            "protocol_sha256": sha(output / "protocol.json"),
            "parents": results,
            "new_simulator_decisions": 0,
            "new_training_updates": 0,
            "replayed_optimizer_steps": 256,
        },
    )


if __name__ == "__main__":
    run()
